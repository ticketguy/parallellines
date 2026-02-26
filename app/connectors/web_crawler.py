"""
Web crawler connector — fetches and parses arbitrary web pages.

Configured via a list of URLs + topic tags in settings or via the
/api/v1/ingest/sources API (stored in DB).

Each page is fetched, cleaned to plain text, scored for sentiment,
and stored as a NEWS or SOCIAL layer signal depending on the source type.

To add a news source:
    POST /api/v1/ingest/sources
    {
        "url": "https://example.com/rss-or-page",
        "topic_tags": ["bitcoin", "crypto"],
        "layer": "news",
        "label": "Example News"
    }
"""
import hashlib
import logging
import re

import httpx

from app.constants import LayerType
from app.connectors.base import BaseConnector, RawSignal
from app.connectors.registry import register_connector

logger = logging.getLogger(__name__)

# Default pages to crawl if no DB sources configured
_DEFAULT_SOURCES: list[dict] = []  # empty by default — add via API

_USER_AGENT = (
    "Mozilla/5.0 (compatible; IntuOneBot/1.0; +https://github.com/parallellines)"
)


def _clean_html(html: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    # Remove scripts + styles
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove all tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_sentences(text: str, max_chars: int = 2000) -> str:
    """Return the first max_chars characters of plain text."""
    return text[:max_chars]


def _rough_sentiment(text: str) -> float:
    """
    Very lightweight sentiment proxy — just counts positive/negative words.
    Replaced by the NLP layer scorer for real analysis; this is a fast hint.
    """
    positive = {"surge", "rally", "gain", "rise", "bullish", "win", "record",
                 "up", "growth", "strong", "high", "beat", "exceed"}
    negative = {"crash", "fall", "drop", "decline", "bearish", "lose", "loss",
                 "low", "weak", "miss", "fail", "collapse", "down"}
    words = set(re.findall(r"\b[a-z]+\b", text.lower()))
    pos = len(words & positive)
    neg = len(words & negative)
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 3)


@register_connector
class WebCrawlerConnector(BaseConnector):
    """
    Generic web crawler. Fetches pages from a configurable list of URLs.

    Sources are loaded from the DB (CrawlSource table) + any hardcoded
    defaults. New sources can be added via the /api/v1/ingest/sources API
    without restarting the server.
    """

    name = "web_crawler"
    layer = LayerType.MEMORY  # default; overridden per-source
    enabled = True

    def __init__(self, sources: list[dict] | None = None):
        # sources: [{"url": str, "topic_tags": [...], "layer": str, "label": str}]
        self._static_sources: list[dict] = sources or _DEFAULT_SOURCES

    @classmethod
    def from_env(cls) -> "WebCrawlerConnector":
        return cls()

    async def _fetch_page(
        self,
        client: httpx.AsyncClient,
        source: dict,
    ) -> RawSignal | None:
        url: str = source["url"]
        try:
            resp = await client.get(url, follow_redirects=True)
            resp.raise_for_status()
            html = resp.text
        except Exception as exc:
            logger.debug("[web_crawler] Failed to fetch %s: %s", url, exc)
            return None

        text = _clean_html(html)
        snippet = _extract_sentences(text)
        sentiment = _rough_sentiment(snippet)
        layer = source.get("layer", LayerType.MEMORY)
        domain = source.get("domain")  # free-form, e.g. "news", "social", "crypto"
        topic_tags = source.get("topic_tags", [])
        label = source.get("label", url)

        # Stable external ID so we don't re-ingest identical content
        content_hash = hashlib.md5(snippet[:500].encode()).hexdigest()[:12]
        external_id = f"web:{content_hash}"

        return RawSignal(
            source="web_crawler",
            layer=layer,
            domain=domain,
            topic_tags=topic_tags,
            signal_strength=abs(sentiment),
            raw_data={"url": url, "label": label, "text_length": len(text)},
            processed_data={
                "text": snippet,
                "headline": label,
                "sentiment_score": sentiment,
                "url": url,
            },
            url=url,
            external_id=external_id,
        )

    async def fetch(self) -> list[RawSignal]:
        # Load sources from DB (if available) + static list
        sources = list(self._static_sources)
        try:
            from app.services.crawl_sources import load_active_sources
            db_sources = await load_active_sources()
            sources.extend(db_sources)
        except Exception:
            pass  # DB not available or table not yet created

        if not sources:
            logger.debug("[web_crawler] No sources configured — skipping.")
            return []

        signals: list[RawSignal] = []
        async with httpx.AsyncClient(
            headers={"User-Agent": _USER_AGENT}, timeout=20
        ) as client:
            for source in sources:
                sig = await self._fetch_page(client, source)
                if sig:
                    signals.append(sig)

        logger.info("[web_crawler] Crawled %d pages, got %d signals", len(sources), len(signals))
        return signals
