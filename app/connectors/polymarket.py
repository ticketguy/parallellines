"""
Polymarket connector — live prediction market data.

API: https://gamma-api.polymarket.com/markets
No API key required for public market data.

Fetches the top markets by 24-hour volume, normalises them into
MARKET-layer RawSignal dicts, and tags topics automatically from
the market question + category tags.
"""
import json
import logging
import re

import httpx

from app.config import settings
from app.constants import LayerType
from app.connectors.base import BaseConnector, RawSignal
from app.connectors.registry import register_connector

logger = logging.getLogger(__name__)

# Top N markets by 24h volume to ingest per cycle
_FETCH_LIMIT = 100
_GAMMA_URL = "https://gamma-api.polymarket.com"


def _extract_topics(question: str, tags: list[str]) -> list[str]:
    """Derive topic tags from a market question + category tags."""
    topics: list[str] = []

    # Use provided tags first (lowercase + slugified)
    for tag in tags:
        clean = re.sub(r"[^a-z0-9 ]", "", tag.lower()).strip()
        if clean and len(clean) > 2:
            topics.append(clean)

    # Extract capitalised proper-noun phrases from the question
    proper = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", question)
    topics.extend(p.lower() for p in proper)

    # Deduplicate, keep order, cap at 8
    seen: set[str] = set()
    result: list[str] = []
    for t in topics:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result[:8]


def _market_to_signal(market: dict) -> RawSignal | None:
    """Convert a single Polymarket market dict into a RawSignal."""
    try:
        question = market.get("question", "") or market.get("title", "")
        if not question:
            return None

        # outcomePrices is a JSON string: '["0.73", "0.27"]'
        raw_prices = market.get("outcomePrices", "[]")
        prices = json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices
        yes_price = float(prices[0]) if prices else 0.5

        raw_outcomes = market.get("outcomes", '["Yes","No"]')
        outcomes = json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else raw_outcomes

        volume_24h = float(market.get("volume24hr", 0) or 0)
        volume_total = float(market.get("volume", 0) or 0)
        liquidity = float(market.get("liquidity", 0) or 0)

        # Signal strength: combination of YES conviction + liquidity proxy
        # Scale: yes_price drives direction; confidence = distance from 0.5 * liquidity weight
        conviction = abs(yes_price - 0.5) * 2  # 0 at 50/50, 1 at 100% or 0%
        liq_weight = min(liquidity / 50_000, 1.0) if liquidity else 0.3
        signal_strength = conviction * (0.6 + 0.4 * liq_weight)

        # Directional score: positive = YES favoured, negative = NO favoured
        directional_score = (yes_price - 0.5) * 2  # -1 to +1

        tags_raw = market.get("tags", []) or []
        topics = _extract_topics(question, tags_raw)

        external_id = f"polymarket:{market.get('id', question[:40])}"

        return RawSignal(
            source="polymarket",
            layer=LayerType.PROBABILITY,
            topic_tags=topics,
            signal_strength=round(signal_strength, 4),
            raw_data={
                "id": market.get("id"),
                "question": question,
                "outcomes": outcomes,
                "outcomePrices": prices,
                "volume": volume_total,
                "volume24hr": volume_24h,
                "liquidity": liquidity,
            },
            processed_data={
                "question": question,
                "yes_price": round(yes_price, 4),
                "no_price": round(1 - yes_price, 4),
                "outcomes": outcomes,
                "volume_24h": volume_24h,
                "liquidity": liquidity,
                "directional_score": round(directional_score, 4),
            },
            url=f"https://polymarket.com/event/{market.get('slug', '')}",
            external_id=external_id,
        )
    except Exception as exc:
        logger.debug("Skipping malformed Polymarket market: %s", exc)
        return None


@register_connector
class PolymarketConnector(BaseConnector):
    """
    Fetches top prediction markets from Polymarket's public Gamma API.
    No API key needed. Sorted by 24h volume descending.
    """

    name = "polymarket"
    layer = LayerType.PROBABILITY
    enabled = True

    def __init__(self, gamma_url: str = _GAMMA_URL, limit: int = _FETCH_LIMIT):
        self.gamma_url = gamma_url
        self.limit = limit

    @classmethod
    def from_env(cls) -> "PolymarketConnector":
        return cls(gamma_url=settings.GAMMA_API_URL)

    async def fetch(self) -> list[RawSignal]:
        url = (
            f"{self.gamma_url}/markets"
            f"?closed=false&limit={self.limit}&order=volume24hr&ascending=false"
        )
        logger.info("[polymarket] Fetching top %d markets …", self.limit)
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                markets: list[dict] = resp.json()
        except Exception as exc:
            logger.error("[polymarket] Fetch failed: %s", exc)
            return []

        signals: list[RawSignal] = []
        for m in markets:
            sig = _market_to_signal(m)
            if sig:
                signals.append(sig)

        logger.info("[polymarket] Ingested %d signals", len(signals))
        return signals
