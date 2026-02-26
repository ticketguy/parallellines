"""Helper to load active CrawlSource rows for the web crawler connector."""
from sqlalchemy import select

from app.database import AsyncSessionLocal
from app.models.crawl_source import CrawlSource


async def load_active_sources() -> list[dict]:
    """Return active CrawlSource rows as dicts for the web crawler."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(CrawlSource).where(CrawlSource.is_active == True)  # noqa: E712
        )
        rows = result.scalars().all()
        return [
            {
                "url": r.url,
                "label": r.label,
                "layer": r.layer,
                "domain": r.domain,  # free-form world layer, e.g. "news", "crypto"
                "topic_tags": r.topic_tags or [],
            }
            for r in rows
        ]
