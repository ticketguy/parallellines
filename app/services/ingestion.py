"""
Background ingestion service.

Polls every registered connector on a fixed interval, deduplicates
against existing signals, and stores new ones in the DB.

Started automatically in the FastAPI lifespan (app/main.py).
Can also be triggered on-demand via POST /api/v1/ingest/run.
"""
import asyncio
import logging
from datetime import datetime, timezone

from sqlalchemy import select

from app.config import settings
from app.connectors import get_active_connectors
from app.connectors.base import RawSignal
from app.database import AsyncSessionLocal
from app.models.signal import Signal

logger = logging.getLogger(__name__)

# Ingestion statistics (in-memory, reset on restart)
_stats: dict = {
    "last_run_at": None,
    "total_ingested": 0,
    "total_skipped": 0,
    "runs": 0,
    "errors": 0,
}


def get_stats() -> dict:
    return dict(_stats)


async def _upsert_signal(db, raw: RawSignal) -> bool:
    """
    Insert a signal if it doesn't already exist (dedup by external_id).
    Returns True if inserted, False if skipped.
    """
    external_id = raw.get("external_id")

    # Dedup check
    if external_id:
        existing = await db.execute(
            select(Signal).where(Signal.external_id == external_id)
        )
        if existing.scalar_one_or_none():
            return False

    signal = Signal(
        source=raw["source"],
        layer=raw["layer"],
        topic_tags=raw.get("topic_tags", []),
        signal_strength=raw.get("signal_strength"),
        raw_data=raw.get("raw_data", {}),
        processed_data=raw.get("processed_data", {}),
        url=raw.get("url"),
        external_id=external_id,
    )
    db.add(signal)
    return True


async def run_ingestion_cycle() -> dict:
    """
    Run one full ingestion cycle across all active connectors.
    Returns a summary dict.
    """
    connectors = get_active_connectors()
    if not connectors:
        logger.info("[ingestion] No active connectors — nothing to do.")
        return {"connectors": 0, "ingested": 0, "skipped": 0, "errors": 0}

    ingested = 0
    skipped = 0
    errors = 0

    for connector in connectors:
        logger.info("[ingestion] Polling connector: %s", connector.name)
        try:
            raw_signals = await connector.fetch()
        except Exception as exc:
            logger.error("[ingestion] Connector %r failed: %s", connector.name, exc)
            errors += 1
            continue

        if not raw_signals:
            continue

        async with AsyncSessionLocal() as db:
            for raw in raw_signals:
                try:
                    inserted = await _upsert_signal(db, raw)
                    if inserted:
                        ingested += 1
                    else:
                        skipped += 1
                except Exception as exc:
                    logger.debug("[ingestion] Signal store failed: %s", exc)
                    errors += 1

            try:
                await db.commit()
            except Exception as exc:
                logger.error("[ingestion] DB commit failed: %s", exc)
                errors += 1

    now = datetime.now(timezone.utc).isoformat()
    _stats["last_run_at"] = now
    _stats["total_ingested"] += ingested
    _stats["total_skipped"] += skipped
    _stats["runs"] += 1
    _stats["errors"] += errors

    logger.info(
        "[ingestion] Cycle done — ingested=%d skipped=%d errors=%d",
        ingested, skipped, errors,
    )
    return {
        "connectors": len(connectors),
        "ingested": ingested,
        "skipped": skipped,
        "errors": errors,
        "at": now,
    }


async def ingestion_loop() -> None:
    """
    Infinite background loop — polls connectors every INGEST_INTERVAL_SECONDS.
    Runs as an asyncio task created in the FastAPI lifespan.
    """
    logger.info(
        "[ingestion] Background loop started (interval=%ds)",
        settings.INGEST_INTERVAL_SECONDS,
    )
    # Initial run immediately on startup
    await asyncio.sleep(5)
    while True:
        try:
            await run_ingestion_cycle()
        except Exception as exc:
            logger.error("[ingestion] Unexpected loop error: %s", exc)
        await asyncio.sleep(settings.INGEST_INTERVAL_SECONDS)
