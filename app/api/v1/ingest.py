"""
Ingestion management endpoints.

GET  /api/v1/ingest/status          — stats + connector list
POST /api/v1/ingest/run             — trigger an immediate ingestion cycle
GET  /api/v1/ingest/sources         — list configured crawl sources
POST /api/v1/ingest/sources         — add a new crawl source
DELETE /api/v1/ingest/sources/{id}  — remove a crawl source
"""
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, HttpUrl
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.connectors.registry import get_registry
from app.database import get_db
from app.models.crawl_source import CrawlSource
from app.services.ingestion import get_stats, run_ingestion_cycle

router = APIRouter(prefix="/ingest", tags=["ingestion"])


# ── schemas ───────────────────────────────────────────────────────────────────

class CrawlSourceIn(BaseModel):
    label: str
    url: str
    layer: str = "memory"
    topic_tags: list[str] = []


class CrawlSourceOut(CrawlSourceIn):
    id: uuid.UUID
    is_active: bool

    model_config = {"from_attributes": True}


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.get("/status")
async def ingestion_status():
    """Connector registry + ingestion loop statistics."""
    registry = get_registry()
    stats = get_stats()
    return {
        "stats": stats,
        "connectors": [
            {"name": name, "layer": cls.layer, "enabled": cls.enabled}
            for name, cls in registry.items()
        ],
    }


@router.post("/run")
async def trigger_ingestion():
    """Immediately run one ingestion cycle across all active connectors."""
    result = await run_ingestion_cycle()
    return result


@router.get("/sources", response_model=list[CrawlSourceOut])
async def list_sources(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(CrawlSource).order_by(CrawlSource.label))
    return result.scalars().all()


@router.post("/sources", response_model=CrawlSourceOut, status_code=201)
async def add_source(payload: CrawlSourceIn, db: AsyncSession = Depends(get_db)):
    source = CrawlSource(
        label=payload.label,
        url=str(payload.url),
        layer=payload.layer,
        topic_tags=payload.topic_tags or [],
    )
    db.add(source)
    await db.commit()
    await db.refresh(source)
    return source


@router.delete("/sources/{source_id}", status_code=204)
async def remove_source(source_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(CrawlSource).where(CrawlSource.id == source_id))
    source = result.scalar_one_or_none()
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    source.is_active = False
    await db.commit()
