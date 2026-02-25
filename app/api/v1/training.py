"""
Training management endpoints.

POST /api/v1/training/generate         — run teacher model on a topic list
POST /api/v1/training/generate/markets — auto-generate from top Polymarket markets
GET  /api/v1/training/examples         — list training examples
GET  /api/v1/training/examples/{id}    — fetch one
POST /api/v1/training/briefing         — run IntuOne live (local model or teacher)
GET  /api/v1/training/status           — dataset stats + model load status
"""
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.inference.pipeline import is_model_loaded, run_intuone
from app.models.signal import Signal
from app.models.training_example import TrainingExample
from app.schemas.signal import SignalRead
from app.schemas.training_example import TrainingExampleRead
from app.training.generator import batch_generate

router = APIRouter(prefix="/training", tags=["training"])


# ── request / response bodies ─────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    topics: list[str]
    time_window: str = "24h"


class BriefingRequest(BaseModel):
    topic: str
    time_window: str = "24h"
    # If True, always use teacher (Claude) even if local model is loaded
    force_teacher: bool = False


class GenerateResponse(BaseModel):
    generated: int
    failed: int
    details: dict[str, int]


class BriefingResponse(BaseModel):
    topic: str
    time_window: str
    overall_score: float
    confidence: float
    layer_scores: dict
    briefing: str
    model_used: str
    signal_count: int


class StatusResponse(BaseModel):
    local_model_loaded: bool
    training_examples: dict[str, int]   # split → count
    total_signals: int


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.post("/generate", response_model=GenerateResponse)
async def generate_training_data(
    payload: GenerateRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    For each topic, fetch signals from the DB, score all layers,
    call Claude Sonnet (teacher) to generate a gold briefing, and
    store the result as a TrainingExample.
    """
    results = await batch_generate(payload.topics, db, payload.time_window)
    generated = sum(v for v in results.values() if v == 1)
    return GenerateResponse(
        generated=generated,
        failed=len(results) - generated,
        details=results,
    )


@router.post("/generate/markets", response_model=GenerateResponse)
async def generate_from_top_markets(
    time_window: str = "24h",
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
):
    """
    Auto-discover topics from the top Polymarket signals in the DB
    and generate training examples for each.
    """
    result = await db.execute(
        select(Signal.topic_tags)
        .where(Signal.layer == "market")
        .where(Signal.topic_tags != None)  # noqa: E711
        .order_by(Signal.created_at.desc())
        .limit(200)
    )
    # Flatten tags and deduplicate
    all_tags: list[str] = []
    for (tags,) in result.all():
        if tags:
            all_tags.extend(tags)

    topics = list(dict.fromkeys(t for t in all_tags if t))[:limit]
    if not topics:
        raise HTTPException(
            status_code=422,
            detail="No topic tags found in market signals. "
                   "Run the Polymarket submind first to ingest data.",
        )

    results = await batch_generate(topics, db, time_window)
    generated = sum(v for v in results.values() if v == 1)
    return GenerateResponse(
        generated=generated,
        failed=len(results) - generated,
        details=results,
    )


@router.get("/examples", response_model=list[TrainingExampleRead])
async def list_examples(
    split: str | None = None,
    topic: str | None = None,
    validated_only: bool = False,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    q = select(TrainingExample)
    if split:
        q = q.where(TrainingExample.split == split)
    if topic:
        q = q.where(TrainingExample.topic == topic)
    if validated_only:
        q = q.where(TrainingExample.is_validated == True)  # noqa: E712
    q = q.order_by(TrainingExample.created_at.desc()).limit(limit).offset(offset)
    result = await db.execute(q)
    return result.scalars().all()


@router.get("/examples/{example_id}", response_model=TrainingExampleRead)
async def get_example(example_id: UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(TrainingExample).where(TrainingExample.id == example_id)
    )
    ex = result.scalar_one_or_none()
    if not ex:
        raise HTTPException(status_code=404, detail="Training example not found")
    return ex


@router.post("/briefing", response_model=BriefingResponse)
async def live_briefing(
    payload: BriefingRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Generate a live IntuOne briefing for a topic.
    Uses the fine-tuned local model if loaded, otherwise falls back to Claude.
    """
    result = await db.execute(
        select(Signal)
        .where(Signal.topic_tags.any(payload.topic))  # type: ignore[attr-defined]
        .order_by(Signal.created_at.desc())
        .limit(50)
    )
    raw_signals = result.scalars().all()
    signals = [SignalRead.model_validate(s) for s in raw_signals]

    output = await run_intuone(
        topic=payload.topic,
        signals=signals,
        time_window=payload.time_window,
        force_teacher=payload.force_teacher,
    )
    return BriefingResponse(**output)


@router.get("/status", response_model=StatusResponse)
async def training_status(db: AsyncSession = Depends(get_db)):
    """Return dataset counts per split and whether the local model is loaded."""
    split_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        result = await db.execute(
            select(func.count()).where(TrainingExample.split == split)
        )
        split_counts[split] = result.scalar_one()

    total_signals_result = await db.execute(select(func.count()).select_from(Signal))
    total_signals = total_signals_result.scalar_one()

    return StatusResponse(
        local_model_loaded=is_model_loaded(),
        training_examples=split_counts,
        total_signals=total_signals,
    )
