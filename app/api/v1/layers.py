from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.layer import LayerScore
from app.schemas.layer import LayerScoreCreate, LayerScoreRead

router = APIRouter(prefix="/layers", tags=["layers"])


@router.post("/scores", response_model=LayerScoreRead, status_code=201)
async def create_layer_score(payload: LayerScoreCreate, db: AsyncSession = Depends(get_db)):
    score = LayerScore(**payload.model_dump())
    db.add(score)
    await db.commit()
    await db.refresh(score)
    return score


@router.get("/scores", response_model=list[LayerScoreRead])
async def list_layer_scores(
    layer: str | None = None,
    topic: str | None = None,
    time_window: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    q = select(LayerScore)
    if layer:
        q = q.where(LayerScore.layer == layer)
    if topic:
        q = q.where(LayerScore.topic == topic)
    if time_window:
        q = q.where(LayerScore.time_window == time_window)
    q = q.order_by(LayerScore.computed_at.desc()).limit(limit).offset(offset)
    result = await db.execute(q)
    return result.scalars().all()


@router.get("/scores/{score_id}", response_model=LayerScoreRead)
async def get_layer_score(score_id: UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(LayerScore).where(LayerScore.id == score_id))
    score = result.scalar_one_or_none()
    if not score:
        raise HTTPException(status_code=404, detail="Layer score not found")
    return score
