from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.market import PredictionMarket
from app.schemas.market import PredictionMarketCreate, PredictionMarketRead

router = APIRouter(prefix="/markets", tags=["markets"])


@router.post("/", response_model=PredictionMarketRead, status_code=201)
async def create_market(payload: PredictionMarketCreate, db: AsyncSession = Depends(get_db)):
    market = PredictionMarket(**payload.model_dump())
    db.add(market)
    await db.commit()
    await db.refresh(market)
    return market


@router.get("/", response_model=list[PredictionMarketRead])
async def list_markets(
    category: str | None = None,
    active_only: bool = True,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    q = select(PredictionMarket)
    if active_only:
        q = q.where(PredictionMarket.is_active == True)  # noqa: E712
    if category:
        q = q.where(PredictionMarket.category == category)
    q = q.order_by(PredictionMarket.volume_24h.desc().nullslast()).limit(limit).offset(offset)
    result = await db.execute(q)
    return result.scalars().all()


@router.get("/{market_id}", response_model=PredictionMarketRead)
async def get_market(market_id: UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(PredictionMarket).where(PredictionMarket.id == market_id)
    )
    market = result.scalar_one_or_none()
    if not market:
        raise HTTPException(status_code=404, detail="Market not found")
    return market
