from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.signal import Signal
from app.schemas.signal import SignalCreate, SignalRead

router = APIRouter(prefix="/signals", tags=["signals"])


@router.post("/", response_model=SignalRead, status_code=201)
async def create_signal(payload: SignalCreate, db: AsyncSession = Depends(get_db)):
    signal = Signal(**payload.model_dump())
    db.add(signal)
    await db.commit()
    await db.refresh(signal)
    return signal


@router.get("/", response_model=list[SignalRead])
async def list_signals(
    layer: str | None = None,
    source: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    q = select(Signal)
    if layer:
        q = q.where(Signal.layer == layer)
    if source:
        q = q.where(Signal.source == source)
    q = q.order_by(Signal.created_at.desc()).limit(limit).offset(offset)
    result = await db.execute(q)
    return result.scalars().all()


@router.get("/{signal_id}", response_model=SignalRead)
async def get_signal(signal_id: UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Signal).where(Signal.id == signal_id))
    signal = result.scalar_one_or_none()
    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")
    return signal
