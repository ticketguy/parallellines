from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel


class PredictionMarketBase(BaseModel):
    external_id: str
    question: str
    description: str | None = None
    category: str | None = None
    yes_price: float | None = None
    no_price: float | None = None
    volume_24h: float | None = None
    total_volume: float | None = None
    liquidity: float | None = None
    end_date: datetime | None = None
    resolution: str | None = None
    is_active: bool = True
    raw_data: dict[str, Any] | None = None


class PredictionMarketCreate(PredictionMarketBase):
    pass


class PredictionMarketRead(PredictionMarketBase):
    id: UUID
    fetched_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
