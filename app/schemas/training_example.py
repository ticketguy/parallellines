from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class TrainingExampleBase(BaseModel):
    topic: str
    time_window: str = "24h"
    context: str
    analysis: str
    source: str = "synthetic_claude"
    quality_score: float | None = Field(None, ge=0.0, le=1.0)
    layer_breakdown: dict[str, Any] | None = None
    signal_ids: list[str] | None = None
    is_validated: bool = False
    split: str = "train"


class TrainingExampleCreate(TrainingExampleBase):
    pass


class TrainingExampleRead(TrainingExampleBase):
    id: UUID
    created_at: datetime

    model_config = {"from_attributes": True}
