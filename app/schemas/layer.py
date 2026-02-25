from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class LayerScoreBase(BaseModel):
    layer: str
    topic: str
    score: float = Field(..., ge=-1.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    signal_count: int = 0
    time_window: str = "24h"
    metadata: dict[str, Any] | None = None


class LayerScoreCreate(LayerScoreBase):
    pass


class LayerScoreRead(LayerScoreBase):
    id: UUID
    computed_at: datetime

    model_config = {"from_attributes": True}
