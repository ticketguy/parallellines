from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class NarrativeBase(BaseModel):
    topic: str
    narrative_text: str
    narrative_type: str
    strength: float = Field(..., ge=0.0, le=1.0)
    entities: list[str] | None = None
    supporting_signal_ids: list[str] | None = None
    layer_scores: dict[str, Any] | None = None


class NarrativeCreate(NarrativeBase):
    pass


class NarrativeRead(NarrativeBase):
    id: UUID
    created_at: datetime

    model_config = {"from_attributes": True}
