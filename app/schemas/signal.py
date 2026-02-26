from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class SignalBase(BaseModel):
    source: str
    layer: str          # perception layer (probability, conviction, echo, memory, shadow)
    domain: str | None = None  # world layer — free-form, e.g. "market", "news",
                               # "social", "crypto". No enum; add new domains freely.
    raw_data: dict[str, Any]
    processed_data: dict[str, Any] | None = None
    signal_strength: float | None = Field(None, ge=0.0, le=1.0)
    confidence: float | None = Field(None, ge=0.0, le=1.0)
    topic_tags: list[str] | None = None
    entity_tags: list[str] | None = None
    signal_timestamp: datetime


class SignalCreate(SignalBase):
    pass


class SignalRead(SignalBase):
    id: UUID
    created_at: datetime

    model_config = {"from_attributes": True}
