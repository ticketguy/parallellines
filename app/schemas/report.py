from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class IntuOneReportBase(BaseModel):
    topic: str
    overall_score: float = Field(..., ge=-1.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    layer_breakdown: dict[str, Any]
    key_signals: list[str] | None = None
    narrative_summary: str | None = None
    prediction: str | None = None
    time_window: str = "24h"
    extra_data: dict[str, Any] | None = None


class IntuOneReportCreate(IntuOneReportBase):
    pass


class IntuOneReportRead(IntuOneReportBase):
    id: UUID
    generated_at: datetime

    model_config = {"from_attributes": True}
