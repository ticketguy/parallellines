import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, String, Text, text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class IntuOneReport(Base):
    """Final synthesised output from the IntuOne perception engine."""

    __tablename__ = "intuone_reports"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    topic: Mapped[str] = mapped_column(String(255), index=True)

    # -1.0 (strongly negative) to +1.0 (strongly positive)
    overall_score: Mapped[float] = mapped_column(Float)
    # 0.0–1.0
    confidence: Mapped[float] = mapped_column(Float)

    # {layer_name: {score, confidence, signal_count}}
    layer_breakdown: Mapped[dict] = mapped_column(JSONB)
    # UUIDs of the most influential signals
    key_signals: Mapped[list | None] = mapped_column(ARRAY(String), nullable=True)

    narrative_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    prediction: Mapped[str | None] = mapped_column(Text, nullable=True)

    time_window: Mapped[str] = mapped_column(String(20), default="24h")
    extra_data: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )
