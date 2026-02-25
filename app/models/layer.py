import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class LayerScore(Base):
    """Aggregated directional score produced by a single layer processor."""

    __tablename__ = "layer_scores"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    # LayerType value
    layer: Mapped[str] = mapped_column(String(50), index=True)
    topic: Mapped[str] = mapped_column(String(255), index=True)

    # -1.0 (strongly bearish/negative) to +1.0 (strongly bullish/positive)
    score: Mapped[float] = mapped_column(Float)
    # 0.0–1.0
    confidence: Mapped[float] = mapped_column(Float)
    signal_count: Mapped[int] = mapped_column(Integer, default=0)

    # e.g. "1h", "24h", "7d"
    time_window: Mapped[str] = mapped_column(String(20), default="24h")
    metadata: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )
