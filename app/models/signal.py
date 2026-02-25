import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, String, text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Signal(Base):
    """Raw signal ingested by a submind agent from any data source."""

    __tablename__ = "signals"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    # e.g. "polymarket", "twitter", "reuters", "reddit"
    source: Mapped[str] = mapped_column(String(100), index=True)
    # LayerType value — which of the 6 layers this signal belongs to
    layer: Mapped[str] = mapped_column(String(50), index=True)

    raw_data: Mapped[dict] = mapped_column(JSONB)
    processed_data: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # 0.0–1.0: how strong/relevant the signal is
    signal_strength: Mapped[float | None] = mapped_column(Float, nullable=True)
    # 0.0–1.0: how confident the submind is in this signal
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    topic_tags: Mapped[list | None] = mapped_column(ARRAY(String), nullable=True)
    entity_tags: Mapped[list | None] = mapped_column(ARRAY(String), nullable=True)

    # Original timestamp of the source event
    signal_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )
