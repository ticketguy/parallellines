import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, String, Text, text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class NarrativeAnalysis(Base):
    """A coherent narrative pattern extracted from cross-layer signal analysis."""

    __tablename__ = "narrative_analyses"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    topic: Mapped[str] = mapped_column(String(255), index=True)
    narrative_text: Mapped[str] = mapped_column(Text)

    # NarrativeType value: emerging | dominant | fading | contrarian
    narrative_type: Mapped[str] = mapped_column(String(50))
    # 0.0–1.0: how strongly this narrative is present
    strength: Mapped[float] = mapped_column(Float)

    entities: Mapped[list | None] = mapped_column(ARRAY(String), nullable=True)
    # UUIDs of the Signal rows that support this narrative
    supporting_signal_ids: Mapped[list | None] = mapped_column(ARRAY(String), nullable=True)
    # Snapshot of each layer's score at the time this narrative was identified
    layer_scores: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )
