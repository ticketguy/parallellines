import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class ConversationSession(Base):
    """
    A single chat session with IntuOne.

    Holds the message history and topic context for one conversation.
    Multiple sessions build up the long-term MemoryEntry store over time.
    """

    __tablename__ = "conversation_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    # Auto-generated from the first user message
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # Topics discussed (for retrieval and memory tagging)
    topics: Mapped[list | None] = mapped_column(ARRAY(String), nullable=True)
    # Compressed summary of the session (written at session close or after N turns)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )

    messages: Mapped[list["ConversationMessage"]] = relationship(
        "ConversationMessage",
        back_populates="session",
        order_by="ConversationMessage.created_at",
    )


class ConversationMessage(Base):
    """
    One message turn within a ConversationSession.

    role: "user" | "assistant" | "system"
    """

    __tablename__ = "conversation_messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversation_sessions.id", ondelete="CASCADE"),
        index=True,
    )
    role: Mapped[str] = mapped_column(String(20))  # user | assistant | system
    content: Mapped[str] = mapped_column(Text)

    # Topics IntuOne identified in this turn (set on assistant messages)
    topics_referenced: Mapped[list | None] = mapped_column(ARRAY(String), nullable=True)
    # Snapshot of layer scores used when generating this response
    layer_scores_snapshot: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    # Number of signals retrieved for this turn
    signal_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )

    session: Mapped["ConversationSession"] = relationship(
        "ConversationSession", back_populates="messages"
    )


class MemoryEntry(Base):
    """
    A persistent long-term memory extracted from conversations and analysis.

    Three types:
      topic_insight   — "Markets show high confidence (87%) that X will happen"
      entity_fact     — "Entity Y has consistently been a leading indicator for Z"
      pattern         — "When market layer diverges from social layer by > 0.4, outcome is usually Y"

    Entries are retrieved by keyword/topic match and injected into the
    model's context before each generation.
    """

    __tablename__ = "memory_entries"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    # topic_insight | entity_fact | pattern
    memory_type: Mapped[str] = mapped_column(String(50), index=True)

    topic: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    entity: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)

    # The memory text injected into context
    content: Mapped[str] = mapped_column(Text)

    # 0.0–1.0 — higher = more likely to be included when context window is limited
    importance: Mapped[float] = mapped_column(Float, default=0.5)
    # Keyword list for lightweight retrieval (no vector DB needed at this stage)
    keywords: Mapped[list | None] = mapped_column(ARRAY(String), nullable=True)

    recall_count: Mapped[int] = mapped_column(Integer, default=0)
    last_recalled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Whether this memory has been superseded by a newer one
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Source session this was extracted from
    source_session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversation_sessions.id", ondelete="SET NULL"),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )
