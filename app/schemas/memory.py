from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


# ── Conversation ──────────────────────────────────────────────────────────────

class MessageIn(BaseModel):
    """Incoming user message to the chat endpoint."""
    content: str
    # If omitted, a new session is created
    session_id: UUID | None = None


class MessageOut(BaseModel):
    """Response from IntuOne."""
    session_id: UUID
    message_id: UUID
    role: str = "assistant"
    content: str
    topics_referenced: list[str] | None = None
    layer_scores_snapshot: dict[str, Any] | None = None
    signal_count: int | None = None
    model_used: str


class ConversationMessageRead(BaseModel):
    id: UUID
    session_id: UUID
    role: str
    content: str
    topics_referenced: list[str] | None = None
    layer_scores_snapshot: dict[str, Any] | None = None
    signal_count: int | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationSessionRead(BaseModel):
    id: UUID
    title: str | None = None
    topics: list[str] | None = None
    summary: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ConversationSessionDetail(ConversationSessionRead):
    messages: list[ConversationMessageRead] = []


# ── Memory entries ────────────────────────────────────────────────────────────

class MemoryEntryCreate(BaseModel):
    memory_type: str
    topic: str | None = None
    entity: str | None = None
    content: str
    importance: float = Field(0.5, ge=0.0, le=1.0)
    keywords: list[str] | None = None


class MemoryEntryRead(MemoryEntryCreate):
    id: UUID
    recall_count: int
    last_recalled_at: datetime | None = None
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}
