"""
Chat endpoint — the conversational interface to IntuOne.

POST /api/v1/chat/message   — send a message, get a briefing back
GET  /api/v1/chat/sessions  — list sessions
GET  /api/v1/chat/sessions/{id}          — session + messages
DELETE /api/v1/chat/sessions/{id}/close  — summarise + close
GET  /api/v1/chat/memory    — browse long-term memory entries
DELETE /api/v1/chat/memory/{id}          — retire a memory
"""
import logging
import re
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.memory.extraction import extract_memories, summarise_session
from app.memory.retrieval import (
    format_history_for_context,
    format_memories_for_context,
    get_session_history,
    retrieve_memories,
)
from app.models.memory import ConversationMessage, ConversationSession, MemoryEntry
from app.models.signal import Signal
from app.schemas.memory import (
    ConversationSessionDetail,
    ConversationSessionRead,
    MemoryEntryCreate,
    MemoryEntryRead,
    MessageIn,
    MessageOut,
)
from app.schemas.signal import SignalRead
from app.inference.pipeline import run_intuone

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

# Summarise a session after this many assistant turns
_SUMMARISE_EVERY_N_TURNS = 10


# ── helpers ───────────────────────────────────────────────────────────────────

def _extract_topics(text: str) -> list[str]:
    """
    Lightweight topic extraction: return capitalised noun phrases and
    any quoted terms from the user's message as topic candidates.
    """
    quoted = re.findall(r'"([^"]+)"', text)
    # Simple heuristic: title-cased runs of words (proper nouns)
    titled = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)
    topics = list(dict.fromkeys(quoted + titled))  # deduplicate, preserve order
    return topics[:8]


async def _get_or_create_session(
    session_id: UUID | None,
    db: AsyncSession,
    title: str | None = None,
) -> ConversationSession:
    if session_id:
        result = await db.execute(
            select(ConversationSession).where(ConversationSession.id == session_id)
        )
        session = result.scalar_one_or_none()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session

    session = ConversationSession(title=title)
    db.add(session)
    await db.flush()  # get the id without committing
    return session


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.post("/message", response_model=MessageOut)
async def chat_message(payload: MessageIn, db: AsyncSession = Depends(get_db)):
    """
    Main chat turn. Full memory-augmented flow:

    1. Load / create session
    2. Extract topics from the user's message
    3. Retrieve relevant long-term memories + session history
    4. Fetch live signals for the identified topics
    5. Build enriched context (memories + history + signals + user message)
    6. Generate briefing (local model or teacher fallback)
    7. Persist messages + extract new memories asynchronously
    """
    # 1. Session
    session = await _get_or_create_session(
        payload.session_id,
        db,
        title=payload.content[:80],
    )

    # 2. Topics + keywords from user message
    topics = _extract_topics(payload.content)
    keywords = [w.lower() for w in payload.content.split() if len(w) > 4][:20]

    # 3. Memory retrieval
    memories = await retrieve_memories(db, topics, keywords)
    history = await get_session_history(db, session.id)

    memory_block = format_memories_for_context(memories)
    history_block = format_history_for_context(history)

    # 4. Live signals for identified topics
    signals: list[SignalRead] = []
    if topics:
        result = await db.execute(
            select(Signal)
            .where(Signal.topic_tags.overlap(topics))  # type: ignore[attr-defined]
            .order_by(Signal.created_at.desc())
            .limit(40)
        )
        signals = [SignalRead.model_validate(s) for s in result.scalars().all()]

    # 5. Build enriched user content
    topic_str = ", ".join(topics) if topics else payload.content[:60]
    enriched_content = ""
    if memory_block:
        enriched_content += memory_block + "\n"
    if history_block:
        enriched_content += history_block + "\n"
    enriched_content += f"USER QUERY: {payload.content}"

    # 6. Generate briefing
    output = await run_intuone(
        topic=topic_str,
        signals=signals,
        time_window="24h",
        user_message=payload.content,
        extra_context=enriched_content if (memory_block or history_block) else None,
    )

    # 7a. Persist user message
    user_msg = ConversationMessage(
        session_id=session.id,
        role="user",
        content=payload.content,
    )
    db.add(user_msg)

    # 7b. Persist assistant message
    assistant_msg = ConversationMessage(
        session_id=session.id,
        role="assistant",
        content=output["briefing"],
        topics_referenced=topics or None,
        layer_scores_snapshot=output.get("layer_scores"),
        signal_count=output.get("signal_count", 0),
    )
    db.add(assistant_msg)

    # 7c. Update session topics
    existing_topics = set(session.topics or [])
    session.topics = list(existing_topics | set(topics))

    # 7d. Auto-summarise at N turns
    all_messages = list(history) + [user_msg, assistant_msg]
    assistant_turns = sum(1 for m in all_messages if m.role == "assistant")
    if assistant_turns > 0 and assistant_turns % _SUMMARISE_EVERY_N_TURNS == 0:
        session.summary = await summarise_session(all_messages, topic_str)

    await db.commit()
    await db.refresh(assistant_msg)

    # 7e. Extract long-term memories from the response (best-effort, non-blocking)
    try:
        new_memories = await extract_memories(output["briefing"], session_id=session.id)
        for m in new_memories:
            db.add(m)
        await db.commit()
    except Exception as exc:
        logger.warning("Memory extraction failed (non-fatal): %s", exc)

    return MessageOut(
        session_id=session.id,
        message_id=assistant_msg.id,
        content=output["briefing"],
        topics_referenced=topics or None,
        layer_scores_snapshot=output.get("layer_scores"),
        signal_count=output.get("signal_count", 0),
        model_used=output.get("model_used", "unknown"),
    )


@router.get("/sessions", response_model=list[ConversationSessionRead])
async def list_sessions(
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ConversationSession)
        .order_by(ConversationSession.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return result.scalars().all()


@router.get("/sessions/{session_id}", response_model=ConversationSessionDetail)
async def get_session(session_id: UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ConversationSession)
        .options(selectinload(ConversationSession.messages))
        .where(ConversationSession.id == session_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.delete("/sessions/{session_id}/close", response_model=ConversationSessionRead)
async def close_session(session_id: UUID, db: AsyncSession = Depends(get_db)):
    """Summarise and close a session."""
    result = await db.execute(
        select(ConversationSession)
        .options(selectinload(ConversationSession.messages))
        .where(ConversationSession.id == session_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session.summary and session.messages:
        session.summary = await summarise_session(list(session.messages))

    await db.commit()
    await db.refresh(session)
    return session


@router.get("/memory", response_model=list[MemoryEntryRead])
async def list_memories(
    topic: str | None = None,
    memory_type: str | None = None,
    active_only: bool = True,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    q = select(MemoryEntry)
    if topic:
        q = q.where(MemoryEntry.topic == topic)
    if memory_type:
        q = q.where(MemoryEntry.memory_type == memory_type)
    if active_only:
        q = q.where(MemoryEntry.is_active == True)  # noqa: E712
    q = q.order_by(MemoryEntry.importance.desc(), MemoryEntry.recall_count.desc()).limit(limit)
    result = await db.execute(q)
    return result.scalars().all()


@router.post("/memory", response_model=MemoryEntryRead, status_code=201)
async def create_memory(payload: MemoryEntryCreate, db: AsyncSession = Depends(get_db)):
    """Manually add a long-term memory (e.g. from a human analyst)."""
    entry = MemoryEntry(**payload.model_dump())
    db.add(entry)
    await db.commit()
    await db.refresh(entry)
    return entry


@router.delete("/memory/{memory_id}", status_code=204)
async def retire_memory(memory_id: UUID, db: AsyncSession = Depends(get_db)):
    """Soft-delete a memory (marks is_active=False)."""
    result = await db.execute(select(MemoryEntry).where(MemoryEntry.id == memory_id))
    entry = result.scalar_one_or_none()
    if not entry:
        raise HTTPException(status_code=404, detail="Memory not found")
    entry.is_active = False
    await db.commit()
