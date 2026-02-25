"""
Memory retrieval — fetches relevant MemoryEntry rows and recent
ConversationMessages to inject into the model's context.

Uses keyword overlap for semantic retrieval (no vector DB needed at this
stage — add pgvector later for embedding-based search).
"""
import logging
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.memory import ConversationMessage, ConversationSession, MemoryEntry

logger = logging.getLogger(__name__)

# Maximum memories to inject per call
MAX_MEMORIES = 6
# Maximum recent conversation turns to include
MAX_HISTORY_TURNS = 10


async def retrieve_memories(
    db: AsyncSession,
    topics: list[str],
    query_keywords: list[str],
    max_results: int = MAX_MEMORIES,
) -> list[MemoryEntry]:
    """
    Find active MemoryEntry rows relevant to the current query.

    Scores by:
      1. Topic match (topic column matches any of the query topics)
      2. Keyword overlap (keywords array overlaps with query_keywords)
      3. Importance (tiebreaker)

    Updates recall_count + last_recalled_at on returned entries.
    """
    # Pull all active memories matching topic or any keyword
    all_keywords = list(set(topics + query_keywords))

    q = (
        select(MemoryEntry)
        .where(MemoryEntry.is_active == True)  # noqa: E712
        .where(
            MemoryEntry.topic.in_(topics)
            | MemoryEntry.keywords.overlap(all_keywords)  # type: ignore[attr-defined]
        )
        .order_by(MemoryEntry.importance.desc(), MemoryEntry.recall_count.asc())
        .limit(max_results * 2)  # fetch extra, re-rank locally
    )
    result = await db.execute(q)
    candidates = result.scalars().all()

    # Local re-rank: score = importance + 0.1 * keyword_overlap_count
    def _score(m: MemoryEntry) -> float:
        overlap = len(set(m.keywords or []) & set(all_keywords))
        topic_bonus = 0.2 if m.topic in topics else 0.0
        return float(m.importance) + 0.05 * overlap + topic_bonus

    ranked = sorted(candidates, key=_score, reverse=True)[:max_results]

    # Mark as recalled
    if ranked:
        ids = [m.id for m in ranked]
        await db.execute(
            update(MemoryEntry)
            .where(MemoryEntry.id.in_(ids))
            .values(
                recall_count=MemoryEntry.recall_count + 1,
                last_recalled_at=datetime.now(timezone.utc),
            )
        )
        await db.commit()

    return ranked


async def get_session_history(
    db: AsyncSession,
    session_id,
    max_turns: int = MAX_HISTORY_TURNS,
) -> list[ConversationMessage]:
    """Return the most recent N messages for a session, oldest first."""
    result = await db.execute(
        select(ConversationMessage)
        .where(ConversationMessage.session_id == session_id)
        .order_by(ConversationMessage.created_at.desc())
        .limit(max_turns)
    )
    messages = result.scalars().all()
    return list(reversed(messages))  # chronological order


def format_memories_for_context(memories: list[MemoryEntry]) -> str:
    """Render retrieved memories as a context block."""
    if not memories:
        return ""

    lines = ["[LONG-TERM MEMORY]"]
    for m in memories:
        tag = m.memory_type.upper().replace("_", " ")
        topic_str = f"[{m.topic}] " if m.topic else ""
        lines.append(f"  [{tag}] {topic_str}{m.content}")
    lines.append("")
    return "\n".join(lines)


def format_history_for_context(messages: list[ConversationMessage]) -> str:
    """Render conversation history as a context block."""
    if not messages:
        return ""

    lines = ["[CONVERSATION HISTORY]"]
    for msg in messages:
        prefix = "User" if msg.role == "user" else "IntuOne"
        # Truncate very long messages
        text = msg.content[:600] + ("…" if len(msg.content) > 600 else "")
        lines.append(f"  {prefix}: {text}")
    lines.append("")
    return "\n".join(lines)
