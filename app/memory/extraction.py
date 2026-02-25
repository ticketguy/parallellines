"""
Memory extraction — after IntuOne generates a response, this module
uses Claude to identify key insights worth storing as long-term memories.

Extracted memories are stored in the MemoryEntry table and will be
retrieved and injected into future relevant conversations.
"""
import logging

import anthropic

from app.config import settings
from app.models.memory import MemoryEntry

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """\
You are a memory extractor for an intelligence system called IntuOne.

Given a briefing that IntuOne just produced, extract 0–3 long-term memories
worth storing. Only extract genuinely useful, generalisable insights — not
obvious or ephemeral statements.

Memory types:
  topic_insight  — a directional claim about a topic backed by data
  entity_fact    — a fact about a named entity (person, org, country)
  pattern        — a cross-layer pattern or correlation observed

For each memory, output a JSON object on its own line:
{
  "memory_type": "topic_insight" | "entity_fact" | "pattern",
  "topic": "<topic string or null>",
  "entity": "<entity string or null>",
  "content": "<the memory text, 1–2 sentences, present tense>",
  "importance": <0.0–1.0>,
  "keywords": ["kw1", "kw2", ...]
}

If there are no memories worth extracting, output nothing.

BRIEFING:
{briefing}
"""


async def extract_memories(
    briefing: str,
    session_id=None,
) -> list[MemoryEntry]:
    """
    Call Claude to extract memory entries from a briefing.
    Returns a list of (unsaved) MemoryEntry objects.
    """
    if not settings.ANTHROPIC_API_KEY:
        return []

    prompt = _EXTRACTION_PROMPT.format(briefing=briefing)

    try:
        client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",  # cheap + fast for extraction
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
    except Exception as exc:
        logger.warning("Memory extraction failed: %s", exc)
        return []

    memories: list[MemoryEntry] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            import json
            data = json.loads(line)
            memories.append(
                MemoryEntry(
                    memory_type=data.get("memory_type", "topic_insight"),
                    topic=data.get("topic"),
                    entity=data.get("entity"),
                    content=data["content"],
                    importance=float(data.get("importance", 0.5)),
                    keywords=data.get("keywords", []),
                    source_session_id=session_id,
                )
            )
        except Exception as exc:
            logger.debug("Skipping malformed memory line: %s — %s", line, exc)

    return memories


async def summarise_session(messages: list, topic_hint: str = "") -> str:
    """
    Produce a short session summary (stored on ConversationSession.summary).
    Called when a session is closed or reaches a turn threshold.
    """
    if not settings.ANTHROPIC_API_KEY or not messages:
        return ""

    history = "\n".join(
        f"{'User' if m.role == 'user' else 'IntuOne'}: {m.content[:300]}"
        for m in messages
    )
    prompt = (
        f"Summarise this IntuOne conversation in 2–3 sentences. "
        f"Focus on what topics were analysed and what the key conclusions were.\n\n"
        f"{history}"
    )

    try:
        client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as exc:
        logger.warning("Session summarisation failed: %s", exc)
        return ""
