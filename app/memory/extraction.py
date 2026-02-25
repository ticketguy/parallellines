"""
Memory extraction — the model extracts its own long-term memories.

After each briefing, the local model scans its own output for insights
worth storing as MemoryEntry rows. No external LLMs. No Claude.

When the model is not yet trained, a fast keyword heuristic is used
to extract any directional claims or notable facts.
"""
import logging
import re

from app.models.memory import MemoryEntry

logger = logging.getLogger(__name__)

_EXTRACT_PROMPT = """\
[EXTRACT-MEMORIES]
Read the briefing below and extract 0-3 facts worth remembering long-term.
Only extract genuinely useful, specific insights -- not generic statements.

Types:
  topic_insight  -- directional claim about a topic backed by data
  entity_fact    -- a fact about a named entity (person, org, country, asset)
  pattern        -- a cross-layer or historical pattern observed

Output one JSON object per line (nothing else):
{{"type": "topic_insight", "topic": "...", "content": "...", "importance": 0.0-1.0, "keywords": ["..."]}}

BRIEFING:
{briefing}
[/EXTRACT-MEMORIES]
"""

_SUMMARISE_PROMPT = """\
[SUMMARISE]
Summarise this conversation in 2-3 sentences.
Focus on: what topics were analysed and what the key conclusions were.

CONVERSATION:
{history}
[/SUMMARISE]
"""

# Patterns that signal a directional claim worth remembering
_DIRECTIONAL_PATTERNS = re.compile(
    r"(bullish|bearish|neutral|positive|negative|likely|unlikely"
    r"|confidence|probability|\d+%|trending|surge|decline|rising|falling)",
    re.IGNORECASE,
)


def _heuristic_extract(briefing: str, session_id=None) -> list[MemoryEntry]:
    """
    Fast heuristic extraction when the local model is not loaded.
    Finds sentences with directional claims and stores them.
    """
    sentences = re.split(r"(?<=[.!?])\s+", briefing)
    memories = []
    for sentence in sentences:
        s = sentence.strip()
        if len(s) < 40 or not _DIRECTIONAL_PATTERNS.search(s):
            continue
        keywords = [
            w.lower() for w in re.findall(r"\b[a-zA-Z]{5,}\b", s)
        ][:8]
        memories.append(MemoryEntry(
            memory_type="topic_insight",
            content=s[:300],
            importance=0.5,
            keywords=keywords,
            source_session_id=session_id,
        ))
        if len(memories) >= 2:
            break
    return memories


async def extract_memories(
    briefing: str,
    session_id=None,
) -> list[MemoryEntry]:
    """
    Extract long-term memories from a briefing.
    Uses the local model if loaded; falls back to heuristic extraction.
    """
    from app.inference.pipeline import generate_briefing_local, is_model_loaded

    if not is_model_loaded():
        return _heuristic_extract(briefing, session_id)

    prompt = _EXTRACT_PROMPT.format(briefing=briefing[:1000])
    try:
        raw = generate_briefing_local(
            prompt,
            max_new_tokens=300,
            temperature=0.2,
            repetition_penalty=1.0,
        )

        memories: list[MemoryEntry] = []
        import json
        for line in raw.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
                memories.append(MemoryEntry(
                    memory_type=data.get("type", "topic_insight"),
                    topic=data.get("topic"),
                    entity=data.get("entity"),
                    content=data["content"][:400],
                    importance=float(data.get("importance", 0.5)),
                    keywords=data.get("keywords", [])[:10],
                    source_session_id=session_id,
                ))
            except Exception:
                continue
        return memories
    except Exception as exc:
        logger.warning("Model memory extraction failed, using heuristic: %s", exc)
        return _heuristic_extract(briefing, session_id)


async def summarise_session(messages: list, topic_hint: str = "") -> str:
    """
    Summarise a conversation session using the local model.
    Returns "" if the model is not loaded.
    """
    from app.inference.pipeline import generate_briefing_local, is_model_loaded

    if not is_model_loaded() or not messages:
        return ""

    history = "\n".join(
        f"{'User' if m.role == 'user' else 'IntuOne'}: {m.content[:200]}"
        for m in messages
        if m.role in ("user", "assistant")
    )
    prompt = _SUMMARISE_PROMPT.format(history=history[:1200])

    try:
        raw = generate_briefing_local(
            prompt,
            max_new_tokens=150,
            temperature=0.3,
            repetition_penalty=1.0,
        )
        # Strip the [SUMMARISE] tags if the model echoes them
        inner = re.search(r"\[SUMMARISE\](.*?)\[/SUMMARISE\]", raw, re.DOTALL)
        return inner.group(1).strip() if inner else raw.strip()
    except Exception as exc:
        logger.warning("Session summarisation failed: %s", exc)
        return ""
