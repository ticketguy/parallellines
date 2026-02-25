"""
IntuOne reasoning module — "think before you speak."

Two phases run before every generation:

1. THINK  — private scratchpad where IntuOne reasons through all signals,
            identifies patterns, cross-layer convergence / divergence, and
            forms a directional hypothesis. This is NOT shown to the user
            but is injected into the generation context.

2. EXPLORE — after thinking, IntuOne explicitly looks for non-obvious insights
            across every layer: what's surprising, what's missing, what would
            change the read. These become memory candidates.

The thinking output is stored as `ConversationMessage(role="thinking")` so
the self-improvement loop can review it later.
"""
import logging

import anthropic

from app.config import settings

logger = logging.getLogger(__name__)

_THINK_SYSTEM = """\
You are IntuOne's internal reasoning engine. Your job is to THINK — not to
produce a final answer, but to reason carefully through all available data
before one is generated.

This thinking is private. Be completely honest, speculative where needed, and
thorough. Explore every angle. You can be wrong in the thinking — that's fine,
the goal is to surface all considerations before committing to a response.

Think through:
1. SIGNAL STRENGTH — Which layers have the strongest data? Which are thin?
2. DIRECTION — What direction are the strong signals pointing? How consistent?
3. CONVERGENCE — Do layers agree? Where do they disagree? What does that mean?
4. SURPRISES — What's unexpected in this data? What's notably absent?
5. KEY DRIVER — If you had to pick ONE signal that's driving the read, what is it?
6. COUNTERARGUMENT — What's the strongest case for the opposite view?
7. UNCERTAINTY — What don't you know that would matter most?
8. CROSS-TOPIC PATTERNS — Does this connect to anything you know about related topics?

Write continuous prose, not a list. Think out loud like a senior analyst
working through a problem privately before presenting to a client.\
"""

_EXPLORE_SYSTEM = """\
You are IntuOne's insight engine. Given signal data and a reasoning scratchpad,
your job is to find the non-obvious insights that a surface-level read would miss.

Look for:
- Anomalies: signals that are outliers compared to what you'd expect
- Silences: important layers that are quiet when they shouldn't be
- Leads and lags: is one layer typically ahead of another? Is it moving first?
- Cross-layer patterns: does the divergence between market and social predict something?
- Historical echoes: does this pattern match something that has happened before?
- Second-order effects: if the primary signal is right, what else follows?

Output 2–4 insight statements. Each should be one sentence, concrete, and
novel — something that wouldn't be obvious from reading the raw data alone.
Format: one insight per line, no bullets, no numbering.\
"""


async def think(
    topic: str,
    user_message: str,
    signal_context: str,
    memory_context: str = "",
) -> str:
    """
    Private reasoning step: IntuOne thinks through all signals before responding.
    Returns the thinking text (injected into final generation context).
    """
    if not settings.ANTHROPIC_API_KEY:
        return ""

    user_turn = (
        f"USER ASKED: {user_message}\n\n"
        f"SIGNAL DATA:\n{signal_context}\n"
    )
    if memory_context:
        user_turn += f"\nRELEVANT MEMORIES:\n{memory_context}\n"
    user_turn += f"\nTopic: {topic}\n\nThink through this carefully."

    try:
        client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",  # fast model for private thinking
            max_tokens=600,
            system=_THINK_SYSTEM,
            messages=[{"role": "user", "content": user_turn}],
        )
        return response.content[0].text.strip()
    except Exception as exc:
        logger.warning("Thinking step failed (non-fatal): %s", exc)
        return ""


async def explore_insights(
    topic: str,
    signal_context: str,
    thinking: str,
) -> list[str]:
    """
    Explore cross-layer patterns and non-obvious insights.
    Returns a list of insight strings (candidates for MemoryEntry).
    """
    if not settings.ANTHROPIC_API_KEY or not thinking:
        return []

    user_turn = (
        f"TOPIC: {topic}\n\n"
        f"SIGNAL DATA:\n{signal_context}\n\n"
        f"REASONING SCRATCHPAD:\n{thinking}\n\n"
        "What non-obvious insights does this data contain?"
    )

    try:
        client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=_EXPLORE_SYSTEM,
            messages=[{"role": "user", "content": user_turn}],
        )
        raw = response.content[0].text.strip()
        return [line.strip() for line in raw.splitlines() if line.strip()]
    except Exception as exc:
        logger.warning("Insight exploration failed (non-fatal): %s", exc)
        return []


def format_thinking_for_context(thinking: str) -> str:
    """Inject the private thinking into the generation context."""
    if not thinking:
        return ""
    return (
        "[INTUONE REASONING — use this to inform your response]\n"
        + thinking
        + "\n[END REASONING]\n\n"
    )
