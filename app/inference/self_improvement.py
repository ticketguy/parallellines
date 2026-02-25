"""
Self-improvement loop.

After every response IntuOne generates, this module:

1. SCORES the response — asks the teacher model to evaluate quality (0–1).
2. REFLECTS — identifies what was weak or missing in the reasoning.
3. STORES a TrainingExample — high-quality responses become training data.
4. SEEDS MEMORIES — insights from explore_insights() are stored as MemoryEntry.

Over time this builds a self-curating dataset: every good conversation
strengthens the model and every weak one surfaces actionable feedback.
"""
import logging
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.memory import MemoryEntry
from app.models.training_example import TrainingExample

logger = logging.getLogger(__name__)

_EVAL_SYSTEM = """\
You are evaluating the quality of an AI-generated intelligence briefing.

Score the response on four dimensions (0.0–1.0 each):
  groundedness    — every claim is backed by signal data (not hallucinated)
  clarity         — the response is easy to understand and well-structured
  directness      — opens with a clear directional read; does not hedge everything
  completeness    — addresses the user's actual question; doesn't miss key signals

Also write one sentence of feedback: what is the single most important thing
this response should have done differently?

Output format (JSON, one line):
{"groundedness": 0.0, "clarity": 0.0, "directness": 0.0, "completeness": 0.0, "feedback": "..."}
"""


async def score_response(
    user_message: str,
    briefing: str,
    signal_context: str,
) -> tuple[float, str]:
    """
    Ask the teacher model to score the briefing.
    Returns (score 0-1, feedback string).
    """
    if not settings.ANTHROPIC_API_KEY:
        return 0.5, ""

    import anthropic

    prompt = (
        f"USER ASKED: {user_message}\n\n"
        f"SIGNAL DATA:\n{signal_context[:1500]}\n\n"
        f"INTUONE RESPONSE:\n{briefing}\n\n"
        "Evaluate the response."
    )

    try:
        client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=_EVAL_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        import json
        data = json.loads(response.content[0].text.strip())
        dims = ["groundedness", "clarity", "directness", "completeness"]
        score = sum(float(data.get(d, 0.5)) for d in dims) / len(dims)
        feedback = data.get("feedback", "")
        return score, feedback
    except Exception as exc:
        logger.debug("Response scoring failed (non-fatal): %s", exc)
        return 0.5, ""


async def store_training_example(
    context: str,
    briefing: str,
    thinking: str,
    quality_score: float,
    feedback: str,
    topic: str,
    db: AsyncSession,
) -> None:
    """
    Store the (context, briefing) pair as a TrainingExample.

    High-quality responses (score >= 0.75) go to 'train'.
    Medium quality (0.5–0.75) go to 'val' for review.
    Low quality (< 0.5) go to 'test' as negative examples.
    """
    if quality_score >= 0.75:
        split = "train"
    elif quality_score >= 0.5:
        split = "val"
    else:
        split = "test"

    # Prepend thinking to context so the model sees the reasoning chain
    full_context = (
        f"[THINKING]\n{thinking}\n[/THINKING]\n\n{context}"
        if thinking
        else context
    )

    example = TrainingExample(
        topic=topic,
        context=full_context,
        analysis=briefing,
        quality_score=quality_score,
        split=split,
        notes=feedback or None,
    )
    db.add(example)
    # Caller is responsible for commit


async def seed_memories_from_insights(
    insights: list[str],
    topic: str,
    session_id: UUID | None,
    db: AsyncSession,
) -> None:
    """
    Convert explore_insights() output into MemoryEntry rows.
    Each insight becomes a 'pattern' memory with medium importance.
    """
    for insight in insights:
        if not insight:
            continue
        # Extract rough keywords from the insight text
        keywords = [
            w.lower()
            for w in insight.split()
            if len(w) > 4 and w.isalpha()
        ][:10]

        entry = MemoryEntry(
            memory_type="pattern",
            topic=topic,
            content=insight,
            importance=0.6,
            keywords=keywords,
            source_session_id=session_id,
        )
        db.add(entry)
    # Caller is responsible for commit
