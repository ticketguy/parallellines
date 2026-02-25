"""
Self-improvement loop — IntuOne evaluates and improves itself.

Every response is:
  1. Scored by the model itself (self-evaluation prompt)
  2. Stored as a training example (high/med/low quality buckets)
  3. Cross-layer insights seeded into long-term memory

No external LLMs. No Claude. The model grows from its own experience.

The training cycle:
  - Good responses (score >= 0.7) -> train split  -> used in next fine-tune
  - Medium (0.4-0.7)              -> val split    -> reviewed before use
  - Weak  (< 0.4)                 -> test split   -> negative examples
  - Claude-generated labels       -> train split   -> gold standard base

Continuously improving: every real conversation makes the model smarter.
"""
import logging
import re
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.memory import MemoryEntry
from app.models.training_example import TrainingExample

logger = logging.getLogger(__name__)

# ── Self-evaluation prompt ────────────────────────────────────────────────────
# Injected into the model to score its own output — no external LLM needed.

_SELF_EVAL_PROMPT = """\
[SELF-EVAL]
Evaluate this response on four dimensions. Answer with a single JSON line.

USER QUESTION: {user_message}
YOUR RESPONSE: {briefing}

Score each dimension 0.0 to 1.0:
  grounded    - every claim is backed by the signal data (not invented)
  direct      - opens with a clear directional read, not vague hedging
  complete    - answers the actual question, covers the key signals
  clear       - readable, well-structured natural English

Output exactly one line: {{"grounded": X, "direct": X, "complete": X, "clear": X}}
[/SELF-EVAL]
"""


def _heuristic_score(briefing: str, user_message: str) -> float:
    """
    Fast heuristic score used when the model is not loaded.
    Evaluates signal presence, length, directional claims.
    """
    score = 0.3  # baseline

    words = len(briefing.split())
    if words >= 80:   score += 0.15
    if words >= 150:  score += 0.10
    if words >= 300:  score += 0.05
    if words > 600:   score -= 0.10  # too long

    # Directional language
    directional = {"bullish", "bearish", "neutral", "positive", "negative",
                   "confidence", "confident", "likely", "probably", "uncertain"}
    found = sum(1 for w in directional if w in briefing.lower())
    score += min(found * 0.05, 0.20)

    # Data references (numbers, percentages)
    if re.search(r"\d+%|\$[\d,]+|\d+\.\d+", briefing):
        score += 0.10

    # Penalise if response is just "not enough data" type
    weak_phrases = ["no data", "not enough", "cannot determine", "insufficient"]
    if any(p in briefing.lower() for p in weak_phrases):
        score -= 0.15

    return round(max(0.0, min(1.0, score)), 3)


def _model_score(briefing: str, user_message: str) -> tuple[float, str]:
    """
    Use the local model to evaluate its own response.
    Returns (score, feedback_string).
    """
    from app.inference.pipeline import generate_briefing_local, is_model_loaded

    if not is_model_loaded():
        s = _heuristic_score(briefing, user_message)
        return s, "heuristic"

    prompt = _SELF_EVAL_PROMPT.format(
        user_message=user_message[:300],
        briefing=briefing[:600],
    )
    try:
        raw = generate_briefing_local(
            prompt,
            max_new_tokens=80,
            temperature=0.1,
            repetition_penalty=1.0,
        )
        # Parse the JSON line
        import json
        line = next(
            (l for l in raw.splitlines() if l.strip().startswith("{")), ""
        )
        if line:
            data = json.loads(line)
            dims = ["grounded", "direct", "complete", "clear"]
            score = sum(float(data.get(d, 0.5)) for d in dims) / len(dims)
            return round(score, 3), "model_self_eval"
    except Exception as exc:
        logger.debug("Model self-eval parse failed: %s", exc)

    # Fall back to heuristic if model eval fails
    return _heuristic_score(briefing, user_message), "heuristic_fallback"


async def score_response(
    user_message: str,
    briefing: str,
    signal_context: str,  # kept for API compatibility, not used directly
) -> tuple[float, str]:
    """
    Score a response. Uses the local model for self-evaluation.
    Returns (score 0-1, method string).
    """
    score, method = _model_score(briefing, user_message)
    logger.debug("Self-eval score=%.2f method=%s", score, method)
    return score, method


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
    Thinking is prepended to context so the model learns the reasoning chain.

    Quality buckets:
      >= 0.70 → train  (used in next fine-tune directly)
       0.40-0.69 → val (reviewed before promotion)
      < 0.40  → test   (negative example / low-quality)
    """
    split = "train" if quality_score >= 0.70 else ("val" if quality_score >= 0.40 else "test")

    full_context = (
        f"[THINKING]\n{thinking}\n[/THINKING]\n\n{context}" if thinking else context
    )

    example = TrainingExample(
        topic=topic,
        context=full_context,
        analysis=briefing,
        quality_score=quality_score,
        split=split,
        notes=feedback or None,
        source="self_generated",
    )
    db.add(example)


async def seed_memories_from_insights(
    insights: list[str],
    topic: str,
    session_id: UUID | None,
    db: AsyncSession,
) -> None:
    """Convert model-generated insights into long-term MemoryEntry rows."""
    for insight in insights:
        if not insight or len(insight) < 20:
            continue
        keywords = [
            w.lower() for w in insight.split()
            if len(w) > 4 and w.isalpha()
        ][:10]
        db.add(MemoryEntry(
            memory_type="pattern",
            topic=topic,
            content=insight,
            importance=0.6,
            keywords=keywords,
            source_session_id=session_id,
        ))
