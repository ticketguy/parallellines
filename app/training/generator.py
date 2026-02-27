"""
Teacher-model label generator.

Uses Claude Sonnet (the teacher) to generate gold-standard intelligence
briefings from formatted signal contexts.  These (context, analysis) pairs
become the fine-tuning dataset for our own model.

Flow:
  1. Pull recent signals from the DB for a given topic.
  2. Run layer scoring (same as production).
  3. Run submind audits (same as production — training context must match).
  4. Format into a context string via formatter.py (audit block included).
  5. Call Claude Sonnet with the context.
  6. Store the result as a TrainingExample row.

The audit block must be included in training contexts so IntuOne learns
to read and respond to submind challenges — the training format must be
identical to the production format.
"""
import asyncio
import logging
import random
from datetime import datetime, timezone

import anthropic
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents import REGISTERED_SUBMINDS
from app.config import settings
from app.models.signal import Signal
from app.models.training_example import TrainingExample
from app.schemas.signal import SignalRead
from app.services.intuone import PRIMARY_LAYERS, _synthesis
from app.training.formatter import build_chat_messages, format_audit_context, format_context

logger = logging.getLogger(__name__)

_TEACHER_SYSTEM = """\
You are IntuOne, a senior intelligence analyst. You have access to live signals
from prediction markets, social media, news, NLP sentiment, and geopolitical
sources. You think deeply before responding and always speak in clear,
opinionated natural English — like a brilliant analyst, not a data report.

Your responses:
- Open with your directional read: "My read: [bullish/bearish/neutral], \
~[X]% confidence."
- Explain in plain language what is driving the signal — which layers are \
moving, why it matters, what the data actually says.
- Highlight where layers converge (strengthens conviction) or diverge \
(raises uncertainty) and what that divergence means.
- Identify the single most important signal — the one thing that would most \
change your view if it moved.
- Close with what to watch — the key variable or upcoming event.
- Write 200–350 words. Flowing prose. First-person perspective. No bullet
  point dumps. Ground every claim in the data. Acknowledge uncertainty honestly.\
"""


async def generate_example(
    topic: str,
    signals: list[SignalRead],
    layer_scores: dict,
    audit_context: str = "",
    time_window: str = "24h",
    split: str = "train",
) -> TrainingExample | None:
    """
    Generate one TrainingExample for `topic` using Claude as the teacher.

    Returns None if the API call fails or yields an unusable response.
    audit_context: pre-rendered submind audit block from format_audit_context().
    """
    context = format_context(
        topic=topic,
        time_window=time_window,
        signals=signals,
        layer_scores=layer_scores,
        audit_context=audit_context,
    )

    try:
        client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        response = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=_TEACHER_SYSTEM,
            messages=[{"role": "user", "content": context}],
        )
        analysis: str = response.content[0].text.strip()
    except Exception as exc:
        logger.error("Teacher model call failed for topic=%r: %s", topic, exc)
        return None

    if len(analysis) < 50:
        logger.warning("Teacher returned suspiciously short analysis for topic=%r", topic)
        return None

    return TrainingExample(
        topic=topic,
        time_window=time_window,
        context=context,
        analysis=analysis,
        source="synthetic_claude",
        layer_breakdown=layer_scores,
        signal_ids=[str(s.id) for s in signals],
        split=split,
    )


async def generate_examples_for_topic(
    topic: str,
    db: AsyncSession,
    time_window: str = "24h",
    max_signals: int = 50,
) -> TrainingExample | None:
    """
    Full pipeline: fetch signals from DB → score → generate → store.
    Returns the stored TrainingExample or None on failure.
    """
    # 1. Fetch recent signals for this topic
    result = await db.execute(
        select(Signal)
        .where(Signal.topic_tags.any(topic))  # type: ignore[attr-defined]
        .order_by(Signal.created_at.desc())
        .limit(max_signals)
    )
    raw_signals = result.scalars().all()

    if not raw_signals:
        logger.warning("No signals found for topic=%r — skipping", topic)
        return None

    signals = [SignalRead.model_validate(s) for s in raw_signals]

    # 2. Score each layer — all signals go through every layer
    layer_scores: dict = {}
    for layer in PRIMARY_LAYERS:
        ls = await layer.score(signals, topic, time_window)
        layer_scores[layer.layer_name] = {
            "score": ls.score,
            "confidence": ls.confidence,
            "signal_count": ls.signal_count,
            **({"extra_data": ls.extra_data} if ls.extra_data else {}),
        }
    layer_scores["synthesis"] = await _synthesis.synthesize(layer_scores, topic=topic)

    # 3. Run submind audits — training context must match production context
    active_subminds = [
        s for s in REGISTERED_SUBMINDS
        if any(sig.source == s.name for sig in signals)
    ]
    audits = list(
        await asyncio.gather(*[
            s.audit(layer_scores, signals)
            for s in active_subminds
        ])
    )
    audit_context = format_audit_context(audits) if audits else ""

    # 4. Assign train/val/test split (80/10/10)
    r = random.random()
    split = "train" if r < 0.8 else ("val" if r < 0.9 else "test")

    # 5. Generate example via teacher
    example = await generate_example(
        topic=topic,
        signals=signals,
        layer_scores=layer_scores,
        audit_context=audit_context,
        time_window=time_window,
        split=split,
    )
    if example is None:
        return None

    # 5. Persist
    db.add(example)
    await db.commit()
    await db.refresh(example)
    logger.info("Generated training example %s for topic=%r", example.id, topic)
    return example


async def batch_generate(
    topics: list[str],
    db: AsyncSession,
    time_window: str = "24h",
) -> dict[str, int]:
    """Generate examples for a list of topics. Returns {topic: 1|0} status map."""
    results: dict[str, int] = {}
    for topic in topics:
        example = await generate_examples_for_topic(topic, db, time_window)
        results[topic] = 1 if example else 0
    return results
