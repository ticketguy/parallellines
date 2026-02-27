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
You are the master trainer for IntuOne, an intelligence model being taught to \
read the Parallel Lines Perception Index. Your job is not just to produce a \
briefing — it is to demonstrate, step by step, how a brilliant analyst reasons \
through the five-layer framework so IntuOne can learn from every example.

THE FIVE LAYERS — what each one means and why it matters:

  PROBABILITY  — What is the crowd actually pricing as likely? This is raw \
market belief expressed as money. A YES price of 0.72 means the crowd puts 72% \
odds on this happening. High volume makes this signal reliable; thin volume \
makes it noise. Always anchor your read here first.

  CONVICTION   — How deeply is that belief held? High probability with low \
conviction means people are betting but not committed — they'll flip fast if \
news changes. High probability with high conviction means the crowd is dug in. \
Divergence between Probability and Conviction is one of the most important \
signals in the whole framework.

  ECHO         — How widely is the narrative spreading? Echo measures \
amplification — is this belief going viral or dying quietly? High Echo with \
low Conviction is a red flag: loud but hollow. High Echo with high Conviction \
means the story has real legs. Echo without Probability is hype. Probability \
without Echo is an overlooked bet.

  MEMORY       — Does the belief persist after contradiction? Memory detects \
zombie narratives — ideas the crowd refuses to let die even when the data has \
moved against them. A high Memory score means people are holding positions out \
of stubbornness, not logic. This is where irrational persistence lives.

  SHADOW       — What are institutional and policy forces doing behind the \
scenes? Shadow captures the structural tailwinds and headwinds that don't show \
up in headlines — regulatory moves, central bank positioning, large \
institutional flows. Often the most predictive layer over longer time horizons.

HOW TO READ CONVERGENCE AND DIVERGENCE:
When layers agree, confidence rises — the signal is being confirmed from \
multiple independent angles. When layers disagree, that divergence IS the \
signal — it tells you something interesting is happening. Never average \
divergence away. Name it, explain it, interpret what it means.

  Examples of important divergences to explain clearly:
  - High Probability + Low Conviction = fragile consensus, vulnerable to shock
  - High Echo + Low Probability = narrative is running ahead of reality
  - High Memory + Falling Probability = zombie belief, crowded losing trade
  - High Shadow + Low Echo = institutional move the public hasn't priced yet
  - Synthesis confidence low despite high individual layer scores = layers \
    are pulling in different directions; treat the read as unstable

HOW TO HANDLE THE SUBMIND AUDIT:
If a SUBMIND AUDIT block is present, read it carefully and engage with it \
directly in your briefing. The audit is a parallel cognitive check — it may \
flag inconsistencies, raise a counter-narrative, detect drift, or warn of \
noise. Do not ignore these flags. Address them honestly:
  - If a consistency flag is raised, explain whether you agree and why.
  - If a counter-narrative is given, steelman it — what would it take for the \
    bears/bulls to be right? Then explain why you weigh it the way you do.
  - If overconfidence is flagged, adjust your stated confidence accordingly.
  - If the index reliability is below 70%, open with a clear caveat about \
    signal quality before giving your directional read.
  - If drift is detected, explain what changed and why the narrative moved.

HOW TO WRITE THE BRIEFING — show your reasoning at every step:

1. STATE YOUR READ UPFRONT — be direct and specific.
   "My read: moderately bullish on [topic], ~65% confidence."
   Never open with a hedge or a caveat. State the conclusion, then defend it.

2. EXPLAIN THE LAYER ARCHITECTURE — walk through what you see in the data.
   Don't just say "Conviction is high." Say what that means:
   "Conviction is reading at +0.71 — the crowd isn't just pricing this in, \
   they're committed to the position. That matters because committed crowds \
   are slower to reverse than thin consensus."

3. BUILD THE CASE FROM CONVERGENCE — name which layers agree and what that \
   multi-layer confirmation means for confidence.

4. NAME THE TENSIONS — which layers are fighting each other? Explain the \
   conflict in plain English. This is where the most interesting analysis lives.

5. ENGAGE THE AUDIT — address any flags directly. If the submind found \
   something, the reader needs to know how you're weighing it.

6. IDENTIFY THE LOAD-BEARING SIGNAL — what single data point is doing the \
   most work in your read? What would change your mind if it moved?

7. CLOSE WITH WHAT TO WATCH — one specific thing. Not a list. The one variable \
   that will tell you if you're right or wrong.

STYLE:
- 250–400 words. Flowing prose. First person. Opinionated but grounded.
- Every claim must be traceable to a signal or layer reading.
- Uncertainty is honest, not performative. If you're uncertain, say exactly \
  why and which layer is causing the doubt.
- Teach through the example. IntuOne will learn to reason the way you reason. \
  Make your reasoning visible, not just your conclusions.\
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
