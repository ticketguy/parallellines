#!/usr/bin/env python
"""
Standalone training data generator.

Runs the full pipeline without the API server:
  1. Fetch live Polymarket signals
  2. Score all five perception layers
  3. Run submind audits
  4. Call Claude Sonnet (teacher) to generate gold-standard briefings
  5. Store TrainingExample rows in the database

Usage:
    python scripts/generate.py --topics "bitcoin" "ethereum" "AI regulation"
    python scripts/generate.py --markets       # auto-discover from top Polymarket markets
    python scripts/generate.py --markets --n 30 --window 48h

Requirements:
    - ANTHROPIC_API_KEY set in .env
    - PostgreSQL running and migrated (alembic upgrade head)
    - pip install -e ".[dev]"
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Allow running from project root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate")


async def fetch_polymarket_signals() -> list:
    """Pull fresh Polymarket signals and return as SignalRead list (not persisted)."""
    from app.agents.polymarket import PolymarketSubmind
    from app.schemas.signal import SignalRead

    logger.info("Fetching live Polymarket data …")
    submind = PolymarketSubmind()
    signal_creates = await submind.fetch()
    logger.info("  Fetched %d markets", len(signal_creates))

    # Convert SignalCreate → SignalRead (assign fake IDs for scoring)
    import uuid
    from datetime import datetime, timezone

    reads = []
    for sc in signal_creates:
        reads.append(SignalRead(
            id=uuid.uuid4(),
            source=sc.source,
            layer=sc.layer,
            domain=sc.domain,
            raw_data=sc.raw_data,
            processed_data=sc.processed_data,
            signal_strength=sc.signal_strength,
            confidence=sc.confidence,
            topic_tags=sc.topic_tags,
            entity_tags=sc.entity_tags,
            signal_timestamp=sc.signal_timestamp,
            created_at=datetime.now(timezone.utc),
        ))
    return reads


async def score_layers(signals: list, topic: str, time_window: str) -> dict:
    """Score all five layers + synthesis. Returns layer_scores dict."""
    from app.services.intuone import PRIMARY_LAYERS, _synthesis

    layer_scores: dict = {}
    for layer in PRIMARY_LAYERS:
        ls = await layer.score(signals, topic, time_window)
        layer_scores[layer.layer_name] = {
            "score": ls.score,
            "confidence": ls.confidence,
            "signal_count": ls.signal_count,
            **({"extra_data": ls.extra_data} if ls.extra_data else {}),
        }
        logger.debug(
            "  [%s] score=%+.2f conf=%.0f%% n=%d",
            layer.layer_name,
            ls.score,
            ls.confidence * 100,
            ls.signal_count,
        )

    layer_scores["synthesis"] = await _synthesis.synthesize(layer_scores, topic=topic)
    synth = layer_scores["synthesis"]
    logger.info(
        "  Synthesis: %+.2f (conf=%.0f%%)",
        synth.get("score", 0.0),
        synth.get("confidence", 0.0) * 100,
    )
    return layer_scores


async def run_audits(layer_scores: dict, signals: list) -> str:
    """Run all registered submind audits and return formatted audit block."""
    from app.agents import REGISTERED_SUBMINDS
    from app.training.formatter import format_audit_context

    active = [s for s in REGISTERED_SUBMINDS if any(sig.source == s.name for sig in signals)]
    if not active:
        return ""

    audits = list(await asyncio.gather(*[s.audit(layer_scores, signals) for s in active]))
    for a in audits:
        logger.info(
            "  Audit [%s]: intensity=%.0f%% reliability=%.0f%%",
            a.submind,
            a.challenge_intensity * 100,
            a.index_reliability * 100,
        )
    return format_audit_context(audits)


async def call_teacher(topic: str, context: str) -> str | None:
    """Call Claude Sonnet and return the generated briefing text."""
    import anthropic
    from app.config import settings
    from app.training.generator import _TEACHER_SYSTEM

    if not settings.ANTHROPIC_API_KEY or settings.ANTHROPIC_API_KEY == "your-anthropic-api-key-here":
        logger.error(
            "ANTHROPIC_API_KEY is not set. "
            "Add your key to .env before running this script."
        )
        return None

    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    try:
        response = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=_TEACHER_SYSTEM,
            messages=[{"role": "user", "content": context}],
        )
        return response.content[0].text.strip()
    except Exception as exc:
        logger.error("Teacher call failed for topic=%r: %s", topic, exc)
        return None


async def generate_for_topics(
    topics: list[str],
    time_window: str = "24h",
    use_db: bool = True,
) -> int:
    """
    Full pipeline for a list of topics.
    If use_db=True, persists results to the database.
    Returns number of examples successfully generated.
    """
    import random
    from app.training.formatter import format_context

    signals = await fetch_polymarket_signals()
    if not signals:
        logger.error("No signals fetched — check Polymarket connectivity.")
        return 0

    generated = 0

    if use_db:
        from app.database import AsyncSessionLocal
        db_factory = AsyncSessionLocal
    else:
        db_factory = None

    for topic in topics:
        logger.info("── Topic: %r ──", topic)

        # Filter signals relevant to this topic (by category/tag match)
        topic_lower = topic.lower()
        topic_signals = [
            s for s in signals
            if any(
                topic_lower in (tag or "").lower()
                for tag in (s.topic_tags or [])
            )
        ] or signals  # fall back to all signals if no tag match

        logger.info("  %d signals for this topic", len(topic_signals))

        layer_scores = await score_layers(topic_signals, topic, time_window)
        audit_block = await run_audits(layer_scores, topic_signals)

        context = format_context(
            topic=topic,
            time_window=time_window,
            signals=topic_signals,
            layer_scores=layer_scores,
            audit_context=audit_block,
        )

        analysis = await call_teacher(topic, context)
        if not analysis or len(analysis) < 50:
            logger.warning("  Skipping %r — teacher returned no usable output", topic)
            continue

        logger.info("  Generated briefing (%d chars)", len(analysis))

        if use_db:
            from app.models.training_example import TrainingExample
            r = random.random()
            split = "train" if r < 0.8 else ("val" if r < 0.9 else "test")

            async with db_factory() as db:
                example = TrainingExample(
                    topic=topic,
                    time_window=time_window,
                    context=context,
                    analysis=analysis,
                    source="synthetic_claude",
                    layer_breakdown=layer_scores,
                    signal_ids=[str(s.id) for s in topic_signals],
                    split=split,
                )
                db.add(example)
                await db.commit()
                logger.info("  Saved to DB as split=%r", split)
        else:
            # Dry-run: just print
            print(f"\n{'='*60}")
            print(f"TOPIC: {topic}")
            print(f"{'='*60}")
            print(context[:800] + "\n…")
            print(f"\n[BRIEFING]\n{analysis}")

        generated += 1

    return generated


async def generate_from_markets(
    n: int = 20,
    time_window: str = "24h",
    use_db: bool = True,
) -> int:
    """Auto-discover topics from live Polymarket category tags."""
    from app.agents.polymarket import PolymarketSubmind

    submind = PolymarketSubmind()
    logger.info("Discovering topics from live Polymarket markets …")
    signal_creates = await submind.fetch()

    # Extract unique categories as topics
    categories: list[str] = []
    seen: set[str] = set()
    for sc in signal_creates:
        proc = sc.processed_data or {}
        cat = proc.get("category")
        if cat and cat not in seen:
            seen.add(cat)
            categories.append(cat)
        if len(categories) >= n:
            break

    if not categories:
        logger.error("No categories found in Polymarket data.")
        return 0

    logger.info("Discovered %d topics: %s", len(categories), categories)
    return await generate_for_topics(categories, time_window=time_window, use_db=use_db)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate IntuOne training data")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--topics", nargs="+", metavar="TOPIC",
        help='Topics to generate examples for, e.g. --topics "bitcoin" "AI"',
    )
    group.add_argument(
        "--markets", action="store_true",
        help="Auto-discover topics from live Polymarket category tags",
    )
    parser.add_argument("--n", type=int, default=20, help="Max topics (--markets only)")
    parser.add_argument("--window", default="24h", help="Time window (default: 24h)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print context + briefing without writing to DB",
    )
    args = parser.parse_args()

    use_db = not args.dry_run

    if args.markets:
        count = asyncio.run(
            generate_from_markets(n=args.n, time_window=args.window, use_db=use_db)
        )
    else:
        count = asyncio.run(
            generate_for_topics(args.topics, time_window=args.window, use_db=use_db)
        )

    print(f"\nDone. {count} training example(s) generated.")


if __name__ == "__main__":
    main()
