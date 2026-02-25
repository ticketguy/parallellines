"""
HuggingFace dataset builder.

Exports TrainingExample rows from the DB into a HuggingFace Dataset
ready for SFTTrainer.  Each row is tokenised into the Llama-3 chat
template format.

Usage (standalone):
    python -m app.training.dataset --output ./data/intuone-v1
"""
import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.database import AsyncSessionLocal
from app.models.training_example import TrainingExample
from app.training.formatter import apply_llama3_chat_template, build_chat_messages

logger = logging.getLogger(__name__)

# ── export ────────────────────────────────────────────────────────────────────

async def _fetch_examples(
    split: str | None = None,
    validated_only: bool = False,
    min_quality: float = 0.0,
) -> list[dict[str, Any]]:
    async with AsyncSessionLocal() as db:
        q = select(TrainingExample)
        if split:
            q = q.where(TrainingExample.split == split)
        if validated_only:
            q = q.where(TrainingExample.is_validated == True)  # noqa: E712
        if min_quality > 0:
            q = q.where(TrainingExample.quality_score >= min_quality)
        result = await db.execute(q)
        rows = result.scalars().all()

    records = []
    for row in rows:
        messages = build_chat_messages(row.context, row.analysis)
        # Full chat-formatted text (used by SFTTrainer with dataset_text_field)
        text = apply_llama3_chat_template(messages, add_generation_prompt=False)
        records.append(
            {
                "id": str(row.id),
                "topic": row.topic,
                "time_window": row.time_window,
                "split": row.split,
                "source": row.source,
                "context": row.context,
                "analysis": row.analysis,
                # Pre-formatted text for SFTTrainer
                "text": text,
            }
        )
    return records


def build_hf_dataset(
    output_dir: str | Path,
    validated_only: bool = False,
    min_quality: float = 0.0,
):
    """
    Pull examples from the DB and write a HuggingFace DatasetDict to disk.

    Directory layout:
        output_dir/
          train/   data-00000-of-00001.arrow
          val/     data-00000-of-00001.arrow
          test/    data-00000-of-00001.arrow
          dataset_info.json
    """
    try:
        from datasets import Dataset, DatasetDict
    except ImportError:
        raise RuntimeError(
            "HuggingFace `datasets` is not installed. "
            "Run: pip install datasets"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits: dict[str, list] = {"train": [], "val": [], "test": []}
    for split_name in splits:
        rows = asyncio.run(_fetch_examples(split_name, validated_only, min_quality))
        splits[split_name] = rows
        logger.info("Split '%s': %d examples", split_name, len(rows))

    total = sum(len(v) for v in splits.values())
    if total == 0:
        raise ValueError(
            "No training examples found in the DB. "
            "Run the generator first: POST /api/v1/training/generate"
        )

    dataset_dict = DatasetDict(
        {name: Dataset.from_list(rows) for name, rows in splits.items() if rows}
    )
    dataset_dict.save_to_disk(str(output_dir))

    # Also write a plain JSONL for inspection
    jsonl_path = output_dir / "train.jsonl"
    with jsonl_path.open("w") as f:
        for row in splits["train"]:
            f.write(json.dumps(row) + "\n")

    logger.info("Dataset saved to %s  (total=%d)", output_dir, total)
    return dataset_dict


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Build IntuOne training dataset")
    parser.add_argument("--output", default="./data/intuone-v1", help="Output directory")
    parser.add_argument("--validated-only", action="store_true")
    parser.add_argument("--min-quality", type=float, default=0.0)
    args = parser.parse_args()

    build_hf_dataset(
        output_dir=args.output,
        validated_only=args.validated_only,
        min_quality=args.min_quality,
    )
