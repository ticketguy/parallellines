"""
QLoRA fine-tuning trainer for IntuOne.

Fine-tunes an open-weights base model (default: Llama-3.1-8B-Instruct)
on our (context, analysis) training examples using:
  - 4-bit NF4 quantisation (bitsandbytes)
  - LoRA adapters on all attention projections (PEFT)
  - SFTTrainer from TRL

Usage:
    python -m app.training.trainer \\
        --dataset ./data/intuone-v1 \\
        --output  ./checkpoints/intuone-v1 \\
        --base-model meta-llama/Meta-Llama-3.1-8B-Instruct

Requirements:
    pip install torch transformers peft trl bitsandbytes accelerate datasets
"""
import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ── config dataclass ─────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    # Model
    base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    dataset_path: str = "./data/intuone-v1"
    output_dir: str = "./checkpoints/intuone-v1"

    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    # Target all attention + feedforward projections
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Quantisation
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8   # effective batch = 16
    learning_rate: float = 2e-4
    max_seq_length: int = 4096
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    bf16: bool = True
    fp16: bool = False

    # Logging / saving
    logging_steps: int = 10
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    report_to: str = "none"          # set to "wandb" when you have a project


# ── trainer ──────────────────────────────────────────────────────────────────

def train(cfg: TrainingConfig) -> None:
    """
    Load base model, apply QLoRA, run SFT on the IntuOne dataset.
    Saves the final LoRA adapter to cfg.output_dir.
    """
    try:
        import torch
        from datasets import load_from_disk
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            f"Missing ML dependency: {exc}. "
            "Install with: pip install torch transformers peft trl bitsandbytes accelerate datasets"
        ) from exc

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Quantisation config ────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.use_4bit,
        bnb_4bit_compute_dtype=getattr(torch, cfg.bnb_4bit_compute_dtype),
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.use_nested_quant,
    )

    # ── 2. Base model ─────────────────────────────────────────────────────
    logger.info("Loading base model: %s", cfg.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=bnb_config if cfg.use_4bit else None,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(model)

    # ── 3. Tokeniser ──────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # required for SFT with causal LM

    # ── 4. LoRA adapters ─────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 5. Dataset ────────────────────────────────────────────────────────
    logger.info("Loading dataset from %s", cfg.dataset_path)
    ds = load_from_disk(cfg.dataset_path)
    train_ds = ds["train"]
    eval_ds = ds.get("val")

    # ── 6. SFT training ──────────────────────────────────────────────────
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        max_seq_length=cfg.max_seq_length,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        evaluation_strategy=cfg.evaluation_strategy if eval_ds else "no",
        load_best_model_at_end=cfg.load_best_model_at_end if eval_ds else False,
        report_to=cfg.report_to,
        dataset_text_field="text",   # the pre-formatted chat text column
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_config,
    )

    # ── 7. Train ─────────────────────────────────────────────────────────
    logger.info("Starting training …")
    trainer.train()

    # ── 8. Save adapter ───────────────────────────────────────────────────
    adapter_path = output_dir / "final_adapter"
    trainer.model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    logger.info("LoRA adapter saved to %s", adapter_path)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Fine-tune IntuOne")
    parser.add_argument("--base-model", default=TrainingConfig.base_model)
    parser.add_argument("--dataset", default=TrainingConfig.dataset_path)
    parser.add_argument("--output", default=TrainingConfig.output_dir)
    parser.add_argument("--epochs", type=int, default=TrainingConfig.num_train_epochs)
    parser.add_argument("--lora-r", type=int, default=TrainingConfig.lora_r)
    parser.add_argument("--lr", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--report-to", default=TrainingConfig.report_to)
    args = parser.parse_args()

    cfg = TrainingConfig(
        base_model=args.base_model,
        dataset_path=args.dataset,
        output_dir=args.output,
        num_train_epochs=args.epochs,
        lora_r=args.lora_r,
        learning_rate=args.lr,
        report_to=args.report_to,
    )
    train(cfg)
