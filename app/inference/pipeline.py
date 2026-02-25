"""
Inference pipeline for the fine-tuned IntuOne model.

Loads the LoRA adapter on top of the base model and generates
intelligence briefings from live signal data.

Two modes:
  1. Fine-tuned model  — loads adapter from disk, runs locally on GPU.
  2. Teacher fallback  — uses Claude Sonnet when no local model is ready.
     Useful during early development before the first training run.
"""
import logging
from typing import Any

from app.config import settings
from app.schemas.signal import SignalRead
from app.services.intuone import PRIMARY_LAYERS, _synthesis
from app.training.formatter import (
    apply_llama3_chat_template,
    build_chat_messages,
    format_context,
)
from app.inference.reasoning import (
    explore_insights,
    format_thinking_for_context,
    think,
)

logger = logging.getLogger(__name__)


# ── local model loader (lazy, singleton) ─────────────────────────────────────

_model = None
_tokenizer = None


def load_model(adapter_path: str, base_model: str | None = None) -> None:
    """
    Load the fine-tuned LoRA adapter into memory.

    Call once at startup (e.g. in a FastAPI lifespan event).
    Raises RuntimeError if ML deps are not installed.
    """
    global _model, _tokenizer
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            f"ML dependencies not installed: {exc}. "
            "Run: pip install torch transformers peft bitsandbytes accelerate"
        ) from exc

    resolved_base = base_model or settings.BASE_MODEL_NAME

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Loading base model %s …", resolved_base)
    base = AutoModelForCausalLM.from_pretrained(
        resolved_base,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    logger.info("Applying LoRA adapter from %s …", adapter_path)
    _model = PeftModel.from_pretrained(base, adapter_path)
    _model.eval()

    _tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    _tokenizer.pad_token = _tokenizer.eos_token
    logger.info("IntuOne model loaded.")


def is_model_loaded() -> bool:
    return _model is not None and _tokenizer is not None


# ── generation ───────────────────────────────────────────────────────────────

def generate_briefing_local(
    context: str,
    max_new_tokens: int = 600,
    temperature: float = 0.3,
    repetition_penalty: float = 1.1,
) -> str:
    """Run inference with the locally loaded fine-tuned model."""
    import torch

    if not is_model_loaded():
        raise RuntimeError("Model not loaded. Call load_model() first.")

    messages = build_chat_messages(context, analysis=None)
    prompt = apply_llama3_chat_template(messages, add_generation_prompt=True)

    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            repetition_penalty=repetition_penalty,
            pad_token_id=_tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


async def generate_briefing_teacher(context: str) -> str:
    """
    Fallback: call Claude Sonnet when no local model is available.
    Used during development / before first training run.
    """
    import anthropic

    from app.training.generator import _TEACHER_SYSTEM

    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    response = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        system=_TEACHER_SYSTEM,
        messages=[{"role": "user", "content": context}],
    )
    return response.content[0].text.strip()


# ── high-level API ───────────────────────────────────────────────────────────

async def run_intuone(
    topic: str,
    signals: list[SignalRead],
    time_window: str = "24h",
    force_teacher: bool = False,
    user_message: str | None = None,
    extra_context: str | None = None,
) -> dict[str, Any]:
    """
    End-to-end: signals → layer scores → think → explore → briefing.

    Flow:
      1. Score all five layers + synthesis
      2. Format signal context
      3. THINK — private reasoning scratchpad (Haiku, fast)
      4. EXPLORE — cross-layer insight mining
      5. Build full generation context (memory + history + thinking + signals)
      6. Generate English briefing (local model or teacher Claude)

    Returns a dict with the briefing, metadata, thinking, and insights.
    """
    # 1. Score layers
    layer_scores: dict[str, Any] = {}
    for layer in PRIMARY_LAYERS:
        layer_sigs = [s for s in signals if s.layer == layer.layer_name]
        ls = await layer.score(layer_sigs, topic, time_window)
        layer_scores[layer.layer_name] = {
            "score": ls.score,
            "confidence": ls.confidence,
            "signal_count": ls.signal_count,
        }
    layer_scores["synthesis"] = _synthesis.synthesize(layer_scores)

    # 2. Format signal context
    signal_context = format_context(
        topic=topic,
        time_window=time_window,
        signals=signals,
        layer_scores=layer_scores,
        user_message=user_message,
    )

    # 3. THINK — private reasoning (runs in parallel with signal context build)
    memory_context = extra_context or ""
    thinking = await think(
        topic=topic,
        user_message=user_message or f"Analyse signals for: {topic}",
        signal_context=signal_context,
        memory_context=memory_context,
    )

    # 4. EXPLORE — mine non-obvious cross-layer insights
    insights = await explore_insights(
        topic=topic,
        signal_context=signal_context,
        thinking=thinking,
    )

    # 5. Build full context: memory/history + thinking + signals
    parts: list[str] = []
    if extra_context:
        parts.append(extra_context.rstrip())
    if thinking:
        parts.append(format_thinking_for_context(thinking))
    parts.append(signal_context)
    full_context = "\n\n".join(parts)

    # 6. Generate English briefing
    if not force_teacher and is_model_loaded():
        briefing = generate_briefing_local(full_context)
        model_used = "local"
    else:
        briefing = await generate_briefing_teacher(full_context)
        model_used = "teacher_claude"

    return {
        "topic": topic,
        "time_window": time_window,
        "layer_scores": layer_scores,
        "overall_score": layer_scores.get("synthesis", {}).get("score", 0.0),
        "confidence": layer_scores.get("synthesis", {}).get("confidence", 0.0),
        "briefing": briefing,
        "model_used": model_used,
        "signal_count": len(signals),
        "thinking": thinking,          # stored by chat endpoint, not shown to user
        "insights": insights,          # seeded as MemoryEntry by chat endpoint
    }
