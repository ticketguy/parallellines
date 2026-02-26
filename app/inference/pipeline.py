"""
IntuOne inference pipeline — the interpreter layer.

No external LLMs in the inference path. Claude is only used in
app/training/generator.py to generate gold-standard perception analyses
for training data.

Architecture:
  1. Score all five parallel perception layers simultaneously
  2. Synthesise into the Perception Index
  3. THINK  — IntuOne reasons privately through the layer readings
  4. EXPLORE — surfaces non-obvious cross-layer patterns (e.g. high
               Conviction diverging from falling Probability)
  5. Generate the final natural-language perception analysis

When the model is not yet loaded:
  - Steps 3 and 4 are skipped
  - Step 5 returns the raw Perception Index data so the UI still has
    something meaningful to display

IntuOne reads resonance, not correctness. It translates the Perception
Index into language — interpreting what the belief topology means, not
forecasting what will happen. The model improves with every training run.
"""
import logging
from typing import Any

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
    Raises RuntimeError if ML dependencies are not installed.
    """
    global _model, _tokenizer
    from app.config import settings

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            f"ML dependencies not installed: {exc}. "
            "Run: pip install parallellines[ml]"
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
    logger.info("IntuOne model loaded and ready.")


def is_model_loaded() -> bool:
    return _model is not None and _tokenizer is not None


# ── generation ────────────────────────────────────────────────────────────────

def generate_briefing_local(
    context: str,
    max_new_tokens: int = 600,
    temperature: float = 0.7,
    repetition_penalty: float = 1.1,
) -> str:
    """
    Run inference with the locally loaded fine-tuned model.
    All reasoning — thinking, response generation, self-evaluation,
    memory extraction — flows through this function.
    """
    import torch

    if not is_model_loaded():
        raise RuntimeError("IntuOne model not loaded.")

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

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _not_loaded_response(topic: str, layer_scores: dict, signal_count: int) -> str:
    """
    Structured 'model not loaded' message shown in the UI before training.
    Shows the raw layer scores so there is still something useful to display.
    """
    synth = layer_scores.get("synthesis", {})
    score = synth.get("score", 0.0)
    conf = synth.get("confidence", 0.0)

    lines = [
        f"**IntuOne model not yet trained.** Train the model first with: `POST /api/v1/training/generate`",
        "",
        f"Raw signal read for **{topic}**:",
        f"Overall score: {score:+.3f} | Confidence: {conf:.0%} | Signals ingested: {signal_count}",
        "",
    ]
    for layer, data in layer_scores.items():
        if layer == "synthesis":
            continue
        n = data.get("signal_count", 0)
        if n:
            lines.append(f"- {layer}: score={data['score']:+.2f}, conf={data['confidence']:.0%}, n={n}")

    lines += [
        "",
        "Once you have training data, run a fine-tuning pass. "
        "IntuOne will then generate natural-language briefings from these signals."
    ]
    return "\n".join(lines)


# ── high-level API ────────────────────────────────────────────────────────────

async def run_intuone(
    topic: str,
    signals: list[SignalRead],
    time_window: str = "24h",
    force_teacher: bool = False,   # kept for API compatibility — ignored
    user_message: str | None = None,
    extra_context: str | None = None,
) -> dict[str, Any]:
    """
    End-to-end inference: signals → Perception Index → think → explore → analysis.

    All five perception layers score simultaneously. IntuOne then interprets
    the resulting Perception Index into natural-language output.
    Returns structured layer readings + the perception analysis text.
    """
    # 1. Score all five perception layers simultaneously
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

    # 3 & 4. THINK + EXPLORE — IntuOne reasons through the Perception Index
    # (both are no-ops if the model isn't loaded yet)
    memory_context = extra_context or ""
    thinking = think(
        topic=topic,
        user_message=user_message or f"Analyse signals for: {topic}",
        signal_context=signal_context,
        memory_context=memory_context,
    )
    insights = explore_insights(
        topic=topic,
        signal_context=signal_context,
        thinking=thinking,
    )

    # 5. Build full context: memory/history + thinking + layer readings
    parts: list[str] = []
    if extra_context:
        parts.append(extra_context.rstrip())
    if thinking:
        parts.append(format_thinking_for_context(thinking))
    parts.append(signal_context)
    full_context = "\n\n".join(parts)

    # 6. Generate perception analysis — local IntuOne model only
    if is_model_loaded():
        briefing = generate_briefing_local(full_context)
        model_used = "intuone_local"
    else:
        briefing = _not_loaded_response(topic, layer_scores, len(signals))
        model_used = "not_loaded"

    return {
        "topic": topic,
        "time_window": time_window,
        "layer_scores": layer_scores,
        "overall_score": layer_scores.get("synthesis", {}).get("score", 0.0),
        "confidence": layer_scores.get("synthesis", {}).get("confidence", 0.0),
        "briefing": briefing,
        "model_used": model_used,
        "signal_count": len(signals),
        "thinking": thinking,
        "insights": insights,
    }
