"""
Signal → model context formatter.

Takes structured signals and layer scores and renders them into the
text format the model will be trained on and queried with.

We use the Llama-3 instruct chat template (<|begin_of_text|> style)
but expose a plain `format_context()` function that returns just the
user-turn content so it can be embedded in any chat template.
"""
from datetime import datetime, timezone
from typing import Any

from app.constants import LayerType
from app.schemas.signal import SignalRead

# ── display helpers ──────────────────────────────────────────────────────────

_LAYER_LABELS: dict[str, str] = {
    LayerType.PROBABILITY: "PROBABILITY",
    LayerType.CONVICTION: "CONVICTION",
    LayerType.ECHO: "ECHO",
    LayerType.MEMORY: "MEMORY",
    LayerType.SHADOW: "SHADOW",
}

_SYSTEM_PROMPT = """\
You are IntuOne — the interpreter layer of the Parallel Lines perception \
framework. You read the Perception Index across five parallel layers: \
Probability, Conviction, Echo, Memory, and Shadow. You translate \
belief topology into clear, natural English — like a brilliant analyst in \
conversation, not a data report.

Your responses are direct, opinionated, and grounded in data. You:
- State your directional read upfront ("My read: bullish, ~72% confidence")
- Explain the *why* in plain language — what signals are driving it
- Note where layers agree or disagree, and what that divergence means
- Acknowledge gaps and uncertainty honestly without hedging everything
- Answer the user's actual question, not a generic briefing template
- Use natural prose — no bullet point dumps, no JSON, no data tables in the
  final answer (unless the user asks for raw numbers)

When you have previous memories or conversation history, use them — build on \
what you already know about a topic rather than starting from scratch each time.\
"""


# ── core formatter ───────────────────────────────────────────────────────────

def format_context(
    topic: str,
    time_window: str,
    signals: list[SignalRead],
    layer_scores: dict[str, dict[str, Any]],
    as_of: datetime | None = None,
    user_message: str | None = None,
) -> str:
    """
    Render the user-turn context string for a given topic snapshot.

    Shows the raw signals first so IntuOne can read the actual content,
    then each layer's score and observations extracted by the model,
    then the synthesis reading.
    """
    as_of = as_of or datetime.now(timezone.utc)
    lines: list[str] = []

    lines.append(f"TOPIC: {topic}")
    lines.append(f"ANALYSIS WINDOW: {time_window}")
    lines.append(f"AS OF: {as_of.strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")

    # ── Raw signals ──────────────────────────────────────────────────────────
    lines.append(f"━━━ SIGNALS ({len(signals)} total) ━━━")
    for i, sig in enumerate(signals[:30], 1):
        proc = sig.processed_data or {}
        raw = sig.raw_data or {}
        content = (
            proc.get("text")
            or proc.get("question")
            or raw.get("question")
            or raw.get("headline")
            or f"signal from {sig.source}"
        )[:200]

        facts = []
        if proc.get("yes_price") is not None:
            facts.append(f"yes_price={proc['yes_price']:.2f}")
        if proc.get("volume_24h"):
            facts.append(f"vol_24h={proc['volume_24h']:,.0f}")
        if proc.get("liquidity"):
            facts.append(f"liq={proc['liquidity']:,.0f}")

        fact_str = f" [{', '.join(facts)}]" if facts else ""
        lines.append(f"  {i}. [{sig.source}]{fact_str} {content}")
    lines.append("")

    # ── Layer readings ───────────────────────────────────────────────────────
    for layer_key, label in _LAYER_LABELS.items():
        score_data = layer_scores.get(layer_key, {})
        score = score_data.get("score", 0.0)
        conf = score_data.get("confidence", 0.0)
        n = score_data.get("signal_count", 0)
        extra = score_data.get("extra_data") or {}

        lines.append(f"━━━ {label} ━━━")

        if conf == 0.0 or n == 0:
            lines.append("  No data available.")
            lines.append("")
            continue

        arrow = "▲" if score > 0.05 else ("▼" if score < -0.05 else "→")
        lines.append(f"  Score: {score:+.3f} {arrow}  |  Confidence: {conf:.0%}  |  Signals: {n}")

        if extra.get("key_data"):
            lines.append("  Key data:")
            for item in extra["key_data"]:
                lines.append(f"    • {item}")

        if extra.get("reasoning"):
            lines.append(f"  Read: {extra['reasoning']}")

        if extra.get("notable"):
            lines.append(f"  Notable: {extra['notable']}")

        lines.append("")

    # ── Synthesis ────────────────────────────────────────────────────────────
    synth = layer_scores.get(LayerType.SYNTHESIS, {})
    overall_score = synth.get("score", 0.0)
    overall_conf = synth.get("confidence", 0.0)
    direction = "POSITIVE" if overall_score > 0.1 else ("NEGATIVE" if overall_score < -0.1 else "NEUTRAL")

    lines.append("━━━ SYNTHESIS ━━━")
    lines.append(
        f"  Overall: {overall_score:+.3f} ({direction})  |  Confidence: {overall_conf:.0%}"
    )

    if synth.get("convergences"):
        lines.append("  Convergences:")
        for c in synth["convergences"]:
            lines.append(f"    • {c}")

    if synth.get("tensions"):
        lines.append("  Tensions:")
        for t in synth["tensions"]:
            lines.append(f"    • {t}")

    if synth.get("perception_read"):
        lines.append(f"  Perception: {synth['perception_read']}")

    lines.append("")

    # ── Instruction ──────────────────────────────────────────────────────────
    if user_message:
        lines.append(
            f"USER QUESTION: {user_message}\n"
            "Answer the user's question directly in natural English. "
            "Draw on the signal data above, your thinking, and any memories you have. "
            "Speak like a sharp analyst in conversation — confident where the data supports it, "
            "honest about uncertainty where it doesn't."
        )
    else:
        lines.append(
            "Write an intelligence briefing in natural English. "
            "Open with your directional read and confidence, then explain the key drivers "
            "and what the signals collectively mean. Be direct and analytical — "
            "this is a briefing for a decision-maker, not a data report."
        )

    return "\n".join(lines)


# ── chat template wrapper ────────────────────────────────────────────────────

def build_chat_messages(context: str, analysis: str | None = None) -> list[dict]:
    """
    Return a list of chat messages in the standard roles format.

    If `analysis` is provided (training mode), includes the assistant turn.
    If None (inference mode), omits it so the model generates it.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": context},
    ]
    if analysis is not None:
        messages.append({"role": "assistant", "content": analysis})
    return messages


def apply_llama3_chat_template(messages: list[dict], add_generation_prompt: bool = False) -> str:
    """
    Manually apply the Llama-3 instruct chat template.

    Used for models without a tokenizer at format time (e.g. during
    dataset construction before the tokenizer is loaded).
    """
    out = "<|begin_of_text|>"
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        out += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
    if add_generation_prompt:
        out += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return out
