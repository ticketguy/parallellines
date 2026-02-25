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
from app.layers.synthesis import LAYER_WEIGHTS
from app.schemas.signal import SignalRead

# ── display helpers ──────────────────────────────────────────────────────────

_LAYER_LABELS: dict[str, str] = {
    LayerType.MARKET: "MARKET",
    LayerType.SOCIAL: "SOCIAL DISCOURSE",
    LayerType.NEWS: "NEWS & MEDIA",
    LayerType.SENTIMENT: "NLP SENTIMENT",
    LayerType.GEOPOLITICAL: "GEOPOLITICAL",
}

_SYSTEM_PROMPT = """\
You are IntuOne — a senior intelligence analyst who synthesises live signals \
from prediction markets, social media, news, NLP sentiment, and geopolitical \
sources. You think carefully before answering and speak in clear, natural \
English — like a brilliant analyst in conversation, not a data report.

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

    Returns plain text suitable for insertion into a chat template.
    user_message: if provided, the actual question the user asked — IntuOne
    should answer it directly rather than producing a generic briefing.
    """
    as_of = as_of or datetime.now(timezone.utc)
    lines: list[str] = []

    lines.append(f"TOPIC: {topic}")
    lines.append(f"ANALYSIS WINDOW: {time_window}")
    lines.append(f"AS OF: {as_of.strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")

    signals_by_layer: dict[str, list[SignalRead]] = {}
    for sig in signals:
        signals_by_layer.setdefault(sig.layer, []).append(sig)

    for layer_key, label in _LAYER_LABELS.items():
        score_data = layer_scores.get(layer_key, {})
        weight_pct = int(LAYER_WEIGHTS.get(layer_key, 0) * 100)
        score = score_data.get("score", 0.0)
        conf = score_data.get("confidence", 0.0)
        n = score_data.get("signal_count", 0)

        lines.append(f"{'━' * 3} {label} [weight {weight_pct}%] {'━' * 3}")

        if conf == 0.0 or n == 0:
            lines.append("  No data available.")
            lines.append("")
            continue

        # Direction arrow
        arrow = "▲" if score > 0.05 else ("▼" if score < -0.05 else "→")
        lines.append(
            f"  Score: {score:+.3f} {arrow}  |  Confidence: {conf:.0%}  |  Signals: {n}"
        )

        # Layer-specific signal formatting
        layer_sigs = signals_by_layer.get(layer_key, [])
        _append_layer_signals(lines, layer_key, layer_sigs)
        lines.append("")

    # Synthesis summary
    synth = layer_scores.get(LayerType.SYNTHESIS, {})
    overall_score = synth.get("score", 0.0)
    overall_conf = synth.get("confidence", 0.0)
    direction = "POSITIVE" if overall_score > 0.1 else ("NEGATIVE" if overall_score < -0.1 else "NEUTRAL")
    lines.append("━━━ SYNTHESIS ━━━")
    lines.append(
        f"  Overall: {overall_score:+.3f} ({direction})  |  Confidence: {overall_conf:.0%}"
    )
    lines.append("")

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


def _append_layer_signals(
    lines: list[str],
    layer_key: str,
    sigs: list[SignalRead],
) -> None:
    """Append signal detail lines for a specific layer."""
    if not sigs:
        return

    # Sort by signal_strength descending, show top 5
    top = sorted(sigs, key=lambda s: s.signal_strength or 0.0, reverse=True)[:5]

    if layer_key == LayerType.MARKET:
        for sig in top:
            pd = sig.processed_data or {}
            q = pd.get("question", "Unknown market")[:90]
            yes = pd.get("yes_price")
            vol = pd.get("volume_24h")
            yes_str = f"YES {yes:.0%}" if yes is not None else "?"
            vol_str = f"${vol:,.0f} vol/24h" if vol else ""
            lines.append(f"  • {q}")
            lines.append(f"    {yes_str}  {vol_str}")

    elif layer_key in (LayerType.SOCIAL, LayerType.NEWS, LayerType.SENTIMENT):
        for sig in top:
            pd = sig.processed_data or {}
            text = pd.get("text") or pd.get("headline") or pd.get("summary", "")
            sentiment = pd.get("sentiment_score")
            sent_str = f"[sentiment {sentiment:+.2f}]" if sentiment is not None else ""
            lines.append(f"  • {text[:100]} {sent_str}".strip())

    elif layer_key == LayerType.GEOPOLITICAL:
        for sig in top:
            pd = sig.processed_data or {}
            summary = pd.get("summary") or pd.get("headline", "")
            direction = pd.get("direction_score")
            dir_str = f"[direction {direction:+.2f}]" if direction is not None else ""
            lines.append(f"  • {summary[:100]} {dir_str}".strip())


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
