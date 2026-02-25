"""
Rule-based inference engine — IntuOne's fully independent, zero-dependency brain.

This module generates natural-English briefings directly from layer scores and
signal data WITHOUT calling any external LLM.  It is the production inference
path when the fine-tuned model is not yet loaded.

As the fine-tuned model improves through training, it will produce better and
better outputs — but this rule-based engine always provides a solid baseline
that works on any machine with no GPU, no API key, and no internet connection.

Output style mirrors the fine-tuned model's target style:
- Direct directional read with confidence
- Layer-by-layer explanation in plain English
- Convergence / divergence commentary
- Key signal callout
- Watch-for closing
"""
import random
from typing import Any

from app.constants import LayerType
from app.schemas.signal import SignalRead

# ── Direction vocabulary ──────────────────────────────────────────────────────

_DIR_LABELS = {
    "strong_bull": ["strongly bullish", "decisively positive", "strongly in YES territory"],
    "bull":        ["moderately bullish", "leaning positive", "cautiously optimistic"],
    "neutral":     ["mixed / neutral", "inconclusive", "sitting on the fence"],
    "bear":        ["moderately bearish", "leaning negative", "cautiously pessimistic"],
    "strong_bear": ["strongly bearish", "decisively negative", "strongly in NO territory"],
}

_CONF_LABELS = {
    "high":   ["high confidence", "strong conviction", "clear signal"],
    "medium": ["moderate confidence", "reasonable conviction", "mixed signals"],
    "low":    ["low confidence", "thin data", "uncertain — treat with caution"],
}

_LAYER_NAMES = {
    LayerType.MARKET:       "prediction market",
    LayerType.SOCIAL:       "social discourse",
    LayerType.NEWS:         "news and media",
    LayerType.SENTIMENT:    "NLP sentiment",
    LayerType.GEOPOLITICAL: "geopolitical",
}

_CONVERGENCE_PHRASES = [
    "Across layers, the read is consistent",
    "All major layers are pointing the same direction",
    "There is notable cross-layer convergence",
]
_DIVERGENCE_PHRASES = [
    "The layers are not fully aligned",
    "There is an interesting divergence between layers",
    "Some layers are pulling in opposite directions",
]

_WATCH_PHRASES = [
    "Watch for any sudden shift in the {layer} layer — that would be the first sign of a reversal.",
    "The key variable to monitor is {layer} data — a move there changes this read.",
    "Keep an eye on {layer} signals; that's where this call is most fragile.",
]


def _direction_key(score: float) -> str:
    if score > 0.35:   return "strong_bull"
    if score > 0.10:   return "bull"
    if score < -0.35:  return "strong_bear"
    if score < -0.10:  return "bear"
    return "neutral"


def _confidence_key(conf: float) -> str:
    if conf >= 0.70: return "high"
    if conf >= 0.40: return "medium"
    return "low"


def _pct(v: float) -> str:
    return f"{v:.0%}"


def _signed(v: float) -> str:
    return f"{v:+.2f}"


# ── Main generator ────────────────────────────────────────────────────────────

def generate(
    topic: str,
    layer_scores: dict[str, dict[str, Any]],
    signals: list[SignalRead],
    user_message: str | None = None,
    time_window: str = "24h",
) -> str:
    """
    Generate a natural-English briefing from layer scores alone.
    No external calls. No GPU. Works on any machine.
    """
    synth = layer_scores.get(LayerType.SYNTHESIS, {})
    overall_score = float(synth.get("score", 0.0))
    overall_conf = float(synth.get("confidence", 0.0))

    dir_key = _direction_key(overall_score)
    conf_key = _confidence_key(overall_conf)

    direction_str = random.choice(_DIR_LABELS[dir_key])
    conf_str = random.choice(_CONF_LABELS[conf_key])
    conf_pct = _pct(overall_conf)

    # ── Opening line ──────────────────────────────────────────────────────────
    if user_message:
        open_line = (
            f"My read on {topic}: {direction_str}, {conf_pct} confidence ({conf_str})."
        )
    else:
        open_line = (
            f"Looking at {topic} over the last {time_window}: "
            f"the signal is {direction_str} with {conf_pct} confidence."
        )

    paragraphs: list[str] = [open_line]

    # ── Layer-by-layer commentary ─────────────────────────────────────────────
    layer_summaries: list[str] = []
    active_layers: list[tuple[str, dict]] = []

    for layer_key in [LayerType.MARKET, LayerType.SOCIAL, LayerType.NEWS,
                      LayerType.SENTIMENT, LayerType.GEOPOLITICAL]:
        ld = layer_scores.get(layer_key, {})
        sc = float(ld.get("score", 0.0))
        cf = float(ld.get("confidence", 0.0))
        n  = int(ld.get("signal_count", 0))
        if n == 0 or cf < 0.05:
            continue
        active_layers.append((layer_key, ld))
        name = _LAYER_NAMES[layer_key]
        d_key = _direction_key(sc)

        # Pick most relevant signal for this layer
        layer_sigs = [s for s in signals if s.layer == layer_key]
        top_sig = sorted(layer_sigs, key=lambda s: s.signal_strength or 0, reverse=True)[:1]

        detail = ""
        if layer_key == LayerType.MARKET and top_sig:
            pd = top_sig[0].processed_data or {}
            q = pd.get("question", "")[:80]
            yes = pd.get("yes_price")
            vol = pd.get("volume_24h")
            if q and yes is not None:
                vol_str = f" (${vol:,.0f} 24h volume)" if vol else ""
                detail = f" The standout contract is \"{q}\" — currently pricing {_pct(yes)} YES{vol_str}."

        elif layer_key in (LayerType.SOCIAL, LayerType.NEWS, LayerType.SENTIMENT) and top_sig:
            pd = top_sig[0].processed_data or {}
            text = (pd.get("text") or pd.get("headline") or "")[:100]
            if text:
                detail = f" The loudest signal here: \"{text}\"."

        direction_phrase = random.choice(_DIR_LABELS[d_key])
        summary = (
            f"The {name} layer ({n} signal{'s' if n != 1 else ''}) "
            f"reads {direction_phrase} at {_pct(cf)} confidence.{detail}"
        )
        layer_summaries.append(summary)

    if layer_summaries:
        paragraphs.append(" ".join(layer_summaries[:3]))  # top 3 layers
        if len(layer_summaries) > 3:
            paragraphs.append(" ".join(layer_summaries[3:]))

    # ── Convergence / divergence ──────────────────────────────────────────────
    if len(active_layers) >= 2:
        scores = [float(ld.get("score", 0)) for _, ld in active_layers]
        # All same sign = convergence
        all_same_sign = all(s > 0 for s in scores) or all(s < 0 for s in scores)
        spread = max(scores) - min(scores)

        if all_same_sign and spread < 0.3:
            conv = random.choice(_CONVERGENCE_PHRASES)
            paragraphs.append(
                f"{conv} — that level of alignment tends to give me more confidence in the read."
            )
        elif spread > 0.5:
            div = random.choice(_DIVERGENCE_PHRASES)
            # find which layers diverge
            pos = [_LAYER_NAMES[k] for k, ld in active_layers if float(ld.get("score", 0)) > 0.05]
            neg = [_LAYER_NAMES[k] for k, ld in active_layers if float(ld.get("score", 0)) < -0.05]
            if pos and neg:
                paragraphs.append(
                    f"{div}: {', '.join(pos)} are pointing positive while "
                    f"{', '.join(neg)} lean negative. "
                    "Divergence like this raises uncertainty — I'd want to see the layers align "
                    "before having strong conviction."
                )

    # ── Closing: what to watch ────────────────────────────────────────────────
    if active_layers:
        # Watch the layer with the most uncertainty (lowest confidence)
        weakest = min(active_layers, key=lambda x: float(x[1].get("confidence", 1)))
        watch_layer = _LAYER_NAMES[weakest[0]]
        watch = random.choice(_WATCH_PHRASES).format(layer=watch_layer)
        paragraphs.append(watch)
    else:
        paragraphs.append(
            "Data is thin across all layers right now — I'd wait for more signals "
            "before making any strong call on this topic."
        )

    return "\n\n".join(paragraphs)
