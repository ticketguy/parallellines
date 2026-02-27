import json
import logging
import re

from app.constants import LayerType
from app.layers.base import LayerBase
from app.schemas.layer import LayerScoreCreate
from app.schemas.signal import SignalRead

logger = logging.getLogger(__name__)

_SYNTHESIZE_PROMPT = """\
[SYNTHESIZE-PERCEPTION]
Layer readings for "{topic}":
{layer_scores}

Synthesize these five perception layers into the overall Perception Index.
Read the layer scores and their detailed observations together. Consider:
- Where layers converge: that convergence is meaningful, raises confidence
- Where layers diverge: name the specific tension and what it implies
- Cross-layer dynamics: High Conviction against falling Probability = risk not certainty.
  High Echo without Conviction = noise. High Memory with falling Probability = zombie belief.
- What the full picture tells you that no single layer shows on its own

Output ONLY this JSON (no other text):
{{
  "score": <-1.0 to +1.0>,
  "confidence": <0.0 to 1.0>,
  "convergences": ["<layers that agree and what they agree on>", ...],
  "tensions": ["<specific cross-layer conflict and what it means>", ...],
  "perception_read": "<overall interpretation of the belief topology in 2-3 sentences>"
}}

score: overall directional read across all layers
confidence: degree of cross-layer coherence
[/SYNTHESIZE-PERCEPTION]"""


class SynthesisLayer(LayerBase):
    """
    Perception Index — the IntuOne model synthesizes all five layer readings
    into a single coherent directional score. Divergence between layers is
    itself a signal: it lowers confidence and raises the instability reading.

    Not fed raw signals — operates on LayerScore objects via synthesize().
    The score() method is a no-op stub kept for interface compliance.
    """

    layer_name = LayerType.SYNTHESIS

    async def score(
        self,
        signals: list[SignalRead],
        topic: str,
        time_window: str = "24h",
    ) -> LayerScoreCreate:
        return self._empty_score(topic, time_window)

    async def synthesize(
        self,
        layer_scores: dict[str, dict],
        topic: str = "",
    ) -> dict[str, float]:
        """
        Ask IntuOne to synthesize five layer readings into the Perception Index.
        Returns {"score": float, "confidence": float}.
        Returns zeroes if the model is not loaded.
        """
        from app.inference.pipeline import generate_briefing_local, is_model_loaded

        if not is_model_loaded():
            return {"score": 0.0, "confidence": 0.0}

        lines = []
        for name, data in layer_scores.items():
            sc = data.get("score", 0.0)
            conf = data.get("confidence", 0.0)
            n = data.get("signal_count", 0)
            extra = data.get("extra_data") or {}
            lines.append(f"  {name}: score={sc:+.3f}, confidence={conf:.0%}, signals={n}")
            if extra.get("reasoning"):
                lines.append(f"    reasoning: {extra['reasoning']}")
            if extra.get("notable"):
                lines.append(f"    notable: {extra['notable']}")
            if extra.get("key_data"):
                for item in extra["key_data"][:3]:
                    lines.append(f"    • {item}")

        prompt = _SYNTHESIZE_PROMPT.format(
            topic=topic or "the topic",
            layer_scores="\n".join(lines),
        )

        try:
            raw = generate_briefing_local(prompt, max_new_tokens=200, temperature=0.1)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
                score = max(-1.0, min(1.0, float(data.get("score", 0.0))))
                conf = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
                result: dict = {"score": round(score, 4), "confidence": round(conf, 4)}
                for key in ("convergences", "tensions", "perception_read"):
                    if data.get(key):
                        result[key] = data[key]
                return result
        except Exception as exc:
            logger.debug("[synthesis] LLM synthesize failed: %s", exc)

        return {"score": 0.0, "confidence": 0.0}
