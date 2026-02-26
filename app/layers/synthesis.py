from app.constants import LayerType
from app.layers.base import LayerBase
from app.schemas.layer import LayerScoreCreate
from app.schemas.signal import SignalRead

# Relative importance of each layer in the final synthesis.
# Weights sum to 1.0. Adjust as you calibrate the model.
LAYER_WEIGHTS: dict[str, float] = {
    LayerType.MARKET: 0.35,
    LayerType.SOCIAL: 0.20,
    LayerType.NEWS: 0.20,
    LayerType.SENTIMENT: 0.15,
    LayerType.GEOPOLITICAL: 0.10,
}


class SynthesisLayer(LayerBase):
    """
    Perception Index — confidence-weighted composite of all parallel layer readings.

    This is not a sixth perception layer. It is the aggregated reading that
    IntuOne receives. A single compressed score hides instability: high
    Conviction alongside high Fracture signals risk, not certainty. The
    confidence-weighting here ensures layers with no active signal do not
    drag the composite toward zero.

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
        # SynthesisLayer is driven by synthesize(), not raw signals.
        return self._empty_score(topic, time_window)

    def synthesize(self, layer_scores: dict[str, dict]) -> dict[str, float]:
        """
        Combine per-layer scores into a single overall score.

        Each layer's contribution is scaled by both its static weight and
        its computed confidence, so layers with no data don't drag the
        result toward zero.

        Args:
            layer_scores: {layer_name: {score, confidence, signal_count}}

        Returns:
            {score: float, confidence: float}
        """
        total_effective_weight = 0.0
        weighted_score = 0.0
        total_weight_for_conf = 0.0
        weighted_conf = 0.0

        for layer_name, data in layer_scores.items():
            static_weight = LAYER_WEIGHTS.get(layer_name, 0.1)
            conf = float(data.get("confidence", 0.0))
            if conf <= 0:
                continue
            effective_w = static_weight * conf
            weighted_score += float(data["score"]) * effective_w
            weighted_conf += conf * static_weight
            total_effective_weight += effective_w
            total_weight_for_conf += static_weight

        if total_effective_weight == 0:
            return {"score": 0.0, "confidence": 0.0}

        return {
            "score": round(weighted_score / total_effective_weight, 4),
            "confidence": round(
                weighted_conf / total_weight_for_conf if total_weight_for_conf else 0.0, 4
            ),
        }
