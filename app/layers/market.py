from app.constants import LayerType
from app.layers.base import LayerBase
from app.schemas.layer import LayerScoreCreate
from app.schemas.signal import SignalRead


class MarketLayer(LayerBase):
    """
    Probability Layer — "What does the crowd price as likely?"

    Reads prediction market yes_price as a proxy for collective probability
    assignment. Score is the weighted average of (yes_price - 0.5) * 2,
    mapping [0, 1] → [-1, +1], weighted by signal_strength (volume-derived).

    A yes_price of 0.8 → +0.6; yes_price of 0.3 → -0.4.
    """

    layer_name = LayerType.MARKET

    async def score(
        self,
        signals: list[SignalRead],
        topic: str,
        time_window: str = "24h",
    ) -> LayerScoreCreate:
        if not signals:
            return self._empty_score(topic, time_window)

        weighted_scores: list[float] = []
        weights: list[float] = []

        for sig in signals:
            pd = sig.processed_data or {}
            yes_price = pd.get("yes_price")
            if yes_price is None:
                continue
            # Maps [0, 1] → [-1, +1]
            directional = (float(yes_price) - 0.5) * 2.0
            weight = float(sig.signal_strength or 0.5)
            weighted_scores.append(directional * weight)
            weights.append(weight)

        if not weights:
            return self._empty_score(topic, time_window)

        score = sum(weighted_scores) / sum(weights)
        avg_conf = sum(sig.confidence or 0.0 for sig in signals) / len(signals)

        return LayerScoreCreate(
            layer=self.layer_name,
            topic=topic,
            score=round(score, 4),
            confidence=round(avg_conf, 4),
            signal_count=len(signals),
            time_window=time_window,
        )
