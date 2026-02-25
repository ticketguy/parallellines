from app.constants import LayerType
from app.layers.base import LayerBase
from app.schemas.layer import LayerScoreCreate
from app.schemas.signal import SignalRead


class GeopoliticalLayer(LayerBase):
    """
    Scores geopolitical / institutional signals (government statements,
    policy moves, regulatory filings, etc.).

    Stub — returns 0/0 until a geopolitical submind is wired up.
    When live: score will reflect directional bias from official
    statements and policy actions affecting the topic.
    """

    layer_name = LayerType.GEOPOLITICAL

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
            direction = pd.get("direction_score")  # expected -1.0 to 1.0
            if direction is None:
                continue
            weight = float(sig.signal_strength or 0.5)
            weighted_scores.append(float(direction) * weight)
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
