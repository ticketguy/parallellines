from app.constants import LayerType
from app.layers.base import LayerBase
from app.schemas.layer import LayerScoreCreate
from app.schemas.signal import SignalRead


class ConvictionLayer(LayerBase):
    """
    Conviction Layer — "How deeply is the belief held?"

    Measures emotional intensity and depth of commitment in expressed belief,
    not just positive/negative polarity. High conviction can exist on both
    sides of a probability. Stub until a conviction submind is wired up.

    Signal input: processed_data["sentiment_score"] in range -1.0 to +1.0,
    where magnitude reflects depth of conviction, sign reflects direction.
    """

    layer_name = LayerType.CONVICTION

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
            sentiment = pd.get("sentiment_score")
            if sentiment is None:
                continue
            weight = float(sig.signal_strength or 0.5)
            weighted_scores.append(float(sentiment) * weight)
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
