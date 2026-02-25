from app.constants import LayerType
from app.layers.base import LayerBase
from app.schemas.layer import LayerScoreCreate
from app.schemas.signal import SignalRead


class SentimentLayer(LayerBase):
    """
    Scores NLP-derived sentiment signals (e.g. from LLM analysis of text).

    Stub — returns 0/0 until a sentiment submind is wired up.
    When live: score will aggregate structured sentiment outputs
    from LLM-based text analysis across multiple sources.
    """

    layer_name = LayerType.SENTIMENT

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
