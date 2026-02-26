from app.constants import LayerType
from app.layers.base import LayerBase
from app.schemas.layer import LayerScoreCreate
from app.schemas.signal import SignalRead


class ProbabilityLayer(LayerBase):
    """
    Probability Layer — "What does the crowd price as likely?"

    Reads the collective probability assignment across all available signals.
    A prediction market at YES=0.73 means the crowd gives 73% probability.
    High-volume markets carry more epistemic weight than thin ones.
    """

    layer_name = LayerType.PROBABILITY

    _layer_prompt = """\
[SCORE-LAYER: PROBABILITY]
Topic: {topic} | Window: {time_window}

Signals:
{signals}

Question: What does available evidence collectively price as the probability of this outcome? \
Consider prediction market prices, trading volume, and liquidity depth.

Output ONLY this JSON (no other text):
{{"score": <-1.0 to +1.0>, "confidence": <0.0 to 1.0>}}

score: -1.0=crowd prices outcome as very unlikely, +1.0=very likely, 0.0=uncertain/50-50
confidence: how strongly the signals support this reading
[/SCORE-LAYER]"""

    async def score(
        self,
        signals: list[SignalRead],
        topic: str,
        time_window: str = "24h",
    ) -> LayerScoreCreate:
        if not signals:
            return self._empty_score(topic, time_window)
        sc, conf = await self._llm_score(signals, topic, time_window)
        return LayerScoreCreate(
            layer=self.layer_name,
            topic=topic,
            score=round(sc, 4),
            confidence=round(conf, 4),
            signal_count=len(signals),
            time_window=time_window,
        )
