from app.constants import LayerType
from app.layers.base import LayerBase
from app.schemas.layer import LayerScoreCreate
from app.schemas.signal import SignalRead


class ShadowLayer(LayerBase):
    """
    Shadow Layer — "What unspoken forces are driving belief?"

    Captures institutional, political, and structural pressures that shape
    belief indirectly — policy signals, regulatory posture, official statements.
    Shadow forces often move before they are visible in price or sentiment.
    """

    layer_name = LayerType.SHADOW

    _layer_prompt = """\
[SCORE-LAYER: SHADOW]
Topic: {topic} | Window: {time_window}

Signals:
{signals}

Question: What unspoken institutional, political, or structural forces are shaping belief \
about this topic? Look for policy signals, regulatory posture, official framing, and hidden flows \
that move before they appear in price or public sentiment.

Output ONLY this JSON (no other text):
{{"score": <-1.0 to +1.0>, "confidence": <0.0 to 1.0>}}

score: -1.0=strong institutional headwind, +1.0=strong institutional tailwind, 0.0=neutral or unknown
confidence: how much signal data supports this shadow reading
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
