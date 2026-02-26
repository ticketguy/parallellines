from app.constants import LayerType
from app.layers.base import LayerBase
from app.schemas.layer import LayerScoreCreate
from app.schemas.signal import SignalRead


class MemoryLayer(LayerBase):
    """
    Memory Layer — "Is belief persisting after contradiction?"

    Tracks narrative persistence: how a belief continues to circulate even
    after data has moved against it. High Memory with falling Probability
    signals a zombie belief — refusing to die despite counter-evidence.
    """

    layer_name = LayerType.MEMORY

    _layer_prompt = """\
[SCORE-LAYER: MEMORY]
Topic: {topic} | Window: {time_window}

Signals:
{signals}

Question: Is belief about this topic persisting even after contradicting evidence? \
Are narratives stubbornly recurring in coverage, or is belief shifting as new data arrives?

Output ONLY this JSON (no other text):
{{"score": <-1.0 to +1.0>, "confidence": <0.0 to 1.0>}}

score: -1.0=negative narrative persisting despite counter-evidence, +1.0=positive narrative persisting, \
0.0=belief tracking reality normally
confidence: how much signal data supports this memory reading
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
