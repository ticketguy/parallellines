from app.constants import LayerType
from app.layers.base import LayerBase
from app.schemas.layer import LayerScoreCreate
from app.schemas.signal import SignalRead


class EchoLayer(LayerBase):
    """
    Echo Layer — "How widely is belief spreading?"

    Measures repetition, reinforcement, and contagion across sources.
    Echo is distinct from Conviction: a belief can spread widely without being
    deeply held. High Echo without Conviction is noise amplification.
    """

    layer_name = LayerType.ECHO

    _layer_prompt = """\
[SCORE-LAYER: ECHO]
Topic: {topic} | Window: {time_window}

Signals:
{signals}

Question: How widely is belief about this topic spreading and being repeated across sources? \
Look for repetition, reinforcement, and narrative contagion — not depth of belief, but reach and spread.

Output ONLY this JSON (no other text):
{{"score": <-1.0 to +1.0>, "confidence": <0.0 to 1.0>}}

score: -1.0=negative narrative spreading widely, +1.0=positive narrative spreading widely, 0.0=no amplification
confidence: how much signal data supports this echo reading
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
