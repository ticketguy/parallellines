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

Read every signal carefully. How widely is belief about this topic spreading and being reinforced \
across sources? Look for repetition of the same narrative across multiple sources, viral framing, \
amplification patterns, and contagion of specific claims. High volume and wide reach without deep \
analysis is Echo. The same talking point appearing everywhere is Echo.

Extract the specific repeated narratives, shared framing, or amplified claims you find. \
Note where reach is high but depth is low — that gap is the Echo signal.

Output ONLY this JSON (no other text):
{{
  "score": <-1.0 to +1.0>,
  "confidence": <0.0 to 1.0>,
  "key_data": ["<specific repeated narrative or amplification pattern>", ...],
  "reasoning": "<what the signals reveal about spread and repetition of belief>",
  "notable": "<any unusually viral narrative or striking absence of amplification — or null>"
}}

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
        sc, conf, extra = await self._llm_score(signals, topic, time_window)
        return LayerScoreCreate(
            layer=self.layer_name,
            topic=topic,
            score=round(sc, 4),
            confidence=round(conf, 4),
            signal_count=len(signals),
            time_window=time_window,
            extra_data=extra or None,
        )
