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

Read every signal carefully. What institutional, political, regulatory, or structural forces \
are operating underneath the surface of this topic? Look for policy language, regulatory signals, \
official positioning, coordinated framing by institutions, and hidden incentive structures. \
These forces often don't show up in prices yet — they shape what's coming before it arrives.

Extract the specific policy signals, official statements, regulatory moves, or structural \
pressures you can identify. Name the institutions and the direction of their force.

Output ONLY this JSON (no other text):
{{
  "score": <-1.0 to +1.0>,
  "confidence": <0.0 to 1.0>,
  "key_data": ["<specific institution, policy signal, or structural force>", ...],
  "reasoning": "<what hidden forces are operating and which direction they push>",
  "notable": "<any significant institutional move or absence of expected shadow force — or null>"
}}

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
