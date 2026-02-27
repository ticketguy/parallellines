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

Read every signal carefully. Is belief about this topic persisting even where contradicting \
evidence exists? Look for narratives that keep recurring despite counter-data, claims that \
were already refuted but are still circulating, and framing that hasn't updated to reflect \
new information. Also look for belief that IS shifting — that is low Memory.

Extract the specific stale narratives, contradicted claims that persist, or evidence of \
belief updating. Name the narratives that are refusing to die.

Output ONLY this JSON (no other text):
{{
  "score": <-1.0 to +1.0>,
  "confidence": <0.0 to 1.0>,
  "key_data": ["<specific persisting narrative or updating pattern>", ...],
  "reasoning": "<what the signals reveal about narrative persistence or decay>",
  "notable": "<any zombie belief or striking narrative shift worth flagging — or null>"
}}

score: -1.0=negative narrative persisting despite counter-evidence, +1.0=positive narrative \
persisting, 0.0=belief tracking reality normally
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
