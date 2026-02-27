from app.constants import LayerType
from app.layers.base import LayerBase
from app.schemas.layer import LayerScoreCreate
from app.schemas.signal import SignalRead


class ConvictionLayer(LayerBase):
    """
    Conviction Layer — "How deeply is the belief held?"

    Measures emotional intensity and depth of commitment in expressed belief,
    not just positive/negative polarity. High conviction can exist on both
    sides of a probability. Conviction is depth; Echo is spread.
    """

    layer_name = LayerType.CONVICTION

    _layer_prompt = """\
[SCORE-LAYER: CONVICTION]
Topic: {topic} | Window: {time_window}

Signals:
{signals}

Read every signal carefully. How deeply is belief held in what you're reading? Look for intensity \
of language, certainty of claims, emotional commitment, and whether sources express conviction \
or hedged uncertainty. A market priced at 0.99 shows deep conviction. Strongly worded articles \
show conviction. Vague or balanced coverage shows shallow belief.

Extract specific phrases, price levels, or data points that reveal depth of belief. \
Note sources where conviction is unusually strong or strikingly absent.

Output ONLY this JSON (no other text):
{{
  "score": <-1.0 to +1.0>,
  "confidence": <0.0 to 1.0>,
  "key_data": ["<specific evidence of conviction depth>", ...],
  "reasoning": "<what the signals reveal about how deeply belief is held>",
  "notable": "<any unusually strong or absent conviction worth flagging — or null>"
}}

score: -1.0=deep conviction against the outcome, +1.0=deep conviction for it, 0.0=shallow or divided
confidence: how much signal data supports this conviction reading
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
