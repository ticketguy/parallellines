import json
import logging
import re
from abc import ABC, abstractmethod

from app.schemas.layer import LayerScoreCreate
from app.schemas.signal import SignalRead

logger = logging.getLogger(__name__)


class LayerBase(ABC):
    """
    Abstract base for all five perception layer processors.

    Each layer asks a specific perceptual question about the input signals
    and returns a directional score (-1.0 to +1.0) with confidence and
    detailed observations extracted from the raw content.
    Scoring is done by the local IntuOne model — no arithmetic, no heuristics.
    When the model is not loaded, the layer returns an empty score (confidence=0).
    """

    layer_name: str

    # Each subclass defines its perceptual question as a prompt template.
    # Available variables: {topic}, {time_window}, {signals}
    _layer_prompt: str = ""

    @abstractmethod
    async def score(
        self,
        signals: list[SignalRead],
        topic: str,
        time_window: str = "24h",
    ) -> LayerScoreCreate:
        """Score this layer over the given signals via the local model."""
        ...

    def _empty_score(self, topic: str, time_window: str) -> LayerScoreCreate:
        return LayerScoreCreate(
            layer=self.layer_name,
            topic=topic,
            score=0.0,
            confidence=0.0,
            signal_count=0,
            time_window=time_window,
        )

    def _format_signals(self, signals: list[SignalRead]) -> str:
        """Render signals as plain text for the model prompt."""
        lines = []
        for i, sig in enumerate(signals[:20], 1):
            proc = sig.processed_data or {}
            raw = sig.raw_data or {}
            content = (
                proc.get("text")
                or proc.get("question")
                or raw.get("question")
                or raw.get("headline")
                or f"signal from {sig.source}"
            )[:300]

            facts = []
            if proc.get("yes_price") is not None:
                facts.append(f"yes_price={proc['yes_price']:.2f}")
            if proc.get("volume_24h"):
                facts.append(f"volume_24h={proc['volume_24h']:.0f}")
            if proc.get("liquidity"):
                facts.append(f"liquidity={proc['liquidity']:.0f}")

            fact_str = f" [{', '.join(facts)}]" if facts else ""
            lines.append(f"{i}. [{sig.source}]{fact_str} {content}")

        return "\n".join(lines) if lines else "No signals."

    async def _llm_score(
        self,
        signals: list[SignalRead],
        topic: str,
        time_window: str,
    ) -> tuple[float, float, dict]:
        """
        Ask the local IntuOne model to score this layer.
        Returns (score, confidence, extra_data).
        All zeroes/empty if model not loaded.
        extra_data contains the model's detailed observations from the content.
        """
        from app.inference.pipeline import generate_briefing_local, is_model_loaded

        if not is_model_loaded():
            return 0.0, 0.0, {}

        prompt = self._layer_prompt.format(
            topic=topic,
            time_window=time_window,
            signals=self._format_signals(signals),
        )

        try:
            raw = generate_briefing_local(prompt, max_new_tokens=200, temperature=0.1)
            # Extract the JSON block — may be multi-line
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
                score = max(-1.0, min(1.0, float(data.get("score", 0.0))))
                conf = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
                extra: dict = {
                    k: v for k, v in data.items() if k not in ("score", "confidence")
                }
                return score, conf, extra
        except Exception as exc:
            logger.debug("[%s] LLM scoring failed: %s", self.layer_name, exc)

        return 0.0, 0.0, {}
