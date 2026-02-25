from abc import ABC, abstractmethod

from app.schemas.layer import LayerScoreCreate
from app.schemas.signal import SignalRead


class LayerBase(ABC):
    """
    Abstract base for all six layer processors.

    Each layer receives signals that belong to it and computes a
    directional score (-1.0 to +1.0) with an associated confidence.
    """

    layer_name: str

    @abstractmethod
    async def score(
        self,
        signals: list[SignalRead],
        topic: str,
        time_window: str = "24h",
    ) -> LayerScoreCreate:
        """Compute a layer score from the given signals."""
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
