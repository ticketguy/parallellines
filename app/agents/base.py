from abc import ABC, abstractmethod
from typing import Any

from app.schemas.audit import SubmindAudit
from app.schemas.signal import SignalCreate, SignalRead


class SubmindBase(ABC):
    """
    Abstract base for all submind agents.

    Each submind is responsible for a single data source (e.g. Polymarket,
    Twitter, Reuters). It has two roles:

    FETCH
        Pull raw data and normalise it into SignalCreate objects — objective
        facts only, no interpretation. The LLM layers read these and produce
        all perceptual readings.

    AUDIT
        Fire in parallel with IntuOne after all five layers have scored.
        The submind inspects its own signals against the Perception Index and
        returns a SubmindAudit covering five cognitive checks:

          1. Checking       — does IntuOne's direction match the raw signals?
          2. Challenging    — what is the counter-narrative? Where is confidence unearned?
          3. Drift          — has interpretation shifted significantly from prior?
          4. Gaps           — which signals were underweighted or missed?
          5. Meta-awareness — how reliable is the Perception Index itself?

        Audit results are injected into IntuOne's briefing context. IntuOne
        decides how to incorporate the challenges — nothing is overridden.

    `layer` is metadata describing the primary nature of the source
    (e.g. "probability" for prediction markets, "echo" for social feeds).
    It is not a routing key — all signals reach all layers.
    """

    name: str       # e.g. "polymarket", "twitter"
    layer: str      # Source metadata — primary nature of this data source
    domain: str     # World domain — free-form, e.g. "market", "social", "news"

    @abstractmethod
    async def fetch(self) -> list[SignalCreate]:
        """Fetch data from the source and return normalised signals."""
        ...

    @abstractmethod
    async def process(self, raw: dict) -> dict:
        """
        Extract objective facts from a single raw API record.
        Returns a dict of factual fields for processed_data.
        No formulas, no computed scores — only what the source directly provides.
        """
        ...

    @abstractmethod
    async def audit(
        self,
        layer_scores: dict[str, Any],
        signals: list[SignalRead],
        prior_score: float | None = None,
    ) -> SubmindAudit:
        """
        Parallel cognitive audit of the Perception Index.

        Called after all five layers have scored but before IntuOne generates
        its briefing. The submind filters to its own signals, inspects them
        against the layer readings, and returns structured challenges.

        prior_score: the synthesis score from the last known report on this
        topic, used for drift detection. None skips drift detection.
        """
        ...
