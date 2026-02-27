from abc import ABC, abstractmethod

from app.schemas.signal import SignalCreate


class SubmindBase(ABC):
    """
    Abstract base for all submind agents.

    Each submind is responsible for a single data source (e.g. Polymarket,
    Twitter, Reuters). It knows how to fetch raw data and normalise it into
    SignalCreate objects that the ingestion pipeline can store.

    Subminds produce objective, factual signals — no computed scores,
    no interpretation. The LLM layers read the raw content and produce
    all perceptual readings.

    `layer` is metadata describing the primary nature of the source
    (e.g. "probability" for prediction markets, "echo" for social feeds).
    It is not a routing key — all signals reach all layers.

    `process()` must return only objective facts extracted from the raw
    API response. No formulas, no sentiment heuristics, no derived scores.
    """

    name: str       # e.g. "polymarket", "twitter"
    layer: str      # Source metadata — primary nature of this data source
    domain: str     # World domain — free-form, e.g. "market", "social",
                    # "news", "geopolitical", "crypto", "politics"

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
