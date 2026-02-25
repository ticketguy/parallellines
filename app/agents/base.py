from abc import ABC, abstractmethod

from app.schemas.signal import SignalCreate


class SubmindBase(ABC):
    """
    Abstract base for all submind agents.

    Each submind is responsible for a single data source (e.g. Polymarket,
    Twitter, Reuters). It knows how to fetch raw data and translate it into
    normalised SignalCreate objects that the pipeline can ingest.
    """

    name: str       # e.g. "polymarket", "twitter"
    layer: str      # LayerType value — which layer this submind feeds

    @abstractmethod
    async def fetch(self) -> list[SignalCreate]:
        """Fetch data from the source and return normalised signals."""
        ...

    @abstractmethod
    async def process(self, raw: dict) -> dict:
        """Translate a single raw record into processed feature dict."""
        ...
