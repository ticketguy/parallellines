"""
Plug-and-play connector base class.

To add a new data source:
  1. Create a file in app/connectors/ (e.g. reddit.py)
  2. Subclass BaseConnector
  3. Decorate the class with @register_connector
  4. Implement async fetch() -> list[RawSignal]

The ingestion service will automatically discover and poll your connector.
"""
from abc import ABC, abstractmethod
from typing import Any, TypedDict


class RawSignal(TypedDict, total=False):
    """Normalised signal dict produced by any connector."""
    # Required
    source: str           # e.g. "polymarket", "reddit", "web_crawler"
    layer: str            # LayerType value — which perception layer this feeds
    domain: str           # World layer — free-form domain string, e.g. "market",
                          # "news", "social", "geopolitical", "crypto", "politics".
                          # No enum — new domains require no code changes.
    topic_tags: list[str] # topics this signal relates to

    # Scoring hint (0.0–1.0 signal strength; set by connector if known)
    signal_strength: float | None

    # Raw data (preserved for debugging / future re-processing)
    raw_data: dict[str, Any]
    # Structured data the layer scorer will use
    processed_data: dict[str, Any]

    # Optional metadata
    url: str | None
    external_id: str | None  # dedup key


class BaseConnector(ABC):
    """
    Abstract base for all data-source connectors.

    Class attributes:
        name   — unique slug, e.g. "polymarket"
        layer  — which perception layer this connector feeds (LayerType value)
        domain — world layer / data domain, free-form string e.g. "market",
                 "news", "social". New domains need no code changes.
        enabled — set to False to skip without removing
    """

    name: str
    layer: str
    domain: str = ""
    enabled: bool = True

    @abstractmethod
    async def fetch(self) -> list[RawSignal]:
        """
        Pull the latest signals from the source.

        Should be idempotent — the ingestion service deduplicates by
        external_id so duplicate fetches are safe.
        """

    @classmethod
    def from_env(cls) -> "BaseConnector":
        """
        Instantiate from environment / settings.
        Override in subclasses that need API keys or config.
        """
        return cls()  # type: ignore[call-arg]

    def __repr__(self) -> str:
        return f"<Connector name={self.name!r} layer={self.layer!r} enabled={self.enabled}>"
