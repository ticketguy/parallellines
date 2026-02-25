"""
Connector package — plug-and-play data source adapters.

All connectors in this directory are auto-discovered by the registry.
Import this package to trigger discovery.
"""
from app.connectors.base import BaseConnector, RawSignal
from app.connectors.registry import get_active_connectors, get_registry, register_connector

__all__ = [
    "BaseConnector",
    "RawSignal",
    "register_connector",
    "get_registry",
    "get_active_connectors",
]
