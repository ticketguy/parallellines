"""
Connector registry — auto-discovers all connectors in this package.

Usage:
    from app.connectors.registry import register_connector, get_active_connectors

    @register_connector
    class MyConnector(BaseConnector):
        name = "my_source"
        ...
"""
import importlib
import logging
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.connectors.base import BaseConnector

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type["BaseConnector"]] = {}


def register_connector(cls: type["BaseConnector"]) -> type["BaseConnector"]:
    """Decorator that registers a connector class by its `name` attribute."""
    _REGISTRY[cls.name] = cls
    logger.debug("Registered connector: %s", cls.name)
    return cls


def get_registry() -> dict[str, type["BaseConnector"]]:
    return dict(_REGISTRY)


def get_active_connectors() -> list["BaseConnector"]:
    """
    Instantiate and return all registered, enabled connectors.
    Each connector is created via cls.from_env() so it can read
    its own config from environment variables.
    """
    connectors: list["BaseConnector"] = []
    for name, cls in _REGISTRY.items():
        if not getattr(cls, "enabled", True):
            continue
        try:
            connectors.append(cls.from_env())
        except Exception as exc:
            logger.warning("Failed to instantiate connector %r: %s", name, exc)
    return connectors


def _autodiscover() -> None:
    """
    Import every module in app/connectors/ so their @register_connector
    decorators fire. Called once at import time.
    """
    pkg_path = Path(__file__).parent
    pkg_name = "app.connectors"
    for info in pkgutil.iter_modules([str(pkg_path)]):
        if info.name in ("base", "registry", "__init__"):
            continue
        try:
            importlib.import_module(f"{pkg_name}.{info.name}")
        except Exception as exc:
            logger.warning("Could not load connector module %r: %s", info.name, exc)


_autodiscover()
