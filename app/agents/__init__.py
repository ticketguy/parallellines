from app.agents.polymarket import PolymarketSubmind

# All active Subminds. The pipeline iterates this list to run parallel audits
# alongside IntuOne after layer scoring.
REGISTERED_SUBMINDS = [PolymarketSubmind()]

__all__ = ["PolymarketSubmind", "REGISTERED_SUBMINDS"]
