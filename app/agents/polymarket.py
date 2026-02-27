from datetime import datetime, timezone

import httpx

from app.agents.base import SubmindBase
from app.config import settings
from app.constants import LayerType
from app.schemas.signal import SignalCreate


class PolymarketSubmind(SubmindBase):
    """
    Fetches active prediction markets from Polymarket's Gamma API and
    normalises them into signals carrying objective market facts.

    processed_data holds only what the API directly provides:
    question text, yes/no prices, volume, liquidity, category.
    No computed scores — the LLM layers read and interpret the content.
    """

    name = "polymarket"
    layer = LayerType.PROBABILITY
    domain = "market"

    async def fetch(self) -> list[SignalCreate]:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{settings.GAMMA_API_URL}/markets",
                params={
                    "limit": 100,
                    "active": "true",
                    "order": "volume24Hr",
                    "ascending": "false",
                },
            )
            resp.raise_for_status()
            markets: list[dict] = resp.json()

        return [
            self._to_signal(m)
            for m in markets
            if float(m.get("volume24Hr") or 0) > 0
        ]

    def _to_signal(self, market: dict) -> SignalCreate:
        tokens: list[dict] = market.get("tokens") or []
        yes_price = next(
            (float(t["price"]) for t in tokens if t.get("outcome") == "Yes"), None
        )
        no_price = next(
            (float(t["price"]) for t in tokens if t.get("outcome") == "No"), None
        )

        category = market.get("category")

        return SignalCreate(
            source=self.name,
            layer=self.layer,
            domain=self.domain,
            raw_data=market,
            processed_data={
                "question": market.get("question"),
                "yes_price": yes_price,
                "no_price": no_price,
                "volume_24h": market.get("volume24Hr"),
                "total_volume": market.get("volume"),
                "liquidity": market.get("liquidity"),
                "category": category,
                "end_date": market.get("endDate"),
                "market_slug": market.get("slug"),
            },
            confidence=0.9,
            topic_tags=[category] if category else [],
            entity_tags=[],
            signal_timestamp=datetime.now(timezone.utc),
        )

    async def process(self, raw: dict) -> dict:
        return self._to_signal(raw).processed_data
