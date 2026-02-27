import statistics
from datetime import datetime, timezone
from typing import Any

import httpx

from app.agents.base import SubmindBase
from app.config import settings
from app.constants import LayerType
from app.schemas.signal import SignalCreate, SignalRead


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

    # ── fetch / process ───────────────────────────────────────────────────────

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

    # ── audit hooks ───────────────────────────────────────────────────────────

    def _consistency_check(
        self,
        layer_scores: dict[str, Any],
        my_signals: list[SignalRead],
    ) -> list[str]:
        """
        Does IntuOne's synthesis direction match the crowd YES prices?
        Checks: majority direction, and highest-volume market alignment.
        """
        synthesis_score: float = layer_scores.get("synthesis", {}).get("score", 0.0)
        markets = [s.processed_data or {} for s in my_signals]
        yes_prices = [m["yes_price"] for m in markets if m.get("yes_price") is not None]
        flags: list[str] = []

        if not yes_prices:
            return flags

        bullish_fraction = sum(1 for p in yes_prices if p > 0.50) / len(yes_prices)

        if synthesis_score > 0.40 and bullish_fraction < 0.35:
            flags.append(
                f"Synthesis reads {synthesis_score:+.2f} (bullish) but only "
                f"{bullish_fraction:.0%} of markets have YES > 0.50"
            )
        elif synthesis_score < -0.40 and bullish_fraction > 0.65:
            flags.append(
                f"Synthesis reads {synthesis_score:+.2f} (bearish) but "
                f"{bullish_fraction:.0%} of markets price YES above 0.50"
            )

        # Highest-volume market alignment
        top = max(markets, key=lambda m: float(m.get("volume_24h") or 0), default=None)
        if top and top.get("yes_price") is not None:
            top_yes = top["yes_price"]
            top_q = (top.get("question") or "unknown")[:80]
            if synthesis_score > 0.30 and top_yes < 0.40:
                flags.append(
                    f"Highest-volume market ({top_q!r}) prices YES at {top_yes:.2f} "
                    f"— contradicts bullish synthesis"
                )
            elif synthesis_score < -0.30 and top_yes > 0.60:
                flags.append(
                    f"Highest-volume market ({top_q!r}) prices YES at {top_yes:.2f} "
                    f"— contradicts bearish synthesis"
                )

        return flags

    def _counter_narrative(
        self,
        layer_scores: dict[str, Any],
        my_signals: list[SignalRead],
    ) -> tuple[str | None, list[str]]:
        """
        Find high-volume markets betting against the synthesis direction.
        Returns (counter_narrative text, overconfidence_warnings).
        Overconfidence warnings beyond the generic base checks go here.
        """
        synthesis_score: float = layer_scores.get("synthesis", {}).get("score", 0.0)
        markets = [s.processed_data or {} for s in my_signals]

        opposite: list[tuple[float, dict]] = []
        for m in markets:
            yp = m.get("yes_price")
            vol = float(m.get("volume_24h") or 0)
            if yp is None or vol == 0:
                continue
            if synthesis_score > 0.10 and yp < 0.40:
                opposite.append((vol, m))
            elif synthesis_score < -0.10 and yp > 0.60:
                opposite.append((vol, m))

        counter_narrative: str | None = None
        if opposite:
            opposite.sort(key=lambda x: x[0], reverse=True)
            top_opp = opposite[0][1]
            opp_q = (top_opp.get("question") or "unknown")[:100]
            opp_yp = top_opp.get("yes_price", 0.0)
            opp_vol = float(top_opp.get("volume_24h") or 0)
            direction_word = "NO" if synthesis_score > 0.10 else "YES"
            counter_narrative = (
                f"{len(opposite)} market(s) with significant volume are pricing "
                f"{direction_word}. Largest: {opp_q!r} "
                f"(yes_price={opp_yp:.2f}, 24h_vol=${opp_vol:,.0f})"
            )

        return counter_narrative, []

    def _gap_check(self, my_signals: list[SignalRead]) -> list[str]:
        """
        Category concentration risk and thin sample detection.
        """
        markets = [s.processed_data or {} for s in my_signals]
        gaps: list[str] = []

        categories = [m.get("category") for m in markets if m.get("category")]
        if len(categories) > 3:
            most_common = max(set(categories), key=categories.count)
            cat_fraction = categories.count(most_common) / len(categories)
            if cat_fraction > 0.60:
                gaps.append(
                    f"{cat_fraction:.0%} of markets are in category '{most_common}' "
                    "— other domains may be underrepresented"
                )

        if len(my_signals) < 5:
            gaps.append(
                f"Only {len(my_signals)} Polymarket signals in pool — "
                "broader market sweep may surface missed signals"
            )

        return gaps

    def _noise_check(self, my_signals: list[SignalRead]) -> tuple[float, list[str]]:
        """
        YES price spread and volume thinness.
        Returns (reliability_penalty, noise_flags).
        """
        markets = [s.processed_data or {} for s in my_signals]
        yes_prices = [m["yes_price"] for m in markets if m.get("yes_price") is not None]
        flags: list[str] = []
        penalty = 0.0

        if len(yes_prices) > 2:
            std = statistics.stdev(yes_prices)
            if std > 0.30:
                flags.append(
                    f"YES price spread is wide (std={std:.2f}) — "
                    "no clear directional consensus in prediction markets"
                )
                penalty += 0.25

        total_vol = sum(float(m.get("volume_24h") or 0) for m in markets)
        if total_vol < 10_000:
            flags.append(
                f"Total 24h volume is thin (${total_vol:,.0f}) — "
                "markets may not reflect informed crowd belief"
            )
            penalty += 0.20

        if len(my_signals) < 3:
            flags.append(
                f"Only {len(my_signals)} markets sampled — index is low-confidence"
            )
            penalty += 0.30

        return penalty, flags
