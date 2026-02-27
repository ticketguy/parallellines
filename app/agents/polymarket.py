import statistics
from datetime import datetime, timezone
from typing import Any

import httpx

from app.agents.base import SubmindBase
from app.config import settings
from app.constants import LayerType
from app.schemas.audit import SubmindAudit
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

    # ── audit ─────────────────────────────────────────────────────────────────

    async def audit(
        self,
        layer_scores: dict[str, Any],
        signals: list[SignalRead],
        prior_score: float | None = None,
    ) -> SubmindAudit:
        """
        Parallel cognitive audit of the Perception Index using Polymarket data.

        Five checks:
          1. Checking       — do raw YES prices align with IntuOne's direction?
          2. Challenging    — which high-volume markets bet the opposite way?
          3. Drift          — has synthesis score moved significantly from prior?
          4. Gaps           — is the market sample concentrated or thin?
          5. Meta-awareness — is YES price consensus noisy? Is volume thin?
        """
        my_signals = [s for s in signals if s.source == self.name]

        if not my_signals:
            return SubmindAudit(
                submind=self.name,
                noise_flags=["No Polymarket signals in pool — audit skipped"],
                index_reliability=0.0,
                challenge_intensity=0.3,
                summary="No prediction market data in signal pool — audit could not run.",
            )

        markets = [s.processed_data or {} for s in my_signals]

        synth = layer_scores.get("synthesis", {})
        synthesis_score: float = synth.get("score", 0.0)
        synthesis_conf: float = synth.get("confidence", 0.0)
        prob_conf: float = layer_scores.get("probability", {}).get("confidence", 0.0)

        consistency_flags: list[str] = []
        counter_narrative: str | None = None
        overconfidence_warnings: list[str] = []
        drift_detected = False
        drift_explanation: str | None = None
        underweighted_signals: list[str] = []
        noise_flags: list[str] = []
        index_reliability = 1.0

        yes_prices = [
            m["yes_price"] for m in markets if m.get("yes_price") is not None
        ]

        # ── 1. CHECKING ───────────────────────────────────────────────────────
        # Does the synthesis direction match the crowd YES prices?
        if yes_prices:
            bullish_count = sum(1 for p in yes_prices if p > 0.50)
            bullish_fraction = bullish_count / len(yes_prices)

            if synthesis_score > 0.40 and bullish_fraction < 0.35:
                consistency_flags.append(
                    f"Synthesis reads {synthesis_score:+.2f} (bullish) but only "
                    f"{bullish_fraction:.0%} of markets have YES > 0.50"
                )
            elif synthesis_score < -0.40 and bullish_fraction > 0.65:
                consistency_flags.append(
                    f"Synthesis reads {synthesis_score:+.2f} (bearish) but "
                    f"{bullish_fraction:.0%} of markets price YES above 0.50"
                )

            # Highest-volume market alignment
            top = max(markets, key=lambda m: float(m.get("volume_24h") or 0), default=None)
            if top and top.get("yes_price") is not None:
                top_yes = top["yes_price"]
                top_q = (top.get("question") or "unknown")[:80]
                if synthesis_score > 0.30 and top_yes < 0.40:
                    consistency_flags.append(
                        f"Highest-volume market ({top_q!r}) prices YES at {top_yes:.2f} "
                        f"— contradicts bullish synthesis"
                    )
                elif synthesis_score < -0.30 and top_yes > 0.60:
                    consistency_flags.append(
                        f"Highest-volume market ({top_q!r}) prices YES at {top_yes:.2f} "
                        f"— contradicts bearish synthesis"
                    )

        # ── 2. CHALLENGING ────────────────────────────────────────────────────
        # Find high-volume markets betting against the synthesis direction
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

        # Overconfidence on thin sample
        if prob_conf > 0.85 and len(my_signals) < 5:
            overconfidence_warnings.append(
                f"Probability layer confidence {prob_conf:.0%} on only "
                f"{len(my_signals)} markets — insufficient sample for this certainty"
            )
        if synthesis_conf > 0.80 and len(consistency_flags) > 0:
            overconfidence_warnings.append(
                f"Synthesis confidence {synthesis_conf:.0%} despite "
                f"{len(consistency_flags)} consistency flag(s)"
            )

        # ── 3. DRIFT ──────────────────────────────────────────────────────────
        if prior_score is not None:
            delta = abs(synthesis_score - prior_score)
            if delta > 0.30:
                drift_detected = True
                direction = "strengthened" if synthesis_score > prior_score else "weakened"
                drift_explanation = (
                    f"Synthesis moved from {prior_score:+.2f} to {synthesis_score:+.2f} "
                    f"(Δ{delta:.2f}) — narrative has {direction} significantly"
                )

        # ── 4. GAPS ───────────────────────────────────────────────────────────
        # Category concentration
        categories = [m.get("category") for m in markets if m.get("category")]
        if len(categories) > 3:
            most_common = max(set(categories), key=categories.count)
            cat_fraction = categories.count(most_common) / len(categories)
            if cat_fraction > 0.60:
                underweighted_signals.append(
                    f"{cat_fraction:.0%} of markets are in category '{most_common}' "
                    "— other domains may be underrepresented"
                )

        if len(my_signals) < 5:
            underweighted_signals.append(
                f"Only {len(my_signals)} Polymarket signals in pool — "
                "broader market sweep may surface missed signals"
            )

        # ── 5. META-AWARENESS ─────────────────────────────────────────────────
        # YES price spread — high variance = noisy consensus
        if len(yes_prices) > 2:
            std = statistics.stdev(yes_prices)
            if std > 0.30:
                noise_flags.append(
                    f"YES price spread is wide (std={std:.2f}) — "
                    "no clear directional consensus in prediction markets"
                )
                index_reliability -= 0.25

        # Thin volume pool
        total_vol = sum(float(m.get("volume_24h") or 0) for m in markets)
        if total_vol < 10_000:
            noise_flags.append(
                f"Total 24h volume is thin (${total_vol:,.0f}) — "
                "markets may not reflect informed crowd belief"
            )
            index_reliability -= 0.20

        # Very few markets
        if len(my_signals) < 3:
            noise_flags.append(
                f"Only {len(my_signals)} markets sampled — index is low-confidence"
            )
            index_reliability -= 0.30

        index_reliability = max(0.0, min(1.0, index_reliability))

        # ── CHALLENGE INTENSITY ───────────────────────────────────────────────
        flag_count = (
            len(consistency_flags)
            + len(overconfidence_warnings)
            + len(noise_flags)
            + len(underweighted_signals)
            + (1 if drift_detected else 0)
            + (1 if counter_narrative else 0)
        )
        challenge_intensity = min(1.0, flag_count * 0.15)

        # ── SUMMARY ───────────────────────────────────────────────────────────
        if challenge_intensity < 0.20:
            summary = (
                f"Polymarket signals broadly endorse the synthesis ({synthesis_score:+.2f}). "
                f"{len(my_signals)} markets read, index reliability {index_reliability:.0%}."
            )
        elif challenge_intensity < 0.50:
            summary = (
                f"Polymarket audit raises moderate concerns "
                f"({len(consistency_flags)} consistency flag(s), "
                f"{len(noise_flags)} noise flag(s)). "
                f"Synthesis ({synthesis_score:+.2f}) should be read with caution."
            )
        else:
            summary = (
                f"Polymarket audit is significantly challenging the synthesis "
                f"({synthesis_score:+.2f}). Multiple flags — treat index "
                "confidence as overstated."
            )

        return SubmindAudit(
            submind=self.name,
            consistency_flags=consistency_flags,
            counter_narrative=counter_narrative,
            overconfidence_warnings=overconfidence_warnings,
            drift_detected=drift_detected,
            drift_explanation=drift_explanation,
            underweighted_signals=underweighted_signals,
            index_reliability=index_reliability,
            noise_flags=noise_flags,
            challenge_intensity=challenge_intensity,
            summary=summary,
        )
