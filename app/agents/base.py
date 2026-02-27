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
        `audit()` is a concrete orchestrator defined here — it runs the five
        cognitive checks, computes challenge intensity and summary, and returns
        a SubmindAudit. Subminds implement four source-specific hooks:

          _consistency_check   — does the synthesis direction match raw signal data?
          _counter_narrative   — what is the opposite read, and where is confidence unearned?
          _gap_check           — what was underweighted or missed?
          _noise_check         — how reliable is this signal pool? (returns penalty + flags)

        Two checks are generic and handled entirely in this base class:

          Drift                — synthesis score shift vs prior (source-agnostic)
          Overconfidence       — high layer confidence on thin sample (source-agnostic)

    `layer` is metadata describing the primary nature of the source
    (e.g. "probability" for prediction markets, "echo" for social feeds).
    It is not a routing key — all signals reach all layers.
    """

    name: str       # e.g. "polymarket", "twitter"
    layer: str      # Source metadata — primary nature of this data source
    domain: str     # World domain — free-form, e.g. "market", "social", "news"

    # ── fetch / process (data ingestion) ─────────────────────────────────────

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

    # ── audit hooks (source-specific, implement in each submind) ─────────────

    @abstractmethod
    def _consistency_check(
        self,
        layer_scores: dict[str, Any],
        my_signals: list[SignalRead],
    ) -> list[str]:
        """
        Check 1 — Checking.
        Does IntuOne's synthesis direction match what the raw signals say?
        Return a list of flag strings describing each inconsistency found.
        Return [] if the signals broadly support the synthesis reading.
        """
        ...

    @abstractmethod
    def _counter_narrative(
        self,
        layer_scores: dict[str, Any],
        my_signals: list[SignalRead],
    ) -> tuple[str | None, list[str]]:
        """
        Check 2 — Challenging.
        What does the opposite read look like? Where is confidence unearned?
        Return (counter_narrative_text | None, overconfidence_warnings list).
        """
        ...

    @abstractmethod
    def _gap_check(
        self,
        my_signals: list[SignalRead],
    ) -> list[str]:
        """
        Check 4 — Gaps.
        Which signals or domains were underweighted or missing?
        Return a list of gap description strings.
        Return [] if coverage looks adequate.
        """
        ...

    @abstractmethod
    def _noise_check(
        self,
        my_signals: list[SignalRead],
    ) -> tuple[float, list[str]]:
        """
        Check 5 — Meta-awareness.
        How reliable is this signal pool?
        Return (reliability_penalty: float, noise_flags: list[str]).
        penalty is subtracted from 1.0 to produce index_reliability — clamp [0, 1].
        Return (0.0, []) if the pool looks clean.
        """
        ...

    # ── audit orchestrator (generic, runs for every submind) ─────────────────

    async def audit(
        self,
        layer_scores: dict[str, Any],
        signals: list[SignalRead],
        prior_score: float | None = None,
    ) -> SubmindAudit:
        """
        Parallel cognitive audit of the Perception Index.

        Called after all five layers have scored but before IntuOne generates
        its briefing. Orchestrates the five checks — two generic (drift,
        overconfidence) handled here, three delegated to source-specific hooks.

        prior_score: synthesis score from the last known report on this topic,
        used for drift detection. None skips drift detection.
        """
        my_signals = [s for s in signals if s.source == self.name]

        if not my_signals:
            return SubmindAudit(
                submind=self.name,
                noise_flags=[f"No {self.name} signals in pool — audit skipped"],
                index_reliability=0.0,
                challenge_intensity=0.3,
                summary=f"No {self.name} data in signal pool — audit could not run.",
            )

        synth = layer_scores.get("synthesis", {})
        synthesis_score: float = synth.get("score", 0.0)
        synthesis_conf: float = synth.get("confidence", 0.0)
        layer_conf: float = layer_scores.get(self.layer, {}).get("confidence", 0.0)

        # ── 1. Checking (source-specific) ─────────────────────────────────────
        consistency_flags = self._consistency_check(layer_scores, my_signals)

        # ── 2. Challenging (source-specific) ──────────────────────────────────
        counter_narrative, overconfidence_warnings = self._counter_narrative(
            layer_scores, my_signals
        )

        # Generic overconfidence checks (apply to every submind)
        if layer_conf > 0.85 and len(my_signals) < 5:
            overconfidence_warnings = list(overconfidence_warnings) + [
                f"{self.layer.capitalize()} layer confidence {layer_conf:.0%} on only "
                f"{len(my_signals)} signals — insufficient sample for this certainty"
            ]
        if synthesis_conf > 0.80 and len(consistency_flags) > 0:
            overconfidence_warnings = list(overconfidence_warnings) + [
                f"Synthesis confidence {synthesis_conf:.0%} despite "
                f"{len(consistency_flags)} consistency flag(s)"
            ]

        # ── 3. Drift (generic — source-agnostic) ──────────────────────────────
        drift_detected = False
        drift_explanation: str | None = None
        if prior_score is not None:
            delta = abs(synthesis_score - prior_score)
            if delta > 0.30:
                drift_detected = True
                direction = "strengthened" if synthesis_score > prior_score else "weakened"
                drift_explanation = (
                    f"Synthesis moved from {prior_score:+.2f} to {synthesis_score:+.2f} "
                    f"(Δ{delta:.2f}) — narrative has {direction} significantly"
                )

        # ── 4. Gaps (source-specific) ──────────────────────────────────────────
        underweighted_signals = self._gap_check(my_signals)

        # ── 5. Meta-awareness (source-specific) ───────────────────────────────
        noise_penalty, noise_flags = self._noise_check(my_signals)
        index_reliability = max(0.0, min(1.0, 1.0 - noise_penalty))

        # ── Challenge intensity ────────────────────────────────────────────────
        flag_count = (
            len(consistency_flags)
            + len(overconfidence_warnings)
            + len(noise_flags)
            + len(underweighted_signals)
            + (1 if drift_detected else 0)
            + (1 if counter_narrative else 0)
        )
        challenge_intensity = min(1.0, flag_count * 0.15)

        # ── Summary ───────────────────────────────────────────────────────────
        if challenge_intensity < 0.20:
            summary = (
                f"{self.name.capitalize()} signals broadly endorse the synthesis "
                f"({synthesis_score:+.2f}). {len(my_signals)} signals read, "
                f"index reliability {index_reliability:.0%}."
            )
        elif challenge_intensity < 0.50:
            summary = (
                f"{self.name.capitalize()} audit raises moderate concerns "
                f"({len(consistency_flags)} consistency flag(s), "
                f"{len(noise_flags)} noise flag(s)). "
                f"Synthesis ({synthesis_score:+.2f}) should be read with caution."
            )
        else:
            summary = (
                f"{self.name.capitalize()} audit is significantly challenging the "
                f"synthesis ({synthesis_score:+.2f}). Multiple flags — treat "
                "index confidence as overstated."
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
