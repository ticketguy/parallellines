"""
SubmindAudit — the output of a Submind's parallel cognitive audit.

Each Submind fires alongside IntuOne after layer scoring, performing five
parallel checks before the briefing is generated:

  1. Checking       — does IntuOne's direction match the raw signals?
  2. Challenging    — what is the counter-narrative? Where is confidence unearned?
  3. Drift          — has interpretation shifted significantly from prior?
  4. Gaps           — which signals did IntuOne underweight or miss?
  5. Meta-awareness — how reliable is the Perception Index itself?

The audit is injected into IntuOne's context. It informs but does not
override — IntuOne decides how to incorporate the challenges into its briefing.
"""
from dataclasses import dataclass, field


@dataclass
class SubmindAudit:
    submind: str

    # 1. Checking — signal direction vs. IntuOne conclusion
    consistency_flags: list[str] = field(default_factory=list)

    # 2. Challenging — counter-narrative and overconfidence
    counter_narrative: str | None = None
    overconfidence_warnings: list[str] = field(default_factory=list)

    # 3. Drift — significant shift from prior reading
    drift_detected: bool = False
    drift_explanation: str | None = None

    # 4. Gaps — signals underweighted or missed
    underweighted_signals: list[str] = field(default_factory=list)

    # 5. Meta-awareness — how trustworthy is the Perception Index?
    index_reliability: float = 1.0   # 0.0 = very noisy, 1.0 = clean
    noise_flags: list[str] = field(default_factory=list)

    # Summary
    challenge_intensity: float = 0.0  # 0.0 = endorses, 1.0 = fully challenges
    summary: str = ""
