"""Shared constants for audit scenarios and labels.

Single source of truth — import from here, do not redefine locally.
"""

from __future__ import annotations

# ── Reliance interventions ────────────────────────────────────────────────────

INTERVENTION_LABELS: dict[str, str] = {
    "none": "Normal (learned Δτ)",
    "clamp_1": "Δτ clamped to 1.0",
    "reverse": "Δτ reversed (2.0 − learned)",
    "random": "Δτ ~ Uniform(0.5, 1.5)",
}
"""Human-readable labels for reliance interventions."""

# ── Robustness scenarios ──────────────────────────────────────────────────────
# Order matters for display — nominal first, then deployment, then stress.

ROBUSTNESS_SCENARIO_LABELS: dict[str, str] = {
    "nominal": "Nominal (speed=1, no wrapper)",
    "speed_5x": "5× speed (unseen frequency)",
    "jitter": "Speed jitter (2 ± 1)",
    "delay": "Observation delay (1 step)",
    "spike": "Mid-episode speed spike (1→5→1)",
    "obs_noise": "Observation noise (σ=0.1)",
    "adversarial_jitter": "Adversarial Jitter (Worst-case Δτ)",
}
"""Human-readable labels for robustness scenarios."""

DEPLOYMENT_SCENARIOS: list[str] = ["jitter", "delay", "spike", "obs_noise"]
"""Scenarios that count toward the Deployment Robustness badge."""

STRESS_SCENARIOS: list[str] = ["speed_5x", "adversarial_jitter"]
"""Scenarios that count toward the Stress Robustness badge."""

ALL_ROBUSTNESS_SCENARIOS: list[str] = DEPLOYMENT_SCENARIOS + STRESS_SCENARIOS
"""All non-nominal robustness scenarios in display order."""

ALL_ROBUSTNESS_SCENARIOS_WITH_NOMINAL: list[str] = ["nominal"] + ALL_ROBUSTNESS_SCENARIOS
"""All robustness scenarios including nominal baseline."""
