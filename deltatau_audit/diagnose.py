"""Failure diagnostics: map scenario failures to patterns, root causes, and fixes.

For each failing or degraded robustness scenario, generate a structured
diagnosis that explains *why* the agent failed and *how* to fix it.
"""

from typing import Dict, List, Optional


# ── Scenario pattern taxonomy ────────────────────────────────────────────────

_SCENARIO_PATTERNS: Dict[str, Dict[str, str]] = {
    "jitter": {
        "pattern": "Speed Jitter Sensitivity",
        "cause": (
            "The agent is sensitive to step-by-step speed fluctuations. "
            "Its value function or policy has learned a fixed implicit time-step "
            "and cannot adapt when dt varies each step."
        ),
        "fix": (
            "Retrain with speed jitter: randomize the environment speed per-step "
            "(e.g., Uniform[0.5, 2.0]). Wrap your training env with JitterWrapper "
            "or use fix-sb3 / fix-cleanrl to automate this."
        ),
    },
    "delay": {
        "pattern": "Observation Recency Dependency",
        "cause": (
            "The agent relies on receiving the most recent observation. "
            "Even a single-step delay in observation delivery degrades performance, "
            "indicating the policy depends on the freshness of state information."
        ),
        "fix": (
            "Add observation delay augmentation during training: wrap your env with "
            "ObservationDelayWrapper(delay=1) for a fraction of training episodes. "
            "Consider switching to a recurrent policy (GRU/LSTM) if not already used."
        ),
    },
    "spike": {
        "pattern": "Frequency Spike Fragility",
        "cause": (
            "The agent cannot handle sudden mid-episode frequency changes. "
            "The policy was trained at a fixed execution speed and its implicit "
            "timing assumptions break when frequency doubles or quadruples mid-episode."
        ),
        "fix": (
            "Train with piecewise speed schedules: use PiecewiseSwitchWrapper or "
            "curriculum training with speed changes within an episode. "
            "This forces the policy to condition on the current execution frequency."
        ),
    },
    "obs_noise": {
        "pattern": "Observation Noise Sensitivity",
        "cause": (
            "The agent's policy is fragile to small Gaussian observation noise (σ=0.1). "
            "This indicates the policy over-fitted to clean training observations "
            "and lacks robustness to sensor noise or partial observability."
        ),
        "fix": (
            "Add observation noise augmentation during training: wrap your env with "
            "ObsNoiseWrapper(std=0.1). Alternatively, use domain randomization on "
            "sensor precision to build natural noise robustness."
        ),
    },
    "speed_5x": {
        "pattern": "Extreme Frequency Fragility",
        "cause": (
            "The agent fails at 5x its training frequency. "
            "The temporal assumptions baked into the policy do not extrapolate "
            "beyond the training speed distribution."
        ),
        "fix": (
            "Train with wide speed randomization (e.g., Uniform[0.5, 5.0]). "
            "For maximum robustness, pair with an internal time module (Δτ) that "
            "explicitly encodes execution frequency so the policy generalizes "
            "beyond training speeds."
        ),
    },
}

# Severity order: lower = worse
_RATING_ORDER = {"FAIL": 0, "DEGRADED": 1, "MILD": 2, "PASS": 3, "N/A": 4}


def _return_ratio_to_rating(ratio: float) -> str:
    """Map a return ratio to a rating string (mirrors auditor.robustness_rating)."""
    if ratio >= 0.95:
        return "PASS"
    elif ratio >= 0.80:
        return "MILD"
    elif ratio >= 0.60:
        return "DEGRADED"
    else:
        return "FAIL"


def generate_diagnosis(summary: Dict, robustness: Dict) -> Dict:
    """Analyse audit results and generate a structured failure diagnosis.

    Parameters
    ----------
    summary : dict
        The ``summary`` sub-dict from ``run_full_audit()`` result.
    robustness : dict
        The ``robustness`` sub-dict from ``run_full_audit()`` result,
        which includes ``per_scenario_scores``.

    Returns
    -------
    dict with keys:
        status           : "pass" | "warn" | "fail"
        failing_scenarios: List[str]  — names of FAIL/DEGRADED scenarios
        issues           : List[dict] — per-scenario diagnosis entries
        primary_pattern  : str | None — pattern name of worst failure
        root_cause       : str | None — explanation for worst failure
        fix_recommendation: str | None — fix suggestion for worst failure
        summary_line     : str        — one-line human-readable verdict
    """
    per_scenario = robustness.get("per_scenario_scores", {})

    issues: List[Dict] = []
    for scenario, sc in per_scenario.items():
        return_ratio = sc.get("return_ratio", 1.0)
        rating = _return_ratio_to_rating(return_ratio)
        if rating in ("FAIL", "DEGRADED"):
            info = _SCENARIO_PATTERNS.get(
                scenario,
                {
                    "pattern": f"{scenario.replace('_', ' ').title()} Sensitivity",
                    "cause": (
                        f"The agent's performance degrades under the '{scenario}' "
                        "perturbation. Review your training environment."
                    ),
                    "fix": (
                        "Add augmentation for this perturbation type during training."
                    ),
                },
            )
            issues.append(
                {
                    "scenario": scenario,
                    "rating": rating,
                    "return_ratio": return_ratio,
                    "return_drop_pct": sc.get("return_drop_pct", 0.0),
                    "pattern": info["pattern"],
                    "cause": info["cause"],
                    "fix": info["fix"],
                }
            )

    # Sort: worst rating first, then by largest return drop
    issues.sort(
        key=lambda x: (_RATING_ORDER.get(x["rating"], 99), x["return_ratio"])
    )

    # Overall status based on deployment/stress rating
    dep_rating = summary.get("deployment_rating", "N/A")
    str_rating = summary.get("stress_rating", "N/A")
    if dep_rating == "FAIL" or str_rating == "FAIL":
        status = "fail"
    elif dep_rating in ("DEGRADED", "MILD") or str_rating in ("DEGRADED", "MILD"):
        status = "warn"
    else:
        status = "pass"

    if not issues:
        return {
            "status": status,
            "failing_scenarios": [],
            "issues": [],
            "primary_pattern": None,
            "root_cause": None,
            "fix_recommendation": None,
            "summary_line": "No significant timing failures detected.",
        }

    primary = issues[0]
    failing_names = [i["scenario"] for i in issues]

    n_fail = sum(1 for i in issues if i["rating"] == "FAIL")
    n_deg = sum(1 for i in issues if i["rating"] == "DEGRADED")
    parts = []
    if n_fail:
        parts.append(f"{n_fail} FAIL")
    if n_deg:
        parts.append(f"{n_deg} DEGRADED")
    count_str = ", ".join(parts)
    sc_str = ", ".join(failing_names)
    plural = "s" if len(issues) > 1 else ""
    summary_line = f"{count_str} scenario{plural}: {sc_str}"

    return {
        "status": status,
        "failing_scenarios": failing_names,
        "issues": issues,
        "primary_pattern": primary["pattern"],
        "root_cause": primary["cause"],
        "fix_recommendation": primary["fix"],
        "summary_line": summary_line,
    }
