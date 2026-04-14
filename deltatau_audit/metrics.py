"""Audit metrics: 2-axis evaluation (Reliance × Robustness).

Axis 1 — Timing Reliance:
    Does the agent USE internal timing? Measured via intervention ablation.
    HIGH reliance = agent depends on Δτ = timing channel is functional.

Axis 2 — Timing Robustness:
    Does the agent SURVIVE realistic timing perturbations? Measured via env wrappers.
    PASS = performance maintained under jitter/delay/speed changes.

All functions return plain dicts/floats for easy JSON serialization.
"""

from typing import Any, Dict, List

import numpy as np

from ._theme import (
    DEGRADED_THRESHOLD,
    MILD_THRESHOLD,
    PASS_THRESHOLD,
    RATING_COLORS,
    RELIANCE_COLORS,
    RELIANCE_THRESHOLD,
)

# ── Value prediction metrics ──────────────────────────────────────────


def compute_value_rmse(values: List[float], returns: List[float]) -> float:
    """RMSE between predicted values and actual discounted returns."""
    v = np.array(values)
    g = np.array(returns)
    return float(np.sqrt(np.mean((v - g) ** 2)))


def compute_value_bias(values: List[float], returns: List[float]) -> float:
    """Mean signed error (positive = overestimate)."""
    return float(np.mean(np.array(values) - np.array(returns)))


def compute_value_mae(values: List[float], returns: List[float]) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.array(values) - np.array(returns))))


def compute_discounted_returns(rewards: List[float], gamma: float = 0.99) -> List[float]:
    """Compute discounted return from each timestep."""
    T = len(rewards)
    returns = np.zeros(T)
    G = 0.0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns.tolist()


# ── Aggregation ───────────────────────────────────────────────────────


def aggregate_episode_metrics(episode_results: List[Dict]) -> Dict:
    """Aggregate metrics across multiple episodes."""
    n = len(episode_results)
    if n == 0:
        return {"n_episodes": 0}

    keys = ["rmse", "mae", "bias", "total_reward", "length"]
    agg: Dict[str, Any] = {"n_episodes": n}

    for key in keys:
        vals = [ep[key] for ep in episode_results if key in ep]
        if vals:
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
            agg[f"{key}_se"] = float(np.std(vals) / np.sqrt(len(vals)))

    dt_means = [ep["dt_mean"] for ep in episode_results if ep.get("dt_mean") is not None]
    if dt_means:
        agg["dt_mean"] = float(np.mean(dt_means))
        agg["dt_std"] = float(np.std(dt_means))

    return agg


# ── Degradation & ratios ─────────────────────────────────────────────


def compute_degradation(baseline_rmse: float, intervention_rmse: float) -> Dict:
    """Compute degradation metrics for an intervention vs baseline."""
    if baseline_rmse > 1e-10:
        pct = (intervention_rmse / baseline_rmse - 1) * 100
        ratio = intervention_rmse / baseline_rmse
    else:
        pct = 0.0 if intervention_rmse < 1e-10 else float("inf")
        ratio = 1.0 if intervention_rmse < 1e-10 else float("inf")

    return {
        "baseline_rmse": baseline_rmse,
        "intervention_rmse": intervention_rmse,
        "absolute_increase": intervention_rmse - baseline_rmse,
        "percent_increase": pct,
        "ratio": ratio,
    }


def compute_return_ratio(nominal_return: float, perturbed_return: float) -> float:
    """Ratio measuring perturbed performance relative to nominal.

    Semantics: 1.0 = same, < 1.0 = worse, > 1.0 = better.

    Handles negative nominal returns (e.g. penalty-heavy envs) correctly:
    - nominal=-100, perturbed=-50  → 1.5  (less penalty = improvement)
    - nominal=-100, perturbed=-150 → 0.5  (more penalty = degradation)
    - nominal=+100, perturbed=+50  → 0.5  (lower return = degradation)
    """
    if abs(nominal_return) < 1e-10:
        return 1.0 if abs(perturbed_return) < 1e-10 else 0.0
    if nominal_return > 0:
        return perturbed_return / nominal_return
    else:
        # nominal < 0: measure relative change preserving sign semantics
        # ratio = 1 + (improvement) / |nominal|
        return 1.0 + (perturbed_return - nominal_return) / abs(nominal_return)


def compute_cohens_d(nominal_returns: List[float], perturbed_returns: List[float]) -> float:
    """Compute Cohen's d for perturbed-vs-nominal return distributions.

    Positive d means perturbed > nominal (improvement).
    Negative d means perturbed < nominal (degradation).
    """
    nom = np.array(nominal_returns, dtype=float)
    pert = np.array(perturbed_returns, dtype=float)

    if len(nom) == 0 or len(pert) == 0:
        return 0.0

    nom_var = float(np.var(nom, ddof=1)) if len(nom) > 1 else 0.0
    pert_var = float(np.var(pert, ddof=1)) if len(pert) > 1 else 0.0

    dof = len(nom) + len(pert) - 2
    if dof <= 0:
        return 0.0

    pooled_var = (((len(nom) - 1) * nom_var) + ((len(pert) - 1) * pert_var)) / dof
    if pooled_var <= 1e-12:
        return 0.0

    pooled_std = float(np.sqrt(pooled_var))
    return float((np.mean(pert) - np.mean(nom)) / pooled_std)


def compute_cliffs_delta(nominal_returns: List[float], perturbed_returns: List[float]) -> float:
    """Compute Cliff's delta for perturbed-vs-nominal return distributions.

    Range: [-1, 1].
    Positive delta means perturbed tends to be larger than nominal.
    """
    nom = np.array(nominal_returns, dtype=float)
    pert = np.array(perturbed_returns, dtype=float)

    if len(nom) == 0 or len(pert) == 0:
        return 0.0

    # Pairwise comparison matrix: pert rows, nominal cols.
    diff = pert[:, None] - nom[None, :]
    wins = int(np.sum(diff > 0))
    losses = int(np.sum(diff < 0))
    total = diff.size

    if total == 0:
        return 0.0
    return float((wins - losses) / total)


def effect_size_magnitude(cohens_d: float) -> str:
    """Qualitative label for absolute Cohen's d."""
    ad = abs(cohens_d)
    if ad < 0.2:
        return "NEGLIGIBLE"
    if ad < 0.5:
        return "SMALL"
    if ad < 0.8:
        return "MEDIUM"
    return "LARGE"


# ── Bootstrap confidence intervals ────────────────────────────────────


def bootstrap_ci(data: List[float], n_bootstrap: int = 2000, ci: float = 0.95, seed: int = 42) -> Dict:
    """Compute bootstrap confidence interval for the mean.

    Returns:
        Dict with mean, ci_lower, ci_upper, std, n.
    """
    arr = np.array(data)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0, "n": 0}
    if n == 1:
        v = float(arr[0])
        return {"mean": v, "ci_lower": v, "ci_upper": v, "std": 0.0, "n": 1}

    rng = np.random.RandomState(seed)
    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = arr[rng.randint(0, n, size=n)]
        means[i] = sample.mean()

    alpha = (1 - ci) / 2
    lower = float(np.percentile(means, alpha * 100))
    upper = float(np.percentile(means, (1 - alpha) * 100))

    return {
        "mean": float(arr.mean()),
        "ci_lower": lower,
        "ci_upper": upper,
        "std": float(arr.std()),
        "n": n,
    }


def _safe_return_ratio(nominal_mean: float, pert_mean: float) -> float:
    """Sign-aware return ratio consistent with compute_return_ratio."""
    if abs(nominal_mean) < 1e-10:
        return 1.0 if abs(pert_mean) < 1e-10 else 0.0
    if nominal_mean > 0:
        return pert_mean / nominal_mean
    else:
        return 1.0 + (pert_mean - nominal_mean) / abs(nominal_mean)


def bootstrap_return_ratio(
    nominal_returns: List[float],
    perturbed_returns: List[float],
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict:
    """Bootstrap CI for the return ratio.

    Uses sign-aware ratio so negative nominal returns are handled correctly.

    Returns:
        Dict with ratio/CI significance and effect-size diagnostics.
        `significant=True` means CI excludes 1.0 on the drop side (upper < 1.0).
    """
    nom = np.array(nominal_returns)
    pert = np.array(perturbed_returns)

    mean_nom = float(nom.mean()) if len(nom) > 0 else 0.0
    mean_pert = float(pert.mean()) if len(pert) > 0 else 0.0
    mean_diff = float(mean_pert - mean_nom)
    cohens_d = compute_cohens_d(nominal_returns, perturbed_returns)
    cliffs_delta = compute_cliffs_delta(nominal_returns, perturbed_returns)
    cles = (cliffs_delta + 1.0) / 2.0  # Common-language effect size in [0,1]

    if len(nom) == 0 or len(pert) == 0 or abs(mean_nom) < 1e-10:
        return {
            "ratio": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "significant": False,
            "significant_change": False,
            "mean_nominal": mean_nom,
            "mean_perturbed": mean_pert,
            "mean_difference": mean_diff,
            "cohens_d": cohens_d,
            "cohens_d_magnitude": effect_size_magnitude(cohens_d),
            "cliffs_delta": cliffs_delta,
            "common_language_effect": cles,
        }

    rng = np.random.RandomState(seed)
    ratios = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        nom_sample = nom[rng.randint(0, len(nom), size=len(nom))]
        pert_sample = pert[rng.randint(0, len(pert), size=len(pert))]
        ratios[i] = _safe_return_ratio(nom_sample.mean(), pert_sample.mean())

    alpha = (1 - ci) / 2
    lower = float(np.percentile(ratios, alpha * 100))
    upper = float(np.percentile(ratios, (1 - alpha) * 100))
    ratio = _safe_return_ratio(mean_nom, mean_pert)
    sig_drop = upper < 1.0
    sig_change = (upper < 1.0) or (lower > 1.0)

    return {
        "ratio": ratio,
        "ci_lower": lower,
        "ci_upper": upper,
        "significant": sig_drop,  # Backward-compatible: statistically significant drop
        "significant_change": sig_change,
        "mean_nominal": mean_nom,
        "mean_perturbed": mean_pert,
        "mean_difference": mean_diff,
        "cohens_d": cohens_d,
        "cohens_d_magnitude": effect_size_magnitude(cohens_d),
        "cliffs_delta": cliffs_delta,
        "common_language_effect": cles,
    }


# ══════════════════════════════════════════════════════════════════════
# 2-AXIS RATING SYSTEM
# ══════════════════════════════════════════════════════════════════════

# ── Axis 1: Reliance ─────────────────────────────────────────────────
# Based on RMSE ratio (intervention / none). HIGH = timing IS used.


def reliance_rating(rmse_ratio: float) -> str:
    """Rate timing reliance from RMSE ratio (intervention/baseline).

    HIGH reliance = the agent's value function depends on Δτ.
    This is INFORMATIONAL — high reliance on a time-aware agent is expected.
    """
    # Preserve historical 4-band semantics around RELIANCE_THRESHOLD=2.0.
    low = 1.05
    moderate = 1.20

    if rmse_ratio < low:
        return "LOW"
    elif rmse_ratio < moderate:
        return "MODERATE"
    elif rmse_ratio < RELIANCE_THRESHOLD:
        return "HIGH"
    else:
        return "VERY_HIGH"


def reliance_color(rating: str) -> str:
    """Color for reliance badge (informational blue spectrum)."""
    return RELIANCE_COLORS.get(rating, RELIANCE_COLORS["LOW"])


# ── Axis 2: Robustness ───────────────────────────────────────────────
# Based on return ratio (wrapper / nominal). FAIL = agent breaks.


def robustness_rating(return_ratio: float) -> str:
    """Rate operational robustness from worst-case return ratio.

    PASS = performance maintained under realistic timing perturbations.
    FAIL = significant performance loss in deployment conditions.
    """
    if return_ratio > PASS_THRESHOLD:
        return "PASS"
    elif return_ratio > MILD_THRESHOLD:
        return "MILD"
    elif return_ratio > DEGRADED_THRESHOLD:
        return "DEGRADED"
    else:
        return "FAIL"


def robustness_color(rating: str) -> str:
    """Color for robustness badge (green=good, red=bad)."""
    return RATING_COLORS.get(rating, "#6c757d")


# ── Legacy single-axis (kept for backward compat) ────────────────────


def severity_rating(pct_increase: float) -> str:
    """Legacy single-axis severity rating."""
    if pct_increase < 5:
        return "PASS"
    elif pct_increase < 20:
        return "MILD"
    elif pct_increase < 50:
        return "MODERATE"
    elif pct_increase < 100:
        return "SEVERE"
    else:
        return "CRITICAL"


def severity_color(rating: str) -> str:
    """Legacy color for single-axis severity."""
    return {
        "PASS": "#28a745",
        "MILD": "#ffc107",
        "MODERATE": "#fd7e14",
        "SEVERE": "#dc3545",
        "CRITICAL": "#721c24",
    }.get(rating, "#6c757d")
