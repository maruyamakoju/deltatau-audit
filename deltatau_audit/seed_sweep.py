"""Seed-sweep utilities for protocol-level robustness reporting.

This module adds a reproducible "multiple-seed" aggregation layer on top of
``run_full_audit()`` so users can report central tendency and uncertainty
across random seeds, not only single-run snapshots.
"""

from __future__ import annotations

import datetime
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, Iterable, List, Mapping

from ._theme import DEPLOY_THRESHOLD_DEFAULT, STRESS_THRESHOLD_DEFAULT
from .metrics import bootstrap_ci, effect_size_magnitude, reliance_rating, robustness_rating
from .schema import SCHEMA_VERSION


def _metric_stats(values: List[float]) -> Dict[str, float | int]:
    """Aggregate a scalar metric with bootstrap CI."""
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "n": 0,
            "min": 0.0,
            "max": 0.0,
        }
    ci = bootstrap_ci(values)
    return {
        "mean": float(ci["mean"]),
        "std": float(ci["std"]),
        "ci_lower": float(ci["ci_lower"]),
        "ci_upper": float(ci["ci_upper"]),
        "n": int(ci["n"]),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _as_float(value: Any) -> float | None:
    """Return float(value) for numeric inputs, else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _aggregate_scenario_metrics(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate per-scenario robustness statistics across seed results."""
    scenario_values: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    scenario_flags: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for result in results:
        if not isinstance(result, dict):
            continue
        robustness = result.get("robustness")
        if not isinstance(robustness, dict):
            continue
        per_scenario = robustness.get("per_scenario_scores")
        if not isinstance(per_scenario, dict):
            continue

        for scenario, sc in per_scenario.items():
            if not isinstance(sc, dict):
                continue
            ret = _as_float(sc.get("return_ratio"))
            rmse = _as_float(sc.get("rmse_ratio"))
            d = _as_float(sc.get("cohens_d"))

            if ret is not None:
                scenario_values[scenario]["return_ratio"].append(ret)
            if rmse is not None:
                scenario_values[scenario]["rmse_ratio"].append(rmse)
            if d is not None:
                scenario_values[scenario]["cohens_d"].append(d)

            sig = sc.get("significant")
            if isinstance(sig, bool):
                scenario_flags[scenario]["significant"].append(1.0 if sig else 0.0)
            sig_change = sc.get("significant_change")
            if isinstance(sig_change, bool):
                scenario_flags[scenario]["significant_change"].append(
                    1.0 if sig_change else 0.0
                )

    out: Dict[str, Dict[str, Any]] = {}
    for scenario in sorted(scenario_values.keys()):
        entry: Dict[str, Any] = {}
        ret_vals = scenario_values[scenario].get("return_ratio", [])
        rmse_vals = scenario_values[scenario].get("rmse_ratio", [])
        d_vals = scenario_values[scenario].get("cohens_d", [])

        if ret_vals:
            entry["return_ratio"] = _metric_stats(ret_vals)
        if rmse_vals:
            entry["rmse_ratio"] = _metric_stats(rmse_vals)
        if d_vals:
            entry["cohens_d"] = _metric_stats(d_vals)

        sig_vals = scenario_flags[scenario].get("significant", [])
        if sig_vals:
            entry["significant_rate"] = sum(sig_vals) / len(sig_vals)
        sig_change_vals = scenario_flags[scenario].get("significant_change", [])
        if sig_change_vals:
            entry["significant_change_rate"] = (
                sum(sig_change_vals) / len(sig_change_vals)
            )

        out[scenario] = entry

    return out


def summarize_seed_sweep(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize multiple audit results (one per seed).

    Args:
        results: Iterable of run_full_audit() result dicts OR dicts containing
            a ``summary`` field with the same keys.

    Returns:
        Dict with aggregate scalar stats, pass rates, and quadrant histogram.
    """
    result_list = list(results)
    summaries = [r.get("summary", r) for r in result_list if isinstance(r, dict)]
    summaries = [s for s in summaries if isinstance(s, dict)]

    n = len(summaries)
    if n == 0:
        return {
            "n_seeds": 0,
            "metrics": {},
            "pass_rates": {},
            "quadrant_counts": {},
            "scenario_metrics": {},
        }

    deploy_scores = [float(s.get("deployment_score", 0.0)) for s in summaries]
    stress_scores = [float(s.get("stress_score", 0.0)) for s in summaries]
    robustness_scores = [float(s.get("robustness_score", 0.0)) for s in summaries]

    rel_scores: List[float] = []
    for s in summaries:
        rel = s.get("reliance_score")
        if rel is not None:
            rel_scores.append(float(rel))

    dep_thr = float(summaries[0].get("deploy_threshold", DEPLOY_THRESHOLD_DEFAULT))
    str_thr = float(summaries[0].get("stress_threshold", STRESS_THRESHOLD_DEFAULT))

    quadrant_counts = Counter(str(s.get("quadrant", "unknown")) for s in summaries)
    pass_rates = {
        "deployment": sum(v >= dep_thr for v in deploy_scores) / n,
        "stress": sum(v >= str_thr for v in stress_scores) / n,
    }

    metrics = {
        "deployment_score": _metric_stats(deploy_scores),
        "stress_score": _metric_stats(stress_scores),
        "robustness_score": _metric_stats(robustness_scores),
    }
    if rel_scores:
        metrics["reliance_score"] = _metric_stats(rel_scores)

    scenario_metrics = _aggregate_scenario_metrics(result_list)

    return {
        "n_seeds": n,
        "thresholds": {"deployment": dep_thr, "stress": str_thr},
        "metrics": metrics,
        "pass_rates": pass_rates,
        "quadrant_counts": dict(quadrant_counts),
        "scenario_metrics": scenario_metrics,
    }


def _safe_float(value: Any, default: float) -> float:
    """Convert numeric values to float, otherwise return default."""
    if isinstance(value, bool):
        return float(default)
    if isinstance(value, (int, float)):
        return float(value)
    return float(default)


def _metric_mean(metrics: Mapping[str, Any], name: str, default: float) -> float:
    """Return aggregate metric mean if available, else fallback default."""
    stat = metrics.get(name, {})
    if isinstance(stat, Mapping):
        val = stat.get("mean")
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
    return float(default)


def _scenario_scores_from_aggregate(
    scenario_metrics: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Convert aggregate scenario stats into per-scenario score records."""
    per_scenario_scores: Dict[str, Dict[str, Any]] = {}
    for scenario, sc in scenario_metrics.items():
        if not isinstance(sc, Mapping):
            continue
        ret_stats = sc.get("return_ratio", {})
        if not isinstance(ret_stats, Mapping):
            continue
        ret_mean = ret_stats.get("mean")
        if not isinstance(ret_mean, (int, float)) or isinstance(ret_mean, bool):
            continue
        ret_mean_f = float(ret_mean)

        rmse_stats = sc.get("rmse_ratio", {})
        rmse_mean = (
            rmse_stats.get("mean", 1.0) if isinstance(rmse_stats, Mapping) else 1.0
        )
        rmse_mean_f = _safe_float(rmse_mean, 1.0)

        d_stats = sc.get("cohens_d", {})
        d_mean = d_stats.get("mean", 0.0) if isinstance(d_stats, Mapping) else 0.0
        d_mean_f = _safe_float(d_mean, 0.0)

        ci_lower = _safe_float(ret_stats.get("ci_lower", ret_mean_f), ret_mean_f)
        ci_upper = _safe_float(ret_stats.get("ci_upper", ret_mean_f), ret_mean_f)

        sig_rate = _safe_float(sc.get("significant_rate", 0.0), 0.0)
        sig_change_rate = _safe_float(sc.get("significant_change_rate", 0.0), 0.0)

        per_scenario_scores[str(scenario)] = {
            "return_ratio": ret_mean_f,
            "return_drop_pct": (1 - ret_mean_f) * 100,
            "rmse_ratio": rmse_mean_f,
            "rmse_increase_pct": (rmse_mean_f - 1) * 100,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "significant": sig_rate >= 0.5,
            "significant_change": sig_change_rate >= 0.5,
            "cohens_d": d_mean_f,
            "cohens_d_magnitude": effect_size_magnitude(d_mean_f),
            "significant_rate": sig_rate,
            "significant_change_rate": sig_change_rate,
        }
    return per_scenario_scores


def _sub_score(
    per_scenario_scores: Mapping[str, Any],
    scenarios: Iterable[str],
    fallback_return: float,
) -> Dict[str, Any]:
    """Build deployment/stress sub-score from scenario-level scores."""
    available: List[tuple[str, Mapping[str, Any]]] = []
    for scenario in scenarios:
        info = per_scenario_scores.get(scenario)
        if isinstance(info, Mapping):
            available.append((scenario, info))
    if not available:
        fallback = float(fallback_return)
        return {
            "return_score": fallback,
            "rmse_score": 1.0,
            "rating": robustness_rating(fallback),
            "worst_case": {
                "scenario": None,
                "return_ratio": fallback,
                "return_drop_pct": (1 - fallback) * 100,
            },
        }

    worst_name, worst_info = min(
        available, key=lambda item: _safe_float(item[1].get("return_ratio"), 1.0)
    )
    worst_ret = _safe_float(worst_info.get("return_ratio"), fallback_return)
    worst_rmse = max(
        _safe_float(info.get("rmse_ratio"), 1.0) for _, info in available
    )
    return {
        "return_score": worst_ret,
        "rmse_score": worst_rmse,
        "rating": robustness_rating(worst_ret),
        "worst_case": {
            "scenario": worst_name,
            "return_ratio": worst_ret,
            "return_drop_pct": (1 - worst_ret) * 100,
        },
    }


def _quadrant_prescription(quadrant: str) -> str:
    """Default aggregate prescription text for multi-seed summaries."""
    mapping = {
        "time_aware_robust": (
            "Agent actively uses internal timing and remains robust under deployment timing shifts."
        ),
        "time_aware_fragile": (
            "Timing is used but not stable across deployment shifts; calibrate with timing-augmented training."
        ),
        "time_blind_fragile": (
            "Agent is both timing-unaware and fragile; introduce explicit timing features and robustness training."
        ),
        "time_blind_robust": (
            "Performance is robust without explicit timing use; consider timing-aware features for harder regimes."
        ),
        "deployment_ready": (
            "Deployment robustness is stable under tested timing perturbations."
        ),
        "deployment_fragile": (
            "Deployment robustness is fragile; prioritize jitter/delay/spike augmentation before release."
        ),
    }
    return mapping.get(
        quadrant,
        "Review scenario-level timing failures and retrain with targeted timing augmentations.",
    )


def seed_sweep_payload_to_result(
    seed_payload: Mapping[str, Any],
    *,
    speeds: Iterable[int] | None = None,
    n_episodes: int = 0,
    deploy_threshold: float = DEPLOY_THRESHOLD_DEFAULT,
    stress_threshold: float = STRESS_THRESHOLD_DEFAULT,
) -> Dict[str, Any]:
    """Convert `run_seed_sweep` payload into a report/CI-compatible result."""
    from ._constants import DEPLOYMENT_SCENARIOS, STRESS_SCENARIOS
    from .diagnose import generate_diagnosis

    aggregate_raw = seed_payload.get("aggregate", {})
    aggregate = aggregate_raw if isinstance(aggregate_raw, Mapping) else {}
    metrics_raw = aggregate.get("metrics", {})
    metrics = metrics_raw if isinstance(metrics_raw, Mapping) else {}
    thresholds_raw = aggregate.get("thresholds", {})
    thresholds = thresholds_raw if isinstance(thresholds_raw, Mapping) else {}

    results_raw = seed_payload.get("results", [])
    results: list[Dict[str, Any]] = []
    if isinstance(results_raw, list):
        for row in results_raw:
            if isinstance(row, dict):
                results.append(row)

    template: Dict[str, Any] = results[0] if results else {}
    template_summary_raw = template.get("summary", {})
    template_summary = (
        template_summary_raw if isinstance(template_summary_raw, Mapping) else {}
    )
    template_robustness_raw = template.get("robustness", {})
    template_robustness = (
        template_robustness_raw if isinstance(template_robustness_raw, Mapping) else {}
    )
    template_reliance_raw = template.get("reliance", {})
    template_reliance = (
        template_reliance_raw if isinstance(template_reliance_raw, Mapping) else {}
    )

    dep_mean = _metric_mean(
        metrics,
        "deployment_score",
        _safe_float(template_summary.get("deployment_score"), 0.0),
    )
    str_mean = _metric_mean(
        metrics,
        "stress_score",
        _safe_float(template_summary.get("stress_score"), 0.0),
    )
    rob_mean = _metric_mean(metrics, "robustness_score", min(dep_mean, str_mean))

    rel_stat = metrics.get("reliance_score", {})
    rel_mean: float | None = None
    if isinstance(rel_stat, Mapping):
        rel_val = rel_stat.get("mean")
        if isinstance(rel_val, (int, float)) and not isinstance(rel_val, bool):
            rel_mean = float(rel_val)

    dep_threshold = _safe_float(
        thresholds.get("deployment"), float(deploy_threshold)
    )
    str_threshold = _safe_float(
        thresholds.get("stress"), float(stress_threshold)
    )

    scenario_metrics_raw = aggregate.get("scenario_metrics", {})
    scenario_metrics = (
        scenario_metrics_raw if isinstance(scenario_metrics_raw, Mapping) else {}
    )
    per_scenario_scores = _scenario_scores_from_aggregate(scenario_metrics)
    if not per_scenario_scores:
        fallback_scores = template_robustness.get("per_scenario_scores", {})
        if isinstance(fallback_scores, Mapping):
            per_scenario_scores = {
                str(name): dict(values)
                for name, values in fallback_scores.items()
                if isinstance(values, Mapping)
            }

    deployment = _sub_score(per_scenario_scores, DEPLOYMENT_SCENARIOS, dep_mean)
    stress = _sub_score(per_scenario_scores, STRESS_SCENARIOS, str_mean)
    overall_return = min(
        _safe_float(deployment.get("return_score"), dep_mean),
        _safe_float(stress.get("return_score"), str_mean),
    )
    overall_rmse = max(
        _safe_float(deployment.get("rmse_score"), 1.0),
        _safe_float(stress.get("rmse_score"), 1.0),
    )

    robustness = dict(template_robustness)
    robustness.update(
        {
            "per_scenario_scores": per_scenario_scores,
            "deployment": deployment,
            "stress": stress,
            "return_score": overall_return,
            "rmse_score": overall_rmse,
            "rating": robustness_rating(overall_return),
        }
    )

    if rel_mean is None:
        rel_rating = "N/A"
        rel_score: float | None = None
    else:
        rel_rating = reliance_rating(rel_mean)
        rel_score = rel_mean

    quadrant_counts_raw = aggregate.get("quadrant_counts", {})
    quadrant_counts = (
        quadrant_counts_raw if isinstance(quadrant_counts_raw, Mapping) else {}
    )
    quadrant: str | None = None
    if quadrant_counts:
        quadrant = max(
            quadrant_counts.items(),
            key=lambda item: (int(item[1]), str(item[0])),
        )[0]
    if not quadrant:
        q = template_summary.get("quadrant")
        if isinstance(q, str):
            quadrant = q
    if not quadrant:
        quadrant = (
            "deployment_ready"
            if dep_mean >= dep_threshold
            else "deployment_fragile"
        )

    summary = {
        "reliance_rating": rel_rating,
        "reliance_score": rel_score,
        "robustness_rating": robustness_rating(rob_mean),
        "robustness_score": rob_mean,
        "robustness_rmse_score": overall_rmse,
        "deployment_rating": robustness_rating(dep_mean),
        "deployment_score": dep_mean,
        "stress_rating": robustness_rating(str_mean),
        "stress_score": str_mean,
        "sensitivity_mean": template_summary.get("sensitivity_mean"),
        "deploy_threshold": dep_threshold,
        "stress_threshold": str_threshold,
        "quadrant": str(quadrant),
        "prescription": _quadrant_prescription(str(quadrant)),
    }

    reliance = dict(template_reliance)
    reliance["rating"] = rel_rating
    reliance["score"] = rel_score

    diagnosis = generate_diagnosis(summary, robustness)

    speed_list = list(speeds) if speeds is not None else []

    seed_sweep = {
        key: value for key, value in seed_payload.items() if key != "results"
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "speeds": template.get("speeds", speed_list),
        "n_episodes": template.get("n_episodes", int(n_episodes)),
        "supports_intervention": bool(template.get("supports_intervention", False)),
        "reliance": reliance,
        "robustness": robustness,
        "sensitivity": template.get("sensitivity"),
        "summary": summary,
        "diagnosis": diagnosis,
        "seed_sweep": seed_sweep,
        "manifest": {},
    }


def run_seed_sweep(
    adapter_factory: Callable[[int | None], Any],
    env_factory: Callable[[], Any],
    seeds: Iterable[int],
    *,
    keep_full_results: bool = False,
    **audit_kwargs: Any,
) -> Dict[str, Any]:
    """Run ``run_full_audit`` across multiple seeds and aggregate the results.

    Args:
        adapter_factory: Callable(seed) -> adapter instance.
        env_factory: Callable returning a fresh environment.
        seeds: Seed list/iterable.
        keep_full_results: Include full run_full_audit payloads in output.
        **audit_kwargs: Extra kwargs forwarded to run_full_audit().

    Returns:
        Dict with per-seed summaries + aggregate statistics.
    """
    from . import __version__
    from .auditor import run_full_audit

    seed_list = list(seeds)
    if not seed_list:
        raise ValueError("seeds must not be empty")

    per_seed: List[Dict[str, Any]] = []
    full_results: List[Dict[str, Any]] = []

    for seed in seed_list:
        adapter = adapter_factory(seed)
        result = run_full_audit(
            adapter,
            env_factory,
            seed=seed,
            **audit_kwargs,
        )
        summary = result.get("summary", {})
        per_seed.append(
            {
                "seed": seed,
                "deployment_score": summary.get("deployment_score"),
                "deployment_rating": summary.get("deployment_rating"),
                "stress_score": summary.get("stress_score"),
                "stress_rating": summary.get("stress_rating"),
                "reliance_score": summary.get("reliance_score"),
                "reliance_rating": summary.get("reliance_rating"),
                "quadrant": summary.get("quadrant"),
            }
        )
        full_results.append(result)

    aggregate = summarize_seed_sweep(full_results)
    payload: Dict[str, Any] = {
        "_version": __version__,
        "_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seeds": seed_list,
        "n_seeds": len(seed_list),
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    if keep_full_results:
        payload["results"] = full_results
    return payload
