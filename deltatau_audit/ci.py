"""CI integration: exit codes, summary JSON/MD for pipeline gates.

Exit codes:
    0 = pass  (deployment >= threshold, stress >= threshold)
    1 = warn  (deployment passes but stress fails)
    2 = fail  (deployment below threshold)
"""

import datetime
import json
import os
from typing import Dict

from ._constants import DEPLOYMENT_SCENARIOS, STRESS_SCENARIOS


def _status_from_exit_code(exit_code: int) -> str:
    return {0: "pass", 1: "warn", 2: "fail"}[exit_code]


def _worst_ci_lower(robustness: Dict, scenarios: list[str]) -> float | None:
    per_scenario = robustness.get("per_scenario_scores", {})
    if not isinstance(per_scenario, dict):
        return None
    vals: list[float] = []
    for sc in scenarios:
        info = per_scenario.get(sc, {})
        if not isinstance(info, dict):
            continue
        ci_lower = info.get("ci_lower")
        if isinstance(ci_lower, bool):
            continue
        if isinstance(ci_lower, (int, float)):
            vals.append(float(ci_lower))
    if not vals:
        return None
    return min(vals)


def _gate_scores(summary: Dict, robustness: Dict | None, gate_mode: str) -> Dict:
    if gate_mode == "worst_ci_lower" and isinstance(robustness, dict):
        dep = _worst_ci_lower(robustness, DEPLOYMENT_SCENARIOS)
        stress = _worst_ci_lower(robustness, STRESS_SCENARIOS)
        if dep is not None and stress is not None:
            return {
                "mode": "worst_ci_lower",
                "deployment": dep,
                "stress": stress,
            }
    dep_score = summary.get("deployment_score", 1.0)
    str_score = summary.get("stress_score", 1.0)
    return {
        "mode": "score",
        "deployment": float(dep_score),
        "stress": float(str_score),
    }


def compute_exit_code(
    summary: Dict,
    deploy_threshold: float = 0.80,
    stress_threshold: float = 0.50,
    robustness: Dict | None = None,
    gate_mode: str = "score",
) -> int:
    """Determine CI exit code from audit summary.

    Returns:
        0 = pass, 1 = warn (stress only), 2 = fail (deployment)
    """
    gate_scores = _gate_scores(summary, robustness, gate_mode)
    dep_score = gate_scores["deployment"]
    str_score = gate_scores["stress"]

    if dep_score < deploy_threshold:
        return 2  # fail
    if str_score < stress_threshold:
        return 1  # warn
    return 0  # pass


def compute_seed_sweep_exit_code(
    pass_rates: Dict,
    min_deploy_pass_rate: float = 0.80,
    min_stress_pass_rate: float = 0.50,
) -> int:
    """Determine CI exit code from multi-seed pass rates.

    Returns:
        0 = pass, 1 = warn (stress pass-rate only), 2 = fail (deployment pass-rate)
    """
    dep_rate = float(pass_rates.get("deployment", 0.0))
    str_rate = float(pass_rates.get("stress", 0.0))

    if dep_rate < min_deploy_pass_rate:
        return 2
    if str_rate < min_stress_pass_rate:
        return 1
    return 0


def _build_ci_summary_payload(
    summary: Dict,
    robustness: Dict,
    deploy_threshold: float,
    stress_threshold: float,
    gate_mode: str,
) -> tuple[dict, int]:
    gate_scores = _gate_scores(summary, robustness, gate_mode)
    exit_code = compute_exit_code(
        summary,
        deploy_threshold,
        stress_threshold,
        robustness=robustness,
        gate_mode=gate_mode,
    )
    status = _status_from_exit_code(exit_code)

    ci_json = {
        "status": status,
        "exit_code": exit_code,
        "deployment_score": summary.get("deployment_score"),
        "deployment_rating": summary.get("deployment_rating"),
        "stress_score": summary.get("stress_score"),
        "stress_rating": summary.get("stress_rating"),
        "thresholds": {
            "deployment": deploy_threshold,
            "stress": stress_threshold,
        },
        "gate_mode": gate_scores["mode"],
        "gate_scores": {
            "deployment": gate_scores["deployment"],
            "stress": gate_scores["stress"],
        },
        "deploy_pass": bool(gate_scores["deployment"] >= deploy_threshold),
        "stress_pass": bool(gate_scores["stress"] >= stress_threshold),
    }

    deploy_info = robustness.get("deployment", {})
    stress_info = robustness.get("stress", {})
    per_scenario = robustness.get("per_scenario_scores", {})
    if deploy_info:
        wc = deploy_info.get("worst_case", {})
        ci_json["deployment_worst"] = wc.get("scenario")
    if stress_info:
        wc = stress_info.get("worst_case", {})
        ci_json["stress_worst"] = wc.get("scenario")

    if per_scenario:
        ci_json["significant_drop_count"] = sum(
            1 for s in per_scenario.values() if s.get("significant")
        )
        ci_json["significant_change_count"] = sum(
            1 for s in per_scenario.values() if s.get("significant_change")
        )
        ci_json["scenario_effect_sizes"] = {
            name: {
                "cohens_d": sc.get("cohens_d"),
                "magnitude": sc.get("cohens_d_magnitude"),
            }
            for name, sc in per_scenario.items()
        }

    return ci_json, exit_code


def write_ci_summary(summary: Dict, robustness: Dict,
                     output_dir: str,
                     deploy_threshold: float = 0.80,
                     stress_threshold: float = 0.50,
                     gate_mode: str = "score") -> int:
    """Write ci_summary.json and ci_summary.md, return exit code.

    Args:
        summary: audit_result["summary"]
        robustness: audit_result["robustness"]
        output_dir: where to write files
        deploy_threshold: deployment return ratio threshold for pass
        stress_threshold: stress return ratio threshold for warn

    Returns:
        Exit code (0/1/2)
    """
    os.makedirs(output_dir, exist_ok=True)
    ci_json, exit_code = _build_ci_summary_payload(
        summary,
        robustness,
        deploy_threshold,
        stress_threshold,
        gate_mode,
    )
    gate_scores = ci_json["gate_scores"]
    status = ci_json["status"]
    per_scenario = robustness.get("per_scenario_scores", {})

    # Traceability
    from . import __version__
    ci_json["_version"] = __version__
    ci_json["_timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    json_path = os.path.join(output_dir, "ci_summary.json")
    with open(json_path, "w") as f:
        json.dump(ci_json, f, indent=2)

    # Markdown summary (one-liner for PR comments)
    dep_score = gate_scores["deployment"]
    dep_rating = summary.get("deployment_rating", "?")
    str_score = gate_scores["stress"]
    str_rating = summary.get("stress_rating", "?")

    icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}[status]

    md_lines = [
        f"## {icon} Time Robustness Audit: **{status.upper()}**",
        "",
        "| Axis | Rating | Score | Threshold |",
        "|------|--------|-------|-----------|",
        f"| Deployment | **{dep_rating}** | {dep_score:.2f} | {deploy_threshold:.2f} |",
        f"| Stress | **{str_rating}** | {str_score:.2f} | {stress_threshold:.2f} |",
    ]
    if ci_json["gate_mode"] != "score":
        md_lines += ["", f"- Gate mode: `{ci_json['gate_mode']}`"]

    # Scenario breakdown
    if per_scenario:
        md_lines += ["", "**Scenarios:**", ""]
        for sc_name, sc in per_scenario.items():
            ret = sc["return_ratio"] * 100
            d = sc.get("cohens_d")
            d_str = f", d={d:+.2f}" if isinstance(d, (int, float)) else ""
            flag = " ⚠️" if ret < 80 else ""
            md_lines.append(f"- {sc_name}: {ret:.0f}%{d_str}{flag}")

    md_lines.append("")
    md_lines.append(f"*Generated by deltatau-audit v{__version__}*")

    md_path = os.path.join(output_dir, "ci_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return exit_code


def generate_ci_summary(
    audit_result: Dict,
    *,
    out_dir: str | os.PathLike[str],
    deploy_threshold: float = 0.80,
    stress_threshold: float = 0.50,
    gate_mode: str = "score",
) -> Dict:
    """Backward-compatible wrapper that accepts a full audit result."""
    summary = audit_result.get("summary", audit_result)
    robustness = audit_result.get("robustness", {})
    if not isinstance(summary, dict):
        raise ValueError("audit_result must contain a summary object")
    if not isinstance(robustness, dict):
        robustness = {}

    write_ci_summary(
        summary,
        robustness,
        str(out_dir),
        deploy_threshold=deploy_threshold,
        stress_threshold=stress_threshold,
        gate_mode=gate_mode,
    )
    json_path = os.path.join(str(out_dir), "ci_summary.json")
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def write_seed_sweep_ci_summary(
    summary: Dict,
    seed_sweep_aggregate: Dict,
    output_dir: str,
    deploy_threshold: float = 0.80,
    stress_threshold: float = 0.50,
    min_deploy_pass_rate: float = 0.80,
    min_stress_pass_rate: float = 0.50,
    gate_mode: str = "pass_rate",
) -> int:
    """Write CI summary for multi-seed sweeps (pass-rate gate).

    Pass-rate gate:
        - fail (2): deployment pass-rate < min_deploy_pass_rate
        - warn (1): deployment pass-rate OK but stress pass-rate is low
        - pass (0): both pass-rates satisfy minima
    """
    os.makedirs(output_dir, exist_ok=True)
    pass_rates = seed_sweep_aggregate.get("pass_rates", {})
    dep_rate = float(pass_rates.get("deployment", 0.0))
    str_rate = float(pass_rates.get("stress", 0.0))
    n_seeds = int(seed_sweep_aggregate.get("n_seeds", 0))

    scenario_metrics = seed_sweep_aggregate.get("scenario_metrics", {})
    if gate_mode == "worst_ci_lower" and isinstance(scenario_metrics, dict):
        dep_vals: list[float] = []
        str_vals: list[float] = []
        for sc in DEPLOYMENT_SCENARIOS:
            sc_data = scenario_metrics.get(sc, {})
            if not isinstance(sc_data, dict):
                continue
            rr = sc_data.get("return_ratio", {})
            if isinstance(rr, dict) and isinstance(rr.get("ci_lower"), (int, float)):
                dep_vals.append(float(rr["ci_lower"]))
        for sc in STRESS_SCENARIOS:
            sc_data = scenario_metrics.get(sc, {})
            if not isinstance(sc_data, dict):
                continue
            rr = sc_data.get("return_ratio", {})
            if isinstance(rr, dict) and isinstance(rr.get("ci_lower"), (int, float)):
                str_vals.append(float(rr["ci_lower"]))
        dep_gate = min(dep_vals) if dep_vals else float(summary.get("deployment_score", 0.0) or 0.0)
        str_gate = min(str_vals) if str_vals else float(summary.get("stress_score", 0.0) or 0.0)
        gate_scores = {"deployment": dep_gate, "stress": str_gate}
        if dep_gate < deploy_threshold:
            exit_code = 2
        elif str_gate < stress_threshold:
            exit_code = 1
        else:
            exit_code = 0
    else:
        gate_scores = {
            "deployment": dep_rate,
            "stress": str_rate,
        }
        exit_code = compute_seed_sweep_exit_code(
            pass_rates,
            min_deploy_pass_rate=min_deploy_pass_rate,
            min_stress_pass_rate=min_stress_pass_rate,
        )
        gate_mode = "pass_rate"

    status = _status_from_exit_code(exit_code)

    ci_json = {
        "mode": "multi_seed",
        "status": status,
        "exit_code": exit_code,
        "n_seeds": n_seeds,
        "deployment_score": summary.get("deployment_score"),
        "deployment_rating": summary.get("deployment_rating"),
        "stress_score": summary.get("stress_score"),
        "stress_rating": summary.get("stress_rating"),
        "thresholds": {
            "deployment": deploy_threshold,
            "stress": stress_threshold,
        },
        "pass_rates": {
            "deployment": dep_rate,
            "stress": str_rate,
        },
        "min_pass_rates": {
            "deployment": min_deploy_pass_rate,
            "stress": min_stress_pass_rate,
        },
        "gate_mode": gate_mode,
        "gate_scores": gate_scores,
        "quadrant_counts": seed_sweep_aggregate.get("quadrant_counts", {}),
    }

    from . import __version__

    ci_json["_version"] = __version__
    ci_json["_timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    json_path = os.path.join(output_dir, "ci_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ci_json, f, indent=2)

    dep_score = float(summary.get("deployment_score", 0.0) or 0.0)
    str_score = float(summary.get("stress_score", 0.0) or 0.0)
    dep_rating = summary.get("deployment_rating", "?")
    str_rating = summary.get("stress_rating", "?")
    icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}[status]

    md_lines = [
        f"## {icon} Time Robustness Audit (Multi-Seed): **{status.upper()}**",
        "",
        f"- Seeds: **{n_seeds}**",
        "",
        f"- Gate mode: **{gate_mode}**",
        "",
        "| Axis | Rating | Mean Score | Score Threshold | Pass Rate | Min Pass Rate |",
        "|------|--------|------------|-----------------|-----------|---------------|",
        (
            f"| Deployment | **{dep_rating}** | {dep_score:.2f} | "
            f"{deploy_threshold:.2f} | {dep_rate:.2%} | {min_deploy_pass_rate:.2%} |"
        ),
        (
            f"| Stress | **{str_rating}** | {str_score:.2f} | "
            f"{stress_threshold:.2f} | {str_rate:.2%} | {min_stress_pass_rate:.2%} |"
        ),
        "",
        f"*Generated by deltatau-audit v{__version__}*",
    ]

    md_path = os.path.join(output_dir, "ci_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return exit_code
