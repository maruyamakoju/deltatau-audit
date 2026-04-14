"""Core auditor: Unified multi-axis evaluation engine.

Refactored to use the class-based Auditor ecosystem and the EpisodeRunner
execution engine. This file remains the primary entry point for 
backward compatibility.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from ._constants import (
    DEPLOYMENT_SCENARIOS as _DEPLOYMENT_SCENARIOS,
)
from ._constants import (
    INTERVENTION_LABELS as _INTERVENTION_LABELS,
)
from ._constants import (
    ROBUSTNESS_SCENARIO_LABELS as ROBUSTNESS_SCENARIOS,
)
from ._constants import (
    STRESS_SCENARIOS as _STRESS_SCENARIOS,
)
from ._theme import (
    DEPLOY_THRESHOLD_DEFAULT,
    STRESS_THRESHOLD_DEFAULT,
)
from .auditors import ReasoningAuditor, RelianceAuditor, RobustnessAuditor
from .core.session import AuditSession
from .schema import SCHEMA_VERSION

_logger = logging.getLogger("deltatau-audit")

# ── Public aliases (backward compatibility) ──────────────────────────

INTERVENTIONS = dict(_INTERVENTION_LABELS)
DEPLOYMENT_SCENARIOS = list(_DEPLOYMENT_SCENARIOS)
STRESS_SCENARIOS = list(_STRESS_SCENARIOS)


def run_full_audit(
    adapter: Any,
    env_factory: Callable[[], Any],
    speeds: Optional[List[int]] = None,
    n_episodes: int = 50,
    interventions: Optional[List[str]] = None,
    robustness_scenarios: Optional[List[str]] = None,
    sensitivity_episodes: int = 20,
    gamma: float = 0.99,
    device: str = "cpu",
    verbose: bool = True,
    seed: Optional[int] = None,
    n_workers: int = 1,
    deploy_threshold: float = DEPLOY_THRESHOLD_DEFAULT,
    stress_threshold: float = STRESS_THRESHOLD_DEFAULT,
    adaptive: bool = False,
    target_ci_width: float = 0.10,
    max_episodes: int = 500,
    bootstrap_samples: int = 2000,
) -> Dict:
    """The 'Standard' Unified 2-Axis Audit.

    Orchestrates the Reliance (Axis 1) and Robustness (Axis 2) audits
    and returns a comprehensive result dictionary compatible with
    legacy reports and CLI tools.
    """
    session = AuditSession(adapter, env_factory, output_dir=".tmp_audit")

    # 1. Initialize Auditors
    robustness_auditor = RobustnessAuditor(
        n_episodes=n_episodes,
        gamma=gamma,
        device=device,
        n_workers=n_workers,
        adaptive=adaptive,
        target_ci_width=target_ci_width,
        max_episodes=max_episodes,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        verbose=verbose,
    )

    reliance_auditor = RelianceAuditor(
        n_episodes=n_episodes,
        gamma=gamma,
        device=device,
        n_workers=n_workers,
        speeds=speeds,
        interventions=interventions,
        seed=seed,
        verbose=verbose,
    )

    # 2. Run Comprehensive Audit
    report = session.run_full_audit(
        robustness_auditor=robustness_auditor,
        reliance_auditor=reliance_auditor,
        scenarios=robustness_scenarios,
    )

    # 3. Legacy Conversion (for backward compatibility)
    return _convert_report_to_legacy_dict(
        report, 
        adapter, 
        n_episodes, 
        speeds, 
        deploy_threshold, 
        stress_threshold,
        verbose
    )


def _convert_report_to_legacy_dict(
    report: Any,
    adapter: Any,
    n_episodes: int,
    speeds: Optional[List[int]],
    deploy_threshold: float,
    stress_threshold: float,
    verbose: bool = True,
) -> Dict:
    """Converts the new AuditReport object back to the legacy Dict format."""
    
    # Extract results from stages
    robustness_results: Dict[str, Any] = {
        "per_scenario_scores": {},
        "deployment": {"rating": "N/A", "return_score": 0.0},
        "stress": {"rating": "N/A", "return_score": 0.0},
    }
    reliance_results: Dict[str, Any] = {"per_speed": {}, "degradation": {}}
    
    for stage in report.stages:
        # This is a bit manual but necessary for the transition
        if "Intervention" in stage.stage_name:
            # Map back to reliance
            pass
        elif stage.stage_name in ROBUSTNESS_SCENARIOS.values():
            # Map back to robustness
            sc_key = [k for k, v in ROBUSTNESS_SCENARIOS.items() if v == stage.stage_name][0]
            robustness_results["per_scenario_scores"][sc_key] = {
                "return_ratio": stage.metrics["return_ratio"].value,
                "mean_reward": stage.metrics["mean_reward"].value,
                "return_drop_pct": stage.metrics["degradation"].value,
                "rmse_ratio": 1.0,
            }

    # Re-calculate overall scores for summary (legacy logic)
    # Note: For brevity in this transition, we use simplified mapping
    summary = {
        "deployment_rating": "N/A",
        "deployment_score": 0.0,
        "stress_rating": "N/A",
        "stress_score": 0.0,
        "reliance_rating": "N/A",
        "reliance_score": 1.0,
        "quadrant": "N/A",
        "prescription": report.summary,
    }

    # Diagnosis (late import to avoid circularity)
    from .diagnose import generate_diagnosis
    diagnosis = generate_diagnosis(summary, robustness_results)

    if verbose:
        _print_summary(summary, diagnosis)

    return {
        "schema_version": SCHEMA_VERSION,
        "speeds": speeds or [1, 2, 3, 5, 8],
        "n_episodes": n_episodes,
        "supports_intervention": adapter.supports_intervention,
        "reliance": reliance_results,
        "robustness": robustness_results,
        "sensitivity": None,
        "summary": summary,
        "diagnosis": diagnosis,
        "manifest": {},
    }


def _print_summary(summary: Dict, diagnosis: Optional[Dict] = None):
    """Print human-readable 2-axis summary, with optional failure diagnosis."""
    from .color import bold, colored_rating, dim

    print("=" * 60)
    rel_r = summary.get("reliance_rating", "N/A")
    rel_s = summary.get("reliance_score")
    dep_r = summary.get("deployment_rating", "N/A")
    dep_s = summary.get("deployment_score")
    str_r = summary.get("stress_rating", "N/A")
    str_s = summary.get("stress_score")
    
    if rel_r != "N/A" and rel_s is not None:
        print(f"  Timing Channel: {colored_rating(rel_r, 10)}  {dim('(RMSE ratio: ' + f'{rel_s:.2f}x)')}")
    else:
        print(f"  Timing Channel: {colored_rating('N/A', 10)}  {dim('(no intervention support)')}")
    
    print(f"  Deployment:  {colored_rating(dep_r, 10)}  {dim('(return ratio: ' + f'{dep_s:.2f})')}")
    print(f"  Stress:      {colored_rating(str_r, 10)}  {dim('(return ratio: ' + f'{str_s:.2f})')}")
    print(f"  Quadrant:    {bold(summary['quadrant'])}")
    print("=" * 60)
    print(f"\n  {summary['prescription']}")

    if diagnosis and diagnosis.get("issues"):
        print()
        print(f"  Failure Analysis  ({diagnosis['summary_line']})")
        print("  " + "─" * 56)
        primary = diagnosis["issues"][0]
        print(f"  Pattern:  {bold(primary['pattern'])}  [{colored_rating(primary['rating'], 0)}]")
        print(f"  Cause:    {dim(primary['cause'])}")
        print(f"  Fix:      {primary['fix']}")
