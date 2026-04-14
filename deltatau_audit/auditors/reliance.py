"""Reliance Auditor (Axis 1).

Evaluates whether the agent's internal value function causally depends
on the timing representation (Δτ) using intervention ablation.
"""

from __future__ import annotations

import logging
import time as _time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from deltatau_audit._constants import INTERVENTION_LABELS
from deltatau_audit.core.runner import EpisodeRunner
from deltatau_audit.metrics import (
    aggregate_episode_metrics,
    compute_degradation,
    reliance_rating,
    severity_rating,
)
from deltatau_audit.schema import AuditReport, AuditStageResult, MetricValue
from deltatau_audit.wrappers.factory import create_wrapped_env

from .base import BaseAuditor

logger = logging.getLogger("deltatau-audit")


class RelianceAuditor(BaseAuditor):
    """Axis 1: Timing Channel Analysis (Intervention Ablation).

    Tampers with the agent's internal Δτ to test whether the value
    function causally depends on the timing representation.
    """

    def __init__(
        self,
        n_episodes: int = 50,
        gamma: float = 0.99,
        device: str = "cpu",
        n_workers: int = 1,
        speeds: Optional[List[int]] = None,
        interventions: Optional[List[str]] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
    ):
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.device = device
        self.n_workers = n_workers
        self.speeds = speeds or [1, 2, 3, 5, 8]
        self.interventions = interventions or ["none", "clamp_1", "reverse", "random"]
        self.seed = seed
        self.verbose = verbose

    def run(self, agent: Any, env_id: str, **kwargs: Any) -> AuditReport:
        """Executes the intervention ablation audit."""
        if not agent.supports_intervention:
            if self.verbose:
                print("  Reliance Test: N/A (no intervention support)")
            return self._empty_report(agent)

        if self.verbose:
            print("  Reliance Test (intervention ablation)")

        stages: List[AuditStageResult] = []
        per_speed_results: Dict[str, Dict[str, Any]] = {}

        runner = EpisodeRunner(agent, lambda: create_wrapped_env(env_id, "nominal"), gamma=self.gamma, device=self.device)

        for speed in self.speeds:
            # Factory for speed-shifted env
            scenario = f"speed_{speed}x" if speed > 1 else "nominal"
            runner.env_factory = lambda s=scenario: create_wrapped_env(env_id, s)
            
            interv_results = {}
            for interv in self.interventions:
                label = f"speed={speed} [{interv}]"
                eps = runner.run_many(
                    n_episodes=self.n_episodes,
                    n_workers=self.n_workers,
                    intervention=interv,
                    seed=self.seed,
                    label=label,
                    verbose=self.verbose,
                    seed_offset=speed * 1000,
                )
                agg = aggregate_episode_metrics([e.to_dict() for e in eps])
                interv_results[interv] = agg
            
            per_speed_results[str(speed)] = interv_results

        # Compute Degradation & Summary
        worst_ratio = 1.0
        for interv in [i for i in self.interventions if i != "none"]:
            deg_by_speed = {}
            for s in self.speeds:
                res = per_speed_results[str(s)]
                base_rmse = res["none"].get("rmse_mean", 0.0)
                test_rmse = res[interv].get("rmse_mean", 0.0)
                
                deg = compute_degradation(base_rmse, test_rmse)
                worst_ratio = max(worst_ratio, deg["ratio"])
                
                deg_by_speed[str(s)] = deg

            stages.append(AuditStageResult(
                stage_name=f"Intervention: {interv}",
                pass_rate=float(1.0 / worst_ratio),  # Inverse of degradation
                metrics={
                    "worst_rmse_ratio": MetricValue(worst_ratio),
                    "reliance_rating": MetricValue(0.0, metadata={"rating": reliance_rating(worst_ratio)}),
                },
                success=worst_ratio < 2.0,  # Arbitrary success threshold
            ))

        rating = reliance_rating(worst_ratio)
        interpretation = (
            "Agent has learned timing features" if rating in ("HIGH", "EXTREME")
            else "Agent shows minimal timing dependence"
        )

        return AuditReport(
            agent_id=str(id(agent)),
            timestamp=_time.ctime(),
            reliability_score=float(1.0 / worst_ratio),
            level=self.determine_level(float(1.0 / worst_ratio), stages),
            stages=stages,
            capabilities=agent.get_capabilities(),
            summary=f"Reliance Audit: {interpretation} (worst ratio {worst_ratio:.2f}x)",
        )

    def _empty_report(self, agent: Any) -> AuditReport:
        return AuditReport(
            agent_id=str(id(agent)),
            timestamp=_time.ctime(),
            reliability_score=0.0,
            level=BaseAuditor.determine_level(0, []),
            stages=[],
            capabilities=agent.get_capabilities(),
            summary="Reliance Audit: N/A (no intervention support)",
        )
