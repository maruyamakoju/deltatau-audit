"""Robustness Auditor (Axis 2).

Evaluates the agent's performance under realistic timing perturbations
using environment wrappers. Supports adaptive sampling for high-rigor
statistical confidence.
"""

from __future__ import annotations

import logging
import time as _time
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from deltatau_audit._constants import ROBUSTNESS_SCENARIO_LABELS as ROBUSTNESS_SCENARIOS
from deltatau_audit.core.runner import EpisodeRunner
from deltatau_audit.metrics import (
    aggregate_episode_metrics,
    bootstrap_return_ratio,
    compute_return_ratio,
    robustness_rating,
)
from deltatau_audit.schema import AuditReport, AuditStageResult, MetricValue
from deltatau_audit.wrappers.factory import create_wrapped_env

from .base import BaseAuditor

logger = logging.getLogger("deltatau-audit")


class RobustnessAuditor(BaseAuditor):
    """Axis 2: Realistic timing perturbations via env wrappers.

    The agent runs NORMALLY (no internal intervention). Only the
    environment is perturbed. Measures whether performance holds
    under deployment-realistic timing conditions.
    """

    def __init__(
        self,
        n_episodes: int = 50,
        gamma: float = 0.99,
        device: str = "cpu",
        n_workers: int = 1,
        adaptive: bool = False,
        target_ci_width: float = 0.10,
        max_episodes: int = 500,
        bootstrap_samples: int = 2000,
        seed: Optional[int] = None,
        verbose: bool = True,
    ):
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.device = device
        self.n_workers = n_workers
        self.adaptive = adaptive
        self.target_ci_width = target_ci_width
        self.max_episodes = max_episodes
        self.bootstrap_samples = bootstrap_samples
        self.seed = seed
        self.verbose = verbose

    def run(self, agent: Any, env_id: str, **kwargs: Any) -> AuditReport:
        """Executes a multi-scenario robustness audit."""
        scenarios = kwargs.get("scenarios") or list(ROBUSTNESS_SCENARIOS.keys())
        if "nominal" not in scenarios:
            scenarios = ["nominal"] + list(scenarios)

        if self.verbose:
            mode_str = f" [adaptive: target CI ±{self.target_ci_width:.2f}]" if self.adaptive else ""
            print(f"  Robustness Test (env wrappers){mode_str}")

        # Initialize runner with a factory that applies the scenario wrapper
        def _factory(sc: str):
            return lambda: create_wrapped_env(env_id, sc, adapter=agent)

        runner = EpisodeRunner(agent, _factory("nominal"), gamma=self.gamma, device=self.device)
        
        stages: List[AuditStageResult] = []
        all_results: Dict[str, List[Any]] = {}

        if not self.adaptive:
            # Fixed sampling
            for idx, sc in enumerate(scenarios):
                label = ROBUSTNESS_SCENARIOS.get(sc, sc)
                runner.env_factory = _factory(sc)
                eps = runner.run_many(
                    n_episodes=self.n_episodes,
                    n_workers=self.n_workers,
                    seed=self.seed,
                    label=label,
                    verbose=self.verbose,
                    seed_offset=idx * 1000,
                )
                all_results[sc] = eps
        else:
            # Adaptive sampling
            all_results = self._run_adaptive(agent, _factory, scenarios)

        # Baseline stats
        nom_rets = [ep.total_reward for ep in all_results["nominal"]]
        baseline_mean = float(np.mean(nom_rets))
        
        # Build stage results
        for sc in scenarios:
            eps = all_results[sc]
            agg = aggregate_episode_metrics([e.to_dict() for e in eps])
            pert_rets = [ep.total_reward for ep in eps]
            
            # Compute return ratio vs nominal
            ratio = compute_return_ratio(baseline_mean, float(np.mean(pert_rets)))
            bci = bootstrap_return_ratio(
                nom_rets, pert_rets, n_bootstrap=self.bootstrap_samples
            )
            
            stages.append(AuditStageResult(
                stage_name=ROBUSTNESS_SCENARIOS.get(sc, sc),
                pass_rate=float(np.clip(ratio, 0.0, 1.0)),
                metrics={
                    "n_episodes": MetricValue(float(len(eps))),
                    "mean_reward": MetricValue(float(np.mean(pert_rets)), lower_ci=bci["ci_lower"], upper_ci=bci["ci_upper"]),
                    "return_ratio": MetricValue(ratio),
                    "degradation": MetricValue((1.0 - ratio) * 100),
                },
                success=ratio > 0.8,
            ))

        # Overall score
        scores = [s.pass_rate for s in stages if s.stage_name != "Nominal"]
        score = float(np.mean(scores)) if scores else 1.0
        
        return AuditReport(
            agent_id=str(id(agent)),
            timestamp=_time.ctime(),
            reliability_score=score,
            level=self.determine_level(score, stages),
            stages=stages,
            capabilities=agent.get_capabilities(),
            summary=f"Robustness audit completed across {len(scenarios)} scenarios.",
        )

    def _run_adaptive(self, agent: Any, factory_gen: Callable[[str], Any], scenarios: List[str]) -> Dict[str, List[Any]]:
        """Adaptive sampling implementation."""
        all_eps: Dict[str, List[Any]] = {sc: [] for sc in scenarios}
        batch_size = max(1, self.n_episodes)
        batch_num = 0
        
        runner = EpisodeRunner(agent, factory_gen("nominal"), gamma=self.gamma, device=self.device)

        while True:
            for idx, sc in enumerate(scenarios):
                if len(all_eps[sc]) >= self.max_episodes:
                    continue
                
                runner.env_factory = factory_gen(sc)
                new_eps = runner.run_many(
                    n_episodes=min(batch_size, self.max_episodes - len(all_eps[sc])),
                    n_workers=self.n_workers,
                    seed=self.seed,
                    label=ROBUSTNESS_SCENARIOS.get(sc, sc),
                    verbose=self.verbose if batch_num == 0 else False,
                    seed_offset=idx * 1000 + batch_num * 100_000,
                )
                all_eps[sc].extend(new_eps)

            batch_num += 1
            
            # Check convergence
            nom_rets = [ep.total_reward for ep in all_eps["nominal"]]
            all_converged = True
            for sc in scenarios:
                if sc == "nominal": continue
                pert_rets = [ep.total_reward for ep in all_eps[sc]]
                if len(pert_rets) < 5:
                    all_converged = False
                    break
                bci = bootstrap_return_ratio(nom_rets, pert_rets, n_bootstrap=self.bootstrap_samples)
                if (bci["ci_upper"] - bci["ci_lower"]) > self.target_ci_width:
                    all_converged = False
                    break
            
            if all_converged or len(all_eps["nominal"]) >= self.max_episodes:
                break
                
        return all_eps
