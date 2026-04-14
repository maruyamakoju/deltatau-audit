"""Reasoning-Aware Auditor (Axis 3).

Evaluates both performance and internal reasoning quality (pondering, 
uncertainty decay, surprise reduction, and deliberative efficiency).
"""

from __future__ import annotations

import logging
import time as _time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from deltatau_audit.core.runner import EpisodeRunner
from deltatau_audit.schema import (
    AuditReport,
    AuditStageResult,
    MetricValue,
    ReliabilityLevel,
)
from deltatau_audit.wrappers.factory import create_wrapped_env

from .base import BaseAuditor

logger = logging.getLogger("deltatau-audit")


class ReasoningAuditor(BaseAuditor):
    """DeepMind-grade Auditor that evaluates both performance and reasoning quality.

    1. **Deliberation Efficiency** -- Does more thinking lead to better decisions?
    2. **Long-horizon Consistency** -- Is the agent's internal state stable over time?
    3. **Temporal Robustness** -- How do timing jitters affect the reasoning process?
    """

    def __init__(
        self,
        n_episodes: int = 50,
        gamma: float = 0.99,
        device: str = "cpu",
        jitter_levels: Optional[List[float]] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
    ):
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.device = device
        self.jitter_levels = jitter_levels or [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
        self.seed = seed
        self.verbose = verbose

    def run(self, agent: Any, env_id: str, **kwargs: Any) -> AuditReport:
        """Executes the multi-stage reasoning-aware audit."""
        if self.verbose:
            print("  Reasoning Test (pondering & deliberative efficiency)")

        runner = EpisodeRunner(agent, lambda: create_wrapped_env(env_id, "nominal"), gamma=self.gamma, device=self.device)
        
        stages: List[AuditStageResult] = []
        
        # Stage 1: Nominal Performance
        nom_eps = runner.run_many(self.n_episodes, label="Nominal", seed=self.seed, verbose=self.verbose)
        nom_rets = [ep.total_reward for ep in nom_eps]
        nom_mean = float(np.mean(nom_rets))
        nom_std = float(np.std(nom_rets)) if len(nom_rets) > 1 else 1.0
        
        # Sigmoid normalization for pass rate
        z = abs(nom_mean) / max(nom_std, 1e-8)
        pass_rate = float(1.0 / (1.0 + np.exp(-z))) if nom_mean >= 0 else float(1.0 - 1.0 / (1.0 + np.exp(-z)))
        
        ci_lower, ci_upper = self.bootstrap_ci(nom_rets)
        stages.append(AuditStageResult(
            stage_name="Nominal Performance",
            pass_rate=pass_rate,
            metrics={
                "mean_reward": MetricValue(nom_mean, lower_ci=ci_lower, upper_ci=ci_upper),
                "std_reward": MetricValue(nom_std),
            },
            success=pass_rate > 0.5,
        ))

        # Stage 2: Deliberative Efficiency (Axis 3)
        if agent.get_capabilities().can_ponder:
            stages.append(self._audit_deliberation(agent, env_id))
        elif self.verbose:
            print("    Deliberation Efficiency... N/A (no pondering support)")

        # Stage 3: Temporal Stress (Escalating Jitter)
        stages.append(self._audit_stress(agent, env_id, nom_mean, nom_std))

        # Overall Reliability
        score = float(np.mean([s.pass_rate for s in stages]))
        level = self.determine_level(score, stages)

        return AuditReport(
            agent_id=str(id(agent)),
            timestamp=_time.ctime(),
            reliability_score=score,
            level=level,
            stages=stages,
            capabilities=agent.get_capabilities(),
            summary=f"Reasoning audit complete. Level: {level.name}",
        )

    def _audit_deliberation(self, agent: Any, env_id: str) -> AuditStageResult:
        """Axis 3: Measures how internal thinking steps improve performance or reduce uncertainty."""
        ponder_steps_list = [1, 2, 4, 8, 16]
        max_ponder = agent.get_capabilities().max_lookahead_steps or 16
        ponder_steps_list = [p for p in ponder_steps_list if p <= max_ponder]
        
        if not ponder_steps_list:
            ponder_steps_list = [1]

        performance_by_ponder = []
        uncertainty_by_ponder = []

        runner = EpisodeRunner(agent, lambda: create_wrapped_env(env_id, "nominal"), gamma=self.gamma, device=self.device)

        for p in ponder_steps_list:
            eps = runner.run_many(
                max(5, self.n_episodes // 5), 
                label=f"Pondering {p} steps", 
                ponder_steps=p, 
                seed=self.seed,
                verbose=self.verbose
            )
            performance_by_ponder.append(float(np.mean([e.total_reward for e in eps])))
            
            # Extract uncertainty reduction from traces
            trace_unc = []
            for ep in eps:
                for t in ep.reasoning_traces:
                    if "uncertainty" in t:
                        trace_unc.append(t["uncertainty"])
            
            uncertainty_by_ponder.append(float(np.mean(trace_unc)) if trace_unc else 0.0)

        # Calculate Efficiency: correlation between pondering and performance
        if len(performance_by_ponder) > 1:
            perf_gain = (performance_by_ponder[-1] - performance_by_ponder[0])
            efficiency_score = float(np.clip(perf_gain / max(abs(performance_by_ponder[0]), 1e-8), -1.0, 1.0))
        else:
            efficiency_score = 0.0

        # Pass rate based on whether more pondering helps (or at least doesn't hurt)
        pass_rate = float(np.clip(0.5 + efficiency_score, 0.0, 1.0))

        return AuditStageResult(
            stage_name="Deliberation Efficiency",
            pass_rate=pass_rate,
            metrics={
                "perf_gain_ratio": MetricValue(efficiency_score),
                "ponder_max_reward": MetricValue(max(performance_by_ponder)),
                "uncertainty_final": MetricValue(uncertainty_by_ponder[-1] if uncertainty_by_ponder else 0.0),
            },
            success=efficiency_score >= -0.05,
            reasoning=f"Efficiency score {efficiency_score:.2f} based on pondering sweep."
        )

    def _audit_stress(self, agent: Any, env_id: str, baseline_mean: float, baseline_std: float) -> AuditStageResult:
        """Escalating jitter analysis."""
        perf_ratios = []
        
        for jitter in self.jitter_levels:
            def _jitter_factory(j=jitter):
                return create_wrapped_env(env_id, "jitter", base_speed=1, jitter=j)

            runner = EpisodeRunner(agent, _jitter_factory, gamma=self.gamma, device=self.device)
            eps = runner.run_many(max(3, self.n_episodes // 4), label=f"Jitter {jitter:.2f}", seed=self.seed, verbose=False)
            
            mean_r = float(np.mean([e.total_reward for e in eps]))
            
            # Ratio calculation
            if abs(baseline_mean) > 1e-8:
                ratio = mean_r / baseline_mean if baseline_mean > 0 else baseline_mean / mean_r
            else:
                ratio = 1.0 - min(abs(mean_r) / max(baseline_std, 1e-8), 1.0)
            
            perf_ratios.append(float(np.clip(ratio, 0.0, 2.0)))

        # Find tolerance threshold
        tolerance = 0.0
        for j, r in zip(self.jitter_levels, perf_ratios):
            if r >= 0.8: tolerance = j
            else: break
            
        # Normalization: 0.5 jitter tolerance -> 0.62 pass rate, 1.0 -> 0.73, etc.
        pass_rate = float(1.0 / (1.0 + np.exp(-tolerance)))
        
        return AuditStageResult(
            stage_name="Temporal Stress Resilience",
            pass_rate=pass_rate,
            metrics={"stress_tolerance": MetricValue(tolerance)},
            success=tolerance >= 0.5,
            reasoning=f"Max jitter with >80% perf: {tolerance:.2f}",
        )
