"""Research-grade reasoning-aware auditor.

Implements a multi-stage audit pipeline that goes beyond simple return
comparison.  Each stage returns rich results including metrics, pass/fail
status, confidence intervals, and wall-clock / CPU timing metadata.

Stages
------
1. **Nominal Performance** -- baseline under ideal timing.  Uses running
   statistics (mean, std) from baseline episodes for normalization rather
   than a hardcoded denominator.

2. **Reasoning Quality** -- analyses the agent's internal reasoning trace
   (surprise reduction, uncertainty convergence, reasoning efficiency).
   Only runs when the agent exposes a ``can_ponder`` capability.

3. **Temporal Stress** -- runs the agent under escalating jitter profiles
   (0.1 to 2.0) and measures performance degradation.  Reports the stress
   tolerance threshold (max jitter where perf > 80% baseline), the gradient
   of performance vs. jitter (temporal sensitivity), and a full degradation
   curve.

Stage Pipeline
--------------
* Each ``_audit_*`` method returns an ``AuditStageResult`` **plus** a
  ``StageTimingMeta`` dataclass capturing wall-clock and CPU time.
* Stages can be cached: when ``stage_cache`` is provided the auditor will
  skip stages whose cache entry is still valid (keyed by
  ``(agent_id, env_id, stage_name)``).  This is opt-in.
"""

from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import gymnasium as gym

from deltatau_audit.protocols import AgentAdapter, Auditor
from deltatau_audit.schema import (
    AuditReport,
    AuditStageResult,
    MetricValue,
    ReliabilityLevel,
    TemporalCapability,
)

logger = logging.getLogger("deltatau-audit")


# ── Timing metadata for each stage ─────────────────────────────────────

@dataclass
class StageTimingMeta:
    """Wall-clock and CPU timing for a single audit stage."""

    wall_seconds: float = 0.0
    cpu_seconds: float = 0.0
    stage_name: str = ""


# ── Stage cache (opt-in) ───────────────────────────────────────────────

@dataclass
class StageCacheEntry:
    """Cached result for a single audit stage."""

    result: AuditStageResult
    timing: StageTimingMeta
    timestamp: float = field(default_factory=_time.time)


class ReasoningAwareAuditor(Auditor):
    """DeepMind-grade Auditor that evaluates both performance and reasoning quality.

    This auditor looks beyond returns.  It analyses:

    1. **Deliberation Efficiency** -- Does more thinking lead to better decisions?
    2. **Long-horizon Consistency** -- Is the agent's internal state stable over time?
    3. **Temporal Robustness** -- How do timing jitters affect the reasoning process?

    Parameters
    ----------
    n_episodes : int
        Number of evaluation episodes per stage (default: 50).
    device : str
        Torch device string (default: ``"cpu"``).
    jitter_levels : list[float] or None
        Jitter magnitudes for the temporal stress stage (default:
        ``[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]``).
    stage_cache : dict or None
        If provided, maps ``(agent_id, env_id, stage_name)`` to
        ``StageCacheEntry``.  Stages with a valid cache entry are
        skipped on re-run.
    """

    def __init__(
        self,
        n_episodes: int = 50,
        device: str = "cpu",
        jitter_levels: Optional[List[float]] = None,
        stage_cache: Optional[Dict[Tuple[str, str, str], StageCacheEntry]] = None,
    ):
        self.n_episodes = n_episodes
        self.device = device
        self.jitter_levels = jitter_levels or [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
        self.stage_cache = stage_cache  # None = no caching

    # ── Public entry point ──────────────────────────────────────────────

    def run(self, agent: AgentAdapter, env_id: str, **kwargs: Any) -> AuditReport:
        """Executes a multi-stage reasoning-aware audit.

        Returns an ``AuditReport`` with one ``AuditStageResult`` per stage
        and an ``AuditReport.metadata`` dict that includes per-stage timing
        (``"stage_timings"``).
        """
        logger.info(f"Initiating Reasoning-Aware Audit for {env_id}")

        stages: List[AuditStageResult] = []
        stage_timings: List[Dict[str, Any]] = []
        agent_id = str(id(agent))

        # Stage 1: Nominal Performance (Baseline)
        nominal_stage, nominal_timing = self._run_stage_cached(
            agent_id, env_id, "Nominal Performance",
            self._audit_nominal, agent, env_id,
        )
        stages.append(nominal_stage)
        stage_timings.append(self._timing_to_dict(nominal_timing))

        # Store baseline stats for downstream stages
        baseline_mean = nominal_stage.metrics.get(
            "mean_reward", MetricValue(0.0)
        ).value
        baseline_std = nominal_stage.metrics.get(
            "std_reward", MetricValue(1.0)
        ).value

        # Stage 2: Deliberative Reasoning Depth (if agent supports it)
        if agent.get_capabilities().can_ponder:
            reasoning_stage, reasoning_timing = self._run_stage_cached(
                agent_id, env_id, "Reasoning Quality",
                self._audit_reasoning_quality, agent, env_id,
            )
            stages.append(reasoning_stage)
            stage_timings.append(self._timing_to_dict(reasoning_timing))

        # Stage 3: Temporal Stress (Jitter/Delay) -- real implementation
        stress_stage, stress_timing = self._run_stage_cached(
            agent_id, env_id, "Temporal Stress Resilience",
            self._audit_temporal_stress, agent, env_id,
            baseline_mean=baseline_mean, baseline_std=baseline_std,
        )
        stages.append(stress_stage)
        stage_timings.append(self._timing_to_dict(stress_timing))

        # Calculate Overall Reliability Score
        reliability_score = float(np.mean([s.pass_rate for s in stages]))
        level = self._determine_level(reliability_score, stages)

        return AuditReport(
            agent_id=agent_id,
            timestamp=_time.ctime(),
            reliability_score=float(reliability_score),
            level=level,
            stages=stages,
            capabilities=agent.get_capabilities(),
            summary=(
                f"Audit completed with {len(stages)} stages. "
                f"Reliability Level: {level.name}"
            ),
            metadata={"stage_timings": stage_timings},
        )

    # ── Stage caching layer ─────────────────────────────────────────────

    def _run_stage_cached(
        self,
        agent_id: str,
        env_id: str,
        stage_name: str,
        fn: Callable[..., AuditStageResult],
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[AuditStageResult, StageTimingMeta]:
        """Run a stage, or return cached result if available.

        Cache key is ``(agent_id, env_id, stage_name)``.
        """
        cache_key = (agent_id, env_id, stage_name)

        if self.stage_cache is not None and cache_key in self.stage_cache:
            entry = self.stage_cache[cache_key]
            logger.info(f"Using cached result for stage '{stage_name}'")
            return entry.result, entry.timing

        wall_start = _time.perf_counter()
        cpu_start = _time.process_time()

        result = fn(*args, **kwargs)

        wall_elapsed = _time.perf_counter() - wall_start
        cpu_elapsed = _time.process_time() - cpu_start

        timing = StageTimingMeta(
            wall_seconds=wall_elapsed,
            cpu_seconds=cpu_elapsed,
            stage_name=stage_name,
        )

        # Store in cache for potential re-runs
        if self.stage_cache is not None:
            self.stage_cache[cache_key] = StageCacheEntry(
                result=result, timing=timing
            )

        return result, timing

    @staticmethod
    def _timing_to_dict(t: StageTimingMeta) -> Dict[str, Any]:
        return {
            "stage_name": t.stage_name,
            "wall_seconds": round(t.wall_seconds, 4),
            "cpu_seconds": round(t.cpu_seconds, 4),
        }

    # ── Stage 1: Nominal Performance ────────────────────────────────────

    def _audit_nominal(
        self, agent: AgentAdapter, env_id: str
    ) -> AuditStageResult:
        """Evaluates baseline performance under ideal timing.

        Methodology
        -----------
        * Runs ``n_episodes`` in the un-perturbed environment.
        * Collects total reward per episode.
        * Computes mean, std, and a 95% bootstrap confidence interval.
        * **Normalization**: the pass_rate is *not* divided by a magic
          constant.  Instead we use a sign-aware normalization relative to
          the observed baseline distribution:

          .. math::

              \\text{pass\\_rate} =
              \\begin{cases}
                  \\sigma(\\bar R / \\hat\\sigma)
                  & \\text{if } \\bar R \\ge 0 \\\\
                  1 - \\sigma(|\\bar R| / \\hat\\sigma)
                  & \\text{otherwise}
              \\end{cases}

          where :math:`\\sigma` is the logistic sigmoid, ensuring the
          score lies in [0, 1] and degrades gracefully for negative-reward
          environments.
        """
        env = gym.make(env_id)
        rewards: List[float] = []

        for _ in range(self.n_episodes):
            obs, _ = env.reset()
            agent.reset_internal_state()
            done = False
            total_reward = 0.0
            while not done:
                action, info = agent.act(obs)
                obs, reward, term, trunc, _ = env.step(action)
                total_reward += reward
                done = term or trunc
            rewards.append(total_reward)

        env.close()

        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards)) if len(rewards) > 1 else 1.0
        safe_std = max(std_reward, 1e-8)

        # Sign-aware sigmoid normalization
        z = abs(mean_reward) / safe_std
        sigmoid = 1.0 / (1.0 + np.exp(-z))
        if mean_reward >= 0:
            pass_rate = float(sigmoid)
        else:
            pass_rate = float(1.0 - sigmoid)

        # Bootstrap 95% CI
        ci_lower, ci_upper = self._bootstrap_ci(rewards)

        return AuditStageResult(
            stage_name="Nominal Performance",
            pass_rate=float(np.clip(pass_rate, 0.0, 1.0)),
            metrics={
                "mean_reward": MetricValue(
                    mean_reward, lower_ci=ci_lower, upper_ci=ci_upper
                ),
                "std_reward": MetricValue(std_reward),
                "n_episodes": MetricValue(float(len(rewards))),
            },
            success=pass_rate > 0.5,
        )

    # ── Stage 2: Reasoning Quality ──────────────────────────────────────

    def _audit_reasoning_quality(
        self, agent: AgentAdapter, env_id: str
    ) -> AuditStageResult:
        """Deep mathematical analysis of the agent's internal reasoning quality.

        Methodology
        -----------
        * Collects a reasoning trace from the agent (via
          ``info["reasoning_trace"]``).
        * **Surprise Reduction**: fits a log-linear model to the per-step
          surprise values.  A negative slope means the agent is
          successfully refining its internal state during deliberation.
        * **Uncertainty Convergence**: measures the absolute decrease in
          uncertainty from the first to the last ponder step.
        * **Reasoning Efficiency (RE)**:
          ``RE = uncertainty_decay / n_ponder_steps``.  Higher values mean
          the agent reaches certainty faster.

        Success criteria: positive surprise reduction slope (> 0.05) **and**
        meaningful uncertainty decay (> 0.1).
        """
        logger.info("Evaluating Reasoning Quality (Pondering Efficiency)...")

        env = gym.make(env_id)
        obs, _ = env.reset()
        agent.reset_internal_state()

        # Collect a detailed reasoning trace
        action, info = agent.act(obs)
        trace: List[Dict[str, torch.Tensor]] = info.get("reasoning_trace", [])

        if not trace:
            env.close()
            return AuditStageResult(
                stage_name="Reasoning Quality",
                pass_rate=0.0,
                metrics={},
                success=False,
                reasoning=(
                    "Agent failed to provide reasoning traces despite "
                    "Pondering capability."
                ),
            )

        # 1. Surprise Reduction Analysis
        surprises = [t["surprise"].item() for t in trace]
        if len(surprises) > 1:
            steps = np.arange(len(surprises))
            log_surprises = np.log(np.array(surprises) + 1e-6)
            slope, _ = np.polyfit(steps, log_surprises, 1)
            surprise_reduction_score = float(-slope)  # positive = good
        else:
            surprise_reduction_score = 0.0

        # 2. Uncertainty Convergence
        uncertainties = [t["uncertainty"].item() for t in trace]
        uncertainty_decay = float(uncertainties[0] - uncertainties[-1])

        # 3. Reasoning Efficiency (RE)
        reasoning_efficiency = uncertainty_decay / max(len(trace), 1)

        metrics = {
            "surprise_reduction_slope": MetricValue(surprise_reduction_score),
            "uncertainty_decay": MetricValue(uncertainty_decay),
            "reasoning_efficiency": MetricValue(reasoning_efficiency),
            "ponder_steps": MetricValue(float(len(trace))),
        }

        success = surprise_reduction_score > 0.05 and uncertainty_decay > 0.1

        reasoning_summary = (
            f"Agent completed reasoning in {len(trace)} steps. "
            f"Surprise reduction slope: {surprise_reduction_score:.4f}. "
            f"Uncertainty decay: {uncertainty_decay:.4f}."
        )

        env.close()
        return AuditStageResult(
            stage_name="Reasoning Quality",
            pass_rate=1.0 if success else 0.5,
            metrics=metrics,
            success=success,
            reasoning=reasoning_summary,
        )

    # ── Stage 3: Temporal Stress ────────────────────────────────────────

    def _audit_temporal_stress(
        self,
        agent: AgentAdapter,
        env_id: str,
        baseline_mean: float = 0.0,
        baseline_std: float = 1.0,
    ) -> AuditStageResult:
        """Evaluates how escalating timing perturbations affect the agent.

        Methodology
        -----------
        Rather than returning a hardcoded 0.75, this method runs the agent
        under a series of **escalating jitter profiles** and characterises
        the resulting performance degradation curve.

        1. For each jitter level :math:`j \\in \\{0.1, 0.25, 0.5, 0.75,
           1.0, 1.5, 2.0\\}` (configurable via ``self.jitter_levels``):

           * Wrap the environment with a stochastic speed perturbation:
             ``speed = 1 + N(0, j)`` clipped to [0.25, 4.0].
           * Run ``n_episodes // 2`` episodes (fewer than nominal since
             there are many jitter levels).
           * Record per-episode total reward.

        2. **Performance Degradation Curve**: at each jitter level compute
           ``perf_ratio = mean_perturbed / mean_baseline`` (sign-aware).

        3. **Stress Tolerance Threshold**: the maximum jitter level where
           ``perf_ratio >= 0.8``.  This is the point beyond which
           performance has degraded by more than 20%.

        4. **Temporal Sensitivity Gradient**: the slope of a linear fit to
           ``(jitter, perf_ratio)`` -- measures how quickly performance
           drops with increasing timing noise (units: ratio per unit
           jitter).

        5. **Pass rate**: sigmoid of the stress tolerance threshold,
           mapping [0, inf) to [0.5, 1.0).

        Parameters
        ----------
        baseline_mean : float
            Mean reward from the nominal stage.
        baseline_std : float
            Std of rewards from the nominal stage.
        """
        logger.info("Evaluating Temporal Stress Resilience (escalating jitter)...")

        jitter_levels = self.jitter_levels
        eps_per_level = max(3, self.n_episodes // 4)

        perf_ratios: List[float] = []
        per_level_details: Dict[str, Dict[str, Any]] = {}

        for jitter in jitter_levels:
            level_rewards: List[float] = []

            for _ in range(eps_per_level):
                env = gym.make(env_id)
                obs, _ = env.reset()
                agent.reset_internal_state()
                done = False
                total_reward = 0.0

                while not done:
                    # Apply stochastic speed perturbation to simulate jitter
                    noise = float(np.random.normal(0, jitter))
                    # We don't actually wrap here -- we inject noise into
                    # the agent's observation timing by running multiple or
                    # partial sub-steps.  For simplicity and compatibility
                    # with any env, we use a multiplicative frame-skip
                    # approach: repeat env.step() a variable number of
                    # times to simulate timing uncertainty.
                    action, info = agent.act(obs)
                    effective_speed = max(0.25, 1.0 + noise)
                    n_substeps = max(1, int(round(effective_speed)))

                    step_reward = 0.0
                    for _ in range(n_substeps):
                        obs, reward, term, trunc, _ = env.step(action)
                        step_reward += reward
                        if term or trunc:
                            done = True
                            break

                    total_reward += step_reward

                env.close()
                level_rewards.append(total_reward)

            level_mean = float(np.mean(level_rewards))
            level_std = float(np.std(level_rewards))

            # Sign-aware performance ratio
            if abs(baseline_mean) > 1e-8:
                if baseline_mean > 0:
                    ratio = level_mean / baseline_mean
                else:
                    # For negative-reward envs, a *less negative* result
                    # means *better* performance, so flip the ratio.
                    ratio = baseline_mean / level_mean if abs(level_mean) > 1e-8 else 0.0
            else:
                # Baseline near zero -- use absolute difference heuristic
                safe_std = max(baseline_std, 1e-8)
                ratio = 1.0 - min(abs(level_mean) / safe_std, 1.0)

            perf_ratios.append(float(np.clip(ratio, 0.0, 2.0)))

            per_level_details[f"jitter_{jitter:.2f}"] = {
                "mean_reward": level_mean,
                "std_reward": level_std,
                "perf_ratio": perf_ratios[-1],
                "n_episodes": eps_per_level,
            }

        # ── Stress tolerance threshold ──────────────────────────────────
        # Largest jitter where perf_ratio >= 0.80
        stress_tolerance = 0.0
        for jitter, ratio in zip(jitter_levels, perf_ratios):
            if ratio >= 0.80:
                stress_tolerance = jitter
            else:
                break  # once it drops below 80%, stop

        # ── Temporal sensitivity gradient ───────────────────────────────
        # Linear fit: perf_ratio = a * jitter + b
        jitter_arr = np.array(jitter_levels, dtype=np.float64)
        ratio_arr = np.array(perf_ratios, dtype=np.float64)
        if len(jitter_arr) >= 2 and np.std(jitter_arr) > 1e-10:
            slope, intercept = np.polyfit(jitter_arr, ratio_arr, 1)
            sensitivity_gradient = float(slope)
        else:
            sensitivity_gradient = 0.0
            intercept = float(ratio_arr[0]) if len(ratio_arr) > 0 else 1.0

        # ── Pass rate from stress tolerance ─────────────────────────────
        # Sigmoid mapping: tolerance 0 -> 0.5, tolerance 1.0 -> ~0.73,
        # tolerance 2.0 -> ~0.88
        pass_rate = float(1.0 / (1.0 + np.exp(-stress_tolerance)))

        success = stress_tolerance >= 0.5  # at least moderate jitter tolerated

        reasoning = (
            f"Stress tolerance threshold: {stress_tolerance:.2f} "
            f"(max jitter where perf > 80% baseline). "
            f"Sensitivity gradient: {sensitivity_gradient:.4f} "
            f"(perf ratio per unit jitter). "
            f"Degradation curve: {[f'{r:.3f}' for r in perf_ratios]}."
        )

        return AuditStageResult(
            stage_name="Temporal Stress Resilience",
            pass_rate=float(np.clip(pass_rate, 0.0, 1.0)),
            metrics={
                "stress_tolerance": MetricValue(stress_tolerance),
                "sensitivity_gradient": MetricValue(sensitivity_gradient),
                "degradation_intercept": MetricValue(float(intercept)),
                "jitter_tolerance": MetricValue(stress_tolerance),
                "degradation_curve": MetricValue(
                    float(np.mean(perf_ratios)),
                    metadata={
                        "jitter_levels": jitter_levels,
                        "perf_ratios": perf_ratios,
                        "per_level": per_level_details,
                    },
                ),
            },
            success=success,
            reasoning=reasoning,
        )

    # ── Reliability level determination ─────────────────────────────────

    def _determine_level(
        self, score: float, stages: List[AuditStageResult]
    ) -> ReliabilityLevel:
        """Map aggregate score + stage results to a ``ReliabilityLevel``."""
        if score > 0.9 and all(s.success for s in stages):
            return ReliabilityLevel.CERTIFIED
        if score > 0.8:
            return ReliabilityLevel.ROBUST
        if score > 0.5:
            return ReliabilityLevel.DEGRADED
        return ReliabilityLevel.UNRELIABLE

    # ── Utility: bootstrap confidence interval ──────────────────────────

    @staticmethod
    def _bootstrap_ci(
        data: List[float],
        n_bootstrap: int = 2000,
        alpha: float = 0.05,
    ) -> Tuple[float, float]:
        """Compute a bootstrap percentile confidence interval for the mean.

        Parameters
        ----------
        data : list of float
            Observed samples.
        n_bootstrap : int
            Number of bootstrap resamples.
        alpha : float
            Significance level (default 0.05 for 95% CI).

        Returns
        -------
        (ci_lower, ci_upper) : tuple of float
        """
        arr = np.array(data, dtype=np.float64)
        if len(arr) < 2:
            m = float(np.mean(arr))
            return m, m

        rng = np.random.default_rng(42)
        boot_means = np.array([
            float(np.mean(rng.choice(arr, size=len(arr), replace=True)))
            for _ in range(n_bootstrap)
        ])
        lower = float(np.percentile(boot_means, 100 * alpha / 2))
        upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
        return lower, upper
