"""Temporal Horizon Auditor -- research-grade.

Tests agents on cascading multi-step timing scenarios with multi-scale
analysis, phase transition detection, adaptive cascade scheduling, and
temporal consistency scoring.

Unlike the standard auditor (single-step perturbation), this tests:
- Timing cascade: speed changes propagate over many steps
- Temporal planning horizon: does multi-step lookahead help?
- Long-horizon jitter: cumulative timing drift over full episodes

Key metrics
-----------
horizon_robustness_score
    Performance at step T / performance at step 1.  Measures how timing
    errors compound over time.

planning_advantage
    planning_agent_score / reactive_agent_score.  Tests whether
    anticipatory planning helps vs reactive adaptation.

Multi-Scale Robustness Curve
    Robustness at time scales [10, 50, 100, 500, 1000] steps, showing
    how the agent degrades (or recovers) at different temporal horizons.

Phase Transition Detection
    Uses cumulative sum (CUSUM) change-point detection to identify
    timestamps where performance drops sharply.

Adaptive Cascade Scheduling
    Binary-searches for the precise failure boundary instead of testing
    only a fixed schedule.

Temporal Consistency Score
    Jensen-Shannon divergence between action distributions at different
    timings, measuring how *consistent* the agent's behaviour is under
    repeated timing perturbations.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger("deltatau-audit")

# Minimum window size for early/late comparisons to avoid overlap
_MIN_WINDOW = 5


class TemporalHorizonAuditor:
    """Tests agents on cascading multi-step timing scenarios.

    Unlike standard auditor (single-step perturbation), this tests:
    - Timing cascade: speed changes propagate over 10+ steps
    - Temporal planning horizon: does lookahead help?
    - Long-horizon jitter: cumulative timing drift over episodes

    Key metric: horizon_robustness_score = performance at step T /
    performance at step 1 (measures how timing errors compound over time)

    Parameters
    ----------
    gamma : float
        Discount factor (default 0.99).
    device : str
        Torch device (default ``"cpu"``).
    verbose : bool
        Print progress (default ``True``).
    time_scales : list[int] or None
        Horizons for multi-scale analysis (default
        ``[10, 50, 100, 500, 1000]``).
    """

    def __init__(
        self,
        gamma: float = 0.99,
        device: str = "cpu",
        verbose: bool = True,
        time_scales: Optional[List[int]] = None,
    ):
        self.gamma = gamma
        self.device = device
        self.verbose = verbose
        self.time_scales = time_scales or [10, 50, 100, 500, 1000]

    # ================================================================
    # 1. Core Cascade Audit
    # ================================================================

    def run_cascade_audit(
        self,
        adapter,
        env_factory: Callable[[], Any],
        cascade_schedule: Optional[List[Tuple[int, int]]] = None,
        horizon: int = 50,
        n_episodes: int = 20,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run a timing cascade audit.

        A timing cascade is a sequence of speed changes where each change
        has downstream effects on subsequent steps.

        Args:
            adapter: AgentAdapter instance.
            env_factory: Callable returning a fresh gymnasium env.
            cascade_schedule: List of (step, speed) tuples.  Example::

                    [(0, 1), (10, 3), (20, 5), (30, 2), (40, 1)]

                Default: gentle escalation then recovery.
            horizon: Number of timesteps to track (default: 50).
            n_episodes: Number of episodes to run.
            seed: Random seed.

        Returns:
            Dict with rewards_by_phase, horizon_robustness_score,
            step_means, multi_scale_robustness, phase_transitions,
            temporal_consistency, and cascade_details.
        """
        if cascade_schedule is None:
            cascade_schedule = [
                (0, 1),    # Normal
                (10, 3),   # Sudden speed-up
                (20, 5),   # Extreme speed
                (30, 2),   # Partial recovery
                (40, 1),   # Full recovery
            ]

        from deltatau_audit.wrappers.speed import PiecewiseSwitchWrapper

        if self.verbose:
            print("  Horizon Cascade Audit")
            print(f"    Schedule: {cascade_schedule}")
            print(f"    Horizon: {horizon} steps, {n_episodes} episodes")

        phase_rewards: Dict[int, List[float]] = {}
        all_step_rewards: List[List[float]] = []
        all_actions: List[List[Any]] = []  # for temporal consistency

        for ep_idx in range(n_episodes):
            ep_seed = (None if seed is None else seed + ep_idx)

            env = env_factory()
            env = PiecewiseSwitchWrapper(env, schedule=cascade_schedule)

            reset_kwargs = {"seed": ep_seed} if ep_seed is not None else {}
            obs, _ = env.reset(**reset_kwargs)
            hidden = adapter.reset_hidden(1, self.device)

            step_rewards: List[float] = []
            ep_actions: List[Any] = []
            done = False
            step = 0

            while not done and step < horizon:
                if not isinstance(obs, torch.Tensor):
                    obs_t = torch.tensor(obs, dtype=torch.float32)
                else:
                    obs_t = obs

                action, value, hidden_new, dt = adapter.act(obs_t, hidden)
                hidden = hidden_new

                obs, reward, term, trunc, _ = env.step(action)
                step_rewards.append(float(reward))
                ep_actions.append(action)
                done = term or trunc
                step += 1

            env.close()

            # Pad to horizon with 0s if episode ended early
            while len(step_rewards) < horizon:
                step_rewards.append(0.0)
            while len(ep_actions) < horizon:
                ep_actions.append(None)

            all_step_rewards.append(step_rewards)
            all_actions.append(ep_actions)

            # Assign rewards to phases
            for phase_idx, (phase_start, speed) in enumerate(cascade_schedule):
                next_start = (
                    cascade_schedule[phase_idx + 1][0]
                    if phase_idx + 1 < len(cascade_schedule)
                    else horizon
                )
                phase_steps = step_rewards[phase_start:next_start]
                if phase_idx not in phase_rewards:
                    phase_rewards[phase_idx] = []
                phase_rewards[phase_idx].extend(phase_steps)

        # ── Compute horizon robustness score ────────────────────────────
        step_means = np.mean(all_step_rewards, axis=0)  # (horizon,)

        # Edge case: horizon < 20 -- use full horizon as single window
        if horizon < 2 * _MIN_WINDOW:
            # Not enough steps for two non-overlapping windows
            half = max(1, horizon // 2)
            start_perf = float(np.mean(step_means[:half]))
            end_perf = float(np.mean(step_means[half:]))
        else:
            window = min(10, horizon // 2)
            start_perf = float(np.mean(step_means[:window]))
            end_perf = float(np.mean(step_means[-window:]))

        horizon_robustness = self._safe_ratio(end_perf, start_perf)

        # ── Per-phase performance ───────────────────────────────────────
        rewards_by_phase = {}
        for phase_idx, (phase_start, speed) in enumerate(cascade_schedule):
            if phase_idx in phase_rewards and phase_rewards[phase_idx]:
                rewards_by_phase[f"phase_{phase_idx}_speed{speed}"] = {
                    "mean_reward": float(np.mean(phase_rewards[phase_idx])),
                    "std_reward": float(np.std(phase_rewards[phase_idx])),
                    "speed": speed,
                    "start_step": phase_start,
                }

        # ── Multi-scale temporal analysis ───────────────────────────────
        multi_scale = self._multi_scale_robustness(
            all_step_rewards, horizon
        )

        # ── Phase transition detection (CUSUM) ─────────────────────────
        phase_transitions = self._detect_phase_transitions(
            step_means.tolist()
        )

        # ── Temporal consistency score ──────────────────────────────────
        consistency = self._temporal_consistency(all_actions, cascade_schedule)

        if self.verbose:
            print(
                f"    -> Horizon robustness: {horizon_robustness:.3f} "
                f"(start={start_perf:.3f}, end={end_perf:.3f})"
            )
            if phase_transitions:
                pts = [f"t={p['step']}(mag={p['magnitude']:.3f})"
                       for p in phase_transitions]
                print(f"    -> Phase transitions: {', '.join(pts)}")
            print(f"    -> Temporal consistency: "
                  f"{consistency.get('mean_consistency', 0.0):.3f}")

        return {
            "rewards_by_phase": rewards_by_phase,
            "horizon_robustness_score": horizon_robustness,
            "step_means": step_means.tolist(),
            "n_episodes": n_episodes,
            "horizon": horizon,
            "cascade_schedule": cascade_schedule,
            "multi_scale_robustness": multi_scale,
            "phase_transitions": phase_transitions,
            "temporal_consistency": consistency,
        }

    # ================================================================
    # 2. Multi-Scale Temporal Analysis
    # ================================================================

    def _multi_scale_robustness(
        self,
        all_step_rewards: List[List[float]],
        max_horizon: int,
    ) -> Dict[str, Any]:
        """Compute robustness at multiple time scales.

        For each time scale T in ``self.time_scales``:
        1. Truncate each episode to T steps (or skip if T > max_horizon).
        2. Compare first-window performance to last-window performance.
        3. Report the robustness ratio at each scale.

        This reveals whether the agent degrades gradually or has a
        characteristic "failure horizon" beyond which performance
        collapses.

        Returns
        -------
        dict
            ``"scales"``       : list of tested scales
            ``"robustness"``   : robustness ratio at each scale
            ``"curve"``        : list of (scale, robustness) pairs
        """
        scales_tested: List[int] = []
        robustness_at_scale: List[float] = []

        for scale in self.time_scales:
            if scale > max_horizon:
                continue

            if scale < 2:
                continue

            # Truncate episodes to this scale
            truncated = [ep[:scale] for ep in all_step_rewards]
            step_means = np.mean(truncated, axis=0)

            if scale < 2 * _MIN_WINDOW:
                half = max(1, scale // 2)
                start_p = float(np.mean(step_means[:half]))
                end_p = float(np.mean(step_means[half:]))
            else:
                window = min(10, scale // 2)
                start_p = float(np.mean(step_means[:window]))
                end_p = float(np.mean(step_means[-window:]))

            rob = self._safe_ratio(end_p, start_p)
            scales_tested.append(scale)
            robustness_at_scale.append(rob)

        return {
            "scales": scales_tested,
            "robustness": robustness_at_scale,
            "curve": list(zip(scales_tested, robustness_at_scale)),
        }

    # ================================================================
    # 3. Phase Transition Detection (CUSUM)
    # ================================================================

    def _detect_phase_transitions(
        self,
        step_means: List[float],
        threshold_sigma: float = 2.0,
    ) -> List[Dict[str, Any]]:
        """Identify critical points where agent performance drops sharply.

        Uses the **CUSUM (Cumulative Sum)** change-point detection
        algorithm.

        Algorithm
        ---------
        1. Compute the running mean and std of per-step rewards.
        2. Maintain a cumulative sum of deviations from the mean.
        3. A phase transition is flagged when the CUSUM exceeds
           ``threshold_sigma`` standard deviations.

        The CUSUM statistic at step *t* is:

        .. math::

            S_t = \\max(0,\\; S_{t-1} + (x_t - \\mu) - k)

        where :math:`k = 0.5 \\sigma` is the allowance (slack)
        parameter.

        Parameters
        ----------
        step_means : list of float
            Per-step mean rewards across episodes.
        threshold_sigma : float
            Number of standard deviations for the CUSUM alarm
            (default 2.0).

        Returns
        -------
        list of dict
            Each entry has ``"step"``, ``"magnitude"``, ``"direction"``
            (``"drop"`` or ``"rise"``), and ``"cusum_value"``.
        """
        if len(step_means) < 3:
            return []

        arr = np.array(step_means, dtype=np.float64)
        mu = float(np.mean(arr))
        sigma = float(np.std(arr))

        if sigma < 1e-10:
            return []

        k = 0.5 * sigma  # allowance
        h = threshold_sigma * sigma  # decision threshold

        # Track both positive and negative shifts
        s_pos = 0.0  # detects upward shifts
        s_neg = 0.0  # detects downward shifts
        transitions: List[Dict[str, Any]] = []
        last_alarm_step = -10  # debounce: skip alarms too close together

        for t in range(len(arr)):
            s_pos = max(0.0, s_pos + (arr[t] - mu) - k)
            s_neg = max(0.0, s_neg - (arr[t] - mu) - k)

            if t - last_alarm_step < 3:
                continue  # debounce

            if s_pos > h:
                transitions.append({
                    "step": t,
                    "magnitude": float(s_pos / sigma),
                    "direction": "rise",
                    "cusum_value": float(s_pos),
                })
                s_pos = 0.0
                last_alarm_step = t

            elif s_neg > h:
                transitions.append({
                    "step": t,
                    "magnitude": float(s_neg / sigma),
                    "direction": "drop",
                    "cusum_value": float(s_neg),
                })
                s_neg = 0.0
                last_alarm_step = t

        return transitions

    # ================================================================
    # 4. Adaptive Cascade Scheduling
    # ================================================================

    def run_adaptive_cascade(
        self,
        adapter,
        env_factory: Callable[[], Any],
        horizon: int = 100,
        n_episodes: int = 10,
        seed: Optional[int] = None,
        failure_threshold: float = 0.5,
        max_speed: int = 20,
        precision: float = 0.5,
    ) -> Dict[str, Any]:
        """Find the precise failure boundary via binary search.

        Instead of a fixed cascade schedule, this method adaptively
        increases the perturbation speed until the agent fails, then
        binary-searches for the exact critical threshold.

        Methodology
        -----------
        1. **Exponential search**: test speeds 1, 2, 4, 8, 16, ...
           until ``horizon_robustness < failure_threshold`` or
           ``speed > max_speed``.
        2. **Binary search**: between the last passing speed and the
           first failing speed, find the precise threshold to within
           ``precision`` speed units.
        3. **Confidence interval**: run extra episodes at the
           threshold to estimate a 95% CI on the robustness score.

        Parameters
        ----------
        adapter
            AgentAdapter instance.
        env_factory
            Callable returning a fresh gymnasium env.
        horizon : int
            Episode length (default 100).
        n_episodes : int
            Episodes per speed test (default 10).
        seed : int or None
            Random seed.
        failure_threshold : float
            Robustness score below which the agent is considered to
            have failed (default 0.5).
        max_speed : int
            Maximum speed to test (default 20).
        precision : float
            Binary search stops when the speed interval is narrower
            than this (default 0.5).

        Returns
        -------
        dict
            ``"critical_speed"``: precise failure boundary.
            ``"ci_lower"``, ``"ci_upper"``: 95% CI at the boundary.
            ``"search_trace"``: list of (speed, robustness) tested.
            ``"n_evaluations"``: total episodes run.
        """
        from deltatau_audit.wrappers.speed import FixedSpeedWrapper

        if self.verbose:
            print("  Adaptive Cascade Scheduling (binary search)")

        search_trace: List[Tuple[float, float]] = []
        total_evals = 0

        def _evaluate_speed(speed_val: float) -> float:
            """Run n_episodes at a given speed and return robustness."""
            nonlocal total_evals
            rewards_list: List[float] = []

            for ep_idx in range(n_episodes):
                ep_seed = (None if seed is None else seed + ep_idx + int(speed_val * 1000))
                env = env_factory()
                if speed_val > 1.01:
                    env = FixedSpeedWrapper(env, speed=max(1, int(round(speed_val))))

                reset_kwargs = {"seed": ep_seed} if ep_seed is not None else {}
                obs, _ = env.reset(**reset_kwargs)
                hidden = adapter.reset_hidden(1, self.device)
                done = False
                ep_reward = 0.0
                step = 0

                while not done and step < horizon:
                    if not isinstance(obs, torch.Tensor):
                        obs_t = torch.tensor(obs, dtype=torch.float32)
                    else:
                        obs_t = obs
                    action, value, hidden_new, dt = adapter.act(obs_t, hidden)
                    hidden = hidden_new
                    obs, reward, term, trunc, _ = env.step(action)
                    ep_reward += reward
                    done = term or trunc
                    step += 1
                env.close()
                rewards_list.append(float(ep_reward))

            total_evals += n_episodes

            # Robustness = mean_reward / baseline (speed=1 is first entry)
            if search_trace:
                baseline = search_trace[0][1]  # robustness at speed=1 is ~1.0
            # For the very first evaluation, robustness is 1.0 by definition
            mean_r = float(np.mean(rewards_list))
            return mean_r

        # Phase 1: baseline at speed=1
        baseline_perf = _evaluate_speed(1.0)
        search_trace.append((1.0, 1.0))

        if abs(baseline_perf) < 1e-10:
            # Agent gets zero reward at baseline -- can't measure degradation
            return {
                "critical_speed": 1.0,
                "ci_lower": 1.0,
                "ci_upper": 1.0,
                "search_trace": search_trace,
                "n_evaluations": total_evals,
                "baseline_perf": baseline_perf,
            }

        # Phase 2: exponential search for failure
        lo_speed = 1.0
        hi_speed: Optional[float] = None

        speed = 2.0
        while speed <= max_speed:
            perf = _evaluate_speed(speed)
            rob = self._safe_ratio(perf, baseline_perf)
            search_trace.append((speed, rob))

            if self.verbose:
                print(f"    Speed {speed:.1f}x -> robustness {rob:.3f}")

            if rob < failure_threshold:
                hi_speed = speed
                break
            else:
                lo_speed = speed
                speed *= 2.0

        if hi_speed is None:
            # Never failed -- agent is robust up to max_speed
            critical = float(max_speed)
        else:
            # Phase 3: binary search
            while (hi_speed - lo_speed) > precision:
                mid = (lo_speed + hi_speed) / 2.0
                perf = _evaluate_speed(mid)
                rob = self._safe_ratio(perf, baseline_perf)
                search_trace.append((mid, rob))

                if self.verbose:
                    print(f"    Binary: speed {mid:.1f}x -> robustness {rob:.3f}")

                if rob < failure_threshold:
                    hi_speed = mid
                else:
                    lo_speed = mid

            critical = (lo_speed + hi_speed) / 2.0

        # Phase 4: confidence interval at critical speed
        ci_perfs: List[float] = []
        for _ in range(max(5, n_episodes)):
            p = _evaluate_speed(critical)
            ci_perfs.append(self._safe_ratio(p, baseline_perf))

        ci_arr = np.array(ci_perfs)
        ci_lower = float(np.percentile(ci_arr, 2.5))
        ci_upper = float(np.percentile(ci_arr, 97.5))

        if self.verbose:
            print(f"    -> Critical speed: {critical:.2f}x "
                  f"[{ci_lower:.3f}, {ci_upper:.3f}]")

        return {
            "critical_speed": critical,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "search_trace": search_trace,
            "n_evaluations": total_evals,
            "baseline_perf": baseline_perf,
        }

    # ================================================================
    # 5. Temporal Consistency Score
    # ================================================================

    def _temporal_consistency(
        self,
        all_actions: List[List[Any]],
        cascade_schedule: List[Tuple[int, int]],
    ) -> Dict[str, Any]:
        """Measure how consistent agent actions are under repeated timing perturbations.

        Methodology
        -----------
        For each phase in the cascade schedule, collect the distribution
        of actions across episodes.  Then compute the Jensen-Shannon
        divergence between action distributions in different phases.

        Low JSD = agent behaves consistently regardless of timing.
        High JSD = agent's behaviour changes significantly with timing.

        The **temporal consistency score** is defined as:

        .. math::

            C = 1 - \\text{mean}(\\text{JSD across all phase pairs})

        A score of 1.0 means perfectly consistent; 0.0 means maximally
        inconsistent.

        For continuous actions, we discretise into bins before computing
        JSD.  For ``None`` actions (padding), we skip those steps.

        Returns
        -------
        dict
            ``"mean_consistency"``: overall score in [0, 1].
            ``"per_phase_pair"``: JSD for each pair of phases.
        """
        n_episodes = len(all_actions)
        if n_episodes < 2 or len(cascade_schedule) < 2:
            return {"mean_consistency": 1.0, "per_phase_pair": {}}

        horizon = len(all_actions[0]) if all_actions else 0

        # Collect actions per phase
        phase_actions: Dict[int, List[float]] = {}
        for phase_idx, (phase_start, speed) in enumerate(cascade_schedule):
            next_start = (
                cascade_schedule[phase_idx + 1][0]
                if phase_idx + 1 < len(cascade_schedule)
                else horizon
            )
            vals: List[float] = []
            for ep in all_actions:
                for t in range(phase_start, min(next_start, len(ep))):
                    a = ep[t]
                    if a is None:
                        continue
                    # Convert action to a float for histogram
                    if isinstance(a, (int, float, np.integer, np.floating)):
                        vals.append(float(a))
                    elif isinstance(a, np.ndarray):
                        vals.extend(a.flatten().tolist())
                    elif isinstance(a, torch.Tensor):
                        vals.extend(a.detach().cpu().flatten().tolist())
                    else:
                        try:
                            vals.append(float(a))
                        except (TypeError, ValueError):
                            pass
            phase_actions[phase_idx] = vals

        # Compute pairwise JSD
        n_bins = 20
        per_phase_pair: Dict[str, float] = {}
        jsd_values: List[float] = []

        phases = sorted(phase_actions.keys())
        for i in range(len(phases)):
            for j in range(i + 1, len(phases)):
                a_i = phase_actions[phases[i]]
                a_j = phase_actions[phases[j]]

                if len(a_i) < 2 or len(a_j) < 2:
                    continue

                jsd = self._jensen_shannon_divergence(a_i, a_j, n_bins)
                key = f"phase_{phases[i]}_vs_{phases[j]}"
                per_phase_pair[key] = jsd
                jsd_values.append(jsd)

        if jsd_values:
            mean_jsd = float(np.mean(jsd_values))
            # JSD is bounded in [0, ln(2)] for base-e; normalise to [0, 1]
            normalised_jsd = mean_jsd / max(math.log(2), 1e-10)
            mean_consistency = float(np.clip(1.0 - normalised_jsd, 0.0, 1.0))
        else:
            mean_consistency = 1.0

        return {
            "mean_consistency": mean_consistency,
            "per_phase_pair": per_phase_pair,
        }

    @staticmethod
    def _jensen_shannon_divergence(
        p_data: List[float],
        q_data: List[float],
        n_bins: int = 20,
    ) -> float:
        """Compute JSD between two sets of continuous samples.

        Discretises both samples into a shared histogram and computes
        the symmetric Jensen-Shannon divergence:

        .. math::

            \\text{JSD}(P || Q) = \\frac{1}{2} D_{KL}(P || M)
            + \\frac{1}{2} D_{KL}(Q || M)

        where :math:`M = (P + Q) / 2`.

        Uses Laplace smoothing (add 1 to each bin) to avoid zero
        probabilities.

        Parameters
        ----------
        p_data, q_data : list of float
            Samples from the two distributions.
        n_bins : int
            Number of histogram bins.

        Returns
        -------
        float
            JSD value in [0, ln(2)].
        """
        combined = np.concatenate([p_data, q_data])
        lo, hi = float(np.min(combined)), float(np.max(combined))

        if abs(hi - lo) < 1e-10:
            return 0.0

        bins = np.linspace(lo, hi, n_bins + 1)

        p_hist = np.histogram(p_data, bins=bins)[0].astype(np.float64) + 1.0
        q_hist = np.histogram(q_data, bins=bins)[0].astype(np.float64) + 1.0

        p_prob = p_hist / p_hist.sum()
        q_prob = q_hist / q_hist.sum()

        m = 0.5 * (p_prob + q_prob)

        # KL(p || m) and KL(q || m)
        kl_pm = float(np.sum(p_prob * np.log(p_prob / m)))
        kl_qm = float(np.sum(q_prob * np.log(q_prob / m)))

        return 0.5 * kl_pm + 0.5 * kl_qm

    # ================================================================
    # 6. Reactive vs Planning Comparison (original)
    # ================================================================

    def compare_reactive_vs_planning(
        self,
        reactive_adapter,
        planning_adapter,
        env_factory: Callable[[], Any],
        n_episodes: int = 20,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compare a standard reactive agent vs a planning agent under cascade.

        Args:
            reactive_adapter: Standard (non-planning) AgentAdapter.
            planning_adapter: Planning AgentAdapter (e.g., wrapping
                TemporalPlanningAgent).
            env_factory: Env factory (same env for both agents).
            n_episodes: Episodes per agent.
            seed: Random seed.

        Returns:
            Dict with:
                reactive_score: Mean total reward for reactive agent.
                planning_score: Mean total reward for planning agent.
                planning_advantage: planning_score / reactive_score.
                rating: "PLANNING_WINS" | "REACTIVE_WINS" | "TIED".
        """
        if self.verbose:
            print("  Reactive vs Planning Comparison")

        def _run_agent(adapter, name: str) -> float:
            total_rewards = []
            from deltatau_audit.wrappers.speed import PiecewiseSwitchWrapper
            schedule = [(0, 1), (15, 5), (30, 1)]

            for ep_idx in range(n_episodes):
                ep_seed = (None if seed is None else seed + ep_idx)
                env = PiecewiseSwitchWrapper(env_factory(), schedule=schedule)

                reset_kwargs = {"seed": ep_seed} if ep_seed is not None else {}
                obs, _ = env.reset(**reset_kwargs)
                hidden = adapter.reset_hidden(1, self.device)
                done = False
                ep_reward = 0.0

                while not done:
                    if not isinstance(obs, torch.Tensor):
                        obs_t = torch.tensor(obs, dtype=torch.float32)
                    else:
                        obs_t = obs
                    action, value, hidden_new, dt = adapter.act(obs_t, hidden)
                    hidden = hidden_new
                    obs, reward, term, trunc, _ = env.step(action)
                    ep_reward += float(reward)
                    done = term or trunc

                env.close()
                total_rewards.append(ep_reward)

            mean = float(np.mean(total_rewards))
            if self.verbose:
                print(f"    {name}: mean_reward={mean:.2f}")
            return mean

        reactive_score = _run_agent(reactive_adapter, "Reactive")
        planning_score = _run_agent(planning_adapter, "Planning")

        advantage = self._safe_ratio(planning_score, reactive_score)

        if advantage > 1.05:
            rating = "PLANNING_WINS"
        elif advantage < 0.95:
            rating = "REACTIVE_WINS"
        else:
            rating = "TIED"

        if self.verbose:
            print(f"    -> Planning advantage: {advantage:.3f} ({rating})")

        return {
            "reactive_score": reactive_score,
            "planning_score": planning_score,
            "planning_advantage": advantage,
            "rating": rating,
        }

    # ================================================================
    # Utility methods
    # ================================================================

    @staticmethod
    def _safe_ratio(
        numerator: float,
        denominator: float,
        zero_default: float = 1.0,
    ) -> float:
        """Compute numerator / denominator with zero-division guard.

        Parameters
        ----------
        numerator, denominator : float
            Values to divide.
        zero_default : float
            Value to return when the denominator is near zero. Defaults
            to 1.0 (no degradation).

        Returns
        -------
        float
        """
        if abs(denominator) < 1e-10:
            if abs(numerator) < 1e-10:
                return zero_default
            return 0.0
        return numerator / denominator
