"""Unified Episode Execution Engine.

Centralizes the logic for running episodes, managing hidden states,
applying interventions, and aggregating results across parallel workers.
"""

from __future__ import annotations

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

from deltatau_audit.metrics import (
    compute_discounted_returns,
    compute_value_bias,
    compute_value_mae,
    compute_value_rmse,
)
from deltatau_audit.protocols import AgentAdapter

logger = logging.getLogger("deltatau-audit")

try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


@dataclass
class EpisodeResult:
    """Standardized results for a single agent-environment rollout."""
    total_reward: float
    length: int
    rmse: float
    mae: float
    bias: float
    dt_mean: Optional[float] = None
    dt_trace: List[float] = field(default_factory=list)
    reasoning_traces: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_reward": self.total_reward,
            "length": self.length,
            "rmse": self.rmse,
            "mae": self.mae,
            "bias": self.bias,
            "dt_mean": self.dt_mean,
            "dt_trace": self.dt_trace,
            "reasoning_traces": self.reasoning_traces,
            **self.metadata,
        }


class EpisodeRunner:
    """The 'Engine' that drives episodes under various conditions."""

    def __init__(
        self,
        adapter: AgentAdapter,
        env_factory: Callable[[], gym.Env],
        gamma: float = 0.99,
        device: str = "cpu",
        max_steps: int = 10_000,
    ):
        self.adapter = adapter
        self.env_factory = env_factory
        self.gamma = gamma
        self.device = device
        self.max_steps = max_steps

    def run_single(
        self,
        intervention: str = "none",
        seed: Optional[int] = None,
        deterministic: bool = True,
        ponder_steps: Optional[int] = None,
    ) -> EpisodeResult:
        """Run one episode with optional intervention and deterministic flag."""
        env = self.env_factory()
        reset_kwargs = {"seed": seed} if seed is not None else {}
        obs, info = env.reset(**reset_kwargs)
        
        self.adapter.reset_internal_state()
        done = False
        n_steps = 0

        step_values = []
        step_rewards = []
        step_dts = []
        reasoning_traces = []

        while not done:
            # Multi-axis reasoning check: Pondering vs Determinism vs Intervention
            action, info_step = self.adapter.act(
                obs, deterministic=deterministic, ponder_steps=ponder_steps
            )

            # Axis 1: Timing Intervention (Ablation)
            if intervention != "none":
                dt = info_step.get("dt", 1.0)
                target_dt = self._calculate_target_dt(intervention, dt)
                
                # Apply intervention through the protocol
                info_step = self.adapter.rerun_with_dt(obs, target_dt)
                value = self.adapter.recompute_value(info_step)
            else:
                value = info_step.get("value", 0.0)
                dt = info_step.get("dt")

            step_values.append(value)
            step_dts.append(dt)
            if "reasoning_trace" in info_step:
                reasoning_traces.append(info_step["reasoning_trace"])

            obs, reward, term, trunc, info_env = env.step(action)
            step_rewards.append(reward)
            done = term or trunc
            n_steps += 1

            if n_steps >= self.max_steps and not done:
                warnings.warn(
                    f"Episode exceeded max_steps={self.max_steps}. Truncating.",
                    RuntimeWarning,
                )
                done = True

        env.close()
        returns = compute_discounted_returns(step_rewards, self.gamma)

        return EpisodeResult(
            total_reward=float(sum(step_rewards)),
            length=n_steps,
            rmse=compute_value_rmse(step_values, returns),
            mae=compute_value_mae(step_values, returns),
            bias=compute_value_bias(step_values, returns),
            dt_mean=float(np.mean([d for d in step_dts if d is not None])) if any(d is not None for d in step_dts) else None,
            dt_trace=[float(d) if d is not None else 1.0 for d in step_dts],
            reasoning_traces=reasoning_traces,
        )

    def run_many(
        self,
        n_episodes: int,
        n_workers: int = 1,
        intervention: str = "none",
        seed: Optional[int] = None,
        label: str = "Audit",
        verbose: bool = True,
        seed_offset: int = 0,
    ) -> List[EpisodeResult]:
        """Run multiple episodes in parallel or serial."""
        
        def _one(idx: int) -> EpisodeResult:
            ep_seed = None if seed is None else seed + seed_offset + idx
            return self.run_single(intervention=intervention, seed=ep_seed)

        if n_workers <= 1 or n_episodes <= 1:
            # Serial path
            results = []
            iterator = range(n_episodes)
            if HAS_TQDM and verbose:
                iterator = _tqdm(iterator, desc=f"    {label:<28}", ncols=72, leave=True)
            elif verbose:
                print(f"    {label}...", end="", flush=True)

            for i in iterator:
                res = _one(i)
                results.append(res)
                if HAS_TQDM and verbose and hasattr(iterator, "set_postfix"):
                    iterator.set_postfix(R=f"{res.total_reward:.1f}")

            if not HAS_TQDM and verbose:
                print(" done.")
            return results

        # Parallel path
        if HAS_TQDM and verbose:
            bar = _tqdm(total=n_episodes, desc=f"    {label:<28}", ncols=72, leave=True)
        elif verbose:
            print(f"    {label}...", end="", flush=True)

        results = [None] * n_episodes  # type: ignore
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_one, i): i for i in range(n_episodes)}
            for future in as_completed(futures):
                idx = futures[future]
                res = future.result()
                results[idx] = res
                if HAS_TQDM and verbose:
                    bar.update(1)
                    bar.set_postfix(R=f"{res.total_reward:.1f}")

        if HAS_TQDM and verbose:
            bar.close()
        elif verbose:
            print()

        return results

    @staticmethod
    def _calculate_target_dt(intervention: str, current_dt: float) -> float:
        """Legacy intervention logic moved from auditor.py."""
        if intervention == "clamp_1":
            return 1.0
        elif intervention == "reverse":
            target = 2.0 - current_dt
            return max(0.3, min(2.5, target))
        elif intervention == "random":
            return float(np.random.uniform(0.5, 1.5))
        return 1.0
