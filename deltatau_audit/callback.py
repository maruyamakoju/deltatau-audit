"""SB3 training callback: periodic timing audit during training.

Usage::

    from deltatau_audit.callback import TimingAuditCallback

    callback = TimingAuditCallback(
        env_id="HalfCheetah-v5",
        audit_freq=50_000,       # audit every 50k steps
        n_episodes=10,           # quick 10-episode audit
        output_dir="audit_logs", # save reports (optional)
    )
    model.learn(total_timesteps=1_000_000, callback=callback)

Requires ``pip install "deltatau-audit[sb3]"``.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

import gymnasium as gym


class TimingAuditCallback:
    """Stable-Baselines3 callback that runs a timing robustness audit periodically.

    Logs ``audit/deployment_score``, ``audit/stress_score``, and per-scenario
    return ratios to SB3's logger (TensorBoard, WandB, CSV, etc.).

    Inherits from ``BaseCallback`` at runtime to avoid a hard SB3 dependency
    at import time.
    """

    _base_resolved = False

    def __init__(
        self,
        env_id: str,
        audit_freq: int = 50_000,
        n_episodes: int = 10,
        speeds: Optional[List[int]] = None,
        scenarios: Optional[List[str]] = None,
        n_workers: int = 1,
        device: str = "cpu",
        seed: Optional[int] = None,
        output_dir: Optional[str] = None,
        verbose: int = 0,
    ):
        """
        Args:
            env_id: Gymnasium environment ID (must match training env).
            audit_freq: Run audit every ``audit_freq`` training timesteps.
            n_episodes: Episodes per condition (lower = faster, noisier).
            speeds: Speed multipliers for robustness test (default: [1,2,3,5,8]).
            scenarios: Robustness scenarios (default: all 6).
            n_workers: Parallel workers for episode collection.
            device: Torch device for the adapter.
            seed: Random seed for reproducibility.
            output_dir: If set, save HTML reports to ``{output_dir}/step_{n}/``.
            verbose: 0 = quiet, 1 = print summary each audit.
        """
        self._env_id = env_id
        self._audit_freq = audit_freq
        self._n_episodes = n_episodes
        self._speeds = speeds or [1, 2, 3, 5, 8]
        self._scenarios = scenarios
        self._n_workers = n_workers
        self._device = device
        self._seed = seed
        self._output_dir = output_dir
        self._verbose_level = verbose

        self._last_audit_step = -1
        self._audit_history: List[Dict[str, Any]] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

    def __class_getitem__(cls, item: Any) -> Any:
        return cls

    @property
    def audit_history(self) -> List[Dict[str, Any]]:
        """List of ``{timestep, deployment_score, stress_score, ...}`` dicts."""
        return list(self._audit_history)

    def _init_callback(self) -> None:
        pass

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_audit_step < self._audit_freq:
            return True

        self._last_audit_step = self.num_timesteps
        self._run_audit()
        return True

    def _run_audit(self) -> None:
        from .adapters.sb3 import SB3Adapter
        from .auditor import run_full_audit
        from .tracker import _build_metrics

        adapter = SB3Adapter(self.model, device=self._device)
        env_factory: Callable[[], Any] = lambda: gym.make(self._env_id)

        result = run_full_audit(
            adapter,
            env_factory,
            speeds=self._speeds,
            n_episodes=self._n_episodes,
            robustness_scenarios=self._scenarios,
            sensitivity_episodes=0,
            device=self._device,
            verbose=False,
            seed=self._seed,
            n_workers=self._n_workers,
        )

        # Log to SB3's logger
        metrics = _build_metrics(result)
        for key, val in metrics.items():
            self.logger.record(f"audit/{key}", val)

        summary = result["summary"]
        self.logger.record("audit/deployment_rating", summary["deployment_rating"])
        self.logger.record("audit/stress_rating", summary["stress_rating"])

        # Track history
        entry: Dict[str, Any] = {
            "timestep": self.num_timesteps,
            "deployment_score": summary["deployment_score"],
            "stress_score": summary["stress_score"],
            "deployment_rating": summary["deployment_rating"],
            "stress_rating": summary["stress_rating"],
        }
        self._audit_history.append(entry)

        # Save report if output_dir is set
        if self._output_dir:
            from .report import generate_report

            step_dir = os.path.join(self._output_dir, f"step_{self.num_timesteps}")
            os.makedirs(step_dir, exist_ok=True)
            generate_report(
                result,
                step_dir,
                title=f"Audit @ step {self.num_timesteps}",
            )

        if self._verbose_level >= 1:
            dep = summary["deployment_score"]
            dep_r = summary["deployment_rating"]
            strs = summary["stress_score"]
            strs_r = summary["stress_rating"]
            print(
                f"[deltatau-audit] step={self.num_timesteps}  "
                f"deploy={dep:.2f} ({dep_r})  stress={strs:.2f} ({strs_r})"
            )


def _resolve_base() -> type:
    """Dynamically inherit from SB3 BaseCallback when SB3 is available."""
    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError:
        raise ImportError(
            "stable-baselines3 is required for TimingAuditCallback.\n"
            'Install with: pip install "deltatau-audit[sb3]"'
        )
    return BaseCallback


def _make_callback_class() -> type:
    """Create a TimingAuditCallback that inherits from SB3's BaseCallback."""
    BaseCallback = _resolve_base()

    class _TimingAuditCallback(TimingAuditCallback, BaseCallback):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            TimingAuditCallback.__init__(self, *args, **kwargs)
            BaseCallback.__init__(self, verbose=self._verbose_level)

    _TimingAuditCallback.__name__ = "TimingAuditCallback"
    _TimingAuditCallback.__qualname__ = "TimingAuditCallback"
    _TimingAuditCallback.__doc__ = TimingAuditCallback.__doc__
    return _TimingAuditCallback


def create_timing_audit_callback(*args: Any, **kwargs: Any) -> Any:
    """Factory that returns a ``TimingAuditCallback`` instance.

    This resolves the SB3 dependency at call time, so the module can
    be imported even without SB3 installed.

    Accepts the same arguments as :class:`TimingAuditCallback`.
    """
    cls = _make_callback_class()
    return cls(*args, **kwargs)
