"""Adversarial timing wrappers for robustness auditing.

These wrappers model a worst-case adversary that controls the environment's
effective speed (action-repeat / frame-skip) to *minimise* the agent's
expected return.  Three attack strategies are provided, ordered by strength:

1. **RandomAdversarialWrapper** -- baseline: picks speeds from a
   configurable distribution (uniform, worst-of-N, Gaussian noise).
2. **AdversarialSpeedWrapper** -- search-based: evaluates a discrete set
   of candidate speeds via an optional ``value_fn`` callback and picks the
   one that *minimises* V(s).  Falls back to worst-of-N random if no
   ``value_fn`` is supplied.
3. **GradientAdversarialWrapper** -- gradient-based (strongest): uses
   autograd to compute dV/d(speed) and performs projected gradient descent
   in the negative-gradient direction to find the worst-case continuous
   speed within ``[speed_min, speed_max]``.

All wrappers expose ``actual_speed`` in the returned *info* dict and on the
unwrapped env for downstream consumers (e.g. ``TimeFeatureWrapper``).

Backward compatibility
----------------------
``AdversarialSpeedWrapper`` keeps its original constructor signature
(``env, agent_adapter, possible_speeds, device``) so that existing call
sites in ``auditor.py`` continue to work unchanged.  The new ``value_fn``
parameter is optional and additive.

``ValueAdversarialJitterWrapper`` is retained as a thin alias.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

import gymnasium as gym
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover – torch is optional at import time
    torch = None  # type: ignore[assignment]

from .speed import _set_speed_metadata, _with_speed_info

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_CANDIDATE_SPEEDS: List[float] = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
"""Default discrete speed set used by ``AdversarialSpeedWrapper``."""


# ── Helpers ──────────────────────────────────────────────────────────────────


def _execute_at_speed(env: gym.Env, action: Any, speed: float) -> Tuple:
    """Execute *action* for ``int(round(speed))`` sub-steps (min 1).

    Returns ``(obs, total_reward, terminated, truncated, info)`` with
    rewards accumulated across sub-steps.
    """
    n_repeats = max(1, int(round(speed)))
    total_reward = 0.0
    terminated = False
    truncated = False
    info: dict = {}

    for _ in range(n_repeats):
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    return obs, total_reward, terminated, truncated, info


# ═══════════════════════════════════════════════════════════════════════════════
# 1. RandomAdversarialWrapper  (baseline)
# ═══════════════════════════════════════════════════════════════════════════════


class RandomAdversarialWrapper(gym.Wrapper):
    """Baseline adversarial timing -- random speed perturbation each step.

    Three *modes* are supported:

    * ``"uniform"`` -- sample speed uniformly from *candidate_speeds*.
    * ``"worst_of_n"`` -- draw *n_candidates* speeds uniformly, execute the
      one with the **lowest** single-step reward (greedy worst-case proxy).
    * ``"gaussian"`` -- sample ``speed = base_speed + N(0, noise_std)``
      clamped to ``[speed_min, speed_max]``.

    Parameters
    ----------
    env : gym.Env
        Environment to wrap.
    mode : str
        One of ``"uniform"``, ``"worst_of_n"``, ``"gaussian"``.
    candidate_speeds : list[float]
        Speed pool for ``"uniform"`` and ``"worst_of_n"`` modes.
    n_candidates : int
        How many candidates to draw in ``"worst_of_n"`` mode (default 5).
    base_speed : float
        Centre speed for ``"gaussian"`` mode (default 1.0).
    noise_std : float
        Noise standard deviation for ``"gaussian"`` mode (default 0.5).
    speed_min, speed_max : float
        Clamp bounds for ``"gaussian"`` mode.
    seed : int | None
        RNG seed for reproducibility.
    """

    _VALID_MODES = {"uniform", "worst_of_n", "gaussian"}

    def __init__(
        self,
        env: gym.Env,
        *,
        mode: str = "uniform",
        candidate_speeds: Optional[List[float]] = None,
        n_candidates: int = 5,
        base_speed: float = 1.0,
        noise_std: float = 0.5,
        speed_min: float = 0.5,
        speed_max: float = 3.0,
        seed: Optional[int] = None,
    ):
        super().__init__(env)
        if mode not in self._VALID_MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {self._VALID_MODES}")
        self.mode = mode
        self.candidate_speeds = list(candidate_speeds or DEFAULT_CANDIDATE_SPEEDS)
        self.n_candidates = max(1, int(n_candidates))
        self.base_speed = float(base_speed)
        self.noise_std = float(noise_std)
        self.speed_min = float(speed_min)
        self.speed_max = float(speed_max)
        self._rng = np.random.RandomState(seed)

        # Tracking statistics
        self._speed_history: List[float] = []

    def reset(self, **kwargs):
        self._speed_history.clear()
        return self.env.reset(**kwargs)

    @property
    def speed_history(self) -> List[float]:
        """Speeds selected during the current episode."""
        return list(self._speed_history)

    # ── core step ────────────────────────────────────────────────────────────

    def step(self, action):
        if self.mode == "uniform":
            speed = float(self._rng.choice(self.candidate_speeds))
        elif self.mode == "gaussian":
            speed = float(
                np.clip(
                    self.base_speed + self._rng.randn() * self.noise_std,
                    self.speed_min,
                    self.speed_max,
                )
            )
        else:  # worst_of_n
            speed = self._worst_of_n(action)

        self._speed_history.append(speed)
        _set_speed_metadata(self.env, speed)
        obs, total_reward, terminated, truncated, info = _execute_at_speed(self.env, action, speed)
        return obs, total_reward, terminated, truncated, _with_speed_info(info, speed)

    def _worst_of_n(self, action) -> float:  # noqa: ARG002 – action unused
        """Pick the candidate speed that a random heuristic deems worst.

        Without a value function we cannot do true worst-case evaluation
        across candidates (the env step is destructive).  Instead we draw
        *n_candidates* speeds and pick the one *furthest* from 1.0 (the
        nominal speed), which is a simple but effective proxy for how
        disruptive a speed change is.
        """
        draws = self._rng.choice(self.candidate_speeds, size=self.n_candidates, replace=True)
        # Furthest from nominal speed 1.0 -> most disruptive
        worst_idx = int(np.argmax(np.abs(draws - 1.0)))
        return float(draws[worst_idx])


# ═══════════════════════════════════════════════════════════════════════════════
# 2. AdversarialSpeedWrapper  (search-based, value-function guided)
# ═══════════════════════════════════════════════════════════════════════════════


class AdversarialSpeedWrapper(gym.Wrapper):
    """Search-based adversarial timing attack.

    At each step the wrapper evaluates a discrete set of candidate speeds
    and selects the one that **minimises** the agent's expected future
    return as estimated by ``value_fn``.

    If *value_fn* is ``None`` the wrapper falls back to a
    *worst-of-N random* heuristic (pick the most disruptive speed among
    ``n_random_candidates`` draws).

    Parameters
    ----------
    env : gym.Env
        Environment to wrap.
    value_fn : callable, optional
        ``value_fn(obs) -> float``.  Given the current observation, return
        the agent's scalar state-value estimate.  The wrapper picks the
        speed that yields the *lowest* value after the step.  When the env
        step is destructive (cannot be undone), the wrapper approximates by
        evaluating the *current* observation augmented with each candidate
        speed -- the caller's ``value_fn`` should accept the raw obs (the
        wrapper will call it once per candidate speed before committing).
    agent_adapter : Any, optional
        Legacy parameter kept for backward compatibility with
        ``auditor.py``.  If *value_fn* is not given and *agent_adapter*
        has a ``predict_value`` method, it is used as the value function.
    possible_speeds : list[float]
        Candidate speeds to search over.
    n_random_candidates : int
        Number of random draws when falling back to heuristic mode.
    device : str
        Device hint passed to agent adapter (legacy).
    seed : int | None
        RNG seed.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        value_fn: Optional[Callable[..., float]] = None,
        agent_adapter: Any = None,
        possible_speeds: Optional[List[float]] = None,
        n_random_candidates: int = 5,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        super().__init__(env)
        self.candidate_speeds = list(possible_speeds or DEFAULT_CANDIDATE_SPEEDS)
        self.n_random_candidates = max(1, int(n_random_candidates))
        self.device = device
        self._rng = np.random.RandomState(seed)

        # Resolve value function: explicit > adapter.predict_value > None
        if value_fn is not None:
            self._value_fn: Optional[Callable[..., float]] = value_fn
        elif agent_adapter is not None and hasattr(agent_adapter, "predict_value"):
            self._value_fn = agent_adapter.predict_value
        else:
            self._value_fn = None

        self.adapter = agent_adapter  # kept for backward compat

        # Tracking
        self._speed_history: List[float] = []
        self._value_deltas: List[float] = []
        self._last_obs: Optional[np.ndarray] = None

    def reset(self, **kwargs):
        self._speed_history.clear()
        self._value_deltas.clear()
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    @property
    def speed_history(self) -> List[float]:
        return list(self._speed_history)

    @property
    def value_deltas(self) -> List[float]:
        """Per-step value drop caused by the adversary's speed choice."""
        return list(self._value_deltas)

    # ── core step ────────────────────────────────────────────────────────────

    def step(self, action):
        if self._value_fn is not None and self._last_obs is not None:
            speed = self._value_guided_search(self._last_obs)
        else:
            speed = self._random_worst_of_n()

        self._speed_history.append(speed)
        _set_speed_metadata(self.env, speed)

        obs, total_reward, terminated, truncated, info = _execute_at_speed(self.env, action, speed)

        # Track value delta if possible
        if self._value_fn is not None:
            try:
                v_before = float(self._value_fn(self._last_obs))
                v_after = float(self._value_fn(obs))
                self._value_deltas.append(v_after - v_before)
            except Exception:
                self._value_deltas.append(float("nan"))

        self._last_obs = obs
        info["adversarial_mode"] = "value_guided" if self._value_fn is not None else "random_worst_of_n"
        return obs, total_reward, terminated, truncated, _with_speed_info(info, speed)

    # ── search strategies ────────────────────────────────────────────────────

    def _value_guided_search(self, obs: Any) -> float:
        """Evaluate each candidate speed via ``value_fn`` and pick the worst.

        Since the env step is destructive, we cannot actually *try* each speed.
        Instead we query the value function for the current obs -- the
        ``value_fn`` callback provided by the auditor can internally condition
        on speed (e.g. by modifying a time-feature channel).

        As a practical approximation: we query ``value_fn(obs)`` once to get
        the baseline, then for each candidate speed we set the speed metadata
        on the env and re-query.  Value functions that read
        ``env.unwrapped.current_speed`` will thus give speed-conditioned
        estimates.
        """
        assert self._value_fn is not None
        best_speed = self.candidate_speeds[0]
        lowest_value = float("inf")

        for speed in self.candidate_speeds:
            _set_speed_metadata(self.env, speed)
            try:
                v = float(self._value_fn(obs))
            except Exception:
                v = 0.0
            if v < lowest_value:
                lowest_value = v
                best_speed = speed

        return float(best_speed)

    def _random_worst_of_n(self) -> float:
        """Worst-of-N heuristic: pick the most disruptive speed."""
        draws = self._rng.choice(
            self.candidate_speeds,
            size=min(self.n_random_candidates, len(self.candidate_speeds)),
            replace=True,
        )
        worst_idx = int(np.argmax(np.abs(draws - 1.0)))
        return float(draws[worst_idx])


# Backward-compatibility alias
ValueAdversarialJitterWrapper = AdversarialSpeedWrapper


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GradientAdversarialWrapper  (strongest -- PGD on speed)
# ═══════════════════════════════════════════════════════════════════════════════


class GradientAdversarialWrapper(gym.Wrapper):
    """Projected-gradient-descent adversarial timing attack.

    This is the strongest adversarial test.  It requires a *differentiable*
    value function (PyTorch) and computes:

        speed_{t+1} = Proj_{[s_min, s_max]}( speed_t - eps * sign(dV/d_speed) )

    The sign of the gradient determines whether increasing or decreasing
    speed hurts the agent more, and the projection keeps the speed within
    the physical bounds ``[speed_min, speed_max]``.

    The wrapper maintains a *continuous* speed variable that evolves via PGD
    across steps, but the actual action-repeat is ``int(round(speed))``.

    Parameters
    ----------
    env : gym.Env
        Environment to wrap.
    value_fn : callable
        ``value_fn(obs, speed_tensor) -> scalar_tensor``.  Must accept a
        ``torch.Tensor`` speed (with ``requires_grad=True``) and return a
        differentiable scalar value estimate.
    epsilon : float
        PGD step size per environment step (default 0.1).
    speed_min, speed_max : float
        Projection bounds for the continuous speed variable.
    initial_speed : float
        Starting speed at episode reset (default 1.0).
    momentum : float
        Momentum coefficient for the PGD update (0 = no momentum).
    seed : int | None
        RNG seed (used only for tie-breaking).
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        value_fn: Callable,
        epsilon: float = 0.1,
        speed_min: float = 0.5,
        speed_max: float = 3.0,
        initial_speed: float = 1.0,
        momentum: float = 0.0,
        seed: Optional[int] = None,
    ):
        super().__init__(env)
        if torch is None:
            raise ImportError("GradientAdversarialWrapper requires PyTorch. Install it with: pip install torch")
        self.value_fn = value_fn
        self.epsilon = float(epsilon)
        self.speed_min = float(speed_min)
        self.speed_max = float(speed_max)
        self.initial_speed = float(initial_speed)
        self.momentum = float(momentum)

        self._rng = np.random.RandomState(seed)
        self._speed: float = self.initial_speed
        self._velocity: float = 0.0  # momentum buffer

        # Tracking
        self._speed_history: List[float] = []
        self._gradient_history: List[float] = []
        self._last_obs: Optional[np.ndarray] = None

    def reset(self, **kwargs):
        self._speed = self.initial_speed
        self._velocity = 0.0
        self._speed_history.clear()
        self._gradient_history.clear()
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    @property
    def speed_history(self) -> List[float]:
        return list(self._speed_history)

    @property
    def gradient_history(self) -> List[float]:
        """Raw dV/d(speed) values recorded at each step."""
        return list(self._gradient_history)

    # ── core step ────────────────────────────────────────────────────────────

    def step(self, action):
        # 1. Compute gradient dV/d(speed) at current obs
        grad = self._compute_speed_gradient(self._last_obs)
        self._gradient_history.append(grad)

        # 2. PGD update with momentum: move speed in *negative* gradient
        #    direction to *minimise* value
        self._velocity = self.momentum * self._velocity - self.epsilon * np.sign(grad)
        self._speed = float(np.clip(self._speed + self._velocity, self.speed_min, self.speed_max))

        self._speed_history.append(self._speed)
        _set_speed_metadata(self.env, self._speed)

        # 3. Execute at the (rounded) speed
        obs, total_reward, terminated, truncated, info = _execute_at_speed(self.env, action, self._speed)
        self._last_obs = obs

        info["adversarial_mode"] = "pgd"
        info["adversarial_gradient"] = grad
        info["adversarial_continuous_speed"] = self._speed
        return obs, total_reward, terminated, truncated, _with_speed_info(info, self._speed)

    # ── gradient computation ─────────────────────────────────────────────────

    def _compute_speed_gradient(self, obs: Any) -> float:
        """Compute dV/d(speed) via PyTorch autograd.

        Returns 0.0 if the gradient cannot be computed (e.g. non-differentiable
        value function, numerical issues).
        """
        try:
            speed_t = torch.tensor(self._speed, dtype=torch.float32, requires_grad=True)

            # Convert obs to tensor if needed
            if isinstance(obs, np.ndarray):
                obs_t = torch.from_numpy(obs).float()
            elif isinstance(obs, torch.Tensor):
                obs_t = obs.float().detach()
            else:
                obs_t = torch.tensor(obs, dtype=torch.float32)

            value = self.value_fn(obs_t, speed_t)

            # Ensure scalar
            if isinstance(value, torch.Tensor):
                value = value.squeeze()
                if value.dim() != 0:
                    value = value.sum()
                value.backward()
                grad_val = speed_t.grad
                return float(grad_val.item()) if grad_val is not None else 0.0
            else:
                # value_fn returned a plain float -- no gradient available
                return 0.0
        except Exception:
            return 0.0
