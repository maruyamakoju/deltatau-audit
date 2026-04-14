"""Temporal desynchronization wrappers for multi-agent and single-agent auditing.

These wrappers simulate scenarios where observations are *delayed* relative to
the true environment state.  In multi-agent settings each agent can experience
a different delay and effective speed, modelling heterogeneous communication
latencies and control frequencies.

Two wrappers are provided:

1. **TemporalDesyncWrapper** -- deterministic per-agent delays and speed
   factors.  Supports both single-agent (plain ``gym.Env``) and multi-agent
   (list/dict observations) environments.
2. **StochasticDesyncWrapper** -- random observation delays drawn each step
   from a configurable distribution (Poisson, geometric, uniform).  Tracks
   delay statistics for reporting.

Design decisions
----------------
* **Observations are delayed; actions execute immediately.**  This is the
  standard model for networked control: the controller sends a command that
  arrives instantly at the actuator, but sensor readings take time to travel
  back.
* Internally, observations are buffered in ``collections.deque`` instances
  with ``maxlen`` set to ``max_delay + 1`` so memory usage is bounded.
* Both wrappers expose delay metadata in the *info* dict for downstream
  consumers (e.g. the auditor report generator).

Backward compatibility
----------------------
``TemporalDesyncWrapper`` keeps its original constructor signature so that
any existing call sites continue to work.  The new ``speed_factors``
parameter and single-agent mode are additive.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Union

import gymnasium as gym
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# 1. TemporalDesyncWrapper  (deterministic delays)
# ═══════════════════════════════════════════════════════════════════════════════


class TemporalDesyncWrapper(gym.Wrapper):
    """Deterministic per-agent observation delay and speed-factor injection.

    In **multi-agent mode** (``num_agents >= 2``), the base env is expected
    to accept a list of actions and return a list of observations/rewards.
    Each agent *i* sees its observation delayed by ``agent_delays[i]`` steps,
    and its action is repeated ``int(round(speed_factors[i]))`` times (speed
    factor).

    In **single-agent mode** (the default when ``agent_delays`` has length 1
    or is an int), the wrapper simply delays the single observation stream by
    the given number of steps and optionally applies a speed factor.

    Parameters
    ----------
    env : gym.Env
        Base environment.
    agent_speeds : list[int], optional
        *Legacy* alias for ``agent_delays`` -- kept for backward
        compatibility.  If both ``agent_speeds`` and ``agent_delays`` are
        provided, ``agent_delays`` takes precedence.
    agent_lags : list[int] | None, optional
        *Legacy* alias for ``agent_delays``.
    agent_delays : list[int] | int | None
        Per-agent observation delay in steps.  An integer is broadcast to
        all agents.
    speed_factors : list[float] | float | None
        Per-agent speed multiplier (action-repeat count).  ``1.0`` means
        no speed change.  An float is broadcast to all agents.
    num_agents : int | None
        Explicit agent count.  Inferred from ``agent_delays`` length when
        not given.

    Attributes
    ----------
    step_count : int
        Number of wrapper-level steps since last reset.
    obs_buffers : list[deque]
        Per-agent observation ring buffers.
    """

    def __init__(
        self,
        env: gym.Env,
        agent_speeds: Optional[List[int]] = None,
        agent_lags: Optional[List[int]] = None,
        *,
        agent_delays: Optional[Union[List[int], int]] = None,
        speed_factors: Optional[Union[List[float], float]] = None,
        num_agents: Optional[int] = None,
    ):
        super().__init__(env)

        # ── resolve delays (new param > legacy lags > legacy speeds > default)
        if agent_delays is not None:
            raw_delays = agent_delays
        elif agent_lags is not None:
            raw_delays = agent_lags
        elif agent_speeds is not None:
            # Original API used agent_speeds as tick-rate integers.
            # Re-interpret as delays of 0 with the speeds as speed_factors
            # to maintain back-compat with original constructor.
            raw_delays = [0] * len(agent_speeds)
            if speed_factors is None:
                speed_factors = [float(s) for s in agent_speeds]
        else:
            raw_delays = [0]

        # Normalize to list
        if isinstance(raw_delays, int):
            n = num_agents or 1
            raw_delays = [raw_delays] * n
        delays: List[int] = [max(0, int(d)) for d in raw_delays]

        # Agent count
        self.num_agents: int = num_agents or len(delays)
        if len(delays) < self.num_agents:
            delays = delays + [delays[-1]] * (self.num_agents - len(delays))

        # ── resolve speed factors
        if speed_factors is None:
            speeds: List[float] = [1.0] * self.num_agents
        elif isinstance(speed_factors, (int, float)):
            speeds = [float(speed_factors)] * self.num_agents
        else:
            speeds = [float(s) for s in speed_factors]
        if len(speeds) < self.num_agents:
            speeds = speeds + [speeds[-1]] * (self.num_agents - len(speeds))

        self.agent_delays: List[int] = delays
        self.agent_speeds: List[float] = speeds
        self.agent_lags: List[int] = delays  # back-compat alias

        # Is this effectively single-agent?
        self._single_agent: bool = self.num_agents == 1

        # State
        self.obs_buffers: List[deque] = []
        self.step_count: int = 0
        self._stale_counts: List[int] = [0] * self.num_agents

    # ── reset ────────────────────────────────────────────────────────────────

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0
        self._stale_counts = [0] * self.num_agents

        if self._single_agent:
            delay = self.agent_delays[0]
            buf: deque = deque(maxlen=delay + 1)
            for _ in range(delay + 1):
                buf.append(obs)
            self.obs_buffers = [buf]
            info["desync_delays"] = self.agent_delays
            return obs, info

        # Multi-agent: obs should be indexable per agent
        self.obs_buffers = []
        for i in range(self.num_agents):
            delay = self.agent_delays[i]
            buf = deque(maxlen=delay + 1)
            agent_obs = self._index_obs(obs, i)
            for _ in range(delay + 1):
                buf.append(agent_obs)
            self.obs_buffers.append(buf)

        info["desync_delays"] = self.agent_delays
        info["desync_speeds"] = self.agent_speeds
        return obs, info

    # ── step ─────────────────────────────────────────────────────────────────

    def step(self, action):
        self.step_count += 1

        if self._single_agent:
            return self._step_single(action)
        return self._step_multi(action)

    def _step_single(self, action):
        """Single-agent path: apply speed factor then delay observation."""
        speed = self.agent_speeds[0]
        n_repeats = max(1, int(round(speed)))

        total_reward = 0.0
        terminated = False
        truncated = False
        info: dict = {}

        for _ in range(n_repeats):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        # Buffer and delay
        self.obs_buffers[0].append(obs)
        delayed_obs = self.obs_buffers[0][0]

        staleness = self.agent_delays[0]
        self._stale_counts[0] = staleness
        info["desync_delays"] = self.agent_delays
        info["desync_staleness"] = staleness
        info["desync_speed"] = speed
        return delayed_obs, total_reward, terminated, truncated, info

    def _step_multi(self, actions):
        """Multi-agent path: per-agent speed factors and delays."""
        obs, rewards, terminated, truncated, info = self.env.step(actions)

        # Update observation buffers with the fresh obs
        delayed_obs_list = []
        for i in range(self.num_agents):
            agent_obs = self._index_obs(obs, i)
            self.obs_buffers[i].append(agent_obs)

            # Speed-based staleness: if an agent runs slower, it only
            # updates its "perceived" observation every N steps.
            speed = self.agent_speeds[i]
            update_interval = max(1, int(round(1.0 / speed))) if speed > 0 else 1

            if self.step_count % update_interval == 0:
                # Agent gets its (delayed) observation
                delayed_obs_list.append(self.obs_buffers[i][0])
                self._stale_counts[i] = self.agent_delays[i]
            else:
                # Agent re-uses its last observation (stale)
                delayed_obs_list.append(self.obs_buffers[i][0])
                self._stale_counts[i] += 1

        info["desync_delays"] = self.agent_delays
        info["desync_staleness"] = list(self._stale_counts)
        info["desync_speeds"] = self.agent_speeds
        return delayed_obs_list, rewards, terminated, truncated, info

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _index_obs(obs: Any, i: int) -> Any:
        """Extract agent *i*'s observation from a composite observation."""
        if isinstance(obs, dict):
            # Try agent_0, agent0, 0, str(i) keys
            for key in [f"agent_{i}", f"agent{i}", i, str(i)]:
                if key in obs:
                    return obs[key]
            # Fallback: return full obs (single-agent in dict form)
            return obs
        elif isinstance(obs, (list, tuple)):
            if i < len(obs):
                return obs[i]
            return obs[-1]
        elif isinstance(obs, np.ndarray) and obs.ndim >= 2:
            if i < obs.shape[0]:
                return obs[i]
            return obs[-1]
        # Scalar or single-agent obs
        return obs


# ═══════════════════════════════════════════════════════════════════════════════
# 2. StochasticDesyncWrapper  (random delays)
# ═══════════════════════════════════════════════════════════════════════════════


class StochasticDesyncWrapper(gym.Wrapper):
    """Random observation delays drawn from a distribution each step.

    At every step a fresh delay *d* is sampled and the agent receives the
    observation from *d* steps ago.  A ring buffer of size ``max_delay + 1``
    stores recent observations so that any sampled delay up to ``max_delay``
    can be served.

    Three delay distributions are supported:

    * ``"poisson"``  -- ``d ~ Poisson(mean_delay)``, clamped to
      ``[0, max_delay]``.
    * ``"geometric"`` -- ``d ~ Geometric(p=1/(mean_delay+1))``, clamped to
      ``[0, max_delay]``.  Models memoryless packet-loss / retry latency.
    * ``"uniform"``  -- ``d ~ Uniform{0, 1, ..., max_delay}``.

    Parameters
    ----------
    env : gym.Env
        Base (single-agent) environment.
    distribution : str
        One of ``"poisson"``, ``"geometric"``, ``"uniform"``.
    mean_delay : float
        Mean of the delay distribution (used by Poisson and geometric).
    max_delay : int
        Hard upper bound on delay.  Also sets the buffer size.
    seed : int | None
        RNG seed.

    Attributes
    ----------
    delay_history : list[int]
        Sequence of delays sampled during the current episode.
    """

    _VALID_DISTS = {"poisson", "geometric", "uniform"}

    def __init__(
        self,
        env: gym.Env,
        *,
        distribution: str = "poisson",
        mean_delay: float = 2.0,
        max_delay: int = 10,
        seed: Optional[int] = None,
    ):
        super().__init__(env)
        if distribution not in self._VALID_DISTS:
            raise ValueError(f"Unknown distribution {distribution!r}; expected one of {self._VALID_DISTS}")
        self.distribution = distribution
        self.mean_delay = max(0.0, float(mean_delay))
        self.max_delay = max(0, int(max_delay))
        self._rng = np.random.RandomState(seed)

        self._obs_buffer: deque = deque(maxlen=self.max_delay + 1)
        self.delay_history: List[int] = []

    # ── reset ────────────────────────────────────────────────────────────────

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer.clear()
        for _ in range(self.max_delay + 1):
            self._obs_buffer.append(obs)
        self.delay_history.clear()
        info["stochastic_desync_distribution"] = self.distribution
        info["stochastic_desync_mean_delay"] = self.mean_delay
        return obs, info

    # ── step ─────────────────────────────────────────────────────────────────

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._obs_buffer.append(obs)

        delay = self._sample_delay()
        self.delay_history.append(delay)

        # Buffer is ordered oldest-first; index 0 is the oldest.
        # To get the obs from *delay* steps ago we index from the end.
        buf_len = len(self._obs_buffer)
        idx = max(0, buf_len - 1 - delay)
        delayed_obs = self._obs_buffer[idx]

        info["stochastic_desync_delay"] = delay
        info["stochastic_desync_mean_realized"] = float(np.mean(self.delay_history))
        info["stochastic_desync_max_realized"] = int(np.max(self.delay_history))
        return delayed_obs, reward, terminated, truncated, info

    # ── sampling ─────────────────────────────────────────────────────────────

    def _sample_delay(self) -> int:
        """Draw a single delay from the configured distribution."""
        if self.distribution == "poisson":
            d = self._rng.poisson(lam=self.mean_delay)
        elif self.distribution == "geometric":
            # Geometric with mean = mean_delay  =>  p = 1/(mean_delay + 1)
            p = 1.0 / (self.mean_delay + 1.0) if self.mean_delay > 0 else 1.0
            d = self._rng.geometric(p=p) - 1  # shift to 0-based
        else:  # uniform
            d = self._rng.randint(0, self.max_delay + 1)

        return int(np.clip(d, 0, self.max_delay))

    # ── statistics ───────────────────────────────────────────────────────────

    @property
    def delay_stats(self) -> Dict[str, float]:
        """Summary statistics of delays observed in the current episode."""
        if not self.delay_history:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        arr = np.array(self.delay_history, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "count": len(self.delay_history),
        }
