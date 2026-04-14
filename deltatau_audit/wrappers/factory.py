"""Environment Wrapping Factory.

Centralizes the creation of perturbed environments for robustness auditing.
Supports nominal, speed-shift, jitter, delay, and adversarial scenarios.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Union

import gymnasium as gym

from .adversarial import AdversarialSpeedWrapper
from .latency import ObservationDelayWrapper, ObsNoiseWrapper
from .speed import FixedSpeedWrapper, JitterWrapper, PiecewiseSwitchWrapper

logger = logging.getLogger("deltatau-audit")


def create_wrapped_env(
    env_id_or_factory: Union[str, Callable[[], gym.Env]],
    scenario: str,
    adapter: Optional[Any] = None,
    **kwargs: Any,
) -> gym.Env:
    """Creates a wrapped environment for a specific robustness scenario.

    Args:
        env_id_or_factory: Either a gymnasium env ID or a factory function.
        scenario: The name of the scenario (e.g., 'nominal', 'speed_5x', 'jitter').
        adapter: Optional agent adapter for adversarial scenarios.
        **kwargs: Additional overrides for specific wrappers.

    Returns:
        A wrapped (or nominal) gymnasium environment.
    """
    if isinstance(env_id_or_factory, str):
        env = gym.make(env_id_or_factory)
    else:
        env = env_id_or_factory()

    if scenario == "nominal":
        return env

    # Scenario Mapping
    if scenario == "speed_5x":
        return FixedSpeedWrapper(env, speed=kwargs.get("speed", 5))

    if scenario == "jitter":
        return JitterWrapper(
            env,
            base_speed=kwargs.get("base_speed", 2),
            jitter=kwargs.get("jitter", 1),
        )

    if scenario == "delay":
        return ObservationDelayWrapper(env, delay=kwargs.get("delay", 1))

    if scenario == "spike":
        # Default piecewise schedule: 1x -> 5x (at step 20) -> 1x (at step 40)
        schedule = kwargs.get("schedule", [(0, 1), (20, 5), (40, 1)])
        return PiecewiseSwitchWrapper(env, schedule=schedule)

    if scenario == "obs_noise":
        return ObsNoiseWrapper(env, std=kwargs.get("std", 0.1))

    if scenario == "adversarial_jitter":
        return AdversarialSpeedWrapper(
            env,
            agent_adapter=adapter,
            possible_speeds=kwargs.get("possible_speeds", [1, 2, 3, 5, 8]),
        )

    # Allow for extensible scenario strings like "speed_2x", "speed_8x"
    if scenario.startswith("speed_") and scenario.endswith("x"):
        try:
            val = float(scenario[6:-1])
            return FixedSpeedWrapper(env, speed=val)
        except ValueError:
            pass

    logger.warning(f"Unknown robustness scenario: {scenario}. Returning nominal environment.")
    return env
