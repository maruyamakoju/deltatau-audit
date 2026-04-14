"""Adapter for SB3 models trained on dm_control environments.

Wraps SB3 models (PPO, SAC, TD3, A2C) trained on dm_control tasks
accessed via shimmy. Handles the dm_control observation format transparently.

dm_control environments return OrderedDict observations; shimmy wraps them
into a flattened Box space automatically. This adapter works with that
flattened observation space.

Supported task IDs (shimmy convention):
    dm_control/walker-walk-v0
    dm_control/cheetah-run-v0
    dm_control/reacher-easy-v0
    dm_control/humanoid-stand-v0

Requirements:
    pip install "deltatau-audit[dm_control]"
    # which installs: shimmy[dm-control], dm-control, stable-baselines3

Usage:
    from deltatau_audit.adapters.dm_control import DMControlSB3Adapter

    adapter = DMControlSB3Adapter.from_path(
        model_path="walker_walk.zip",
        env_id="dm_control/walker-walk-v0",
    )

    from deltatau_audit.auditor import run_full_audit
    result = run_full_audit(
        adapter,
        env_factory=lambda: make_dm_control_env("dm_control/walker-walk-v0"),
        speeds=[1, 2, 3, 5, 8],
        n_episodes=20,
    )
"""

from typing import Any, Optional, Tuple, Union

import numpy as np
import torch

from .base import AgentAdapter

# ---------------------------------------------------------------------------
# Known dm_control task identifiers (shimmy convention)
# ---------------------------------------------------------------------------

DM_CONTROL_ENVS = {
    # Short aliases -> canonical shimmy env ID
    "walker-walk": "dm_control/walker-walk-v0",
    "cheetah-run": "dm_control/cheetah-run-v0",
    "reacher-easy": "dm_control/reacher-easy-v0",
    "humanoid-stand": "dm_control/humanoid-stand-v0",
    # Canonical IDs pass through unchanged
    "dm_control/walker-walk-v0": "dm_control/walker-walk-v0",
    "dm_control/cheetah-run-v0": "dm_control/cheetah-run-v0",
    "dm_control/reacher-easy-v0": "dm_control/reacher-easy-v0",
    "dm_control/humanoid-stand-v0": "dm_control/humanoid-stand-v0",
    # Accept without trailing version for convenience
    "dm_control/walker-walk": "dm_control/walker-walk-v0",
    "dm_control/cheetah-run": "dm_control/cheetah-run-v0",
    "dm_control/reacher-easy": "dm_control/reacher-easy-v0",
    "dm_control/humanoid-stand": "dm_control/humanoid-stand-v0",
}


def _resolve_env_id(env_id: str) -> str:
    """Resolve short or alternate env_id forms to the canonical shimmy ID."""
    resolved = DM_CONTROL_ENVS.get(env_id)
    if resolved is not None:
        return resolved
    # Pass unknown IDs through -- user may be using a newer shimmy version
    return env_id


def _check_dependencies() -> None:
    """Raise a clear ImportError if shimmy or dm_control are not installed."""
    missing = []
    try:
        import shimmy  # noqa: F401
    except ImportError:
        missing.append("shimmy[dm-control]")
    try:
        import dm_control  # noqa: F401
    except ImportError:
        missing.append("dm-control")

    if missing:
        parts = ", ".join(missing)
        raise ImportError(
            "dm_control adapter requires: " + parts + ". Install with: pip install deltatau-audit[dm_control]"
        )


def make_dm_control_env(
    env_id: str,
    speed: float = 1.0,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
) -> Any:
    """Create a gymnasium-compatible dm_control environment via shimmy.

    shimmy registers dm_control tasks under the dm_control/ namespace.
    The resulting env has a flattened Box observation space -- the
    OrderedDict observations that dm_control produces natively are
    concatenated into a single float32 vector by shimmy automatically.

    The existing deltatau-audit wrappers (JitterWrapper, FixedSpeedWrapper,
    ObservationDelayWrapper, etc.) operate at the gymnasium level and
    therefore work with shimmy-wrapped dm_control environments out of the
    box.

    Args:
        env_id:      Shimmy/gymnasium env ID. Accepts short aliases like
                     "walker-walk" or canonical IDs like
                     "dm_control/walker-walk-v0".
        speed:       Playback speed multiplier applied via FixedSpeedWrapper.
                     speed=1.0 means no wrapping (native timing).
        seed:        Optional RNG seed passed to env.reset().
        render_mode: Passed to gym.make() if provided (e.g. "rgb_array").

    Returns:
        A gymnasium.Env instance with flattened Box observation space.

    Raises:
        ImportError: If shimmy or dm_control are not installed.
    """
    _check_dependencies()

    import gymnasium as gym
    import shimmy  # noqa: F401 -- needed for gymnasium registration side-effect

    canonical_id = _resolve_env_id(env_id)

    make_kwargs: dict = {}
    if render_mode is not None:
        make_kwargs["render_mode"] = render_mode

    env = gym.make(canonical_id, **make_kwargs)

    # Seed the environment if requested
    if seed is not None:
        env.reset(seed=seed)

    # Wrap with speed controller when speed != 1.0
    if speed != 1.0:
        from deltatau_audit.wrappers.speed import FixedSpeedWrapper

        env = FixedSpeedWrapper(env, speed=speed)

    return env


class DMControlSB3Adapter(AgentAdapter):
    """Adapter for SB3 models trained on dm_control environments via shimmy.

    Works with PPO, SAC, TD3, A2C trained on any dm_control task.
    Observations are expected to be flattened float32 vectors (shimmy
    handles the OrderedDict -> vector conversion automatically).

    This adapter does NOT support intervention (Reliance = N/A) -- same
    behaviour as SB3Adapter.

    Attributes:
        supports_intervention: Always False (no internal delta-tau module).

    Example::

        adapter = DMControlSB3Adapter.from_path(
            "walker_walk_ppo.zip", env_id="dm_control/walker-walk-v0"
        )
        result = run_full_audit(
            adapter,
            env_factory=lambda: make_dm_control_env("dm_control/walker-walk-v0"),
        )
    """

    supports_intervention: bool = False

    def __init__(
        self,
        model: Any,
        env_id: str,
        speed: float = 1.0,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            model:   A loaded SB3 model instance (.predict() + .policy).
            env_id:  Gymnasium env ID for the dm_control task.
            speed:   Default speed multiplier used by make_env().
            seed:    Default RNG seed used by make_env().
            device:  Torch device string ("cpu" or "cuda").
        """
        self.model = model
        self.env_id = _resolve_env_id(env_id)
        self.speed = speed
        self.seed = seed
        self.device = device

    # ------------------------------------------------------------------
    # AgentAdapter interface
    # ------------------------------------------------------------------

    def reset_hidden(self, batch: int = 1, device: str = "cpu") -> None:
        """Return None -- SB3 MLP/CNN policies have no recurrent hidden state."""
        return None

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        hidden: Any,
    ) -> Tuple[Union[int, np.ndarray], float, Any, Optional[float]]:
        """Single-step forward pass.

        Args:
            obs:    Observation tensor, shape (obs_dim,) or (1, obs_dim).
                    obs_dim is the flattened dm_control observation size
                    as returned by shimmy (e.g. 24 for walker-walk).
            hidden: Ignored (always None for non-recurrent policies).

        Returns:
            action:     Selected action. For dm_control tasks this is
                        always a float32 ndarray matching the action space.
            value:      Scalar value estimate V(s).
            hidden_new: None (no hidden state).
            dt:         None (no internal time module).
        """
        # Reshape to (1, obs_dim) for SB3 batch dimension
        if obs.dim() == 1:
            obs_np = obs.cpu().numpy().reshape(1, -1)
        else:
            obs_np = obs.cpu().numpy()

        # SB3 predict returns (action_array, state)
        action_arr, _ = self.model.predict(obs_np, deterministic=False)

        # Value estimate -- policy.predict_values expects a tensor
        obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32, device=self.model.device)
        value_tensor = self.model.policy.predict_values(obs_tensor)
        value_scalar = float(value_tensor.item())

        # Return action in env-compatible form
        if hasattr(self.model.action_space, "n"):
            # Discrete action space (unusual for dm_control but handled)
            if hasattr(action_arr, "__len__"):
                action_out: Any = int(action_arr[0])
            else:
                action_out = int(action_arr)
        else:
            # Continuous action space -- strip batch dimension
            action_out = action_arr[0] if action_arr.ndim > 1 else action_arr

        return (action_out, value_scalar, None, None)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def make_env(
        self,
        speed: Optional[float] = None,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> Any:
        """Create a shimmy-wrapped dm_control env matching this adapter's task.

        Args:
            speed:       Speed multiplier (defaults to self.speed).
            seed:        RNG seed (defaults to self.seed).
            render_mode: Optional render mode string.

        Returns:
            gymnasium.Env instance ready for use with the auditor.
        """
        return make_dm_control_env(
            self.env_id,
            speed=speed if speed is not None else self.speed,
            seed=seed if seed is not None else self.seed,
            render_mode=render_mode,
        )

    @classmethod
    def from_path(
        cls,
        model_path: str,
        env_id: str,
        algo: str = "ppo",
        speed: float = 1.0,
        seed: int = 42,
        device: str = "cpu",
    ) -> "DMControlSB3Adapter":
        """Load an SB3 model from a .zip file and create an adapter.

        Args:
            model_path: Path to the saved SB3 model (.zip).
            env_id:     Gymnasium env ID for the dm_control task.
            algo:       SB3 algorithm ("ppo", "sac", "td3", "a2c").
            speed:      Default speed multiplier for make_env().
            seed:       Default RNG seed for make_env().
            device:     Torch device string.

        Returns:
            DMControlSB3Adapter with model loaded from disk.

        Raises:
            ImportError:       If stable-baselines3 is not installed.
            ValueError:        If algo is not recognised.
            FileNotFoundError: If model_path does not exist.
        """
        _check_dependencies()

        try:
            import stable_baselines3 as sb3
        except ImportError:
            raise ImportError("stable-baselines3 is required. Install with: pip install deltatau-audit[dm_control]")

        algo_map = {
            "ppo": sb3.PPO,
            "sac": sb3.SAC,
            "td3": sb3.TD3,
            "a2c": sb3.A2C,
        }
        algo_cls = algo_map.get(algo.lower())
        if algo_cls is None:
            keys = list(algo_map.keys())
            raise ValueError(f"Unknown algo {algo!r}. Supported: {keys}")

        model = algo_cls.load(model_path, device=device)
        return cls(model, env_id=env_id, speed=speed, seed=seed, device=device)

    def __repr__(self) -> str:
        if self.model is not None:
            algo_name = type(self.model).__name__
        else:
            algo_name = "None"
        env_id_r = repr(self.env_id)
        return (
            f"DMControlSB3Adapter(env_id={env_id_r}, algo={algo_name}, speed={self.speed}, supports_intervention=False)"
        )
