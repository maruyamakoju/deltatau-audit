"""Adapter for standard (non-recurrent) Stable-Baselines3 models.

Wraps PPO, SAC, TD3, A2C, etc. for robustness auditing.
Supports both discrete and continuous action spaces.
Intervention is not supported (Reliance = N/A).

Requires: stable-baselines3 >= 2.0

Usage:
    from stable_baselines3 import PPO
    from deltatau_audit.adapters.sb3 import SB3Adapter

    model = PPO.load("my_model.zip")
    adapter = SB3Adapter(model)

    from deltatau_audit.auditor import run_full_audit
    result = run_full_audit(adapter, env_factory, ...)
"""

from typing import Any, Optional, Tuple, Union

import numpy as np
import torch

from .base import AgentAdapter


class SB3Adapter(AgentAdapter):
    """Adapter for standard (non-recurrent) SB3 models.

    Works with PPO, SAC, TD3, A2C, and any SB3 model that has
    a .predict() method and a .policy.predict_values() method.

    This adapter does NOT support intervention (Reliance = N/A).
    """

    def __init__(self, model, device: str = "cpu"):
        """
        Args:
            model: An SB3 model instance (already loaded).
            device: Device string (default: "cpu").
        """
        self.model = model
        self.device = device

    def reset_hidden(self, batch: int = 1,
                     device: str = "cpu") -> Any:
        return None  # No hidden state for non-recurrent models

    @torch.no_grad()
    def act(self, obs: torch.Tensor, hidden: Any
            ) -> Tuple[Union[int, np.ndarray], float, Any, Optional[float]]:
        # Convert obs to numpy for SB3
        if obs.dim() == 1:
            obs_np = obs.cpu().numpy().reshape(1, -1)
        else:
            obs_np = obs.cpu().numpy()

        # Get action
        action, _ = self.model.predict(
            obs_np,
            deterministic=False,
        )

        # Get value estimate
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32,
                                device=self.model.device)
        value = self.model.policy.predict_values(obs_t)
        value_scalar = value.item()

        # Return action in env-compatible form
        if hasattr(self.model.action_space, 'n'):
            # Discrete action space
            action_out = int(action[0]) if hasattr(action, '__len__') else int(action)
        else:
            # Continuous action space — return array
            action_out = action[0] if action.ndim > 1 else action

        return (action_out, value_scalar, None, None)

    @classmethod
    def from_path(cls, path: str, algo: str = "ppo",
                  device: str = "cpu") -> "SB3Adapter":
        """Load an SB3 model from a .zip file.

        Args:
            path: Path to the saved model (.zip)
            algo: Algorithm name ("ppo", "sac", "td3", "a2c")
            device: Device string

        Returns:
            SB3Adapter instance
        """
        try:
            import stable_baselines3
        except ImportError:
            raise ImportError(
                "stable-baselines3 is required for SB3 adapter. "
                "Install with: pip install stable-baselines3"
            )

        algo_map = {
            "ppo": stable_baselines3.PPO,
            "sac": stable_baselines3.SAC,
            "td3": stable_baselines3.TD3,
            "a2c": stable_baselines3.A2C,
        }
        algo_cls = algo_map.get(algo.lower())
        if algo_cls is None:
            raise ValueError(
                f"Unknown algo '{algo}'. Supported: {list(algo_map.keys())}"
            )

        model = algo_cls.load(path, device=device)
        return cls(model, device=device)

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        algo: str = "ppo",
        filename: Optional[str] = None,
        token: Optional[str] = None,
        device: str = "cpu",
    ) -> "SB3Adapter":
        """Download an SB3 model from HuggingFace Hub and return an adapter.

        Args:
            repo_id: HuggingFace repo ID (e.g. "sb3/ppo-CartPole-v1").
            algo:    Algorithm name ("ppo", "sac", "td3", "a2c").
            filename: Filename inside the repo. Auto-detected if None.
            token:   HuggingFace token for private repos.
            device:  Device string.

        Returns:
            SB3Adapter instance with model loaded from Hub.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for Hub downloads. "
                'Install with: pip install "deltatau-audit[hf]"'
            )

        # Auto-detect filename: try {algo}-{repo_name}.zip, then model.zip
        if filename is None:
            repo_name = repo_id.split("/")[-1]
            candidates = [f"{repo_name}.zip", "model.zip"]
        else:
            candidates = [filename]

        local_path = None
        last_err = None
        for fname in candidates:
            try:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=fname,
                    token=token,
                )
                break
            except Exception as e:
                last_err = e

        if local_path is None:
            raise FileNotFoundError(
                f"Could not find model in '{repo_id}'. "
                f"Tried: {candidates}. Last error: {last_err}\n"
                "Tip: Use --filename to specify the exact filename."
            )

        return cls.from_path(local_path, algo=algo, device=device)
