"""Generic adapter for any PyTorch actor-critic policy.

Works with any framework that produces actions and value estimates
from a PyTorch nn.Module:
- IsaacLab / RSL-RL
- IsaacGym / rl_games
- Custom actor-critic networks
- Any framework not covered by SB3 or CleanRL adapters

Usage (simple callable interface):
    from deltatau_audit.adapters.torch_policy import TorchPolicyAdapter

    def my_act(obs):
        action = actor(obs)
        value = critic(obs)
        return action, value

    adapter = TorchPolicyAdapter(my_act)
    result = run_full_audit(adapter, env_factory, ...)

Usage (nn.Module interface):
    adapter = TorchPolicyAdapter.from_actor_critic(actor, critic)

Usage (IsaacLab / RSL-RL):
    See examples/isaaclab_skeleton.py
"""

from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .base import AgentAdapter


class TorchPolicyAdapter(AgentAdapter):
    """Adapter for any PyTorch policy via a user-provided callable.

    The callable receives an observation tensor and returns
    (action, value) where:
        action: int, np.ndarray, or torch.Tensor
        value: float or scalar torch.Tensor

    Reliance = N/A (no internal Δτ).
    """

    def __init__(
        self,
        act_fn: Callable[[torch.Tensor], Tuple[Any, Any]],
        device: str = "cpu",
    ):
        """
        Args:
            act_fn: Callable(obs: Tensor) -> (action, value).
                    obs shape: (1, obs_dim). Both outputs can be
                    tensors or Python scalars/arrays.
            device: Device to move observations to.
        """
        self._act_fn = act_fn
        self.device = device

    def reset_hidden(self, batch: int = 1, device: str = "cpu") -> Any:
        return None  # Stateless (no hidden state)

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        hidden: Any,
    ) -> Tuple[Union[int, np.ndarray], float, Any, Optional[float]]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)

        action, value = self._act_fn(obs)

        # Normalize action
        if isinstance(action, torch.Tensor):
            action = action.cpu()
            if action.numel() == 1:
                action_out = int(action.item()) if action.dtype in (
                    torch.int32, torch.int64, torch.long
                ) else float(action.item())
            else:
                arr = action.numpy().flatten()
                action_out = arr[0] if len(arr) == 1 else arr
        elif isinstance(action, np.ndarray):
            action_out = int(action.item()) if action.size == 1 else action.flatten()
        else:
            action_out = action

        value_scalar = float(
            value.item() if isinstance(value, torch.Tensor) else float(value)
        )

        return action_out, value_scalar, None, None

    @classmethod
    def from_actor_critic(
        cls,
        actor: nn.Module,
        critic: nn.Module,
        is_discrete: bool = True,
        device: str = "cpu",
    ) -> "TorchPolicyAdapter":
        """Create adapter from separate actor and critic networks.

        Args:
            actor: Policy network. Output is action logits (discrete)
                   or action mean (continuous).
            critic: Value network. Output is a scalar value estimate.
            is_discrete: True for Categorical policy (argmax actions).
            device: Device string.

        Example:
            actor = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 2))
            critic = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 1))
            adapter = TorchPolicyAdapter.from_actor_critic(actor, critic)
        """
        actor = actor.to(device).eval()
        critic = critic.to(device).eval()

        @torch.no_grad()
        def _act_fn(obs):
            logits_or_mean = actor(obs)
            if is_discrete:
                action = logits_or_mean.argmax(dim=-1)
            else:
                action = logits_or_mean  # deterministic
            value = critic(obs)
            return action, value

        return cls(_act_fn, device=device)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        actor: nn.Module,
        critic: Optional[nn.Module] = None,
        is_discrete: bool = True,
        actor_key: str = "actor",
        critic_key: str = "critic",
        device: str = "cpu",
    ) -> "TorchPolicyAdapter":
        """Load actor/critic weights from a checkpoint file.

        Handles common checkpoint formats:
        - {"actor": state_dict, "critic": state_dict}  (RSL-RL style)
        - {"model_state_dict": state_dict}
        - Raw state_dict

        Args:
            checkpoint_path: Path to .pt or .pth checkpoint.
            actor: Actor network (architecture must match checkpoint).
            critic: Critic network (optional).
            is_discrete: True for discrete action spaces.
            actor_key: Key for actor state_dict in checkpoint.
            critic_key: Key for critic state_dict in checkpoint.
            device: Device string.

        Example (RSL-RL / IsaacLab):
            # checkpoint has {"model_state_dict": {"actor.weight": ..., "critic.weight": ...}}
            adapter = TorchPolicyAdapter.from_checkpoint(
                "model.pt", actor=actor_net, critic=critic_net
            )
        """
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

        def _try_load(module, state_dict):
            try:
                module.load_state_dict(state_dict, strict=True)
            except RuntimeError:
                # Try with strict=False (partial load)
                module.load_state_dict(state_dict, strict=False)

        if isinstance(ckpt, dict):
            if actor_key in ckpt and isinstance(ckpt[actor_key], dict):
                # Separate actor/critic keys
                _try_load(actor, ckpt[actor_key])
                if critic is not None and critic_key in ckpt:
                    _try_load(critic, ckpt[critic_key])
            elif "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
                # Try to split by prefix
                actor_sd = {
                    k.replace("actor.", "", 1): v
                    for k, v in state_dict.items()
                    if k.startswith("actor.")
                }
                critic_sd = {
                    k.replace("critic.", "", 1): v
                    for k, v in state_dict.items()
                    if k.startswith("critic.")
                }
                if actor_sd:
                    _try_load(actor, actor_sd)
                    if critic is not None and critic_sd:
                        _try_load(critic, critic_sd)
                else:
                    # No prefix — assume whole dict is for actor
                    _try_load(actor, state_dict)
            else:
                # Raw state_dict — load into actor
                _try_load(actor, ckpt)
        else:
            raise ValueError(
                f"Unexpected checkpoint format: {type(ckpt)}. "
                "Expected a dict."
            )

        if critic is not None:
            return cls.from_actor_critic(actor, critic, is_discrete, device)
        else:
            # Actor-only (no value estimate)
            actor = actor.to(device).eval()

            @torch.no_grad()
            def _act_fn_no_critic(obs):
                logits_or_mean = actor(obs)
                action = logits_or_mean.argmax(dim=-1) if is_discrete else logits_or_mean
                return action, torch.tensor(0.0)

            return cls(_act_fn_no_critic, device=device)
