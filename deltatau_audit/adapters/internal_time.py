"""Adapter for the Internal Time agent from this project."""

from typing import Any, Optional, Tuple

import torch

from .base import AgentAdapter


class InternalTimeAdapter(AgentAdapter):
    """Adapter for InternalTimeAgent / SkipRNNAgent / baseline GRU.

    Works with any agent that has:
      - get_initial_hidden(batch, device) -> hidden
      - get_action_and_value(obs, hidden) -> (action, log_prob, entropy, value, hidden, dt)
      - encoder(obs) -> encoded
      - rnn(encoded, hidden, dt) -> hidden_new  (for internal_time)
      - value_head(hidden) -> value  (for intervention value recompute)
    """

    def __init__(self, agent: torch.nn.Module, device: str = "cpu",
                 agent_type: str = "internal_time"):
        self.agent = agent
        self.device = device
        self.agent_type = agent_type
        self.agent.eval()

    def reset_hidden(self, batch: int = 1,
                     device: str = "cpu") -> Any:
        return self.agent.get_initial_hidden(batch, device or self.device)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, hidden: Any
            ) -> Tuple[int, float, Any, Optional[float]]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)

        action, _, _, value, hidden_new, dt = \
            self.agent.get_action_and_value(obs, hidden)

        return (
            action.item(),
            value.item(),
            hidden_new,
            dt.item() if dt is not None else None,
        )

    @torch.no_grad()
    def rerun_with_dt(self, obs: torch.Tensor, hidden: Any,
                      target_dt: float) -> Any:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)

        encoded = self.agent.encoder(obs)
        dt_tensor = torch.tensor(
            [[target_dt]], dtype=torch.float32, device=self.device
        )
        return self.agent.rnn(encoded, hidden, dt_tensor)

    @torch.no_grad()
    def recompute_value(self, hidden: Any) -> float:
        return self.agent.value_head(hidden).squeeze(-1).item()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, obs_dim: int, act_dim: int,
                        agent_type: str = "internal_time",
                        device: str = "cpu") -> "InternalTimeAdapter":
        """Load from a saved checkpoint file."""
        from internal_time_rl.models.baselines import SkipRNNAgent
        from internal_time_rl.models.policy import InternalTimeAgent

        ckpt = torch.load(checkpoint_path, map_location=device,
                          weights_only=False)

        if agent_type == "baseline":
            agent = InternalTimeAgent(obs_dim, act_dim, use_internal_time=False)
        elif agent_type in ("internal_time", "internal_time_discount"):
            agent = InternalTimeAgent(obs_dim, act_dim, use_internal_time=True)
        elif agent_type == "skip_rnn":
            agent = SkipRNNAgent(obs_dim, act_dim)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        agent.load_state_dict(ckpt["agent"])
        agent.to(device)
        return cls(agent, device=device, agent_type=agent_type)
