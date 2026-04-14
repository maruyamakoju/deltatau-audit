"""Adapter for Causal Resolution Agent (Axis 10).

Bridges the Level 5 internal_time_rl agent to the deltatau_audit engine,
reporting both subjective time (dt) and causal engagement status.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from deltatau_audit.protocols import AgentAdapter
from deltatau_audit.schema import TemporalCapability
from internal_time_rl.models.causal_reasoning import CausalResolutionAgent


class CausalResolutionAdapter(AgentAdapter):
    """Adapter for the Axis 10 Integrated Agent."""

    def __init__(self, agent: CausalResolutionAgent, device: str = "cpu"):
        self.agent = agent
        self.device = torch.device(device)
        self.hidden: Optional[torch.Tensor] = None
        self.supports_intervention = True
        self.supports_value_recompute = True

    def get_capabilities(self) -> TemporalCapability:
        return TemporalCapability(
            can_ponder=True,
            max_lookahead_steps=self.agent.causal_depth, # Reports its System 2 lookahead
            supports_variable_dt=True,
            has_internal_clock=True,
        )

    def reset_internal_state(self) -> None:
        self.hidden = self.agent.get_initial_hidden(1, self.device)

    def act(
        self,
        observation: Any,
        deterministic: bool = True,
        ponder_steps: Optional[int] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        if self.hidden is None:
            self.reset_internal_state()

        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            dist, value, hidden_new, dt, diag = self.agent.forward(
                obs_t, self.hidden, ponder_override=ponder_steps
            )
            
            if deterministic:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()

        self.hidden = hidden_new

        # Export rich diagnostics to the audit report
        info = {
            "value": value.item(),
            "dt": dt.item(),
            "reasoning_trace": {
                "expected_steps": diag["expected_steps"].item(),
                "causal_engaged": diag.get("causal_engaged", False),
            }
        }
        
        return int(action.squeeze(0).cpu().numpy()), info

    def rerun_with_dt(self, observation: Any, target_dt: float) -> Dict[str, Any]:
        """Manual override of delta_tau for intervention audit."""
        if self.hidden is None:
            self.reset_internal_state()

        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        dt_t = torch.tensor([[target_dt]], device=self.device)

        with torch.no_grad():
            dist, value, _, _, diag = self.agent.forward(
                obs_t, self.hidden, dt_override=dt_t
            )
            
        return {
            "value": value.item(),
            "dt": target_dt,
            "reasoning_trace": {
                "expected_steps": diag["expected_steps"].item(),
                "causal_engaged": diag.get("causal_engaged", False),
            }
        }

    def recompute_value(self, info: Dict[str, Any]) -> float:
        return info.get("value", 0.0)
