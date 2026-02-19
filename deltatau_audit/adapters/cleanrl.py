"""Adapter for CleanRL-style agents.

CleanRL (https://github.com/vwxyzjn/cleanrl) trains minimal PyTorch agents
with no external framework dependency. Agents are plain nn.Module subclasses
saved as state_dicts.

Typical CleanRL MLP agent interface:
    agent.get_action_and_value(obs)
        -> (action, logprob, entropy, value)

Typical CleanRL LSTM agent interface:
    agent.get_action_and_value(obs, lstm_state, done)
        -> (action, logprob, entropy, value, lstm_state)

Usage (Python API):
    # Your Agent class (copy from your CleanRL training script)
    class Agent(nn.Module):
        def get_action_and_value(self, obs): ...

    agent = Agent(obs_dim=4, act_dim=2)
    agent.load_state_dict(torch.load("runs/CartPole/agent.pt"))

    from deltatau_audit.adapters.cleanrl import CleanRLAdapter
    adapter = CleanRLAdapter(agent)
    result = run_full_audit(adapter, env_factory, ...)

Usage (CLI):
    deltatau-audit audit-cleanrl \\
        --checkpoint runs/CartPole/agent.pt \\
        --agent-module agent.py \\
        --agent-class Agent \\
        --env CartPole-v1
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .base import AgentAdapter


class CleanRLAdapter(AgentAdapter):
    """Adapter for CleanRL-style PyTorch agents.

    Wraps any CleanRL agent that implements the standard interface:
        agent.get_action_and_value(obs)           # MLP
        agent.get_action_and_value(obs, state, done)  # LSTM

    Works with discrete (Categorical) and continuous (Normal) action spaces.
    Reliance = N/A (no internal Δτ).
    """

    def __init__(
        self,
        agent: nn.Module,
        lstm: bool = False,
        device: str = "cpu",
    ):
        """
        Args:
            agent: CleanRL Agent instance (already instantiated, weights loaded).
            lstm: True if agent uses LSTM (get_action_and_value takes lstm_state).
            device: Device string (default: "cpu").
        """
        self.agent = agent.to(device)
        self.agent.eval()
        self._lstm = lstm
        self.device = device

        # Auto-detect LSTM if requested
        if lstm:
            self._lstm_module = self._find_lstm_module()
        else:
            self._lstm_module = None

    def _find_lstm_module(self) -> Optional[nn.LSTM]:
        """Find LSTM module for hidden state initialization."""
        for module in self.agent.modules():
            if isinstance(module, nn.LSTM):
                return module
        return None

    def reset_hidden(self, batch: int = 1, device: str = "cpu") -> Any:
        if not self._lstm:
            return None
        if self._lstm_module is None:
            return None
        n_layers = self._lstm_module.num_layers
        hidden_dim = self._lstm_module.hidden_size
        h = torch.zeros(n_layers, batch, hidden_dim, device=device)
        c = torch.zeros(n_layers, batch, hidden_dim, device=device)
        return (h, c)

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        hidden: Any,
    ) -> Tuple[Union[int, np.ndarray], float, Any, Optional[float]]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)

        if self._lstm and hidden is not None:
            # LSTM CleanRL: get_action_and_value(obs, lstm_state, done)
            done = torch.zeros(obs.shape[0], device=self.device)
            result = self.agent.get_action_and_value(obs, hidden, done)
            # Returns: action, logprob, entropy, value, lstm_state
            action, _, _, value, new_hidden = result
        else:
            # MLP CleanRL: get_action_and_value(obs)
            result = self.agent.get_action_and_value(obs)
            # Returns: action, logprob, entropy, value
            action, _, _, value = result
            new_hidden = None

        # Normalize action output
        if isinstance(action, torch.Tensor):
            if action.numel() == 1:
                action_out = int(action.item())
            else:
                action_out = action.cpu().numpy().flatten()
                if len(action_out) == 1:
                    action_out = action_out[0]
        else:
            action_out = action

        value_scalar = float(
            value.item() if isinstance(value, torch.Tensor) else float(value)
        )

        return action_out, value_scalar, new_hidden, None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        agent_class: type,
        agent_kwargs: Optional[Dict] = None,
        lstm: bool = False,
        device: str = "cpu",
    ) -> "CleanRLAdapter":
        """Load a CleanRL agent from a checkpoint file.

        Args:
            checkpoint_path: Path to .pt or .pth checkpoint file.
            agent_class: The Agent class (from your training script).
            agent_kwargs: Kwargs to pass to Agent(**agent_kwargs).
            lstm: True if agent uses LSTM.
            device: Device string.

        Example:
            class Agent(nn.Module):
                def __init__(self, obs_dim, act_dim): ...
                def get_action_and_value(self, obs): ...

            adapter = CleanRLAdapter.from_checkpoint(
                "runs/CartPole-v1/agent.pt",
                agent_class=Agent,
                agent_kwargs={"obs_dim": 4, "act_dim": 2},
            )
        """
        agent_kwargs = agent_kwargs or {}
        agent = agent_class(**agent_kwargs)

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

        # Handle various checkpoint formats
        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            elif all(
                isinstance(v, torch.Tensor) for v in ckpt.values()
            ):
                state_dict = ckpt  # raw state_dict
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        agent.load_state_dict(state_dict)
        return cls(agent, lstm=lstm, device=device)

    @classmethod
    def from_module_path(
        cls,
        checkpoint_path: str,
        agent_module_path: str,
        agent_class_name: str = "Agent",
        agent_kwargs: Optional[Dict] = None,
        lstm: bool = False,
        device: str = "cpu",
    ) -> "CleanRLAdapter":
        """Load a CleanRL agent by importing the Agent class from a Python file.

        This is the CLI-friendly version — import the Agent class directly
        from your training script.

        Args:
            checkpoint_path: Path to .pt checkpoint file.
            agent_module_path: Path to Python file containing Agent class.
            agent_class_name: Name of the Agent class (default: "Agent").
            agent_kwargs: Kwargs to pass to Agent(**agent_kwargs).
            lstm: True if agent uses LSTM.
            device: Device string.

        Example:
            adapter = CleanRLAdapter.from_module_path(
                checkpoint_path="runs/CartPole/agent.pt",
                agent_module_path="ppo_cartpole.py",
                agent_class_name="Agent",
                agent_kwargs={"obs_dim": 4, "act_dim": 2},
            )
        """
        module_path = Path(agent_module_path).resolve()
        if not module_path.exists():
            raise FileNotFoundError(f"Agent module not found: {module_path}")

        spec = importlib.util.spec_from_file_location(
            "_cleanrl_agent_module", str(module_path)
        )
        assert spec is not None, f"Cannot create module spec from {module_path}"
        module = importlib.util.module_from_spec(spec)
        sys.modules["_cleanrl_agent_module"] = module
        assert spec.loader is not None, f"Module spec has no loader: {module_path}"
        spec.loader.exec_module(module)

        if not hasattr(module, agent_class_name):
            available = [
                name for name in dir(module)
                if not name.startswith("_")
                and isinstance(getattr(module, name), type)
                and issubclass(getattr(module, name), nn.Module)
            ]
            raise AttributeError(
                f"Class '{agent_class_name}' not found in {module_path}. "
                f"Available nn.Module subclasses: {available}"
            )

        agent_class = getattr(module, agent_class_name)
        return cls.from_checkpoint(
            checkpoint_path,
            agent_class=agent_class,
            agent_kwargs=agent_kwargs,
            lstm=lstm,
            device=device,
        )
