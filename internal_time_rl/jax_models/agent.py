"""
JAX Agent and Auditor Adapter.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple, Any, Optional
import numpy as np

from .cells import TimeModule, TimeAwareGRUCell
from deltatau_audit.adapters.base import AgentAdapter

class JaxInternalTimeAgent(eqx.Module):
    time_module: TimeModule
    rnn: TimeAwareGRUCell
    policy_head: eqx.nn.Linear
    value_head: eqx.nn.Linear
    hidden_dim: int
    obs_dim: int
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int, key: jax.random.PRNGKey):
        t_key, r_key, p_key, v_key = jax.random.split(key, 4)
        self.time_module = TimeModule(hidden_dim, obs_dim, t_key)
        self.rnn = TimeAwareGRUCell(obs_dim, hidden_dim, r_key)
        self.policy_head = eqx.nn.Linear(hidden_dim, act_dim, key=p_key)
        self.value_head = eqx.nn.Linear(hidden_dim, 1, key=v_key)
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim

    def __call__(self, h: jnp.ndarray, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Unbatched call: h is (H,), x is (O,)."""
        dt = self.time_module(h, x)
        h_new = self.rnn(h, x, dt)
        logits = self.policy_head(h_new)
        value = self.value_head(h_new)
        return logits, value, h_new, dt

class JaxAdapter(AgentAdapter):
    """Bridges JAX/Equinox agents to the auditor."""
    def __init__(self, agent: JaxInternalTimeAgent):
        self.agent = agent
        # Vmap the agent call for batched execution
        self._batched_agent = jax.vmap(agent)
        self._batched_rnn = jax.vmap(agent.rnn)
        self._batched_value = jax.vmap(agent.value_head)

    def reset_hidden(self, batch: int = 1, device: str = "cpu") -> Any:
        return jnp.zeros((batch, self.agent.hidden_dim))

    def act(self, obs: Any, hidden: Any) -> Tuple[int, float, Any, Optional[float]]:
        # obs: (batch, obs_dim) or (obs_dim,)
        if hasattr(obs, "numpy"):
            obs = obs.numpy()
        x = jnp.array(obs)
        if x.ndim == 1:
            x = x[None, :]
        
        # hidden: (batch, hidden_dim)
        logits, value, h_new, dt = self._batched_agent(hidden, x)
        
        # Auditor usually runs batch size 1 for single episodes
        action = int(jnp.argmax(logits[0]))
        val = float(value[0, 0])
        dt_val = float(dt[0, 0])
        
        return action, val, h_new, dt_val

    def rerun_with_dt(self, obs: Any, hidden: Any, target_dt: float) -> Any:
        if hasattr(obs, "numpy"):
            obs = obs.numpy()
        x = jnp.array(obs)
        if x.ndim == 1:
            x = x[None, :]
        
        batch_size = x.shape[0]
        dt_input = jnp.full((batch_size, 1), target_dt)
        
        h_new = self._batched_rnn(hidden, x, dt_input)
        return h_new

    def recompute_value(self, hidden: Any) -> float:
        value = self._batched_value(hidden)
        return float(value[0, 0])
