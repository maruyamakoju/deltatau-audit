"""
JAX/Equinox implementation of Internal Time Dynamics.

Provides high-performance, JIT-compilable versions of:
- TimeModule: Learned internal clock.
- TimeAwareGRU: Gated Recurrent Unit with temporal modulation.
- LiquidTimeCell: ODE-based state evolution.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Tuple

class TimeModule(eqx.Module):
    """Predicts delta_tau = g(h, x)."""
    net: eqx.nn.MLP
    dt_min: float = 0.3
    dt_max: float = 2.5

    def __init__(self, hidden_dim: int, obs_dim: int, key: jax.random.PRNGKey):
        self.net = eqx.nn.MLP(
            in_size=hidden_dim + obs_dim,
            out_size=1,
            width_size=32,
            depth=1,
            activation=jax.nn.tanh,
            key=key
        )

    def __call__(self, h: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        combined = jnp.concatenate([h, x], axis=-1)
        raw = self.net(combined)
        # Scale sigmoid to [dt_min, dt_max]
        delta_tau = self.dt_min + (self.dt_max - self.dt_min) * jax.nn.sigmoid(raw)
        return delta_tau

class TimeAwareGRUCell(eqx.Module):
    """GRU cell with temporal modulation: z_eff = 1 - (1-z)^dt."""
    weights_zx: eqx.nn.Linear
    weights_zh: eqx.nn.Linear
    weights_rx: eqx.nn.Linear
    weights_rh: eqx.nn.Linear
    weights_hx: eqx.nn.Linear
    weights_hh: eqx.nn.Linear
    hidden_size: int

    def __init__(self, input_size: int, hidden_size: int, key: jax.random.PRNGKey):
        keys = jax.random.split(key, 6)
        self.weights_zx = eqx.nn.Linear(input_size, hidden_size, key=keys[0])
        self.weights_zh = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=keys[1])
        self.weights_rx = eqx.nn.Linear(input_size, hidden_size, key=keys[2])
        self.weights_rh = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=keys[3])
        self.weights_hx = eqx.nn.Linear(input_size, hidden_size, key=keys[4])
        self.weights_hh = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=keys[5])
        self.hidden_size = hidden_size

    def __call__(self, h: jnp.ndarray, x: jnp.ndarray, dt: jnp.ndarray) -> jnp.ndarray:
        z = jax.nn.sigmoid(self.weights_zx(x) + self.weights_zh(h))
        r = jax.nn.sigmoid(self.weights_rx(x) + self.weights_rh(h))
        h_tilde = jnp.tanh(self.weights_hx(x) + self.weights_hh(r * h))
        
        # z_eff = 1 - (1-z)^dt
        z_eff = 1.0 - jnp.power(jnp.clip(1.0 - z, a_min=1e-7), dt)
        
        h_new = (1.0 - z_eff) * h + z_eff * h_tilde
        return h_new

class LiquidTimeCell(eqx.Module):
    """Continuous-time evolution: dh/dt = -1/tau * h + f."""
    net: eqx.nn.MLP
    hidden_size: int

    def __init__(self, input_size: int, hidden_size: int, key: jax.random.PRNGKey):
        self.net = eqx.nn.MLP(
            in_size=input_size + hidden_size,
            out_size=hidden_size * 2,
            width_size=hidden_size * 2,
            depth=1,
            activation=jax.nn.tanh,
            key=key
        )
        self.hidden_size = hidden_size

    def __call__(self, h: jnp.ndarray, x: jnp.ndarray, dt: jnp.ndarray) -> jnp.ndarray:
        combined = jnp.concatenate([x, h], axis=-1)
        out = self.net(combined)
        inv_tau_raw, f_raw = jnp.split(out, 2, axis=-1)
        
        inv_tau = jax.nn.softplus(inv_tau_raw) + 1e-4
        f = jnp.tanh(f_raw)
        
        # Analytic solution for dh/dt = -inv_tau * h + f over interval dt
        decay = jnp.exp(-dt * inv_tau)
        h_new = h * decay + f * (1.0 - decay)
        return h_new
