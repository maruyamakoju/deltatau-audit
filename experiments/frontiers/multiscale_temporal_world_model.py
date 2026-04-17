"""
Multi-Scale Temporal World Model
================================

A hierarchical RSSM with slow and fast latent variables operating at different
temporal scales, connected by cross-scale attention.

**Core idea**: Standard world models operate at a single temporal scale.  Real-
world dynamics have multiple scales: fast reactive control (reflexes, immediate
responses) and slow strategic planning (goals, phase transitions).  This module
introduces a two-tier categorical RSSM where the fast tier updates every step
and the slow tier updates every K steps, coupled through multi-head cross-scale
attention that lets the fast tier condition on slow strategic context.

Self-contained: no imports from the main ``deltatau_audit`` or
``internal_time_rl`` packages.

License: Apache-2.0
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric log transform (Dreamer v3)."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def categorical_kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """KL(Cat(p) || Cat(q)) summed over the last dim, averaged over batch."""
    p = F.softmax(p_logits, dim=-1)
    log_p = F.log_softmax(p_logits, dim=-1)
    log_q = F.log_softmax(q_logits, dim=-1)
    return (p * (log_p - log_q)).sum(-1).mean()


def kl_balanced(
    posterior_logits: torch.Tensor,
    prior_logits: torch.Tensor,
    alpha: float = 0.8,
) -> torch.Tensor:
    """KL balancing (Dreamer v3): alpha * KL(sg(post)||prior) + (1-a) * KL(post||sg(prior))."""
    return alpha * categorical_kl(posterior_logits.detach(), prior_logits) + (
        1.0 - alpha
    ) * categorical_kl(posterior_logits, prior_logits.detach())


def gumbel_softmax_sample(
    logits: torch.Tensor, temperature: float = 1.0, hard: bool = True
) -> torch.Tensor:
    """Sample from Gumbel-Softmax with optional straight-through gradient."""
    return F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)


# ---------------------------------------------------------------------------
# Cross-Scale Attention
# ---------------------------------------------------------------------------

class CrossScaleAttention(nn.Module):
    """Multi-head attention: fast (query) attends to slow (key/value).

    Query  : fast hidden state  (B, fast_dim)
    Key/Val: slow hidden + slow stochastic  (B, slow_dim + slow_stoch_flat)
    Output : context vector  (B, fast_dim)
    """

    def __init__(
        self,
        fast_dim: int,
        slow_dim: int,
        slow_stoch_flat: int,
        num_heads: int = 4,
        head_dim: int = 32,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner = num_heads * head_dim
        self.q_proj = nn.Linear(fast_dim, inner, bias=False)
        self.k_proj = nn.Linear(slow_dim + slow_stoch_flat, inner, bias=False)
        self.v_proj = nn.Linear(slow_dim + slow_stoch_flat, inner, bias=False)
        self.out_proj = nn.Linear(inner, fast_dim, bias=False)
        self.scale = head_dim ** -0.5

    def forward(
        self, fast_h: torch.Tensor, slow_h: torch.Tensor, slow_z: torch.Tensor
    ) -> torch.Tensor:
        B = fast_h.shape[0]
        kv_input = torch.cat([slow_h, slow_z], dim=-1)  # (B, slow_dim+stoch)

        # Project to multi-head form — treat as single-token sequences
        q = self.q_proj(fast_h).view(B, self.num_heads, 1, self.head_dim)
        k = self.k_proj(kv_input).view(B, self.num_heads, 1, self.head_dim)
        v = self.v_proj(kv_input).view(B, self.num_heads, 1, self.head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, 1, 1)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).view(B, -1)  # (B, inner)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Fast RSSM
# ---------------------------------------------------------------------------

class FastRSSM(nn.Module):
    """Standard RSSM operating at every timestep.

    State: (h, z)  where h is deterministic (GRU) and z is categorical stochastic.
    The GRU receives an additional *context* vector from the slow scale.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        stoch_dim: int = 32,
        num_classes: int = 32,
        context_dim: int = 256,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.stoch_dim = stoch_dim
        self.num_classes = num_classes
        self.stoch_flat = stoch_dim * num_classes

        # Pre-GRU projection: concat(z_flat, action, context) → hidden_dim
        self.pre_gru = nn.Sequential(
            nn.Linear(self.stoch_flat + action_dim + context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Prior: h → logits
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.stoch_flat),
        )

        # Posterior: (h, obs) → logits
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.stoch_flat),
        )

        # Timing head: h → log-normal parameters (mu, log_sigma)
        self.timing_head = nn.Linear(hidden_dim, 2)

    def initial_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        z = torch.zeros(batch_size, self.stoch_flat, device=device)
        return h, z

    def prior(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.prior_net(h)
        return logits.view(-1, self.stoch_dim, self.num_classes)

    def posterior(self, h: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        logits = self.posterior_net(torch.cat([h, obs], dim=-1))
        return logits.view(-1, self.stoch_dim, self.num_classes)

    def step(
        self,
        prev_h: torch.Tensor,
        prev_z: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor,
        obs: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Single transition step.  Returns dict with h, z, prior_logits, posterior_logits, dt."""
        x = self.pre_gru(torch.cat([prev_z, action, context], dim=-1))
        h = self.gru(x, prev_h)

        prior_logits = self.prior(h)

        if obs is not None:
            post_logits = self.posterior(h, obs)
            z = gumbel_softmax_sample(post_logits, temperature=temperature, hard=True)
        else:
            post_logits = prior_logits
            z = gumbel_softmax_sample(prior_logits, temperature=temperature, hard=True)

        z_flat = z.view(-1, self.stoch_flat)

        # Timing prediction
        timing_params = self.timing_head(h)
        dt_mu, dt_log_sigma = timing_params[:, 0], timing_params[:, 1]

        return {
            "h": h,
            "z": z_flat,
            "prior_logits": prior_logits,
            "posterior_logits": post_logits,
            "dt_mu": dt_mu,
            "dt_log_sigma": dt_log_sigma,
        }


# ---------------------------------------------------------------------------
# Slow RSSM
# ---------------------------------------------------------------------------

class SlowRSSM(nn.Module):
    """RSSM operating every K timesteps.

    Before each slow tick, K fast hidden states are aggregated via learned
    attention pooling into a fixed-size summary.
    """

    def __init__(
        self,
        obs_dim: int,
        fast_hidden_dim: int,
        hidden_dim: int = 128,
        stoch_dim: int = 16,
        num_classes: int = 16,
        slow_tick_every: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.stoch_dim = stoch_dim
        self.num_classes = num_classes
        self.stoch_flat = stoch_dim * num_classes
        self.slow_tick_every = slow_tick_every

        # Attention pooling over K fast states
        self.pool_query = nn.Parameter(torch.randn(1, 1, fast_hidden_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=fast_hidden_dim, num_heads=4, batch_first=True
        )
        self.pool_proj = nn.Linear(fast_hidden_dim, hidden_dim)

        # Pre-GRU: concat(Z_flat, aggregated_fast) → hidden_dim
        self.pre_gru = nn.Sequential(
            nn.Linear(self.stoch_flat + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Prior / Posterior
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.stoch_flat),
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.stoch_flat),
        )

        # Timing head — predicts aggregate DT for the slow interval
        self.timing_head = nn.Linear(hidden_dim, 2)

    def initial_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        H = torch.zeros(batch_size, self.hidden_dim, device=device)
        Z = torch.zeros(batch_size, self.stoch_flat, device=device)
        return H, Z

    def aggregate_fast(self, fast_states: torch.Tensor) -> torch.Tensor:
        """Attention-pool K fast hidden states.

        Args:
            fast_states: (B, K, fast_hidden_dim)

        Returns:
            (B, hidden_dim)
        """
        B = fast_states.shape[0]
        query = self.pool_query.expand(B, -1, -1)  # (B, 1, fast_hidden_dim)
        pooled, _ = self.pool_attn(query, fast_states, fast_states)  # (B, 1, fast_hidden_dim)
        return self.pool_proj(pooled.squeeze(1))  # (B, hidden_dim)

    def step(
        self,
        prev_H: torch.Tensor,
        prev_Z: torch.Tensor,
        fast_states: torch.Tensor,
        obs: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        agg = self.aggregate_fast(fast_states)
        x = self.pre_gru(torch.cat([prev_Z, agg], dim=-1))
        H = self.gru(x, prev_H)

        prior_logits = self.prior_net(H).view(-1, self.stoch_dim, self.num_classes)

        if obs is not None:
            post_logits = self.posterior_net(torch.cat([H, obs], dim=-1)).view(
                -1, self.stoch_dim, self.num_classes
            )
            Z = gumbel_softmax_sample(post_logits, temperature=temperature, hard=True)
        else:
            post_logits = prior_logits
            Z = gumbel_softmax_sample(prior_logits, temperature=temperature, hard=True)

        Z_flat = Z.view(-1, self.stoch_flat)

        timing_params = self.timing_head(H)
        DT_mu, DT_log_sigma = timing_params[:, 0], timing_params[:, 1]

        return {
            "H": H,
            "Z": Z_flat,
            "prior_logits": prior_logits,
            "posterior_logits": post_logits,
            "DT_mu": DT_mu,
            "DT_log_sigma": DT_log_sigma,
        }


# ---------------------------------------------------------------------------
# Observation Decoder
# ---------------------------------------------------------------------------

class ObservationDecoder(nn.Module):
    """Decodes (h, z) → observation prediction in symlog space."""

    def __init__(self, hidden_dim: int, stoch_flat: int, obs_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + stoch_flat, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, obs_dim),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h, z], dim=-1))


# ---------------------------------------------------------------------------
# Multi-Scale Temporal World Model
# ---------------------------------------------------------------------------

class MultiScaleTemporalWorldModel(nn.Module):
    """Hierarchical two-tier world model with fast and slow RSSM, coupled
    through cross-scale attention.

    Training operates on observation/action sequences of length T.  The fast
    RSSM processes every step; the slow RSSM fires every ``slow_tick_every``
    steps.  A cross-scale consistency loss aligns fast predictions at slow-tick
    boundaries with the slow tier's predictions.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        fast_hidden_dim: int = 256,
        slow_hidden_dim: int = 128,
        fast_stoch_dim: int = 32,
        slow_stoch_dim: int = 16,
        num_classes: int = 32,
        cross_scale_heads: int = 4,
        slow_tick_every: int = 4,
        kl_alpha: float = 0.8,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.slow_tick_every = slow_tick_every
        self.kl_alpha = kl_alpha

        fast_stoch_flat = fast_stoch_dim * num_classes
        slow_stoch_flat = slow_stoch_dim * num_classes

        self.cross_attn = CrossScaleAttention(
            fast_dim=fast_hidden_dim,
            slow_dim=slow_hidden_dim,
            slow_stoch_flat=slow_stoch_flat,
            num_heads=cross_scale_heads,
            head_dim=fast_hidden_dim // cross_scale_heads,
        )

        self.fast_rssm = FastRSSM(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=fast_hidden_dim,
            stoch_dim=fast_stoch_dim,
            num_classes=num_classes,
            context_dim=fast_hidden_dim,  # cross-attn output dim
        )

        self.slow_rssm = SlowRSSM(
            obs_dim=obs_dim,
            fast_hidden_dim=fast_hidden_dim,
            hidden_dim=slow_hidden_dim,
            stoch_dim=slow_stoch_dim,
            num_classes=num_classes,
            slow_tick_every=slow_tick_every,
        )

        self.fast_decoder = ObservationDecoder(fast_hidden_dim, fast_stoch_flat, obs_dim)
        self.slow_decoder = ObservationDecoder(slow_hidden_dim, slow_stoch_flat, obs_dim)

        # Cross-scale consistency projector: project fast state to slow state space
        self.consistency_proj = nn.Sequential(
            nn.Linear(fast_hidden_dim + fast_stoch_flat, 256),
            nn.SiLU(),
            nn.Linear(256, slow_hidden_dim + slow_stoch_flat),
        )

    def forward(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Process a training sequence.

        Args:
            obs_seq: (B, T, obs_dim)
            action_seq: (B, T, action_dim)
            temperature: Gumbel-Softmax temperature

        Returns:
            dict with losses and metrics
        """
        B, T, _ = obs_seq.shape
        device = obs_seq.device
        K = self.slow_tick_every

        # Initial states
        fast_h, fast_z = self.fast_rssm.initial_state(B, device)
        slow_H, slow_Z = self.slow_rssm.initial_state(B, device)

        # Accumulators
        fast_prior_logits_list: List[torch.Tensor] = []
        fast_post_logits_list: List[torch.Tensor] = []
        fast_recon_list: List[torch.Tensor] = []
        fast_dt_mus: List[torch.Tensor] = []
        fast_dt_log_sigmas: List[torch.Tensor] = []

        slow_prior_logits_list: List[torch.Tensor] = []
        slow_post_logits_list: List[torch.Tensor] = []
        slow_recon_list: List[torch.Tensor] = []
        slow_dt_mus: List[torch.Tensor] = []
        slow_dt_log_sigmas: List[torch.Tensor] = []

        consistency_losses: List[torch.Tensor] = []

        # Buffer for fast states within current slow interval
        fast_h_buffer: List[torch.Tensor] = []

        for t in range(T):
            obs_t = obs_seq[:, t]
            act_t = action_seq[:, t]

            # Cross-scale context
            context = self.cross_attn(fast_h, slow_H, slow_Z)

            # Fast step
            fast_out = self.fast_rssm.step(
                fast_h, fast_z, act_t, context, obs=obs_t, temperature=temperature
            )
            fast_h = fast_out["h"]
            fast_z = fast_out["z"]

            fast_prior_logits_list.append(fast_out["prior_logits"])
            fast_post_logits_list.append(fast_out["posterior_logits"])
            fast_recon_list.append(self.fast_decoder(fast_h, fast_z))
            fast_dt_mus.append(fast_out["dt_mu"])
            fast_dt_log_sigmas.append(fast_out["dt_log_sigma"])

            fast_h_buffer.append(fast_h)

            # Slow tick
            if (t + 1) % K == 0 and len(fast_h_buffer) >= K:
                fast_states = torch.stack(fast_h_buffer[-K:], dim=1)  # (B, K, fast_hidden_dim)
                slow_out = self.slow_rssm.step(
                    slow_H, slow_Z, fast_states, obs=obs_t, temperature=temperature
                )
                slow_H = slow_out["H"]
                slow_Z = slow_out["Z"]

                slow_prior_logits_list.append(slow_out["prior_logits"])
                slow_post_logits_list.append(slow_out["posterior_logits"])
                slow_recon_list.append(self.slow_decoder(slow_H, slow_Z))
                slow_dt_mus.append(slow_out["DT_mu"])
                slow_dt_log_sigmas.append(slow_out["DT_log_sigma"])

                # Cross-scale consistency: fast state at boundary ≈ slow state
                fast_combined = torch.cat([fast_h, fast_z], dim=-1)
                slow_combined = torch.cat([slow_H, slow_Z], dim=-1)
                projected = self.consistency_proj(fast_combined)
                consistency_losses.append(
                    F.mse_loss(projected, slow_combined.detach())
                )

                fast_h_buffer = []

        # ---- Aggregate losses ----
        # Fast KL
        fast_kl = torch.tensor(0.0, device=device)
        for pr, po in zip(fast_prior_logits_list, fast_post_logits_list):
            fast_kl = fast_kl + kl_balanced(po, pr, self.kl_alpha)
        fast_kl = fast_kl / max(len(fast_prior_logits_list), 1)

        # Slow KL
        slow_kl = torch.tensor(0.0, device=device)
        for pr, po in zip(slow_prior_logits_list, slow_post_logits_list):
            slow_kl = slow_kl + kl_balanced(po, pr, self.kl_alpha)
        slow_kl = slow_kl / max(len(slow_prior_logits_list), 1)

        # Fast reconstruction (symlog space)
        fast_preds = torch.stack(fast_recon_list, dim=1)  # (B, T, obs_dim)
        fast_recon_loss = F.mse_loss(fast_preds, symlog(obs_seq))

        # Slow reconstruction at boundary observations
        if slow_recon_list:
            slow_preds = torch.stack(slow_recon_list, dim=1)
            # Gather boundary obs
            boundary_indices = [K * (i + 1) - 1 for i in range(len(slow_recon_list))]
            boundary_obs = obs_seq[:, boundary_indices]
            slow_recon_loss = F.mse_loss(slow_preds, symlog(boundary_obs))
        else:
            slow_recon_loss = torch.tensor(0.0, device=device)

        # Cross-scale consistency
        if consistency_losses:
            cross_consistency = torch.stack(consistency_losses).mean()
        else:
            cross_consistency = torch.tensor(0.0, device=device)

        # Timing loss: fast dt should predict constant 1.0 (unit timestep)
        if fast_dt_mus:
            dt_mus = torch.stack(fast_dt_mus, dim=1)
            dt_target = torch.zeros_like(dt_mus)  # log(1.0) = 0 in log-normal
            timing_loss = F.mse_loss(dt_mus, dt_target)
        else:
            timing_loss = torch.tensor(0.0, device=device)

        total_loss = (
            fast_recon_loss
            + slow_recon_loss
            + 0.1 * fast_kl
            + 0.1 * slow_kl
            + 0.5 * cross_consistency
            + 0.01 * timing_loss
        )

        return {
            "loss": total_loss,
            "fast_recon": fast_recon_loss.detach(),
            "slow_recon": slow_recon_loss.detach(),
            "fast_kl": fast_kl.detach(),
            "slow_kl": slow_kl.detach(),
            "cross_consistency": cross_consistency.detach(),
            "timing_loss": timing_loss.detach(),
        }

    @torch.no_grad()
    def imagine(
        self,
        initial_obs: torch.Tensor,
        actions: torch.Tensor,
        horizon: int,
        temperature: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """Imagine a trajectory from an initial observation.

        Args:
            initial_obs: (B, obs_dim) — used to seed the posterior at step 0
            actions: (B, horizon, action_dim)
            horizon: number of steps to imagine
            temperature: Gumbel temperature (lower → more deterministic)

        Returns:
            dict with predicted observations and timing at both scales
        """
        B = initial_obs.shape[0]
        device = initial_obs.device
        K = self.slow_tick_every

        fast_h, fast_z = self.fast_rssm.initial_state(B, device)
        slow_H, slow_Z = self.slow_rssm.initial_state(B, device)

        # Seed with initial observation
        context = self.cross_attn(fast_h, slow_H, slow_Z)
        seed = self.fast_rssm.step(
            fast_h, fast_z, actions[:, 0], context, obs=initial_obs, temperature=temperature
        )
        fast_h, fast_z = seed["h"], seed["z"]

        preds: List[torch.Tensor] = []
        fast_h_buffer: List[torch.Tensor] = [fast_h]

        for t in range(1, horizon):
            context = self.cross_attn(fast_h, slow_H, slow_Z)
            fast_out = self.fast_rssm.step(
                fast_h, fast_z, actions[:, t], context, obs=None, temperature=temperature
            )
            fast_h = fast_out["h"]
            fast_z = fast_out["z"]
            fast_h_buffer.append(fast_h)

            preds.append(symexp(self.fast_decoder(fast_h, fast_z)))

            if (t + 1) % K == 0 and len(fast_h_buffer) >= K:
                fast_states = torch.stack(fast_h_buffer[-K:], dim=1)
                slow_out = self.slow_rssm.step(
                    slow_H, slow_Z, fast_states, obs=None, temperature=temperature
                )
                slow_H, slow_Z = slow_out["H"], slow_out["Z"]
                fast_h_buffer = []

        return {"predicted_obs": torch.stack(preds, dim=1) if preds else torch.empty(B, 0, self.obs_dim, device=device)}


# ---------------------------------------------------------------------------
# Single-Scale Baseline (for ablation)
# ---------------------------------------------------------------------------

class SingleScaleBaseline(nn.Module):
    """A standard single-scale RSSM for controlled comparison."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        stoch_dim: int = 32,
        num_classes: int = 32,
        kl_alpha: float = 0.8,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.kl_alpha = kl_alpha
        stoch_flat = stoch_dim * num_classes

        # Use a zero-context FastRSSM
        self.rssm = FastRSSM(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            stoch_dim=stoch_dim,
            num_classes=num_classes,
            context_dim=hidden_dim,  # will feed zeros
        )
        self.decoder = ObservationDecoder(hidden_dim, stoch_flat, obs_dim)

    def forward(
        self, obs_seq: torch.Tensor, action_seq: torch.Tensor, temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = obs_seq.shape
        device = obs_seq.device

        h, z = self.rssm.initial_state(B, device)
        zero_ctx = torch.zeros(B, self.rssm.hidden_dim, device=device)

        prior_list, post_list, recon_list = [], [], []

        for t in range(T):
            out = self.rssm.step(h, z, action_seq[:, t], zero_ctx, obs=obs_seq[:, t], temperature=temperature)
            h, z = out["h"], out["z"]
            prior_list.append(out["prior_logits"])
            post_list.append(out["posterior_logits"])
            recon_list.append(self.decoder(h, z))

        kl = torch.tensor(0.0, device=device)
        for pr, po in zip(prior_list, post_list):
            kl = kl + kl_balanced(po, pr, self.kl_alpha)
        kl = kl / max(len(prior_list), 1)

        preds = torch.stack(recon_list, dim=1)
        recon_loss = F.mse_loss(preds, symlog(obs_seq))

        total = recon_loss + 0.1 * kl
        return {"loss": total, "recon": recon_loss.detach(), "kl": kl.detach()}


# ---------------------------------------------------------------------------
# Data Collection
# ---------------------------------------------------------------------------

def collect_cartpole_data(
    num_episodes: int = 200,
    max_steps: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect random trajectories from CartPole.

    Returns:
        obs_data:    (N, T, obs_dim)  — zero-padded to max_steps
        action_data: (N, T, 1)        — one-hot is not needed; raw int → float
        lengths:     (N,)             — actual episode lengths
    """
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]

    obs_list, act_list, len_list = [], [], []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        ep_obs, ep_act = [obs], []

        for _step in range(max_steps - 1):
            action = env.action_space.sample()
            next_obs, _, terminated, truncated, _ = env.step(action)
            ep_act.append([float(action)])
            ep_obs.append(next_obs)
            if terminated or truncated:
                break

        T_ep = len(ep_act)
        len_list.append(T_ep)

        # Pad to max_steps
        obs_arr = np.zeros((max_steps, obs_dim), dtype=np.float32)
        act_arr = np.zeros((max_steps, 1), dtype=np.float32)
        obs_arr[: T_ep + 1] = np.array(ep_obs[: T_ep + 1], dtype=np.float32)
        act_arr[:T_ep] = np.array(ep_act, dtype=np.float32)

        obs_list.append(obs_arr)
        act_list.append(act_arr)

    env.close()

    return (
        np.array(obs_list, dtype=np.float32),
        np.array(act_list, dtype=np.float32),
        np.array(len_list, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------

@dataclass
class MultiScaleWMExperiment:
    """End-to-end experiment: data collection, training, evaluation, and
    comparison against a single-scale baseline.
    """

    # Dimensions
    obs_dim: int = 4
    action_dim: int = 1

    # Fast scale
    fast_hidden_dim: int = 256
    fast_stoch_dim: int = 32

    # Slow scale
    slow_hidden_dim: int = 128
    slow_stoch_dim: int = 16

    # Shared
    num_classes: int = 32
    cross_scale_heads: int = 4
    slow_tick_every: int = 4

    # Training
    sequence_length: int = 48
    batch_size: int = 32
    train_steps: int = 500
    lr: float = 3e-4

    def _build_model(self, device: torch.device) -> MultiScaleTemporalWorldModel:
        return MultiScaleTemporalWorldModel(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            fast_hidden_dim=self.fast_hidden_dim,
            slow_hidden_dim=self.slow_hidden_dim,
            fast_stoch_dim=self.fast_stoch_dim,
            slow_stoch_dim=self.slow_stoch_dim,
            num_classes=self.num_classes,
            cross_scale_heads=self.cross_scale_heads,
            slow_tick_every=self.slow_tick_every,
        ).to(device)

    def _build_baseline(self, device: torch.device) -> SingleScaleBaseline:
        return SingleScaleBaseline(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.fast_hidden_dim,
            stoch_dim=self.fast_stoch_dim,
            num_classes=self.num_classes,
        ).to(device)

    def _sample_batch(
        self,
        obs_data: np.ndarray,
        act_data: np.ndarray,
        lengths: np.ndarray,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of sub-sequences."""
        N = obs_data.shape[0]
        indices = np.random.randint(0, N, size=self.batch_size)
        T = self.sequence_length

        obs_batch = np.zeros((self.batch_size, T, self.obs_dim), dtype=np.float32)
        act_batch = np.zeros((self.batch_size, T, self.action_dim), dtype=np.float32)

        for i, idx in enumerate(indices):
            ep_len = int(lengths[idx])
            max_start = max(0, ep_len - T)
            start = np.random.randint(0, max_start + 1)
            end = min(start + T, ep_len)
            seg_len = end - start
            obs_batch[i, :seg_len] = obs_data[idx, start:end]
            act_batch[i, :seg_len] = act_data[idx, start:end]

        return (
            torch.tensor(obs_batch, device=device),
            torch.tensor(act_batch, device=device),
        )

    def _train_loop(
        self,
        model: nn.Module,
        obs_data: np.ndarray,
        act_data: np.ndarray,
        lengths: np.ndarray,
        device: torch.device,
        label: str,
    ) -> List[Dict[str, float]]:
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.train_steps)

        history: List[Dict[str, float]] = []
        temperature = 1.0

        for step in range(1, self.train_steps + 1):
            # Anneal temperature
            temperature = max(0.1, 1.0 - 0.9 * step / self.train_steps)

            obs_b, act_b = self._sample_batch(obs_data, act_data, lengths, device)
            out = model(obs_b, act_b, temperature=temperature)

            optimizer.zero_grad()
            out["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            optimizer.step()
            scheduler.step()

            record = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in out.items()}
            record["step"] = step
            record["temperature"] = temperature
            history.append(record)

            if step % 100 == 0 or step == 1:
                loss_str = ", ".join(f"{k}={v:.4f}" for k, v in record.items() if k != "step")
                print(f"[{label}] step {step}/{self.train_steps}: {loss_str}")

        return history

    @torch.no_grad()
    def _evaluate(
        self,
        model: nn.Module,
        obs_data: np.ndarray,
        act_data: np.ndarray,
        lengths: np.ndarray,
        device: torch.device,
        n_eval: int = 20,
    ) -> Dict[str, float]:
        """Evaluate reconstruction quality on held-out batches."""
        model.eval()
        total_recon = 0.0
        total_slow_recon = 0.0
        total_cross_consistency = 0.0
        has_slow = False

        for _ in range(n_eval):
            obs_b, act_b = self._sample_batch(obs_data, act_data, lengths, device)
            out = model(obs_b, act_b, temperature=0.1)
            # Use the first recon-related key
            for key in ("fast_recon", "recon"):
                if key in out:
                    total_recon += out[key].item()
                    break
            if "slow_recon" in out:
                total_slow_recon += out["slow_recon"].item()
                has_slow = True
            if "cross_consistency" in out:
                total_cross_consistency += out["cross_consistency"].item()

        model.train()
        result = {"avg_recon": total_recon / n_eval}
        if has_slow:
            result["avg_slow_recon"] = total_slow_recon / n_eval
            result["avg_cross_consistency"] = total_cross_consistency / n_eval
        return result

    def run(self, out_dir: Path) -> Dict[str, float]:
        """Execute the full experiment.

        Steps:
            1. Collect CartPole data
            2. Train multi-scale model
            3. Train single-scale baseline
            4. Evaluate both
            5. Compute composite score
            6. Save results

        Returns:
            Dictionary of final metrics including composite_score.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        # ---- 1. Data ----
        t0 = time.time()
        print("Collecting CartPole data ...")
        obs_data, act_data, lengths = collect_cartpole_data(
            num_episodes=200, max_steps=200
        )
        data_time = time.time() - t0
        print(f"  Collected {len(lengths)} episodes in {data_time:.1f}s  "
              f"(mean length {lengths.mean():.1f})")

        # Simple train/eval split
        n_train = int(0.8 * len(lengths))
        train_obs, eval_obs = obs_data[:n_train], obs_data[n_train:]
        train_act, eval_act = act_data[:n_train], act_data[n_train:]
        train_len, eval_len = lengths[:n_train], lengths[n_train:]

        # ---- 2. Multi-scale model ----
        print("\n=== Training Multi-Scale World Model ===")
        ms_model = self._build_model(device)
        n_params_ms = sum(p.numel() for p in ms_model.parameters())
        print(f"  Parameters: {n_params_ms:,}")

        t1 = time.time()
        ms_history = self._train_loop(
            ms_model, train_obs, train_act, train_len, device, label="MultiScale"
        )
        ms_train_time = time.time() - t1

        ms_eval = self._evaluate(ms_model, eval_obs, eval_act, eval_len, device)
        print(f"  Eval recon: {ms_eval['avg_recon']:.4f}")

        # ---- 3. Baseline ----
        print("\n=== Training Single-Scale Baseline ===")
        bl_model = self._build_baseline(device)
        n_params_bl = sum(p.numel() for p in bl_model.parameters())
        print(f"  Parameters: {n_params_bl:,}")

        t2 = time.time()
        bl_history = self._train_loop(
            bl_model, train_obs, train_act, train_len, device, label="Baseline"
        )
        bl_train_time = time.time() - t2

        bl_eval = self._evaluate(bl_model, eval_obs, eval_act, eval_len, device)
        print(f"  Eval recon: {bl_eval['avg_recon']:.4f}")

        # ---- 4. Metrics ----
        # Normalised quality scores (lower recon = better, map to [0,1])
        # All metrics now use held-out evaluation data (not training batch)
        fast_pred_quality = 1.0 / (1.0 + ms_eval["avg_recon"])
        slow_pred_quality = 1.0 / (1.0 + ms_eval.get("avg_slow_recon", ms_history[-1].get("slow_recon", 1.0)))
        cross_consistency = 1.0 / (1.0 + ms_eval.get("avg_cross_consistency", ms_history[-1].get("cross_consistency", 1.0)))

        # Improvement over baseline (positive = multi-scale is better)
        bl_recon = bl_eval["avg_recon"]
        ms_recon = ms_eval["avg_recon"]
        improvement = max(0.0, (bl_recon - ms_recon) / (bl_recon + 1e-8))
        improvement_score = min(1.0, improvement)  # cap at 1.0

        composite_score = (
            0.30 * fast_pred_quality
            + 0.25 * slow_pred_quality
            + 0.25 * cross_consistency
            + 0.20 * improvement_score
        )

        results: Dict[str, Any] = {
            "composite_score": round(composite_score, 4),
            "fast_pred_quality": round(fast_pred_quality, 4),
            "slow_pred_quality": round(slow_pred_quality, 4),
            "cross_consistency": round(cross_consistency, 4),
            "improvement_over_baseline": round(improvement_score, 4),
            "ms_eval_recon": round(ms_recon, 4),
            "bl_eval_recon": round(bl_recon, 4),
            "ms_params": n_params_ms,
            "bl_params": n_params_bl,
            "ms_train_time_s": round(ms_train_time, 1),
            "bl_train_time_s": round(bl_train_time, 1),
            "data_time_s": round(data_time, 1),
            "device": str(device),
            "train_steps": self.train_steps,
            "slow_tick_every": self.slow_tick_every,
        }

        print(f"\n=== Results ===")
        for k, v in results.items():
            print(f"  {k}: {v}")

        # ---- 5. Save ----
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(results, f, indent=2)

        # Training curves
        curves = {
            "multiscale": ms_history,
            "baseline": bl_history,
        }
        with open(out_dir / "training_curves.json", "w") as f:
            json.dump(curves, f, indent=2)

        # Save model checkpoints
        torch.save(ms_model.state_dict(), out_dir / "multiscale_model.pt")
        torch.save(bl_model.state_dict(), out_dir / "baseline_model.pt")

        print(f"\nArtifacts saved to {out_dir}")
        return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the multi-scale temporal world model experiment."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Scale Temporal World Model — frontier experiment"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/multiscale_wm"),
        help="Output directory for artifacts",
    )
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=48)
    parser.add_argument("--slow-tick-every", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--fast-hidden-dim", type=int, default=256)
    parser.add_argument("--slow-hidden-dim", type=int, default=128)
    parser.add_argument("--fast-stoch-dim", type=int, default=32)
    parser.add_argument("--slow-stoch-dim", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=32)
    parser.add_argument("--cross-scale-heads", type=int, default=4)
    args = parser.parse_args()

    experiment = MultiScaleWMExperiment(
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        slow_tick_every=args.slow_tick_every,
        lr=args.lr,
        fast_hidden_dim=args.fast_hidden_dim,
        slow_hidden_dim=args.slow_hidden_dim,
        fast_stoch_dim=args.fast_stoch_dim,
        slow_stoch_dim=args.slow_stoch_dim,
        num_classes=args.num_classes,
        cross_scale_heads=args.cross_scale_heads,
    )

    results = experiment.run(args.out_dir)
    print(f"\nComposite score: {results['composite_score']}")


if __name__ == "__main__":
    main()
