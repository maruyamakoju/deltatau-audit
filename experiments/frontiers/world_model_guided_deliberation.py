"""World Model-Guided Deliberation — Frontier Research Module.

A novel algorithm where a Dreamer v3-style RSSM world model's uncertainty
predictions guide the halting decisions of an Adaptive Computation Time (ACT)
agent.

**Core insight**: Standard ACT uses a fixed geometric prior for halting.
We replace this with an *uncertainty-adaptive* halting schedule: when the
world model is uncertain about upcoming states, the agent deliberates
longer.  When the future is predictable, the agent halts early and acts
without wasting computation.

This creates a natural connection between *epistemic uncertainty* (what the
agent doesn't know about the world) and *computational budget* (how long it
thinks before acting).  The result is an agent that allocates its reasoning
resources proportionally to the difficulty of the situation — a desirable
property for safe, efficient RL.

Uses the shared ACT bookkeeping helpers so frontier semantics match the core
deliberative agent.

Authors: frontier-experiments team
License: Apache-2.0
"""

from __future__ import annotations

import json
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from act_utils import apply_act_step, halt_distribution_stats, stack_halt_weights

# ---------------------------------------------------------------------------
# Hyperparameter containers
# ---------------------------------------------------------------------------


@dataclass
class RSSMConfig:
    """Configuration for the MiniRSSM world model."""

    obs_dim: int = 4
    action_dim: int = 2
    hidden_dim: int = 128
    stoch_dim: int = 32
    num_classes: int = 32
    gumbel_tau: float = 1.0
    reward_layers: int = 2
    timing_layers: int = 2


@dataclass
class ACTConfig:
    """Configuration for UncertaintyGuidedACT."""

    obs_dim: int = 4
    action_dim: int = 2
    hidden_dim: int = 128
    base_thinking_steps: int = 5
    max_thinking_steps: int = 15
    lambda_geo: float = 0.5
    uncertainty_scale: float = 5.0
    lambda_uncertainty_modulation: float = 0.3
    encoder_hidden: int = 64


@dataclass
class ExperimentConfig:
    """Top-level experiment hyperparameters."""

    env_id: str = "CartPole-v1"
    obs_dim: int = 4
    action_dim: int = 2
    hidden_dim: int = 128
    rssm_stoch_dim: int = 32
    rssm_num_classes: int = 32
    max_thinking_steps: int = 15
    imagination_horizon: int = 5
    uncertainty_threshold: float = 0.5
    lambda_geo: float = 0.5
    n_episodes: int = 50
    max_steps: int = 500
    seed: int = 42
    device: str = "cpu"


# ---------------------------------------------------------------------------
# MiniRSSM — lightweight Dreamer-inspired world model
# ---------------------------------------------------------------------------


class MiniRSSM(nn.Module):
    """Lightweight Dreamer v3-style Recurrent State-Space Model.

    The world model factorises its latent state into a *deterministic*
    recurrent component ``h`` (GRU hidden state) and a *stochastic*
    categorical component ``z`` (a vector of ``stoch_dim`` categoricals,
    each with ``num_classes`` classes).

    Deterministic path::

        h_t = GRU(h_{t-1}, concat(z_{t-1}, a_{t-1}))

    Prior (imagination — no observation)::

        z_t ~ Cat(f(h_t))

    Posterior (observation available)::

        z_t ~ Cat(g(h_t, o_t))

    Auxiliary heads decode observations, rewards, and timing information
    from the joint latent ``(h, z_flat)``.
    """

    def __init__(self, cfg: RSSMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.z_dim = cfg.stoch_dim * cfg.num_classes  # flattened categorical

        # --- Deterministic path (GRU) ---
        gru_input_dim = self.z_dim + cfg.action_dim
        self.gru_cell = nn.GRUCell(gru_input_dim, cfg.hidden_dim)

        # --- Prior network: h → logits for categorical z ---
        self.prior_net = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ELU(),
            nn.Linear(cfg.hidden_dim, self.z_dim),
        )

        # --- Posterior network: (h, o) → logits for categorical z ---
        self.posterior_net = nn.Sequential(
            nn.Linear(cfg.hidden_dim + cfg.obs_dim, cfg.hidden_dim),
            nn.ELU(),
            nn.Linear(cfg.hidden_dim, self.z_dim),
        )

        # --- Observation decoder ---
        self.obs_decoder = nn.Sequential(
            nn.Linear(cfg.hidden_dim + self.z_dim, cfg.hidden_dim),
            nn.ELU(),
            nn.Linear(cfg.hidden_dim, cfg.obs_dim),
        )

        # --- Reward predictor ---
        reward_layers: list[nn.Module] = []
        in_dim = cfg.hidden_dim + self.z_dim
        for _ in range(cfg.reward_layers):
            reward_layers.extend([nn.Linear(in_dim, cfg.hidden_dim), nn.ELU()])
            in_dim = cfg.hidden_dim
        reward_layers.append(nn.Linear(in_dim, 1))
        self.reward_head = nn.Sequential(*reward_layers)

        # --- Timing predictor: (mu, log_sigma) for LogNormal dt ---
        timing_layers: list[nn.Module] = []
        in_dim = cfg.hidden_dim + self.z_dim
        for _ in range(cfg.timing_layers):
            timing_layers.extend([nn.Linear(in_dim, cfg.hidden_dim), nn.ELU()])
            in_dim = cfg.hidden_dim
        timing_layers.append(nn.Linear(in_dim, 2))  # mu, log_sigma
        self.timing_head = nn.Sequential(*timing_layers)

    # -- helpers --

    def _reshape_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """(*batch, z_dim) → (*batch, stoch_dim, num_classes)."""
        *batch, _ = logits.shape
        return logits.view(*batch, self.cfg.stoch_dim, self.cfg.num_classes)

    def _sample_categorical(self, logits: torch.Tensor) -> torch.Tensor:
        """Gumbel-softmax straight-through sample from reshaped logits.

        Args:
            logits: shape ``(*batch, stoch_dim, num_classes)``

        Returns:
            One-hot sample, same shape, with straight-through gradient.
        """
        *batch, S, C = logits.shape
        flat = logits.reshape(-1, C)
        sample = F.gumbel_softmax(flat, tau=self.cfg.gumbel_tau, hard=True)
        return sample.view(*batch, S, C)

    def _flatten_z(self, z: torch.Tensor) -> torch.Tensor:
        """(*batch, stoch_dim, num_classes) → (*batch, z_dim)."""
        *batch, S, C = z.shape
        return z.reshape(*batch, S * C)

    def initial_state(self, batch_size: int = 1) -> Dict[str, torch.Tensor]:
        """Return a zero-initialised latent state dictionary."""
        device = next(self.parameters()).device
        return {
            "h": torch.zeros(batch_size, self.cfg.hidden_dim, device=device),
            "z": torch.zeros(
                batch_size, self.cfg.stoch_dim, self.cfg.num_classes, device=device
            ),
        }

    # -- core dynamics --

    def observe(
        self,
        prev_state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        obs: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """One-step posterior update (observation available).

        Returns:
            ``(new_state, prior_logits, posterior_logits)``
        """
        h_prev = prev_state["h"]
        z_prev_flat = self._flatten_z(prev_state["z"])

        # Deterministic step
        gru_in = torch.cat([z_prev_flat, action], dim=-1)
        h = self.gru_cell(gru_in, h_prev)

        # Prior
        prior_logits = self._reshape_logits(self.prior_net(h))

        # Posterior
        post_input = torch.cat([h, obs], dim=-1)
        post_logits = self._reshape_logits(self.posterior_net(post_input))
        z = self._sample_categorical(post_logits)

        new_state = {"h": h, "z": z}
        return new_state, prior_logits, post_logits

    def imagine_step(
        self,
        prev_state: Dict[str, torch.Tensor],
        action: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """One-step prior prediction (no observation — imagination).

        Returns:
            ``(new_state, prior_logits)``
        """
        h_prev = prev_state["h"]
        z_prev_flat = self._flatten_z(prev_state["z"])

        gru_in = torch.cat([z_prev_flat, action], dim=-1)
        h = self.gru_cell(gru_in, h_prev)

        prior_logits = self._reshape_logits(self.prior_net(h))
        z = self._sample_categorical(prior_logits)

        new_state = {"h": h, "z": z}
        return new_state, prior_logits

    def decode(
        self, state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Decode observation, reward, and timing from a latent state.

        Returns:
            ``(obs_hat, reward_hat, (timing_mu, timing_log_sigma))``
        """
        h = state["h"]
        z_flat = self._flatten_z(state["z"])
        feat = torch.cat([h, z_flat], dim=-1)

        obs_hat = self.obs_decoder(feat)
        reward_hat = self.reward_head(feat).squeeze(-1)
        timing_params = self.timing_head(feat)
        timing_mu = timing_params[..., 0]
        timing_log_sigma = timing_params[..., 1]

        return obs_hat, reward_hat, (timing_mu, timing_log_sigma)

    # -- uncertainty computation --

    @staticmethod
    def _categorical_kl(
        logits_p: torch.Tensor, logits_q: torch.Tensor
    ) -> torch.Tensor:
        """KL(Cat(p) || Cat(q)) summed over stoch_dim categoricals.

        Args:
            logits_p: ``(batch, stoch_dim, num_classes)``
            logits_q: ``(batch, stoch_dim, num_classes)``

        Returns:
            Scalar KL per batch element, shape ``(batch,)``.
        """
        p = F.softmax(logits_p, dim=-1) + 1e-8
        q = F.softmax(logits_q, dim=-1) + 1e-8
        kl = (p * (p.log() - q.log())).sum(dim=-1).sum(dim=-1)
        return kl

    @staticmethod
    def _categorical_entropy(logits: torch.Tensor) -> torch.Tensor:
        """Entropy of a categorical distribution, summed over stoch_dim.

        Args:
            logits: ``(batch, stoch_dim, num_classes)``

        Returns:
            Entropy per batch element, shape ``(batch,)``.
        """
        p = F.softmax(logits, dim=-1) + 1e-8
        return -(p * p.log()).sum(dim=-1).sum(dim=-1)

    def compute_uncertainty(
        self,
        state: Dict[str, torch.Tensor],
        obs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute epistemic uncertainty of the current state.

        If ``obs`` is provided, uncertainty is the KL divergence between
        the posterior and the prior (measures how much the observation
        changes the model's belief).  Otherwise, uncertainty is the
        entropy of the prior (measures how spread the model's prediction
        is without grounding).

        Returns:
            Uncertainty scalar(s), shape ``(batch,)``.
        """
        h = state["h"]
        prior_logits = self._reshape_logits(self.prior_net(h))

        if obs is not None:
            post_input = torch.cat([h, obs], dim=-1)
            post_logits = self._reshape_logits(self.posterior_net(post_input))
            return self._categorical_kl(post_logits, prior_logits)
        else:
            return self._categorical_entropy(prior_logits)

    def imagine(
        self,
        state: Dict[str, torch.Tensor],
        horizon: int,
        action_sampler: Optional[Any] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """Rollout the world model forward using the prior only.

        At each step, if ``action_sampler`` is provided it is called
        with the current state to produce an action tensor; otherwise
        a random action is sampled uniformly.

        Returns:
            List of dicts, each with keys ``h``, ``z``, ``reward_hat``,
            ``uncertainty``, ``timing_mu``, ``timing_log_sigma``.
        """
        trajectory: List[Dict[str, torch.Tensor]] = []
        current = state
        batch_size = state["h"].shape[0]
        device = state["h"].device

        for _ in range(horizon):
            if action_sampler is not None:
                action = action_sampler(current)
            else:
                action = torch.zeros(batch_size, self.cfg.action_dim, device=device)
                rand_idx = torch.randint(0, self.cfg.action_dim, (batch_size,))
                action[torch.arange(batch_size), rand_idx] = 1.0

            current, prior_logits = self.imagine_step(current, action)
            _, reward_hat, (t_mu, t_ls) = self.decode(current)
            uncertainty = self._categorical_entropy(prior_logits)

            trajectory.append(
                {
                    "h": current["h"].detach(),
                    "z": current["z"].detach(),
                    "reward_hat": reward_hat.detach(),
                    "uncertainty": uncertainty.detach(),
                    "timing_mu": t_mu.detach(),
                    "timing_log_sigma": t_ls.detach(),
                }
            )

        return trajectory


# ---------------------------------------------------------------------------
# UncertaintyGuidedACT — adaptive computation guided by world model
# ---------------------------------------------------------------------------


class UncertaintyGuidedACT(nn.Module):
    """Adaptive Computation Time agent whose halting schedule is modulated
    by the epistemic uncertainty of a companion world model.

    **Standard ACT** iterates a recurrent cell up to ``max_steps`` times
    per decision, accumulating a halt probability at each step.  The halt
    prior is geometric with parameter ``lambda_geo``.

    **Our extension** adapts two things based on uncertainty ``u``:

    1. **Dynamic step budget**::

           effective_max_steps = base_steps + floor(u * scale_factor)

       (clamped to ``max_thinking_steps``).

    2. **Modulated halting rate**::

           lambda_eff = lambda_geo * (1 - modulation * tanh(u))

       High uncertainty lowers the halting rate, encouraging the agent
       to ponder longer.  Low uncertainty raises it, encouraging early
       termination.

    The agent outputs an action, a value estimate, and a scalar time
    prediction (dt) at each pondering step; the final output is the
    halt-probability-weighted combination.
    """

    HALT_EPS: float = 1e-4

    def __init__(self, cfg: ACTConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Observation encoder
        self.encoder = nn.Sequential(
            nn.Linear(cfg.obs_dim, cfg.encoder_hidden),
            nn.ReLU(),
            nn.Linear(cfg.encoder_hidden, cfg.hidden_dim),
        )

        # Time-aware GRU: input is hidden_dim + 1 (for dt)
        self.gru_cell = nn.GRUCell(cfg.hidden_dim + 1, cfg.hidden_dim)

        # Halt network: takes concatenated [h, obs_encoded] → [0, 1]
        self.halt_net = nn.Sequential(
            nn.Linear(cfg.hidden_dim + cfg.obs_dim, 1),
            nn.Sigmoid(),
        )

        # Policy head
        self.policy_head = nn.Linear(cfg.hidden_dim, cfg.action_dim)

        # Value head
        self.value_head = nn.Linear(cfg.hidden_dim, 1)

        # Time head: predicts dt > 0
        self.time_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        obs: torch.Tensor,
        uncertainty: torch.Tensor,
        dt_input: float = 0.02,
    ) -> Dict[str, torch.Tensor]:
        """Run the uncertainty-guided ACT forward pass.

        Args:
            obs: Observation tensor, shape ``(batch, obs_dim)``.
            uncertainty: World-model uncertainty, shape ``(batch,)`` or
                scalar.  Used to modulate halting schedule.
            dt_input: Physical time-step fed into the time-aware GRU.

        Returns:
            Dictionary with keys:

            - ``action_logits``: ``(batch, action_dim)``
            - ``value``: ``(batch,)``
            - ``dt``: ``(batch,)``
            - ``ponder_cost``: ``(batch,)`` — cumulative KL from geometric
            - ``n_steps``: ``(batch,)`` — effective ponder steps used
            - ``halt_probs``: ``(batch, effective_max_steps)``
        """
        batch_size = obs.shape[0]
        device = obs.device

        # Encode observation
        obs_enc = self.encoder(obs)

        # Compute effective halting parameters
        u = uncertainty.float()
        if u.dim() == 0:
            u = u.unsqueeze(0).expand(batch_size)

        # Dynamic step budget per sample
        extra_steps = (u * self.cfg.uncertainty_scale).floor().int()
        effective_max = torch.clamp(
            self.cfg.base_thinking_steps + extra_steps,
            min=1,
            max=self.cfg.max_thinking_steps,
        )
        global_max = int(effective_max.max().item())

        # Modulated halting rate
        lambda_eff = self.cfg.lambda_geo * (
            1.0 - self.cfg.lambda_uncertainty_modulation * torch.tanh(u)
        )

        # Initialise pondering state
        h = torch.zeros(batch_size, self.cfg.hidden_dim, device=device)
        dt_vec = torch.full((batch_size, 1), dt_input, device=device)

        # Accumulators (halt-probability weighted)
        action_logits_acc = torch.zeros(
            batch_size, self.cfg.action_dim, device=device
        )
        value_acc = torch.zeros(batch_size, device=device)
        dt_acc = torch.zeros(batch_size, device=device)

        cumulative_halt = torch.zeros(batch_size, 1, device=device)
        remainder = torch.ones(batch_size, 1, device=device)
        ponder_cost = torch.zeros(batch_size, device=device)
        halt_probs_list: list[torch.Tensor] = []
        n_steps = torch.zeros(batch_size, device=device)

        for step in range(global_max):
            active = ((step < effective_max) & (remainder.squeeze(-1) > self.HALT_EPS)).float()
            if active.sum() < 1e-6:
                break

            # GRU step
            gru_input = torch.cat([obs_enc, dt_vec], dim=-1)
            h_candidate = self.gru_cell(gru_input, h)
            h = torch.where(active.unsqueeze(-1).bool(), h_candidate, h)

            # Outputs at this step
            step_action = self.policy_head(h)
            step_value = self.value_head(h).squeeze(-1)
            step_dt = self.time_head(h).squeeze(-1)

            # Halt probability
            halt_input = torch.cat([h, obs], dim=-1)
            p_halt = self.halt_net(halt_input).squeeze(-1)

            # Geometric prior for this step: P(halt at step) = lambda * (1-lambda)^step
            geo_prior = lambda_eff * ((1.0 - lambda_eff) ** step)

            lambda_n, cumulative_halt, remainder, _ = apply_act_step(
                cumulative_halt=cumulative_halt,
                remainder=remainder,
                p_halt=p_halt.unsqueeze(-1),
                still_running=active.unsqueeze(-1),
                force_halt=((step + 1) >= effective_max).float().unsqueeze(-1),
                halt_eps=self.HALT_EPS,
            )
            halt_weight = lambda_n.squeeze(-1)
            halt_probs_list.append(lambda_n)

            # Weighted accumulation
            action_logits_acc += halt_weight.unsqueeze(-1) * step_action
            value_acc += halt_weight * step_value
            dt_acc += halt_weight * step_dt

            # Ponder cost: KL(halt_prob || geo_prior) contribution
            # Using the simplified form: halt_prob * log(halt_prob / geo_prior)
            safe_hp = halt_weight.clamp(min=1e-8)
            safe_gp = geo_prior.clamp(min=1e-8)
            ponder_cost += halt_weight * (safe_hp.log() - safe_gp.log())

            n_steps += active

            # Update dt for next step
            next_dt = step_dt.unsqueeze(-1).detach()
            dt_vec = torch.where(active.unsqueeze(-1).bool(), next_dt, dt_vec)

        # Stack halt probs for diagnostics
        halt_probs, _ = stack_halt_weights(
            step_weights=halt_probs_list,
            batch_size=batch_size,
            device=device,
        )

        return {
            "action_logits": action_logits_acc,
            "value": value_acc,
            "dt": dt_acc,
            "ponder_cost": ponder_cost,
            "n_steps": n_steps,
            "halt_probs": halt_probs,
        }


# ---------------------------------------------------------------------------
# Pondering diagnostics
# ---------------------------------------------------------------------------


def compute_pondering_diagnostics(
    halt_probs: torch.Tensor,
    uncertainties: torch.Tensor,
    n_steps: torch.Tensor,
    max_steps: int,
) -> Dict[str, float]:
    """Compute diagnostic statistics for the pondering process.

    Args:
        halt_probs: ``(n_decisions, max_ponder_steps)``
        uncertainties: ``(n_decisions,)``
        n_steps: ``(n_decisions,)`` — effective step counts
        max_steps: The configured maximum thinking steps.

    Returns:
        Dictionary with:
        - ``mean_ponder_depth``: average number of pondering steps
        - ``uncertainty_correlation``: Pearson correlation between
          uncertainty and ponder depth (should be positive)
        - ``halt_entropy``: mean entropy of halt distributions
        - ``efficiency``: 1 - mean_ponder_depth / max_steps
    """
    stats = halt_distribution_stats(halt_probs)
    ponder_depths = stats["expected_steps"].float()
    mean_ponder = ponder_depths.mean().item()
    mean_active_steps = n_steps.float().mean().item() if n_steps.numel() > 0 else 0.0

    # Pearson correlation between uncertainty and ponder depth
    if uncertainties.numel() > 2:
        u = uncertainties.float()
        d = ponder_depths.float()
        u_centered = u - u.mean()
        d_centered = d - d.mean()
        num = (u_centered * d_centered).sum()
        den = (u_centered.norm() * d_centered.norm()).clamp(min=1e-8)
        corr = (num / den).item()
    else:
        corr = 0.0

    entropy = stats["halt_entropy"].mean().item()
    weight_sum_error = (halt_probs.sum(dim=-1) - 1.0).abs().mean().item()

    # Efficiency: less pondering when unnecessary is better
    efficiency = 1.0 - (mean_ponder / max_steps) if max_steps > 0 else 0.0

    return {
        "mean_ponder_depth": mean_ponder,
        "mean_active_steps": mean_active_steps,
        "uncertainty_correlation": corr,
        "halt_entropy": entropy,
        "efficiency": efficiency,
        "weight_sum_error": weight_sum_error,
    }


# ---------------------------------------------------------------------------
# WMGuidedDeliberationExperiment — full experiment runner
# ---------------------------------------------------------------------------


class WMGuidedDeliberationExperiment:
    """End-to-end experiment: train / evaluate a world-model-guided
    deliberation agent on a Gymnasium environment.

    The experiment loop:

    1. Initialise a ``MiniRSSM`` and ``UncertaintyGuidedACT``.
    2. For each episode:
       a. Reset the environment and world model state.
       b. At each step, feed the observation to both the world model
          (posterior update) and the ACT agent (with uncertainty from
          the world model).
       c. Record returns, ponder depths, uncertainties, and timing.
    3. Compute aggregate metrics including the critical
       *uncertainty–ponder correlation* (positive ⇒ the agent is
       pondering more when the world is unpredictable, as intended).
    4. Save results to ``out_dir``.
    """

    def __init__(self, cfg: Optional[ExperimentConfig] = None) -> None:
        self.cfg = cfg or ExperimentConfig()

    def _make_models(
        self,
    ) -> Tuple[MiniRSSM, UncertaintyGuidedACT]:
        """Instantiate fresh world model and ACT agent."""
        rssm_cfg = RSSMConfig(
            obs_dim=self.cfg.obs_dim,
            action_dim=self.cfg.action_dim,
            hidden_dim=self.cfg.hidden_dim,
            stoch_dim=self.cfg.rssm_stoch_dim,
            num_classes=self.cfg.rssm_num_classes,
        )
        act_cfg = ACTConfig(
            obs_dim=self.cfg.obs_dim,
            action_dim=self.cfg.action_dim,
            hidden_dim=self.cfg.hidden_dim,
            max_thinking_steps=self.cfg.max_thinking_steps,
            lambda_geo=self.cfg.lambda_geo,
        )
        rssm = MiniRSSM(rssm_cfg).to(self.cfg.device)
        act = UncertaintyGuidedACT(act_cfg).to(self.cfg.device)
        return rssm, act

    @staticmethod
    def _obs_to_tensor(
        obs: Any, device: str = "cpu"
    ) -> torch.Tensor:
        """Convert a gymnasium observation to a batched float tensor."""
        if isinstance(obs, torch.Tensor):
            t = obs.float()
        else:
            t = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t.to(device)

    @staticmethod
    def _action_to_onehot(
        action: int, action_dim: int, device: str = "cpu"
    ) -> torch.Tensor:
        """Convert a scalar action to a batched one-hot tensor."""
        vec = torch.zeros(1, action_dim, device=device)
        vec[0, action] = 1.0
        return vec

    def run(self, out_dir: Path) -> Dict[str, float]:
        """Execute the full experiment and return metrics.

        Args:
            out_dir: Directory in which to save results JSON.

        Returns:
            Metrics dictionary containing at least:
            ``mean_return``, ``std_return``, ``mean_ponder_depth``,
            ``uncertainty_ponder_correlation``, ``timing_stability``,
            ``efficiency``, ``composite_score``.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Reproducibility
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        random.seed(self.cfg.seed)

        device = self.cfg.device
        rssm, act = self._make_models()
        rssm.eval()
        act.eval()

        # Collection buffers
        episode_returns: List[float] = []
        all_ponder_depths: List[float] = []
        all_uncertainties: List[float] = []
        all_halt_probs: List[torch.Tensor] = []
        all_n_steps: List[torch.Tensor] = []
        all_dts: List[float] = []

        start_time = time.time()

        for ep in range(self.cfg.n_episodes):
            env = gym.make(self.cfg.env_id)
            obs_raw, _ = env.reset(seed=self.cfg.seed + ep)
            obs_t = self._obs_to_tensor(obs_raw, device)

            wm_state = rssm.initial_state(batch_size=1)
            ep_return = 0.0
            prev_action = torch.zeros(1, self.cfg.action_dim, device=device)

            for step_i in range(self.cfg.max_steps):
                with torch.no_grad():
                    # --- World model: posterior update ---
                    wm_state, prior_logits, post_logits = rssm.observe(
                        wm_state, prev_action, obs_t
                    )

                    # --- Uncertainty from world model ---
                    uncertainty = rssm.compute_uncertainty(wm_state, obs=obs_t)

                    # --- Optionally: imagine future for extra uncertainty signal ---
                    if self.cfg.imagination_horizon > 0:
                        traj = rssm.imagine(
                            wm_state, self.cfg.imagination_horizon
                        )
                        imagined_uncertainty = torch.stack(
                            [t["uncertainty"] for t in traj], dim=0
                        ).mean(dim=0)
                        # Blend current and imagined uncertainty
                        uncertainty = 0.5 * uncertainty + 0.5 * imagined_uncertainty

                    # --- ACT forward pass ---
                    act_out = act(obs_t, uncertainty)

                    # Select action (argmax for evaluation)
                    action_logits = act_out["action_logits"]
                    action_idx = action_logits.argmax(dim=-1).item()

                # Record diagnostics
                all_uncertainties.append(uncertainty.item())
                step_diag = compute_pondering_diagnostics(
                    halt_probs=act_out["halt_probs"],
                    uncertainties=uncertainty.reshape(-1),
                    n_steps=act_out["n_steps"],
                    max_steps=self.cfg.max_thinking_steps,
                )
                all_ponder_depths.append(step_diag["mean_ponder_depth"])
                all_halt_probs.append(act_out["halt_probs"].detach().cpu())
                all_n_steps.append(act_out["n_steps"].detach().cpu())
                all_dts.append(act_out["dt"].item())

                # Step the environment
                obs_raw, reward, terminated, truncated, _ = env.step(action_idx)
                ep_return += float(reward)
                obs_t = self._obs_to_tensor(obs_raw, device)
                prev_action = self._action_to_onehot(
                    action_idx, self.cfg.action_dim, device
                )

                if terminated or truncated:
                    break

            episode_returns.append(ep_return)
            env.close()

        wall_time = time.time() - start_time

        # ----- Aggregate metrics -----
        mean_return = statistics.mean(episode_returns)
        std_return = (
            statistics.stdev(episode_returns) if len(episode_returns) > 1 else 0.0
        )
        mean_ponder = statistics.mean(all_ponder_depths) if all_ponder_depths else 0.0

        # Uncertainty-ponder correlation (the key metric)
        if len(all_uncertainties) > 2:
            u_t = torch.tensor(all_uncertainties)
            d_t = torch.tensor(all_ponder_depths)
            u_c = u_t - u_t.mean()
            d_c = d_t - d_t.mean()
            corr_num = (u_c * d_c).sum()
            corr_den = (u_c.norm() * d_c.norm()).clamp(min=1e-8)
            uncertainty_ponder_corr = (corr_num / corr_den).item()
        else:
            uncertainty_ponder_corr = 0.0

        # Timing stability: coefficient of variation of predicted dt
        if len(all_dts) > 1:
            dt_mean = statistics.mean(all_dts)
            dt_std = statistics.stdev(all_dts)
            timing_stability = 1.0 - min(dt_std / max(dt_mean, 1e-8), 1.0)
        else:
            timing_stability = 1.0

        mean_active_steps = (
            statistics.mean(float(x.mean().item()) for x in all_n_steps)
            if all_n_steps
            else 0.0
        )
        mean_weight_sum_error = (
            statistics.mean(
                float((hp.sum(dim=-1) - 1.0).abs().mean().item()) for hp in all_halt_probs
            )
            if all_halt_probs
            else 0.0
        )
        efficiency = (
            1.0 - (mean_ponder / self.cfg.max_thinking_steps)
            if self.cfg.max_thinking_steps > 0
            else 0.0
        )

        # Normalised return: map [0, max_steps] → [0, 1] for CartPole
        norm_return = min(mean_return / self.cfg.max_steps, 1.0)

        # Composite score
        composite_score = (
            0.30 * norm_return
            + 0.25 * max(uncertainty_ponder_corr, 0.0)
            + 0.25 * timing_stability
            + 0.20 * max(efficiency, 0.0)
        )

        metrics: Dict[str, float] = {
            "mean_return": mean_return,
            "std_return": std_return,
            "normalized_return": norm_return,
            "mean_ponder_depth": mean_ponder,
            "mean_active_steps": mean_active_steps,
            "uncertainty_ponder_correlation": uncertainty_ponder_corr,
            "timing_stability": timing_stability,
            "efficiency": efficiency,
            "mean_halt_weight_sum_error": mean_weight_sum_error,
            "composite_score": composite_score,
            "n_episodes": float(self.cfg.n_episodes),
            "total_decisions": float(len(all_uncertainties)),
            "wall_time_seconds": wall_time,
            "mean_uncertainty": (
                statistics.mean(all_uncertainties) if all_uncertainties else 0.0
            ),
            "mean_dt": statistics.mean(all_dts) if all_dts else 0.0,
        }

        # Save results
        results_path = out_dir / "wm_guided_deliberation_results.json"
        with open(results_path, "w") as f:
            json.dump(
                {
                    "config": {
                        k: v
                        for k, v in self.cfg.__dict__.items()
                    },
                    "metrics": metrics,
                },
                f,
                indent=2,
            )

        return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the world-model-guided deliberation experiment from the CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="World Model-Guided Deliberation experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/wm_guided_deliberation"),
        help="Directory for output artefacts.",
    )
    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-thinking-steps", type=int, default=15)
    parser.add_argument("--imagination-horizon", type=int, default=5)
    parser.add_argument("--lambda-geo", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig(
        env_id=args.env_id,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        hidden_dim=args.hidden_dim,
        max_thinking_steps=args.max_thinking_steps,
        imagination_horizon=args.imagination_horizon,
        lambda_geo=args.lambda_geo,
        seed=args.seed,
        device=args.device,
    )

    experiment = WMGuidedDeliberationExperiment(cfg)
    metrics = experiment.run(args.out_dir)

    print("\n=== World Model-Guided Deliberation Results ===")
    for k, v in metrics.items():
        print(f"  {k:>35s}: {v:>10.4f}")
    print(f"\n  Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
