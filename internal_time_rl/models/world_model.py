"""
Temporal World Model -- Dreamer v3 RSSM with Temporal Uncertainty.

Publication-quality implementation of the Recurrent State Space Model (RSSM)
following Hafner et al. 2023 ("Mastering Diverse Domains through World Models",
arXiv:2301.04104) with extensions for temporal (timing) uncertainty.

Architecture:
    Deterministic:  h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})
    Prior:          z_t ~ Cat(logits_p)              where logits_p = f(h_t)
    Posterior:      z_t ~ Cat(logits_q)              where logits_q = g(h_t, o_t)
    Observation:    o_t ~ p(o | h_t, z_t)            decoded via symlog
    Reward:         r_t ~ p(r | h_t, z_t)            decoded via symlog
    Continue:       c_t ~ Bernoulli(p(h_t, z_t))     episode continuation
    Timing:         dt_t ~ LogNormal(mu, sigma)       strictly positive timing

Key Dreamer v3 innovations implemented here:
    1. Categorical latent variables with straight-through gradients
    2. Symlog transform for observation and reward prediction
    3. Continue predictor for proper discounting
    4. KL balancing (replaces free nats / free bits)
    5. Multi-step prediction loss for temporal consistency
    6. LogNormal timing distribution with consistency regularization

Loss = recon_loss + kl_balanced + continue_loss + timing_loss
       + multistep_loss + timing_consistency_loss

References:
    [1] Hafner et al. 2023, "Mastering Diverse Domains through World Models"
    [2] Hafner et al. 2020, "Dream to Control: Learning Behaviors by
        Latent Imagination" (DreamerV1)
    [3] Hafner et al. 2022, "Mastering Atari with Discrete World Models"
        (DreamerV2)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


# ============================================================================
# Symlog / Symexp transforms (Dreamer v3, Hafner et al. 2023, Sec. 3)
# ============================================================================

def symlog(x: torch.Tensor) -> torch.Tensor:
    r"""Symmetric logarithmic transform (Dreamer v3).

    .. math::
        \text{symlog}(x) = \text{sign}(x) \cdot \ln(|x| + 1)

    Compresses the scale of targets so that the model can learn across
    different reward magnitudes without manual normalization. The gradient
    is bounded: :math:`\frac{d}{dx}\text{symlog}(x) = \frac{1}{|x|+1}`.

    Args:
        x: Input tensor of any shape.

    Returns:
        Transformed tensor with compressed scale.

    Reference:
        Hafner et al. 2023, Section 3: Symlog Predictions.
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    r"""Inverse symmetric logarithmic transform (Dreamer v3).

    .. math::
        \text{symexp}(x) = \text{sign}(x) \cdot (\exp(|x|) - 1)

    Used to convert predictions back from symlog-compressed space to
    the original scale.

    Args:
        x: Tensor in symlog space.

    Returns:
        Tensor in original scale.

    Reference:
        Hafner et al. 2023, Section 3: Symlog Predictions.
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


# ============================================================================
# Straight-Through Categorical (Dreamer v3)
# ============================================================================

def _straight_through_categorical(
    logits: torch.Tensor,
    num_categories: int,
    category_dim: int,
) -> torch.Tensor:
    r"""Sample from categorical distribution with straight-through gradients.

    Given logits of shape (..., num_categories * category_dim), reshape to
    (..., num_categories, category_dim), sample one-hot vectors along the
    last dimension, and use straight-through estimator for backpropagation:

    .. math::
        z_{\text{hard}} = \text{one\_hot}(\arg\max(\text{logits}))

    .. math::
        z_{\text{ST}} = z_{\text{hard}} - \text{sg}(\text{softmax}(\text{logits}))
                        + \text{softmax}(\text{logits})

    where :math:`\text{sg}` is stop-gradient. The forward pass uses hard
    samples; the backward pass passes gradients through the softmax.

    Args:
        logits: Raw logits of shape (..., num_categories * category_dim).
        num_categories: Number of categorical distributions (C).
        category_dim: Number of classes per distribution (D).

    Returns:
        Flattened one-hot tensor of shape (..., num_categories * category_dim)
        with straight-through gradients.

    Reference:
        Hafner et al. 2023, Section C: Categorical Latents;
        Bengio et al. 2013, "Estimating or Propagating Gradients Through
        Stochastic Neurons for Conditional Computation".
    """
    batch_shape = logits.shape[:-1]
    # (..., C, D)
    logits_reshaped = logits.view(*batch_shape, num_categories, category_dim)
    # Soft probabilities for gradient
    probs = F.softmax(logits_reshaped, dim=-1)
    # Hard one-hot sample (no gradient)
    indices = torch.argmax(logits_reshaped + _gumbel_noise(logits_reshaped), dim=-1)
    hard = F.one_hot(indices, num_classes=category_dim).float()
    # Straight-through: forward uses hard, backward uses soft
    z = hard - probs.detach() + probs
    # Flatten back to (..., C * D)
    return z.view(*batch_shape, num_categories * category_dim)


def _gumbel_noise(logits: torch.Tensor) -> torch.Tensor:
    """Generate Gumbel(0, 1) noise for categorical sampling.

    .. math::
        g = -\\log(-\\log(u)), \\quad u \\sim \\text{Uniform}(0, 1)

    Args:
        logits: Tensor whose shape and device to match.

    Returns:
        Gumbel noise tensor of same shape.
    """
    u = torch.rand_like(logits).clamp(1e-8, 1.0 - 1e-8)
    return -torch.log(-torch.log(u))


# ============================================================================
# KL divergence for categorical latents
# ============================================================================

def _categorical_kl(
    posterior_logits: torch.Tensor,
    prior_logits: torch.Tensor,
    num_categories: int,
    category_dim: int,
) -> torch.Tensor:
    r"""KL divergence between two categorical distributions.

    .. math::
        D_{\mathrm{KL}}(q \| p) = \sum_{c=1}^{C} \sum_{d=1}^{D}
            q_{c,d} \log \frac{q_{c,d}}{p_{c,d}}

    summed over C independent categorical distributions, each with D classes.

    Args:
        posterior_logits: Logits of shape (..., C * D) for posterior q.
        prior_logits: Logits of shape (..., C * D) for prior p.
        num_categories: Number of categorical distributions (C).
        category_dim: Number of classes per distribution (D).

    Returns:
        KL divergence of shape (...), summed over all categories.

    Reference:
        Hafner et al. 2023, Equation 4.
    """
    batch_shape = posterior_logits.shape[:-1]
    q = F.softmax(
        posterior_logits.view(*batch_shape, num_categories, category_dim),
        dim=-1,
    )
    log_q = F.log_softmax(
        posterior_logits.view(*batch_shape, num_categories, category_dim),
        dim=-1,
    )
    log_p = F.log_softmax(
        prior_logits.view(*batch_shape, num_categories, category_dim),
        dim=-1,
    )
    # Sum over classes D, then sum over categories C
    kl = (q * (log_q - log_p)).sum(dim=-1).sum(dim=-1)
    return kl


# ============================================================================
# TemporalRSSM — Dreamer v3 RSSM with temporal uncertainty
# ============================================================================

class TemporalRSSM(nn.Module):
    r"""Dreamer v3 Recurrent State Space Model with temporal uncertainty.

    This implements the full RSSM from Hafner et al. 2023 with extensions
    for temporal (timing) prediction. Key improvements over the original
    Dreamer v1/v2 RSSM:

    1. **Categorical latent variables** (replaces Gaussian):
       :math:`z_t \sim \text{Cat}(\text{logits})` with shape
       ``(num_categories, category_dim)``. Uses straight-through gradients
       for discrete sampling.

    2. **Symlog transform** for observation and reward prediction,
       stabilizing training across reward scales.

    3. **Continue predictor**: binary classifier
       :math:`c_t \sim \text{Bernoulli}(\sigma(f(h_t, z_t)))` for
       proper discounting during imagination.

    4. **KL balancing** (replaces free nats):
       :math:`\mathcal{L}_{\text{KL}} = \alpha \cdot
       D_{\text{KL}}[\text{sg}(q) \| p]
       + (1 - \alpha) \cdot D_{\text{KL}}[q \| \text{sg}(p)]`
       with :math:`\alpha = 0.8`.

    5. **Multi-step prediction loss**: auxiliary loss that penalizes
       world models accurate at one step but divergent over longer
       horizons.

    6. **LogNormal timing distribution** (strictly positive) with
       consistency regularization.

    7. **Sequence-level metrics** for ELBO decomposition visualization.

    The flattened latent dimension is ``num_categories * category_dim``.
    For backward compatibility, when ``latent_dim`` is provided and
    ``num_categories`` / ``category_dim`` are not, the model falls back
    to a Gaussian latent variable.

    Args:
        obs_dim: Observation space dimensionality.
        act_dim: Action space dimensionality (discrete: n_actions).
        hidden_dim: Deterministic GRU hidden size (default: 200).
        latent_dim: Stochastic latent variable dimensionality.
            Used as ``num_categories * category_dim`` when categorical
            mode is active. When only ``latent_dim`` is given (no
            ``num_categories``/``category_dim``), falls back to Gaussian
            for backward compatibility.
        min_std: Minimum standard deviation for numerical stability
            (used in Gaussian fallback and timing distribution).
        num_categories: Number of categorical distributions for z
            (Dreamer v3 default: 32). If None, uses Gaussian latent.
        category_dim: Number of classes per categorical distribution
            (Dreamer v3 default: 32). If None, uses Gaussian latent.
        kl_balance_alpha: Mixing coefficient for KL balancing
            (default: 0.8, as in Dreamer v3). Higher values put more
            weight on training the prior to match the posterior.
        multistep_horizon: Number of steps for multi-step prediction
            loss (default: 5). Set to 0 to disable.
        multistep_weight: Weight for multi-step prediction loss
            (default: 0.5).
        continue_weight: Weight for continue predictor loss
            (default: 1.0, as in Dreamer v3).
        timing_consistency_weight: Weight for penalizing large timing
            changes between consecutive steps (default: 0.1).
        timing_concentration: Concentration parameter for LogNormal
            timing distribution, controlling variance (default: 1.0).
            Lower values = higher variance.
        use_symlog: Whether to apply symlog transform to observations
            and rewards (default: True).

    Reference:
        Hafner et al. 2023, "Mastering Diverse Domains through World Models",
        arXiv:2301.04104.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 200,
        latent_dim: int = 30,
        min_std: float = 0.1,
        *,
        num_categories: Optional[int] = None,
        category_dim: Optional[int] = None,
        kl_balance_alpha: float = 0.8,
        multistep_horizon: int = 5,
        multistep_weight: float = 0.5,
        continue_weight: float = 1.0,
        timing_consistency_weight: float = 0.1,
        timing_concentration: float = 1.0,
        use_symlog: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.min_std = min_std
        self.kl_balance_alpha = kl_balance_alpha
        self.multistep_horizon = multistep_horizon
        self.multistep_weight = multistep_weight
        self.continue_weight = continue_weight
        self.timing_consistency_weight = timing_consistency_weight
        self.timing_concentration = timing_concentration
        self.use_symlog = use_symlog

        # ------------------------------------------------------------------
        # Determine latent mode: categorical (Dreamer v3) vs Gaussian (v1/v2)
        # ------------------------------------------------------------------
        if num_categories is not None and category_dim is not None:
            self._categorical = True
            self.num_categories = num_categories
            self.category_dim = category_dim
            self.latent_dim = num_categories * category_dim
        else:
            # Backward compatibility: Gaussian latent
            self._categorical = False
            self.num_categories = 0
            self.category_dim = 0
            self.latent_dim = latent_dim

        # ------------------------------------------------------------------
        # Deterministic recurrent core
        # h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
        # ------------------------------------------------------------------
        self.recurrent = nn.GRUCell(self.latent_dim + act_dim, hidden_dim)

        # ------------------------------------------------------------------
        # Stochastic latent variable networks
        # ------------------------------------------------------------------
        if self._categorical:
            # Prior p(z|h): outputs logits for C independent Cat(D)
            self.prior_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, self.latent_dim),
            )
            # Posterior q(z|h, o): outputs logits for C independent Cat(D)
            self.posterior_net = nn.Sequential(
                nn.Linear(hidden_dim + obs_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, self.latent_dim),
            )
        else:
            # Gaussian prior p(z|h): outputs [mu, log_std]
            self.prior_net = nn.Linear(hidden_dim, 2 * self.latent_dim)
            # Gaussian posterior q(z|h, o): outputs [mu, log_std]
            self.posterior_net = nn.Linear(
                hidden_dim + obs_dim, 2 * self.latent_dim
            )

        # ------------------------------------------------------------------
        # Decoder heads
        # ------------------------------------------------------------------
        feat_dim = hidden_dim + self.latent_dim

        # Observation decoder: (h, z) -> o_hat (in symlog space if enabled)
        self.obs_decoder = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, obs_dim),
        )

        # Reward decoder: (h, z) -> r_hat (in symlog space if enabled)
        self.reward_decoder = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

        # Continue predictor (Dreamer v3): (h, z) -> logit for P(continue)
        self.continue_decoder = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

        # Timing decoder: (h, z) -> [mu_log_dt, log_sigma_dt]
        # Parameterizes LogNormal(mu, sigma) for strictly positive timing
        self.timing_decoder = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ELU(),
            nn.Linear(64, 2),  # [mu_log, log_sigma]
        )

    # ================================================================
    # State initialization
    # ================================================================

    def initial_state(
        self, batch: int, device: torch.device = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (h_0, z_0) zero initial states.

        Args:
            batch: Batch size B.
            device: Target device (default: cpu).

        Returns:
            Tuple of:
                h: (B, hidden_dim) deterministic state, zeros.
                z: (B, latent_dim) stochastic state, zeros.
                   For categorical mode, latent_dim = num_categories * category_dim.
        """
        d = device or torch.device("cpu")
        h = torch.zeros(batch, self.hidden_dim, device=d)
        z = torch.zeros(batch, self.latent_dim, device=d)
        return h, z

    # ================================================================
    # Prior and Posterior distributions
    # ================================================================

    def _prior(self, h: torch.Tensor) -> Union[Normal, torch.Tensor]:
        r"""Prior distribution p(z|h).

        **Categorical mode** (Dreamer v3):
            Returns raw logits of shape (B, num_categories * category_dim).
            The prior is uniform: logits are zero everywhere, meaning
            :math:`p(z_t | h_t) = \text{Cat}(f(h_t))` where f is a learned
            network. During KL computation, we compare against a uniform
            prior.

        **Gaussian mode** (backward compat):
            Returns Normal(mu, std) where (mu, std) = f(h_t).

        Args:
            h: (B, hidden_dim) deterministic hidden state.

        Returns:
            Normal distribution (Gaussian mode) or logits tensor
            (categorical mode).
        """
        if self._categorical:
            return self.prior_net(h)  # (B, C * D) logits
        else:
            params = self.prior_net(h)
            mu, log_std = params.chunk(2, dim=-1)
            std = F.softplus(log_std) + self.min_std
            return Normal(mu, std)

    def _posterior(
        self, h: torch.Tensor, obs: torch.Tensor
    ) -> Union[Normal, torch.Tensor]:
        r"""Posterior distribution q(z|h, o).

        **Categorical mode** (Dreamer v3):
            Returns logits of shape (B, num_categories * category_dim).
            :math:`q(z_t | h_t, o_t) = \text{Cat}(g(h_t, o_t))`.

        **Gaussian mode** (backward compat):
            Returns Normal(mu, std) where (mu, std) = g(h_t, o_t).

        Args:
            h: (B, hidden_dim) deterministic hidden state.
            obs: (B, obs_dim) observation.

        Returns:
            Normal distribution (Gaussian) or logits tensor (categorical).
        """
        if self._categorical:
            return self.posterior_net(
                torch.cat([h, obs], dim=-1)
            )  # (B, C * D) logits
        else:
            params = self.posterior_net(torch.cat([h, obs], dim=-1))
            mu, log_std = params.chunk(2, dim=-1)
            std = F.softplus(log_std) + self.min_std
            return Normal(mu, std)

    def _sample_latent(
        self, dist_or_logits: Union[Normal, torch.Tensor]
    ) -> torch.Tensor:
        r"""Sample latent variable z from prior or posterior.

        **Categorical mode**: uses straight-through gradients.
        **Gaussian mode**: uses reparameterization trick.

        Args:
            dist_or_logits: Normal distribution (Gaussian) or logits
                tensor (categorical).

        Returns:
            z: (B, latent_dim) sampled latent variable.
        """
        if self._categorical:
            return _straight_through_categorical(
                dist_or_logits,
                self.num_categories,
                self.category_dim,
            )
        else:
            return dist_or_logits.rsample()

    # ================================================================
    # Timing distribution (LogNormal)
    # ================================================================

    def _timing_dist(
        self, h: torch.Tensor, z: torch.Tensor
    ) -> Normal:
        r"""Timing distribution p(dt | h, z) as LogNormal.

        We parameterize the **log of timing** as Normal, so that the
        actual timing dt is LogNormal-distributed (strictly positive):

        .. math::
            \log(dt) \sim \mathcal{N}(\mu, \sigma)

        .. math::
            dt = \exp(\log(dt)) > 0

        The returned Normal is over log-timing. To get the timing
        mean and std, use:

        .. math::
            \mathbb{E}[dt] = \exp(\mu + \sigma^2 / 2)

        .. math::
            \text{Var}[dt] = (\exp(\sigma^2) - 1) \exp(2\mu + \sigma^2)

        The concentration parameter controls the prior on sigma:
        lower concentration allows higher variance in timing.

        Args:
            h: (B, hidden_dim) deterministic state.
            z: (B, latent_dim) stochastic state.

        Returns:
            Normal distribution over log-timing. To get actual timing,
            exponentiate samples. The ``.loc`` and ``.scale`` attributes
            correspond to mu and sigma of the LogNormal.

        Note:
            For backward compatibility, ``predict_timing()`` returns
            (mu_dt, sigma_dt) in the *original timing space* (positive),
            not in log-space.
        """
        params = self.timing_decoder(torch.cat([h, z], dim=-1))
        mu_log, log_sigma = params.chunk(2, dim=-1)
        sigma = F.softplus(log_sigma) * self.timing_concentration + self.min_std
        return Normal(mu_log, sigma)

    def _timing_mean_std(
        self, h: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Compute mean and std of timing in original (positive) space.

        From the LogNormal parameterization:

        .. math::
            \mu_{dt} = \exp(\mu + \sigma^2 / 2)

        .. math::
            \sigma_{dt} = \mu_{dt} \sqrt{\exp(\sigma^2) - 1}

        Args:
            h: (B, hidden_dim) deterministic state.
            z: (B, latent_dim) stochastic state.

        Returns:
            Tuple (mu_dt, sigma_dt) each of shape (B, 1), in original
            timing space (positive values).
        """
        log_dist = self._timing_dist(h, z)
        mu_log = log_dist.loc
        sigma_log = log_dist.scale
        # LogNormal mean and std
        mu_dt = torch.exp(mu_log + 0.5 * sigma_log.pow(2))
        var_dt = (torch.exp(sigma_log.pow(2)) - 1.0) * torch.exp(
            2.0 * mu_log + sigma_log.pow(2)
        )
        sigma_dt = torch.sqrt(var_dt + 1e-8)
        return mu_dt, sigma_dt

    # ================================================================
    # KL divergence computation with KL balancing (Dreamer v3)
    # ================================================================

    def _compute_kl(
        self,
        posterior: Union[Normal, torch.Tensor],
        prior: Union[Normal, torch.Tensor],
        kl_weight: float = 1.0,
        free_nats: float = 3.0,
    ) -> torch.Tensor:
        r"""Compute KL divergence loss with KL balancing (Dreamer v3).

        **Categorical mode** — KL balancing (Hafner et al. 2023, Eq. 4):

        .. math::
            \mathcal{L}_{\text{KL}} = \alpha \cdot
            D_{\text{KL}}[\text{sg}(q) \| p]
            + (1 - \alpha) \cdot D_{\text{KL}}[q \| \text{sg}(p)]

        where :math:`\text{sg}` denotes stop-gradient and
        :math:`\alpha = 0.8`. The first term trains the prior to match
        the posterior (dynamics learning). The second term trains the
        posterior to stay close to the prior (regularization).

        **Gaussian mode** — standard KL with free nats (backward compat):

        .. math::
            \mathcal{L}_{\text{KL}} = \max(D_{\text{KL}}[q \| p],
            \text{free\_nats})

        Args:
            posterior: Posterior distribution q(z|h,o). Normal (Gaussian)
                or logits tensor (categorical).
            prior: Prior distribution p(z|h). Normal (Gaussian) or logits
                tensor (categorical).
            kl_weight: Beta weighting (default: 1.0).
            free_nats: Free nats threshold (Gaussian mode only).

        Returns:
            Scalar KL loss.

        Reference:
            Hafner et al. 2023, Section A.1: KL Balancing.
        """
        if self._categorical:
            alpha = self.kl_balance_alpha
            # Uniform prior logits for reference
            uniform_logits = torch.zeros_like(prior)

            # Term 1: train prior to match posterior (stop-grad on posterior)
            kl_prior = _categorical_kl(
                posterior.detach(), prior,
                self.num_categories, self.category_dim,
            )
            # Term 2: train posterior to stay close (stop-grad on prior)
            kl_posterior = _categorical_kl(
                posterior, prior.detach(),
                self.num_categories, self.category_dim,
            )
            kl = alpha * kl_prior + (1.0 - alpha) * kl_posterior
            return kl_weight * kl.mean()
        else:
            # Gaussian mode: standard KL with free nats
            kl = torch.distributions.kl_divergence(posterior, prior).sum(dim=-1)
            kl = torch.clamp(kl, min=free_nats)
            return kl_weight * kl.mean()

    # ================================================================
    # Observation sequence processing (training)
    # ================================================================

    def rssm_observe(
        self,
        obs_seq: torch.Tensor,
        act_seq: torch.Tensor,
        initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, List[Any]]:
        """Process a sequence with posterior inference (used during training).

        Runs the full RSSM forward pass: for each timestep, computes the
        deterministic transition, infers the posterior from the observation,
        samples z, and decodes observations, rewards, continue flags, and
        timing distributions.

        Args:
            obs_seq: (T, B, obs_dim) sequence of observations.
            act_seq: (T, B, act_dim) sequence of one-hot or continuous actions.
            initial_state: Optional (h_0, z_0) tuple. If None, uses zeros.

        Returns:
            Dict with lists of length T:
                priors: list of Normal distributions (Gaussian) or logits
                    tensors (categorical).
                posteriors: list of Normal distributions or logits tensors.
                z_posts: list of sampled posterior z tensors, (B, latent_dim).
                h_dets: list of deterministic hidden states, (B, hidden_dim).
                obs_recon: list of reconstructed observations, (B, obs_dim).
                    In symlog space if use_symlog is True.
                reward_preds: list of predicted rewards, (B, 1).
                    In symlog space if use_symlog is True.
                timing_dists: list of Normal distributions over log-timing.
                continue_logits: list of continue logits, (B, 1).
        """
        T, B = obs_seq.shape[:2]
        device = obs_seq.device

        if initial_state is None:
            h, z = self.initial_state(B, device)
        else:
            h, z = initial_state

        priors: List[Any] = []
        posteriors: List[Any] = []
        z_posts: List[torch.Tensor] = []
        h_dets: List[torch.Tensor] = []
        obs_recons: List[torch.Tensor] = []
        reward_preds: List[torch.Tensor] = []
        timing_dists: List[Normal] = []
        continue_logits: List[torch.Tensor] = []

        for t in range(T):
            obs_t = obs_seq[t]   # (B, obs_dim)
            act_t = act_seq[t]   # (B, act_dim)

            # 1. Deterministic transition: h_t = GRU(h_{t-1}, [z_{t-1}, a])
            h = self.recurrent(torch.cat([z, act_t], dim=-1), h)

            # 2. Prior: p(z_t | h_t)
            prior = self._prior(h)

            # 3. Posterior: q(z_t | h_t, o_t)
            posterior = self._posterior(h, obs_t)

            # 4. Sample from posterior
            z = self._sample_latent(posterior)

            # 5. Decode
            feat = torch.cat([h, z], dim=-1)
            obs_hat = self.obs_decoder(feat)
            r_hat = self.reward_decoder(feat)
            cont_logit = self.continue_decoder(feat)
            dt_dist = self._timing_dist(h, z)

            priors.append(prior)
            posteriors.append(posterior)
            z_posts.append(z)
            h_dets.append(h)
            obs_recons.append(obs_hat)
            reward_preds.append(r_hat)
            timing_dists.append(dt_dist)
            continue_logits.append(cont_logit)

        return {
            "priors": priors,
            "posteriors": posteriors,
            "z_posts": z_posts,
            "h_dets": h_dets,
            "obs_recon": obs_recons,
            "reward_preds": reward_preds,
            "timing_dists": timing_dists,
            "continue_logits": continue_logits,
        }

    # ================================================================
    # Imagination rollout (planning / actor-critic)
    # ================================================================

    def rssm_imagine(
        self,
        initial_h: torch.Tensor,
        initial_z: torch.Tensor,
        horizon: int,
        policy: nn.Module,
    ) -> Dict[str, List[torch.Tensor]]:
        """Multi-step imagination rollout using prior only (no observations).

        Used for temporal planning and actor-critic training in latent space.
        At each step, the policy produces an action from the current feature
        vector, the GRU performs a deterministic transition, and a new
        stochastic state is sampled from the prior.

        Args:
            initial_h: (B, hidden_dim) starting deterministic state.
            initial_z: (B, latent_dim) starting stochastic state.
            horizon: Number of imagination steps.
            policy: Module mapping feature (h, z) -> action distribution.

        Returns:
            Dict with lists of length ``horizon``:
                h_dets: deterministic states.
                z_latents: stochastic states.
                actions: sampled actions.
                reward_preds: predicted rewards (symlog space if enabled).
                timing_mus: timing distribution means (original space).
                timing_stds: timing distribution stds (original space).
                continue_probs: predicted continuation probabilities.
        """
        h, z = initial_h, initial_z
        hs, zs, actions, rewards = [], [], [], []
        timing_mus, timing_stds, continue_probs = [], [], []

        for _ in range(horizon):
            # Get action from policy
            feat = torch.cat([h, z], dim=-1)
            with torch.no_grad():
                try:
                    action_dist = policy(feat)
                    act = action_dist.sample()
                    # One-hot encode if discrete
                    if act.dim() == 1:
                        act_onehot = F.one_hot(
                            act, num_classes=self.act_dim
                        ).float()
                    else:
                        act_onehot = act
                except Exception:
                    act_onehot = torch.zeros(
                        h.shape[0], self.act_dim, device=h.device
                    )
                    act = torch.zeros(
                        h.shape[0], device=h.device, dtype=torch.long
                    )

            # Deterministic transition
            h = self.recurrent(torch.cat([z, act_onehot], dim=-1), h)

            # Sample from prior (no observation available)
            prior = self._prior(h)
            z = self._sample_latent(prior)

            # Decode
            feat = torch.cat([h, z], dim=-1)
            r_hat = self.reward_decoder(feat)
            mu_dt, sigma_dt = self._timing_mean_std(h, z)
            cont_logit = self.continue_decoder(feat)
            cont_prob = torch.sigmoid(cont_logit)

            hs.append(h)
            zs.append(z)
            actions.append(act)
            rewards.append(r_hat)
            timing_mus.append(mu_dt)
            timing_stds.append(sigma_dt)
            continue_probs.append(cont_prob)

        return {
            "h_dets": hs,
            "z_latents": zs,
            "actions": actions,
            "reward_preds": rewards,
            "timing_mus": timing_mus,
            "timing_stds": timing_stds,
            "continue_probs": continue_probs,
        }

    # ================================================================
    # Multi-step prediction loss (temporal consistency)
    # ================================================================

    def _multistep_loss(
        self,
        obs_seq: torch.Tensor,
        act_seq: torch.Tensor,
        h_dets: List[torch.Tensor],
        z_posts: List[torch.Tensor],
    ) -> torch.Tensor:
        r"""Multi-step prediction loss for temporal consistency.

        Penalizes world models that are accurate one-step but diverge
        over longer horizons. Starting from each timestep t, rolls out
        for N steps using only the prior (no observations) and measures
        prediction error against the true future observations.

        .. math::
            \mathcal{L}_{\text{multi}} = \frac{1}{T \cdot N}
            \sum_{t=1}^{T} \sum_{n=1}^{N}
            \| \hat{o}_{t+n} - o_{t+n} \|^2

        where :math:`\hat{o}_{t+n}` is decoded from the prior-only
        rollout starting at time t.

        Args:
            obs_seq: (T, B, obs_dim) observation sequence.
            act_seq: (T, B, act_dim) action sequence.
            h_dets: List of T deterministic states from rssm_observe.
            z_posts: List of T posterior z samples from rssm_observe.

        Returns:
            Scalar multi-step prediction loss.
        """
        T = len(h_dets)
        horizon = min(self.multistep_horizon, T - 1)

        if horizon <= 0:
            return torch.tensor(
                0.0, device=obs_seq.device, requires_grad=True
            )

        total_mse = torch.tensor(0.0, device=obs_seq.device)
        count = 0

        for start_t in range(T - 1):
            h_curr = h_dets[start_t]
            z_curr = z_posts[start_t]
            steps = min(horizon, T - 1 - start_t)

            for n in range(steps):
                future_t = start_t + n + 1
                act_t = act_seq[future_t]

                # Prior-only transition
                h_curr = self.recurrent(
                    torch.cat([z_curr, act_t], dim=-1), h_curr
                )
                prior = self._prior(h_curr)
                z_curr = self._sample_latent(prior)

                # Decode and compare
                feat = torch.cat([h_curr, z_curr], dim=-1)
                obs_pred = self.obs_decoder(feat)

                if self.use_symlog:
                    target = symlog(obs_seq[future_t])
                else:
                    target = obs_seq[future_t]

                total_mse = total_mse + F.mse_loss(obs_pred, target)
                count += 1

        if count == 0:
            return torch.tensor(
                0.0, device=obs_seq.device, requires_grad=True
            )

        return total_mse / count

    # ================================================================
    # Timing consistency loss
    # ================================================================

    def _timing_consistency_loss(
        self,
        timing_dists: List[Normal],
    ) -> torch.Tensor:
        r"""Timing consistency regularization.

        Penalizes large changes in timing distribution parameters between
        consecutive steps. This encourages smooth temporal dynamics:

        .. math::
            \mathcal{L}_{\text{tc}} = \frac{1}{T-1} \sum_{t=1}^{T-1}
            |\mu_t - \mu_{t-1}|^2

        where :math:`\mu_t` is the log-timing mean at step t.

        Args:
            timing_dists: List of T Normal distributions over log-timing.

        Returns:
            Scalar consistency loss.
        """
        if len(timing_dists) < 2:
            device = timing_dists[0].loc.device if timing_dists else "cpu"
            return torch.tensor(0.0, device=device)

        diffs = []
        for t in range(1, len(timing_dists)):
            mu_diff = timing_dists[t].loc - timing_dists[t - 1].loc
            diffs.append(mu_diff.pow(2).mean())

        return torch.stack(diffs).mean()

    # ================================================================
    # Full loss computation
    # ================================================================

    def compute_loss(
        self,
        obs_seq: torch.Tensor,
        act_seq: torch.Tensor,
        dt_seq: Optional[torch.Tensor] = None,
        kl_weight: float = 1.0,
        free_nats: float = 3.0,
        *,
        continue_seq: Optional[torch.Tensor] = None,
        reward_seq: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        r"""Compute full Dreamer v3 RSSM loss.

        Total loss is a weighted sum of all components:

        .. math::
            \mathcal{L} = \mathcal{L}_{\text{recon}}
            + \mathcal{L}_{\text{KL}}
            + \mathcal{L}_{\text{timing}}
            + \mathcal{L}_{\text{continue}}
            + \mathcal{L}_{\text{multistep}}
            + \mathcal{L}_{\text{tc}}

        where:
            - :math:`\mathcal{L}_{\text{recon}}`: observation reconstruction
              (MSE in symlog space if enabled).
            - :math:`\mathcal{L}_{\text{KL}}`: KL divergence with balancing
              (categorical) or free nats (Gaussian).
            - :math:`\mathcal{L}_{\text{timing}}`: timing prediction NLL.
            - :math:`\mathcal{L}_{\text{continue}}`: binary cross-entropy
              for episode continuation prediction.
            - :math:`\mathcal{L}_{\text{multistep}}`: multi-step prediction
              error for temporal consistency.
            - :math:`\mathcal{L}_{\text{tc}}`: timing consistency
              regularization.

        Args:
            obs_seq: (T, B, obs_dim) observation sequence.
            act_seq: (T, B, act_dim) action sequence.
            dt_seq: (T, B, 1) observed timing values; None = skip timing
                loss.
            kl_weight: Beta for KL weighting (default: 1.0).
            free_nats: KL free nats (Gaussian mode only, default: 3.0).
            continue_seq: (T, B, 1) binary continuation labels (1 = episode
                continues, 0 = terminal). If None, continue loss is zero.
            reward_seq: (T, B, 1) reward targets. If None, reward loss is
                zero (only reconstruction loss on observations is computed).

        Returns:
            Dict with scalar tensors:
                total_loss: sum of all components.
                reconstruction_loss: observation reconstruction MSE.
                kl_loss: KL divergence (balanced or free-nats).
                timing_loss: timing NLL.
                continue_loss: continuation BCE.
                multistep_loss: multi-step prediction error.
                timing_consistency_loss: timing smoothness penalty.
                reward_loss: reward prediction MSE (if reward_seq given).

        Reference:
            Hafner et al. 2023, Section A.1: World Model Learning.
        """
        out = self.rssm_observe(obs_seq, act_seq)

        T, B = obs_seq.shape[:2]
        device = obs_seq.device

        # ------------------------------------------------------------------
        # Observation reconstruction loss (symlog space if enabled)
        # ------------------------------------------------------------------
        obs_recon_stack = torch.stack(out["obs_recon"], dim=0)  # (T, B, D)
        if self.use_symlog:
            obs_target = symlog(obs_seq)
        else:
            obs_target = obs_seq
        recon_loss = F.mse_loss(obs_recon_stack, obs_target, reduction="mean")

        # ------------------------------------------------------------------
        # KL divergence: KL balancing (categorical) or free nats (Gaussian)
        # ------------------------------------------------------------------
        kl_vals = []
        for prior, posterior in zip(out["priors"], out["posteriors"]):
            kl_vals.append(
                self._compute_kl(posterior, prior, kl_weight, free_nats)
            )
        kl_loss = torch.stack(kl_vals).mean()

        # ------------------------------------------------------------------
        # Timing prediction loss (NLL under LogNormal)
        # ------------------------------------------------------------------
        if dt_seq is not None:
            timing_nll_vals = []
            for t, dt_dist in enumerate(out["timing_dists"]):
                dt_t = dt_seq[t]  # (B, 1)
                # LogNormal: compute NLL of observed dt under the log-timing
                # Normal distribution. log_prob(log(dt)) - log(dt) gives the
                # LogNormal log-probability.
                log_dt = torch.log(dt_t.clamp(min=1e-6))
                nll = -dt_dist.log_prob(log_dt).mean()
                timing_nll_vals.append(nll)
            timing_loss = torch.stack(timing_nll_vals).mean()
        else:
            timing_loss = torch.tensor(0.0, device=device)

        # ------------------------------------------------------------------
        # Continue predictor loss (Dreamer v3)
        # Binary cross entropy: c_t ~ Bernoulli(sigma(logit))
        # ------------------------------------------------------------------
        if continue_seq is not None:
            cont_logits = torch.stack(
                out["continue_logits"], dim=0
            )  # (T, B, 1)
            continue_loss = self.continue_weight * F.binary_cross_entropy_with_logits(
                cont_logits, continue_seq, reduction="mean"
            )
        else:
            continue_loss = torch.tensor(0.0, device=device)

        # ------------------------------------------------------------------
        # Reward prediction loss (symlog space if enabled)
        # ------------------------------------------------------------------
        if reward_seq is not None:
            reward_preds = torch.stack(
                out["reward_preds"], dim=0
            )  # (T, B, 1)
            if self.use_symlog:
                reward_target = symlog(reward_seq)
            else:
                reward_target = reward_seq
            reward_loss = F.mse_loss(
                reward_preds, reward_target, reduction="mean"
            )
        else:
            reward_loss = torch.tensor(0.0, device=device)

        # ------------------------------------------------------------------
        # Multi-step prediction loss (temporal consistency)
        # ------------------------------------------------------------------
        if self.multistep_horizon > 0 and T > 1:
            multistep_loss = self.multistep_weight * self._multistep_loss(
                obs_seq, act_seq, out["h_dets"], out["z_posts"]
            )
        else:
            multistep_loss = torch.tensor(0.0, device=device)

        # ------------------------------------------------------------------
        # Timing consistency loss
        # ------------------------------------------------------------------
        if self.timing_consistency_weight > 0.0 and len(out["timing_dists"]) > 1:
            tc_loss = (
                self.timing_consistency_weight
                * self._timing_consistency_loss(out["timing_dists"])
            )
        else:
            tc_loss = torch.tensor(0.0, device=device)

        # ------------------------------------------------------------------
        # Total
        # ------------------------------------------------------------------
        total_loss = (
            recon_loss
            + kl_loss
            + timing_loss
            + continue_loss
            + reward_loss
            + multistep_loss
            + tc_loss
        )

        return {
            "total_loss": total_loss,
            "reconstruction_loss": recon_loss,
            "kl_loss": kl_loss,
            "timing_loss": timing_loss,
            "continue_loss": continue_loss,
            "reward_loss": reward_loss,
            "multistep_loss": multistep_loss,
            "timing_consistency_loss": tc_loss,
        }

    # ================================================================
    # Timing prediction (backward-compatible public API)
    # ================================================================

    def predict_timing(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next timing as (mu_dt, sigma_dt) in original space.

        Returns timing mean and standard deviation in the original
        (positive) timing space. This uses the LogNormal parameterization
        internally.

        Args:
            h: (B, hidden_dim) deterministic state.
            z: (B, latent_dim) stochastic state.

        Returns:
            mu_dt: (B, 1) expected next timing (positive).
            sigma_dt: (B, 1) uncertainty about next timing (positive).
        """
        return self._timing_mean_std(h, z)

    # ================================================================
    # Sequence-level metrics (ELBO decomposition for visualization)
    # ================================================================

    def compute_sequence_metrics(
        self,
        obs_seq: torch.Tensor,
        act_seq: torch.Tensor,
        dt_seq: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        r"""Compute per-timestep ELBO decomposition for visualization.

        Returns arrays (one value per timestep) that allow plotting:
        - How KL evolves over the sequence (should be roughly constant
          in a well-trained model, or decrease as the model becomes more
          certain).
        - How reconstruction error evolves (should decrease as the
          model accumulates more context).
        - How timing prediction error evolves.

        Useful for diagnosing:
        - Posterior collapse: KL near zero at all timesteps.
        - Prior mismatch: KL growing over time.
        - Temporal drift: reconstruction error growing over time.

        Args:
            obs_seq: (T, B, obs_dim) observation sequence.
            act_seq: (T, B, act_dim) action sequence.
            dt_seq: (T, B, 1) optional timing targets.

        Returns:
            Dict with:
                per_step_kl: (T,) tensor of KL divergence per timestep,
                    averaged over batch.
                per_step_recon: (T,) tensor of reconstruction MSE per
                    timestep, averaged over batch.
                per_step_timing: (T,) tensor of timing NLL per timestep,
                    averaged over batch (zeros if dt_seq is None).
                mean_kl: scalar mean KL across all timesteps.
                mean_recon: scalar mean reconstruction error.
                mean_timing: scalar mean timing NLL.
                kl_trend: linear slope of KL over timesteps (positive =
                    growing, concerning).
                recon_trend: linear slope of reconstruction error over
                    timesteps (positive = growing, concerning).
        """
        self.eval()
        T, B = obs_seq.shape[:2]
        device = obs_seq.device

        with torch.no_grad():
            out = self.rssm_observe(obs_seq, act_seq)

        per_step_kl = torch.zeros(T, device=device)
        per_step_recon = torch.zeros(T, device=device)
        per_step_timing = torch.zeros(T, device=device)

        for t in range(T):
            # KL divergence
            posterior = out["posteriors"][t]
            prior = out["priors"][t]
            if self._categorical:
                kl_t = _categorical_kl(
                    posterior, prior,
                    self.num_categories, self.category_dim,
                ).mean()
            else:
                kl_t = torch.distributions.kl_divergence(
                    posterior, prior
                ).sum(dim=-1).mean()
            per_step_kl[t] = kl_t

            # Reconstruction
            obs_hat = out["obs_recon"][t]
            if self.use_symlog:
                obs_target = symlog(obs_seq[t])
            else:
                obs_target = obs_seq[t]
            per_step_recon[t] = F.mse_loss(obs_hat, obs_target)

            # Timing
            if dt_seq is not None:
                dt_dist = out["timing_dists"][t]
                log_dt = torch.log(dt_seq[t].clamp(min=1e-6))
                per_step_timing[t] = -dt_dist.log_prob(log_dt).mean()

        # Compute trends (linear regression slope)
        t_idx = torch.arange(T, dtype=torch.float32, device=device)
        t_centered = t_idx - t_idx.mean()
        t_var = (t_centered ** 2).sum().clamp(min=1e-8)

        kl_trend = ((t_centered * (per_step_kl - per_step_kl.mean())).sum()
                     / t_var)
        recon_trend = (
            (t_centered * (per_step_recon - per_step_recon.mean())).sum()
            / t_var
        )

        return {
            "per_step_kl": per_step_kl,
            "per_step_recon": per_step_recon,
            "per_step_timing": per_step_timing,
            "mean_kl": per_step_kl.mean(),
            "mean_recon": per_step_recon.mean(),
            "mean_timing": per_step_timing.mean(),
            "kl_trend": kl_trend,
            "recon_trend": recon_trend,
        }


# ============================================================================
# Backward-compatible alias
# ============================================================================

class TemporalWorldModel(nn.Module):
    """Lightweight deterministic world model (backward compat alias).

    For new code use TemporalRSSM. This class is kept for existing code
    that imports TemporalWorldModel.
    """

    def __init__(self, obs_dim: int, act_dim: int, latent_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, latent_dim), nn.ELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + act_dim, latent_dim), nn.ELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.obs_predictor = nn.Linear(latent_dim, obs_dim)
        self.reward_predictor = nn.Linear(latent_dim, 1)
        # Timing: distribution (mean + log_std) not a point estimate
        self.timing_predictor = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 2),  # [mu_dt, log_std_dt]
        )

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(obs)
        z_next = self.dynamics(torch.cat([z, action], dim=-1))
        obs_pred = self.obs_predictor(z_next)
        reward_pred = self.reward_predictor(z_next)
        timing_params = self.timing_predictor(z_next)
        timing_mu = F.softplus(timing_params[:, :1]) + 0.1
        return obs_pred, reward_pred, timing_mu

    def get_latent(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)
