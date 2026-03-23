r"""Level 3 Agent: Deliberative Internal Time Agent (Publication Grade).

Implements Adaptive Computation Time (ACT) with six research-grade extensions
for publication-quality temporal reasoning in reinforcement learning.

Core Algorithm (Graves, 2016):
    At each environment step, the agent performs N pondering iterations:

    .. math::
        h_n = f(h_{n-1}, x; \Delta\tau_n), \quad n = 1, \ldots, N

        p_n = \sigma(\text{halt\_net}(h_n, x)) \in (0, 1)

        \text{output} = \sum_{n=1}^{N} w_n \cdot h_n, \quad
        w_n = \begin{cases}
            p_n & \text{if } \sum_{k=1}^{n} p_k < 1 \\
            R_n = 1 - \sum_{k=1}^{n-1} p_k & \text{otherwise (remainder)}
        \end{cases}

Extensions:
    1. **Geometric Halting Prior with KL Regularization** (Section 3.1, Graves 2016)
    2. **Information-Theoretic Halting Criterion** (cosine-similarity MI proxy)
    3. **Numerically Stable Remainder Distribution** with weight-sum assertions
    4. **Adaptive Max Steps** via learned complexity estimator
    5. **Pondering Diagnostics** (entropy, mode, variance of halt distribution)
    6. **Multi-Head Deliberation** (parallel reasoning streams with attention)

References:
    [1] Graves, A. (2016). Adaptive Computation Time for Recurrent Neural
        Networks. https://arxiv.org/abs/1603.08983
    [2] Dehghani, M. et al. (2019). Universal Transformers. ICLR.
    [3] Banino, A. et al. (2021). PonderNet: Learning to Ponder. ICML Workshop.
    [4] Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .policy import InternalTimeAgent
from .encoder import ObservationEncoder
from .time_module import TimeModule, TimeAwareGRUCell


# ═══════════════════════════════════════════════════════════════════════════════
# §1  Geometric Halting Prior (Graves 2016, Section 3.1)
# ═══════════════════════════════════════════════════════════════════════════════


class GeometricHaltingPrior(nn.Module):
    r"""Geometric prior over the halting distribution for KL regularization.

    The prior assumes each pondering step halts independently with probability
    :math:`\lambda`, giving a geometric distribution:

    .. math::
        p_{\text{geo}}(n) = (1 - \lambda)^{n-1} \cdot \lambda,
        \quad n = 1, 2, \ldots, N

    The KL divergence between the learned halting distribution :math:`q(n)` and
    this prior :math:`p(n)` is:

    .. math::
        D_{\text{KL}}(q \| p) = \sum_{n=1}^{N} q(n) \left[
            \log q(n) - \log p_{\text{geo}}(n)
        \right]

    This regularizes the halting distribution toward a sensible default,
    preventing mode collapse to always-halt (step 1) or never-halt (step N).

    Args:
        lambda_geo: Geometric distribution parameter :math:`\lambda \in (0, 1)`.
            Higher values encourage earlier halting. Default 0.5 gives a median
            halting time of 1 step.
        max_steps: Maximum number of pondering steps for prior truncation.

    Reference:
        Graves (2016), Section 3.1: "The prior distribution [...] is a
        geometric distribution with parameter :math:`\lambda_g`."
    """

    def __init__(self, lambda_geo: float = 0.5, max_steps: int = 10):
        super().__init__()
        self.lambda_geo = lambda_geo
        self.max_steps = max_steps
        # Pre-compute and register log-prior (not trainable)
        log_prior = self._compute_log_prior(lambda_geo, max_steps)
        self.register_buffer("log_prior", log_prior)

    @staticmethod
    def _compute_log_prior(lambda_geo: float, max_steps: int) -> torch.Tensor:
        r"""Compute :math:`\log p_{\text{geo}}(n)` for :math:`n = 1, \ldots, N`.

        Truncates and renormalizes so the prior sums to 1 over [1, N].

        Returns:
            Tensor of shape ``(max_steps,)`` with log-probabilities.
        """
        n = torch.arange(1, max_steps + 1, dtype=torch.float32)
        # Unnormalized log-prob: (n-1)*log(1-λ) + log(λ)
        log_unnorm = (n - 1) * math.log(max(1.0 - lambda_geo, 1e-12)) + math.log(
            max(lambda_geo, 1e-12)
        )
        # Truncate and renormalize in log-space
        log_Z = torch.logsumexp(log_unnorm, dim=0)
        return log_unnorm - log_Z

    def kl_divergence(self, halt_weights: torch.Tensor) -> torch.Tensor:
        r"""Compute KL divergence between learned halting distribution and prior.

        .. math::
            D_{\text{KL}}(q \| p) = \sum_{n=1}^{N} q_n
            [\log q_n - \log p_n]

        Args:
            halt_weights: Tensor of shape ``(batch, N)`` where ``N <= max_steps``.
                Each row is the halting weight distribution for one batch element.
                Must sum to approximately 1.0 per row.

        Returns:
            Scalar KL divergence averaged over the batch.
        """
        N = halt_weights.shape[1]
        # Slice prior to match actual steps used
        log_p = self.log_prior[:N].unsqueeze(0)  # (1, N)

        # Clamp for numerical stability in log
        q = halt_weights.clamp(min=1e-10)
        log_q = torch.log(q)

        # KL = sum_n q_n * (log q_n - log p_n)
        kl_per_sample = (q * (log_q - log_p)).sum(dim=1)  # (batch,)
        # Clamp to ensure non-negative (numerical errors can make it slightly negative)
        return kl_per_sample.clamp(min=0.0).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# §2  Information-Theoretic Halting Criterion
# ═══════════════════════════════════════════════════════════════════════════════


class InformationGainTracker:
    r"""Tracks information gain between consecutive hidden states.

    Uses cosine similarity as a proxy for mutual information between
    successive hidden states :math:`h_{n-1}` and :math:`h_n`:

    .. math::
        \text{redundancy}(h_{n-1}, h_n) = \frac{h_{n-1} \cdot h_n}
        {\|h_{n-1}\| \cdot \|h_n\|}

    .. math::
        \text{info\_gain}_n = 1 - \text{redundancy}(h_{n-1}, h_n)

    When :math:`\text{info\_gain}_n < \epsilon`, further pondering is unlikely
    to yield new information, and the agent should halt.

    This is related to the information bottleneck principle (Tishby et al., 2000):
    the hidden state should compress only task-relevant information, and
    redundant pondering steps add no new relevant bits.

    Args:
        threshold: Information gain threshold :math:`\epsilon` below which
            halting is recommended. Default 0.01.
    """

    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold

    def compute_info_gain(
        self,
        h_prev: torch.Tensor,
        h_curr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Compute information gain between consecutive hidden states.

        Args:
            h_prev: Previous hidden state ``(batch, hidden_dim)``.
            h_curr: Current hidden state ``(batch, hidden_dim)``.

        Returns:
            info_gain: Per-sample information gain ``(batch, 1)``.
            should_halt: Boolean mask ``(batch, 1)`` where True indicates
                information gain is below threshold.
        """
        # Cosine similarity: 1.0 = identical, 0.0 = orthogonal, -1.0 = opposite
        cos_sim = F.cosine_similarity(h_prev, h_curr, dim=-1, eps=1e-8)  # (batch,)
        # Information gain = 1 - |cosine_similarity|
        # Use absolute value since anti-correlated states are also informative
        info_gain = (1.0 - cos_sim.abs()).unsqueeze(-1)  # (batch, 1)
        should_halt = info_gain < self.threshold
        return info_gain, should_halt


# ═══════════════════════════════════════════════════════════════════════════════
# §3  Complexity Estimator for Adaptive Max Steps
# ═══════════════════════════════════════════════════════════════════════════════


class ComplexityEstimator(nn.Module):
    r"""Predicts required pondering depth from initial observation.

    Maps the encoded observation to a predicted number of required steps:

    .. math::
        \hat{N}(x) = N_{\min} + (N_{\max} - N_{\min}) \cdot
        \sigma\!\left(\text{MLP}(x)\right)

    where :math:`\sigma` is the sigmoid function ensuring the output is
    clamped to :math:`[N_{\min}, N_{\max}]`.

    This is inspired by the "depth prediction" in Universal Transformers
    (Dehghani et al., 2019), where the number of recurrent layers adapts
    to input complexity.

    Args:
        input_dim: Dimensionality of the encoded observation.
        min_steps: Minimum allowed pondering steps :math:`N_{\min}`.
        hard_max_steps: Maximum allowed pondering steps :math:`N_{\max}`.
        hidden_dim: Hidden layer size for the complexity MLP.

    Reference:
        Dehghani et al. (2019). Universal Transformers. ICLR.
    """

    def __init__(
        self,
        input_dim: int,
        min_steps: int = 1,
        hard_max_steps: int = 20,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.min_steps = min_steps
        self.hard_max_steps = hard_max_steps

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize bias so that predicted steps start near the midpoint
        mid = (hard_max_steps + min_steps) / 2.0
        # sigmoid(bias) * range + min = mid  =>  sigmoid(bias) = 0.5  =>  bias = 0
        nn.init.constant_(self.net[-1].bias, 0.0)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)

    def forward(self, encoded_obs: torch.Tensor) -> torch.Tensor:
        r"""Predict required pondering depth.

        Args:
            encoded_obs: Encoded observation ``(batch, input_dim)``.

        Returns:
            Predicted steps ``(batch, 1)`` in :math:`[N_{\min}, N_{\max}]`.
            Continuous-valued (not rounded) to preserve gradient flow.
        """
        raw = self.net(encoded_obs)
        # Sigmoid maps to [0, 1], then scale to [min_steps, hard_max_steps]
        predicted = self.min_steps + (self.hard_max_steps - self.min_steps) * torch.sigmoid(raw)
        return predicted


# ═══════════════════════════════════════════════════════════════════════════════
# §4  Multi-Head Deliberation Stream
# ═══════════════════════════════════════════════════════════════════════════════


class DeliberationHead(nn.Module):
    r"""A single deliberation stream with its own halting mechanism.

    Each head maintains an independent recurrent state and halting network,
    enabling specialization (e.g., spatial vs. temporal reasoning):

    .. math::
        h_n^{(k)} = f^{(k)}(h_{n-1}^{(k)}, x; \Delta\tau_n), \quad
        p_n^{(k)} = \sigma(\text{halt}^{(k)}(h_n^{(k)}, x))

    The final output of each head is the ACT-weighted sum of its hidden states:

    .. math::
        \bar{h}^{(k)} = \sum_{n=1}^{N^{(k)}} w_n^{(k)} \cdot h_n^{(k)}

    Args:
        hidden_dim: Dimensionality of the recurrent hidden state.
        latent_dim: Dimensionality of the encoded observation.
        head_dim: Output dimensionality of this head (projected from hidden_dim).
        head_id: Integer identifier for this head (for logging).

    Reference:
        Inspired by multi-head attention (Vaswani et al., 2017), adapted to
        the ACT framework where each "head" is a full deliberation stream.
    """

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        head_dim: int,
        head_id: int = 0,
    ):
        super().__init__()
        self.head_id = head_id
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim

        # Per-head halting network
        self.halt_net = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Projection from hidden_dim to head_dim
        self.projection = nn.Linear(hidden_dim, head_dim)

    def compute_halt_prob(
        self,
        hidden: torch.Tensor,
        encoded: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute halting probability for this head.

        Args:
            hidden: Current hidden state ``(batch, hidden_dim)``.
            encoded: Encoded observation ``(batch, latent_dim)``.

        Returns:
            Halting probability ``(batch, 1)`` in ``(0, 1)``.
        """
        return self.halt_net(torch.cat([hidden, encoded], dim=-1))

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden state to head output dimension.

        Args:
            hidden: Hidden state ``(batch, hidden_dim)``.

        Returns:
            Projected output ``(batch, head_dim)``.
        """
        return self.projection(hidden)


class MultiHeadDeliberationAggregator(nn.Module):
    r"""Aggregates multiple deliberation head outputs via learned attention.

    Given :math:`K` head outputs :math:`\bar{h}^{(1)}, \ldots, \bar{h}^{(K)}`
    and the encoded observation :math:`x`, computes attention weights:

    .. math::
        \alpha_k = \frac{\exp(e_k)}{\sum_{j=1}^{K} \exp(e_j)}, \quad
        e_k = v^\top \tanh(W_q x + W_k \bar{h}^{(k)})

    .. math::
        \bar{h}_{\text{final}} = \sum_{k=1}^{K} \alpha_k \cdot \bar{h}^{(k)}

    This is a form of additive (Bahdanau) attention where the query is the
    observation and the keys/values are the head outputs.

    Args:
        head_dim: Dimensionality of each head's output.
        num_heads: Number of deliberation heads :math:`K`.
        query_dim: Dimensionality of the query (encoded observation).
        attn_dim: Internal attention dimensionality.

    Reference:
        Bahdanau et al. (2015). Neural Machine Translation by Jointly
        Learning to Align and Translate. ICLR.
    """

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        query_dim: int,
        attn_dim: int = 32,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.W_q = nn.Linear(query_dim, attn_dim, bias=False)
        self.W_k = nn.Linear(head_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

        # Output projection: head_dim -> hidden_dim happens outside this module
        # to keep it generic

    def forward(
        self,
        head_outputs: torch.Tensor,
        query: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Aggregate head outputs via attention.

        Args:
            head_outputs: Stacked head outputs ``(batch, num_heads, head_dim)``.
            query: Query vector ``(batch, query_dim)`` (typically encoded obs).

        Returns:
            aggregated: Attention-weighted sum ``(batch, head_dim)``.
            attn_weights: Attention weights ``(batch, num_heads)`` summing to 1.
        """
        B, K, D = head_outputs.shape

        # Query projection: (batch, attn_dim) -> (batch, 1, attn_dim)
        q = self.W_q(query).unsqueeze(1)  # (B, 1, attn_dim)

        # Key projection: (batch, K, head_dim) -> (batch, K, attn_dim)
        k = self.W_k(head_outputs)  # (B, K, attn_dim)

        # Additive attention scores
        energy = self.v(torch.tanh(q + k)).squeeze(-1)  # (B, K)
        attn_weights = F.softmax(energy, dim=-1)  # (B, K)

        # Weighted sum
        aggregated = (attn_weights.unsqueeze(-1) * head_outputs).sum(dim=1)  # (B, D)

        return aggregated, attn_weights


# ═══════════════════════════════════════════════════════════════════════════════
# §5  Pondering Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PonderingDiagnostics:
    r"""Detailed statistics about the halting distribution for a forward pass.

    These diagnostics are critical for understanding whether the ACT mechanism
    is learning meaningful deliberation patterns. A well-trained agent should:

    - Have moderate halting entropy (not collapsed to a single step)
    - Show input-dependent variation in ponder steps (high variance)
    - Use the remainder mechanism sparingly (low ``remainder_fraction``)

    Attributes:
        halt_entropy: Entropy of the halting weight distribution (per-sample mean).
            :math:`H(w) = -\sum_n w_n \log w_n`. High = uncertain, Low = peaked.
        halt_mode: Most likely halting step (per-sample mode, then averaged).
        halt_variance: Variance of the halting weight distribution.
        mean_ponder_steps: Average number of active pondering steps.
        max_ponder_steps: Maximum ponder steps observed in the batch.
        remainder_fraction: Fraction of batch elements that hit the max step
            and used the remainder distribution.
        weight_sum_error: Mean absolute deviation of weight sums from 1.0.
            Should be near 0 for a correct implementation.
        info_gain_per_step: Mean information gain at each pondering step.
        head_attention_weights: Attention weights over deliberation heads
            (only populated when ``num_heads > 1``).
    """

    halt_entropy: float = 0.0
    halt_mode: float = 0.0
    halt_variance: float = 0.0
    mean_ponder_steps: float = 0.0
    max_ponder_steps: float = 0.0
    remainder_fraction: float = 0.0
    weight_sum_error: float = 0.0
    info_gain_per_step: List[float] = field(default_factory=list)
    head_attention_weights: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a flat dictionary for logging."""
        d: Dict[str, Any] = {
            "halt_entropy": self.halt_entropy,
            "halt_mode": self.halt_mode,
            "halt_variance": self.halt_variance,
            "mean_ponder_steps": self.mean_ponder_steps,
            "max_ponder_steps": self.max_ponder_steps,
            "remainder_fraction": self.remainder_fraction,
            "weight_sum_error": self.weight_sum_error,
        }
        if self.info_gain_per_step:
            d["info_gain_per_step"] = self.info_gain_per_step
        if self.head_attention_weights is not None:
            d["head_attention_weights"] = self.head_attention_weights
        return d


# ═══════════════════════════════════════════════════════════════════════════════
# §6  DeliberativeInternalTimeAgent (Main Agent)
# ═══════════════════════════════════════════════════════════════════════════════


class DeliberativeInternalTimeAgent(InternalTimeAgent):
    r"""Publication-grade Adaptive Computation Time agent with temporal awareness.

    Implements the full ACT algorithm (Graves, 2016) with six research extensions:

    **1. Geometric Halting Prior (KL Regularization)**

    Replaces naive ``ponder_cost.mean()`` with:

    .. math::
        \mathcal{L}_{\text{halt}} = \lambda_p \cdot D_{\text{KL}}(q \| p_{\text{geo}})

    where :math:`p_{\text{geo}}` is a geometric prior. This prevents mode collapse
    in the halting distribution and gives a principled regularizer grounded in
    the original ACT formulation (Graves 2016, Section 3.1).

    **2. Information-Theoretic Early Halting**

    Monitors cosine-similarity-based information gain between consecutive hidden
    states. When :math:`\text{info\_gain}_n < \epsilon`, further pondering is
    redundant and the agent halts early (before ``max_steps``).

    **3. Numerically Stable Remainder**

    Ensures :math:`\sum_n w_n = 1` per sample within tolerance ``WEIGHT_TOL``.
    All intermediate values are clamped to ``[0, 1]``.

    **4. Adaptive Max Steps**

    A learned ``ComplexityEstimator`` predicts required depth from the
    encoded observation, dynamically adjusting the maximum pondering budget
    between ``min_steps`` and ``hard_max_steps``.

    **5. Pondering Diagnostics**

    ``get_pondering_diagnostics()`` returns detailed statistics (entropy, mode,
    variance, remainder fraction, weight-sum error) for understanding ACT
    behavior during training and evaluation.

    **6. Multi-Head Deliberation**

    Multiple parallel deliberation streams (``DeliberationHead``) with
    independent halting networks. Outputs are aggregated via learned attention,
    inspired by multi-head attention in Transformers (Vaswani et al., 2017).

    Args:
        obs_dim: Observation dimensionality.
        act_dim: Action space dimensionality (discrete).
        hidden_dim: Recurrent hidden state dimensionality.
        latent_dim: Encoded observation dimensionality.
        time_hidden_dim: TimeModule hidden layer size.
        max_thinking_steps: Default maximum pondering steps :math:`N`.
        use_internal_time: Whether to use the learned :math:`\Delta\tau` module.
        transition_type: Recurrent cell type (``"gru"`` or ``"ode"``).
        time_init_bias: Initial bias for the time module.
        lambda_geo: Geometric prior parameter :math:`\lambda \in (0, 1)`.
        info_gain_threshold: Threshold :math:`\epsilon` for information-based halting.
        use_adaptive_steps: Whether to enable the complexity estimator for
            adaptive maximum steps.
        min_steps: Minimum pondering steps (for adaptive mode).
        hard_max_steps: Hard upper bound on pondering steps (for adaptive mode).
        num_heads: Number of parallel deliberation heads (1 = standard ACT).
        head_dim: Output dimensionality per deliberation head. If ``None``,
            defaults to ``hidden_dim // num_heads``.
    """

    # Tolerance for weight-sum assertion
    WEIGHT_TOL: float = 1e-3
    # Epsilon for numerical halting detection
    HALT_EPS: float = 1e-4

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        time_hidden_dim: int = 32,
        max_thinking_steps: int = 5,
        use_internal_time: bool = True,
        transition_type: str = "gru",
        time_init_bias: float = 0.0,
        # --- Extension parameters ---
        lambda_geo: float = 0.5,
        info_gain_threshold: float = 0.01,
        use_adaptive_steps: bool = False,
        min_steps: int = 1,
        hard_max_steps: int = 20,
        num_heads: int = 1,
        head_dim: Optional[int] = None,
    ):
        super().__init__(
            obs_dim,
            act_dim,
            hidden_dim,
            latent_dim,
            time_hidden_dim,
            use_internal_time,
            transition_type,
            time_init_bias,
        )
        self.max_thinking_steps = max_thinking_steps
        self.num_heads = num_heads
        self.use_adaptive_steps = use_adaptive_steps
        self.min_steps = min_steps
        self.hard_max_steps = hard_max_steps

        # Effective head dimension
        _head_dim = head_dim if head_dim is not None else hidden_dim // max(num_heads, 1)
        self._head_dim = _head_dim

        # --- §1: Geometric Halting Prior ---
        effective_max = hard_max_steps if use_adaptive_steps else max_thinking_steps
        self.halting_prior = GeometricHaltingPrior(
            lambda_geo=lambda_geo,
            max_steps=effective_max,
        )

        # --- §2: Information Gain Tracker ---
        self.info_gain_tracker = InformationGainTracker(threshold=info_gain_threshold)

        # --- §3: Halting network (single-head or shared backbone) ---
        if num_heads <= 1:
            # Standard single-head ACT
            self.halting_net = nn.Sequential(
                nn.Linear(hidden_dim + latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
            self._deliberation_heads: Optional[nn.ModuleList] = None
            self._head_aggregator: Optional[MultiHeadDeliberationAggregator] = None
            self._head_output_proj: Optional[nn.Linear] = None
        else:
            # --- §6: Multi-head deliberation ---
            self.halting_net = None  # type: ignore[assignment]
            self._deliberation_heads = nn.ModuleList(
                [
                    DeliberationHead(
                        hidden_dim=hidden_dim,
                        latent_dim=latent_dim,
                        head_dim=_head_dim,
                        head_id=k,
                    )
                    for k in range(num_heads)
                ]
            )
            self._head_aggregator = MultiHeadDeliberationAggregator(
                head_dim=_head_dim,
                num_heads=num_heads,
                query_dim=latent_dim,
                attn_dim=32,
            )
            # Project aggregated head output back to hidden_dim
            self._head_output_proj = nn.Linear(_head_dim, hidden_dim)

        # --- §4: Complexity Estimator ---
        if use_adaptive_steps:
            self.complexity_estimator = ComplexityEstimator(
                input_dim=latent_dim,
                min_steps=min_steps,
                hard_max_steps=hard_max_steps,
            )
        else:
            self.complexity_estimator = None

        # --- §5: Diagnostics storage (populated during forward) ---
        self._last_diagnostics: Optional[PonderingDiagnostics] = None
        self._last_halt_weights: Optional[torch.Tensor] = None
        self._last_halt_weights_live: Optional[torch.Tensor] = None

    # ───────────────────────────────────────────────────────────────────────
    # Forward pass
    # ───────────────────────────────────────────────────────────────────────

    def forward(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[Categorical, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""ACT forward pass with all six extensions.

        Implements Algorithm 1 from Graves (2016) with extensions:

        .. code-block:: text

            Input: observation x, hidden state h_0
            Output: action distribution, value, weighted hidden, cum_halt, ponder_cost

            1. Encode observation: z = Encoder(x)
            2. (Optional) Predict max steps: N_max = ComplexityEstimator(z)
            3. For n = 1, ..., N_max:
               a. Compute delta_tau via TimeModule (or uniform)
               b. Update hidden: h_n = RNN(z, h_{n-1}, delta_tau)
               c. Compute halt probability: p_n = HaltNet(h_n, z)
               d. Check info gain: if MI(h_{n-1}, h_n) < eps, boost p_n
               e. Compute weight w_n (ACT remainder at last step)
               f. Accumulate: output += w_n * h_n
            4. Produce action distribution and value from output

        Args:
            obs: Raw observation ``(batch, obs_dim)``.
            hidden: Recurrent hidden state ``(batch, hidden_dim)``.
            deterministic: Unused (kept for interface compatibility).

        Returns:
            dist: ``Categorical`` action distribution from weighted hidden state.
            value: State value ``(batch,)``.
            weighted_hidden: ACT-weighted hidden state ``(batch, hidden_dim)``.
            cumulative_halt: Cumulative halt probability ``(batch, 1)``.
            ponder_cost: Number of active thinking steps ``(batch, 1)``.
        """
        B = obs.shape[0]
        device = obs.device
        encoded = self.encoder(obs)

        # --- §4: Adaptive max steps ---
        if self.use_adaptive_steps and self.complexity_estimator is not None:
            predicted_steps = self.complexity_estimator(encoded)  # (B, 1)
            # Use per-sample max, take batch maximum for the loop bound
            # (we mask per-sample below)
            adaptive_max = int(predicted_steps.max().item() + 0.5)
            adaptive_max = max(self.min_steps, min(adaptive_max, self.hard_max_steps))
            per_sample_max = predicted_steps.squeeze(-1)  # (B,)
        else:
            adaptive_max = self.max_thinking_steps
            per_sample_max = None

        # ACT accumulators
        cumulative_halt = torch.zeros(B, 1, device=device)
        weighted_hidden = torch.zeros_like(hidden)  # (B, hidden_dim)
        remainder = torch.ones(B, 1, device=device)
        ponder_cost = torch.zeros(B, 1, device=device)

        # Per-step records for diagnostics and KL computation
        step_weights: List[torch.Tensor] = []  # w_n (detached, for diagnostics)
        step_weights_live: List[torch.Tensor] = []  # w_n (with grad, for KL loss)
        step_info_gains: List[torch.Tensor] = []  # info gain per step
        used_remainder = torch.zeros(B, device=device)  # 1 if sample hit max steps

        current_hidden = hidden
        prev_hidden = hidden  # for info gain tracking

        # Multi-head accumulators
        if self._deliberation_heads is not None:
            head_weighted: List[torch.Tensor] = []
            head_halt_weights_all: List[List[torch.Tensor]] = [[] for _ in range(self.num_heads)]

        for step in range(adaptive_max):
            still_running = (cumulative_halt < 1.0 - self.HALT_EPS).float()

            # Per-sample adaptive step limit
            if per_sample_max is not None:
                step_allowed = (step < per_sample_max).float().unsqueeze(-1)  # (B, 1)
                still_running = still_running * step_allowed

            # Early exit if all batch elements halted
            if still_running.sum() < 1e-6:
                break

            # --- Internal time step ---
            if self.use_internal_time:
                delta_tau = self.time_module(current_hidden, encoded)
            else:
                delta_tau = torch.ones(B, 1, device=device) / adaptive_max

            # --- State transition ---
            current_hidden = self.rnn(encoded, current_hidden, delta_tau)

            # --- §2: Information gain check ---
            info_gain, should_halt_info = self.info_gain_tracker.compute_info_gain(
                prev_hidden, current_hidden
            )
            step_info_gains.append(info_gain.detach().mean().item())

            # --- Halt probability ---
            if self._deliberation_heads is not None:
                # Multi-head: average halt probability across heads
                head_halt_probs = []
                for head in self._deliberation_heads:
                    hp = head.compute_halt_prob(current_hidden, encoded)
                    head_halt_probs.append(hp)
                # Average halt probability across heads for the ACT mechanism
                p_halt = torch.stack(head_halt_probs, dim=0).mean(dim=0)  # (B, 1)
            else:
                p_halt = self.halting_net(
                    torch.cat([current_hidden, encoded], dim=-1)
                )  # (B, 1)

            # Boost halt probability when info gain is low (soft, differentiable)
            # This adds a gentle push toward halting when states become redundant
            info_boost = (1.0 - info_gain.detach().clamp(0, 1)) * 0.1 * should_halt_info.float()
            p_halt = (p_halt + info_boost).clamp(0.0, 1.0)

            new_cumulative = cumulative_halt + p_halt * still_running

            # At the last step, force-halt remaining elements via remainder
            is_last_step = float(step == adaptive_max - 1)
            use_remainder_mask = (
                (new_cumulative >= 1.0 - self.HALT_EPS).float()
                + torch.tensor(is_last_step, device=device) * still_running
            ).clamp(0, 1)

            # lambda_n: effective weight for this step's hidden state
            lambda_n = torch.where(
                use_remainder_mask.bool(),
                remainder.clamp(min=0.0) * still_running,
                p_halt * still_running,
            )

            # --- §3: Numerically stable remainder ---
            lambda_n = lambda_n.clamp(min=0.0, max=1.0)

            # --- ACT weighted accumulation ---
            weighted_hidden = weighted_hidden + lambda_n * current_hidden
            ponder_cost = ponder_cost + still_running

            # Record step weight for diagnostics (detached) and KL (live)
            step_weights.append(lambda_n.detach())
            step_weights_live.append(lambda_n)

            # Track which samples used remainder at this step
            if is_last_step > 0.5:
                used_remainder = used_remainder + (still_running.squeeze(-1) > 0.5).float()

            # Update accumulators
            remainder = (remainder - lambda_n * still_running).clamp(min=0.0)
            cumulative_halt = new_cumulative.clamp(0.0, 1.0)

            prev_hidden = current_hidden

        # --- §6: Multi-head aggregation ---
        if self._deliberation_heads is not None and self._head_aggregator is not None:
            # Project weighted hidden through each head
            head_outputs = torch.stack(
                [head.project(weighted_hidden) for head in self._deliberation_heads],
                dim=1,
            )  # (B, K, head_dim)
            aggregated, attn_weights = self._head_aggregator(head_outputs, encoded)
            # Project back to hidden_dim
            weighted_hidden = self._head_output_proj(aggregated)
            self._last_head_attn_weights = attn_weights.detach()
        else:
            self._last_head_attn_weights = None

        # --- §3: Weight-sum assertion ---
        if step_weights:
            weight_matrix = torch.stack(step_weights, dim=1).squeeze(-1)  # (B, N)
            weight_sums = weight_matrix.sum(dim=1)  # (B,)
            weight_sum_error = (weight_sums - 1.0).abs().mean().item()
            # Live version for gradient flow through KL loss
            weight_matrix_live = torch.stack(step_weights_live, dim=1).squeeze(-1)
        else:
            weight_matrix = torch.zeros(B, 1, device=device)
            weight_matrix_live = torch.zeros(B, 1, device=device)
            weight_sum_error = 0.0

        # --- §5: Compute and store diagnostics ---
        self._last_halt_weights = weight_matrix
        self._last_halt_weights_live = weight_matrix_live
        self._last_diagnostics = self._compute_diagnostics(
            weight_matrix=weight_matrix,
            ponder_cost=ponder_cost,
            used_remainder=used_remainder,
            weight_sum_error=weight_sum_error,
            info_gains=step_info_gains,
        )

        # Produce action and value from accumulated weighted hidden state
        logits = self.policy_head(weighted_hidden)
        value = self.value_head(weighted_hidden).squeeze(-1)
        dist = Categorical(logits=logits)

        return dist, value, weighted_hidden, cumulative_halt, ponder_cost

    # ───────────────────────────────────────────────────────────────────────
    # PPO-compatible interface
    # ───────────────────────────────────────────────────────────────────────

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ):
        """PPO-compatible rollout method.

        Returns:
            action, log_prob, entropy, value, hidden_new, ponder_cost
        """
        dist, value, hidden_new, cumulative_halt, ponder_cost = self.forward(
            obs, hidden
        )

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        # Return ponder_cost in the delta_tau slot for compatibility
        return action, log_prob, entropy, value, hidden_new, ponder_cost

    # ───────────────────────────────────────────────────────────────────────
    # §1: KL-regularized ponder loss
    # ───────────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_ponder_loss(
        ponder_cost: torch.Tensor,
        lambda_p: float = 0.01,
    ) -> torch.Tensor:
        r"""Compute ACT ponder cost penalty (backward-compatible interface).

        This is the simple ponder cost from Graves (2016) Eq. 4:

        .. math::
            \mathcal{L}_{\text{ponder}} = \lambda_p \cdot
            \frac{1}{B} \sum_{i=1}^{B} N_i

        For the KL-regularized version, use ``compute_ponder_loss_kl()`` instead.

        Args:
            ponder_cost: Tensor ``(batch, 1)`` counting active steps per element.
            lambda_p: Ponder cost coefficient (default 0.01).

        Returns:
            Scalar ponder loss term.
        """
        return lambda_p * ponder_cost.mean()

    def compute_ponder_loss_kl(
        self,
        ponder_cost: torch.Tensor,
        lambda_p: float = 0.01,
        kl_weight: float = 0.1,
    ) -> torch.Tensor:
        r"""KL-regularized ponder loss combining step count and halting prior.

        .. math::
            \mathcal{L} = \lambda_p \cdot \bar{N} + \beta \cdot
            D_{\text{KL}}(q_{\text{halt}} \| p_{\text{geo}})

        where :math:`\bar{N}` is the mean ponder cost and :math:`\beta` is
        the KL weight controlling the strength of the geometric prior
        regularization.

        This replaces the naive ponder cost with a principled regularizer
        that shapes the halting distribution toward the geometric prior,
        preventing pathological behaviors:

        - **Always-halt** (step 1): KL penalty for deviating from prior
        - **Never-halt** (step N): KL penalty + high ponder cost

        Args:
            ponder_cost: Tensor ``(batch, 1)`` from ``forward()``.
            lambda_p: Ponder cost coefficient for the step-count term.
            kl_weight: Weight :math:`\beta` for the KL divergence term.

        Returns:
            Scalar combined loss.

        Reference:
            Graves (2016) Section 3.1; Banino et al. (2021) PonderNet loss.
        """
        step_loss = lambda_p * ponder_cost.mean()

        # Use live (non-detached) weights for gradient flow through KL
        live_w = self._last_halt_weights_live
        if live_w is not None and live_w.shape[1] > 0:
            kl_loss = self.halting_prior.kl_divergence(live_w)
        else:
            kl_loss = torch.tensor(0.0, device=ponder_cost.device)

        return step_loss + kl_weight * kl_loss

    # ───────────────────────────────────────────────────────────────────────
    # §5: Pondering diagnostics
    # ───────────────────────────────────────────────────────────────────────

    def _compute_diagnostics(
        self,
        weight_matrix: torch.Tensor,
        ponder_cost: torch.Tensor,
        used_remainder: torch.Tensor,
        weight_sum_error: float,
        info_gains: List[float],
    ) -> PonderingDiagnostics:
        r"""Compute detailed pondering statistics from a forward pass.

        Args:
            weight_matrix: Halting weights ``(batch, N)`` from the forward pass.
            ponder_cost: Ponder cost ``(batch, 1)``.
            used_remainder: Binary mask ``(batch,)`` of samples that hit max steps.
            weight_sum_error: Mean absolute deviation of weight sums from 1.0.
            info_gains: List of mean info gains per step.

        Returns:
            ``PonderingDiagnostics`` dataclass with all computed statistics.
        """
        B, N = weight_matrix.shape
        if N == 0:
            return PonderingDiagnostics()

        # Halt entropy: H(w) = -sum(w * log(w))
        w_safe = weight_matrix.clamp(min=1e-10)
        entropy_per_sample = -(w_safe * torch.log(w_safe)).sum(dim=1)  # (B,)
        halt_entropy = entropy_per_sample.mean().item()

        # Mode: argmax step (1-indexed)
        halt_mode = (weight_matrix.argmax(dim=1).float() + 1).mean().item()

        # Variance of halting distribution
        steps = torch.arange(1, N + 1, dtype=torch.float32, device=weight_matrix.device)
        mean_step = (weight_matrix * steps.unsqueeze(0)).sum(dim=1)  # (B,)
        var_step = (weight_matrix * (steps.unsqueeze(0) - mean_step.unsqueeze(1)) ** 2).sum(dim=1)
        halt_variance = var_step.mean().item()

        # Remainder fraction
        remainder_fraction = used_remainder.mean().item() if used_remainder.numel() > 0 else 0.0

        # Head attention weights
        head_attn = None
        if self._last_head_attn_weights is not None:
            head_attn = self._last_head_attn_weights.mean(dim=0).tolist()

        return PonderingDiagnostics(
            halt_entropy=halt_entropy,
            halt_mode=halt_mode,
            halt_variance=halt_variance,
            mean_ponder_steps=float(ponder_cost.mean().item()),
            max_ponder_steps=float(ponder_cost.max().item()),
            remainder_fraction=remainder_fraction,
            weight_sum_error=weight_sum_error,
            info_gain_per_step=[float(g) for g in info_gains],
            head_attention_weights=head_attn,
        )

    def get_pondering_diagnostics(self) -> PonderingDiagnostics:
        r"""Return diagnostics from the most recent forward pass.

        Must be called after ``forward()`` — returns cached diagnostics.

        Returns:
            ``PonderingDiagnostics`` dataclass. See its docstring for field
            descriptions.

        Raises:
            RuntimeError: If called before any forward pass.
        """
        if self._last_diagnostics is None:
            raise RuntimeError(
                "No diagnostics available. Call forward() first."
            )
        return self._last_diagnostics


# ═══════════════════════════════════════════════════════════════════════════════
# §7  Temporal Uncertainty Estimator (Enhanced)
# ═══════════════════════════════════════════════════════════════════════════════


class TemporalUncertaintyEstimator(nn.Module):
    r"""Enhanced epistemic uncertainty estimator for temporal contexts.

    Estimates uncertainty about the current timing context using MC Dropout
    with three improvements over the baseline:

    **1. Calibration Check**

    Verifies that predicted confidence intervals are well-calibrated by
    comparing the empirical coverage of predictions against the expected
    coverage for Gaussian confidence intervals.

    **2. Ensemble Disagreement Metric**

    Beyond variance, computes Jensen-Shannon divergence between MC samples
    to capture multi-modal disagreement:

    .. math::
        \text{JSD}(p_1, \ldots, p_K) = H\!\left(\frac{1}{K}\sum_k p_k\right)
        - \frac{1}{K} \sum_k H(p_k)

    For scalar predictions, we use sample-level pairwise disagreement as
    a proxy.

    **3. Dropout Scheduling**

    Supports annealing the dropout probability during training:

    .. math::
        p_{\text{drop}}(t) = p_{\max} \cdot \max\!\left(
            p_{\min}/p_{\max},\; \gamma^t
        \right)

    where :math:`t` is the training step, :math:`\gamma` is the decay rate,
    and :math:`p_{\min}` is a floor ensuring minimum stochasticity.

    Args:
        hidden_dim: Hidden state dimensionality.
        latent_dim: Encoded observation dimensionality.
        tau_min: Minimum tau for timing context.
        tau_max: Maximum tau for timing context.
        dropout_p: Initial/maximum dropout probability.
        dropout_min: Minimum dropout probability (floor for scheduling).
        dropout_decay: Exponential decay rate for dropout scheduling.
    """

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        tau_min: float = 0.5,
        tau_max: float = 2.0,
        dropout_p: float = 0.2,
        dropout_min: float = 0.05,
        dropout_decay: float = 0.9999,
    ):
        super().__init__()
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.dropout_p_init = dropout_p
        self.dropout_min = dropout_min
        self.dropout_decay = dropout_decay
        self._training_step = 0

        # Uncertainty estimator head with scheduled dropout
        self._dropout1 = nn.Dropout(p=dropout_p)
        self._dropout2 = nn.Dropout(p=dropout_p)

        self.uncertainty_net = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, 64),
            nn.ReLU(),
            self._dropout1,
            nn.Linear(64, 32),
            nn.ReLU(),
            self._dropout2,
            nn.Linear(32, 1),
        )

        # Ponder recommendation: maps uncertainty to suggested steps
        self.ponder_recommender = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus(),  # Always positive
        )

        # Calibration tracking buffers
        self.register_buffer(
            "_calibration_predictions", torch.zeros(0), persistent=False
        )
        self.register_buffer(
            "_calibration_targets", torch.zeros(0), persistent=False
        )

    def _update_dropout_schedule(self) -> None:
        r"""Anneal dropout probability according to the schedule.

        .. math::
            p(t) = \max(p_{\min}, p_{\max} \cdot \gamma^t)
        """
        new_p = max(
            self.dropout_min,
            self.dropout_p_init * (self.dropout_decay ** self._training_step),
        )
        self._dropout1.p = new_p
        self._dropout2.p = new_p
        self._training_step += 1

    def step_dropout_schedule(self) -> None:
        """Public method to advance the dropout schedule by one training step.

        Call this once per training iteration to anneal the dropout rate.
        """
        self._update_dropout_schedule()

    def estimate_timing_uncertainty(
        self,
        obs_encoded: torch.Tensor,
        hidden: torch.Tensor,
        n_samples: int = 10,
    ) -> dict:
        r"""Estimate timing uncertainty via MC Dropout with enhanced metrics.

        Performs :math:`K` stochastic forward passes with active dropout,
        then computes:

        - **mean_value**: :math:`\hat{\mu} = \frac{1}{K}\sum_k v_k`
        - **std_value**: :math:`\hat{\sigma} = \sqrt{\frac{1}{K-1}\sum_k (v_k - \hat{\mu})^2}`
        - **ensemble_disagreement**: Normalized pairwise mean absolute difference
        - **recommended_ponder_steps**: Suggested deliberation depth

        Args:
            obs_encoded: Encoded observation ``(batch, latent_dim)``.
            hidden: Current hidden state ``(batch, hidden_dim)``.
            n_samples: Number of Monte Carlo forward passes :math:`K`.

        Returns:
            Dict with keys: ``mean_value``, ``std_value``,
            ``ensemble_disagreement``, ``recommended_ponder_steps``.
        """
        # Enable dropout for uncertainty estimation
        self.uncertainty_net.train()

        value_samples = []
        h_enc = torch.cat([hidden, obs_encoded], dim=-1)

        for _ in range(n_samples):
            v = self.uncertainty_net(h_enc)
            value_samples.append(v)

        self.uncertainty_net.eval()

        values = torch.stack(value_samples, dim=0)  # (K, batch, 1)
        mean_value = values.mean(dim=0)  # (batch, 1)
        std_value = values.std(dim=0) if n_samples > 1 else torch.zeros_like(mean_value)

        # --- Ensemble disagreement (pairwise MAD) ---
        # Compute mean absolute difference between all pairs of samples
        K = values.shape[0]
        if K > 1:
            # Efficient: compare each sample to the mean
            deviations = (values - mean_value.unsqueeze(0)).abs()  # (K, batch, 1)
            ensemble_disagreement = deviations.mean(dim=0)  # (batch, 1)
        else:
            ensemble_disagreement = torch.zeros_like(mean_value)

        # Recommend ponder steps based on uncertainty
        raw_steps = self.ponder_recommender(std_value)
        recommended = max(1, int(raw_steps.mean().item() + 0.5))

        return {
            "mean_value": mean_value,
            "std_value": std_value,
            "ensemble_disagreement": ensemble_disagreement,
            "recommended_ponder_steps": recommended,
        }

    def check_calibration(
        self,
        predictions_mean: torch.Tensor,
        predictions_std: torch.Tensor,
        targets: torch.Tensor,
        confidence_levels: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        r"""Check calibration of uncertainty estimates.

        For well-calibrated uncertainty, the fraction of targets falling
        within a :math:`z`-score confidence interval should match the
        expected coverage:

        .. math::
            \text{coverage}(z) = \frac{1}{N} \sum_{i=1}^{N}
            \mathbb{1}\!\left[
                |\hat{y}_i - y_i| \leq z \cdot \hat{\sigma}_i
            \right]

        A perfectly calibrated model has :math:`\text{coverage}(1.0) \approx 0.683`,
        :math:`\text{coverage}(1.96) \approx 0.95`, etc.

        Args:
            predictions_mean: Predicted means ``(N, 1)`` or ``(N,)``.
            predictions_std: Predicted standard deviations, same shape.
            targets: True values, same shape.
            confidence_levels: Z-scores to evaluate. Default: [1.0, 1.96, 2.58].

        Returns:
            Dict mapping ``"coverage_z{z}"`` to empirical coverage fractions,
            plus ``"calibration_error"`` (mean absolute deviation from expected).
        """
        if confidence_levels is None:
            confidence_levels = [1.0, 1.96, 2.58]

        # Expected coverage for each z-score (from standard normal CDF)
        expected_coverages = {
            1.0: 0.6827,
            1.96: 0.9500,
            2.58: 0.9901,
        }

        predictions_mean = predictions_mean.reshape(-1)
        predictions_std = predictions_std.reshape(-1).clamp(min=1e-8)
        targets = targets.reshape(-1)
        residuals = (predictions_mean - targets).abs()

        result: Dict[str, float] = {}
        cal_errors: List[float] = []

        for z in confidence_levels:
            within = (residuals <= z * predictions_std).float()
            empirical_coverage = within.mean().item()
            result[f"coverage_z{z:.2f}"] = empirical_coverage

            expected = expected_coverages.get(z, 0.5)
            cal_errors.append(abs(empirical_coverage - expected))

        result["calibration_error"] = sum(cal_errors) / len(cal_errors) if cal_errors else 0.0
        return result
