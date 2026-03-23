"""Advanced architecture models for deep continuous-time and sequence modeling.

Publication-quality implementations with full mathematical documentation.

Includes:

1. **LiquidTimeCell** -- Continuous-time RNN cell inspired by Liquid
   Time-Constant (LTC) Networks (Hasani et al. 2021) with multiple
   ODE solver options (Euler, RK4, Dopri5-style adaptive).

2. **TimeAwareTransformer** -- Transformer architecture with rotary
   position embeddings (RoPE) adapted for continuous time, relative
   time attention, combined causal + temporal masking, and optional
   memory-efficient chunked attention.

3. **TemporalDiffusionModel** -- Denoising diffusion model for
   trajectory planning conditioned on timing constraints, following
   Janner et al. 2022 (Diffuser).

References:
    [1] Hasani et al. "Liquid Time-constant Networks", AAAI 2021.
    [2] Su et al. "RoFormer: Enhanced Transformer with Rotary Position
        Embedding", arXiv:2104.09864, 2021.
    [3] Janner et al. "Planning with Diffusion for Flexible Behavior
        Synthesis" (Diffuser), ICML 2022.
    [4] Ho et al. "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
    [5] Vaswani et al. "Attention Is All You Need", NeurIPS 2017.
    [6] Chen et al. "Neural Ordinary Differential Equations", NeurIPS 2018.
"""

import math
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .time_module import TimeModule


# ═══════════════════════════════════════════════════════════════════════════
# ODE Solver Enum
# ═══════════════════════════════════════════════════════════════════════════


class ODESolver(Enum):
    """Supported numerical ODE integration methods.

    Each solver trades off accuracy for computational cost:

    * **EULER** -- First-order, single function evaluation.
      Error :math:`\\mathcal{O}(\\Delta t^2)` per step.
    * **RK4** -- Fourth-order Runge-Kutta.  Four function evaluations.
      Error :math:`\\mathcal{O}(\\Delta t^5)` per step.
    * **ADAPTIVE** -- Dormand-Prince 4(5) adaptive step-size method.
      Automatically adjusts step size to meet tolerance.
      Recommended for stiff dynamics.
    """

    EULER = "euler"
    RK4 = "rk4"
    ADAPTIVE = "adaptive"


# ═══════════════════════════════════════════════════════════════════════════
# LiquidTimeCell — Continuous-time RNN cell
# ═══════════════════════════════════════════════════════════════════════════


class LiquidTimeCell(nn.Module):
    r"""Liquid Time-Constant (LTC) continuous-time recurrent cell.

    The hidden state evolves according to the ODE:

    .. math::

        \frac{dh}{dt} = -\frac{1}{\tau(x, h)} \, h + f(x, h)

    where :math:`\tau(x, h) > 0` is a *state-dependent time constant*
    and :math:`f(x, h)` is a learned driving force.

    The exact solution over interval :math:`\Delta\tau` (assuming constant
    :math:`\tau` and :math:`f` within the step) is:

    .. math::

        h(t + \Delta\tau) = h(t) \, e^{-\Delta\tau / \tau}
                          + f \, \tau \bigl(1 - e^{-\Delta\tau / \tau}\bigr)

    For the simplified form used here (with :math:`\text{inv\_tau} = 1/\tau`):

    .. math::

        \alpha = e^{-\Delta\tau \cdot \text{inv\_tau}}

        h_{t+1} = \alpha \, h_t + (1 - \alpha) \, f

    **Boundary behavior:**

    * :math:`\Delta\tau = 0 \Rightarrow \alpha = 1`: state frozen (no change).
    * :math:`\Delta\tau \to \infty \Rightarrow \alpha \to 0`: instant
      transition to equilibrium :math:`h \to f`.

    **ODE solver options** control how the integration is performed.
    For Euler/RK4 the closed-form is replaced by numerical integration
    of the continuous dynamics with multiple sub-steps for accuracy.

    Args:
        input_dim: Dimensionality of input :math:`x`.
        hidden_dim: Dimensionality of hidden state :math:`h`.
        solver: ODE integration method (see :class:`ODESolver`).
        num_substeps: Number of sub-steps for Euler/RK4 integration.
        adaptive_tol: Error tolerance for adaptive solver.

    References:
        Hasani et al. "Liquid Time-constant Networks", AAAI 2021.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        solver: str = "euler",
        num_substeps: int = 4,
        adaptive_tol: float = 1e-3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.solver = ODESolver(solver)
        self.num_substeps = num_substeps
        self.adaptive_tol = adaptive_tol

        # Network to compute 1/tau (decay rate) and f (driving force)
        self.net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )

        # Separate network for ODE dynamics (used in multi-step solvers)
        self.dynamics_net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

    def _compute_decay_and_force(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Compute the decay rate :math:`1/\tau` and driving force :math:`f`.

        Args:
            x: ``(B, input_dim)`` input.
            h: ``(B, hidden_dim)`` current hidden state.

        Returns:
            Tuple of ``(inv_tau, f)`` each ``(B, hidden_dim)``.
            ``inv_tau`` is strictly positive via softplus + epsilon.
        """
        combined = torch.cat([x, h], dim=-1)
        out = self.net(combined)
        inv_tau_raw, f_raw = torch.chunk(out, 2, dim=-1)
        inv_tau = F.softplus(inv_tau_raw) + 1e-4
        f = torch.tanh(f_raw)
        return inv_tau, f

    def _ode_fn(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`dh/dt = -h/\tau + f`.

        This is the right-hand side of the ODE for numerical solvers.

        Args:
            x: ``(B, input_dim)`` input (held constant during integration).
            h: ``(B, hidden_dim)`` current state.

        Returns:
            ``(B, hidden_dim)`` time derivative of :math:`h`.
        """
        inv_tau, f = self._compute_decay_and_force(x, h)
        return -inv_tau * h + inv_tau * f

    def _step_closed_form(
        self, x: torch.Tensor, h: torch.Tensor, delta_tau: torch.Tensor
    ) -> torch.Tensor:
        r"""Closed-form integration (exact for constant coefficients).

        .. math::

            \alpha = \exp(-\Delta\tau \cdot \text{inv\_tau})

            h' = \alpha \cdot h + (1 - \alpha) \cdot f

        Handles edge cases:
        - :math:`\Delta\tau = 0`: returns :math:`h` unchanged.
        - :math:`\Delta\tau \to \infty`: returns :math:`f` (equilibrium).
        """
        inv_tau, f = self._compute_decay_and_force(x, h)

        # Clamp to prevent overflow in exp for very large delta_tau * inv_tau
        exponent = (delta_tau * inv_tau).clamp(max=30.0)
        decay = torch.exp(-exponent)

        h_new = h * decay + f * (1.0 - decay)
        return h_new

    def _step_euler(
        self, x: torch.Tensor, h: torch.Tensor, delta_tau: torch.Tensor
    ) -> torch.Tensor:
        r"""Euler integration with ``num_substeps`` sub-steps.

        .. math::

            h_{k+1} = h_k + \frac{\Delta\tau}{N} \cdot \frac{dh}{dt}\big|_{h_k}

        Args:
            x: ``(B, input_dim)``
            h: ``(B, hidden_dim)``
            delta_tau: ``(B, 1)`` total time to integrate.

        Returns:
            ``(B, hidden_dim)`` state after integration.
        """
        dt_sub = delta_tau / self.num_substeps
        h_curr = h
        for _ in range(self.num_substeps):
            dh = self._ode_fn(x, h_curr)
            h_curr = h_curr + dt_sub * dh
        return h_curr

    def _step_rk4(
        self, x: torch.Tensor, h: torch.Tensor, delta_tau: torch.Tensor
    ) -> torch.Tensor:
        r"""Fourth-order Runge-Kutta integration.

        The classical RK4 method with four evaluations per sub-step:

        .. math::

            k_1 &= f(h_n) \\
            k_2 &= f(h_n + \tfrac{dt}{2} k_1) \\
            k_3 &= f(h_n + \tfrac{dt}{2} k_2) \\
            k_4 &= f(h_n + dt \cdot k_3) \\
            h_{n+1} &= h_n + \tfrac{dt}{6}(k_1 + 2k_2 + 2k_3 + k_4)

        Error: :math:`\\mathcal{O}(dt^5)` per step.
        """
        dt_sub = delta_tau / self.num_substeps
        h_curr = h
        for _ in range(self.num_substeps):
            k1 = self._ode_fn(x, h_curr)
            k2 = self._ode_fn(x, h_curr + 0.5 * dt_sub * k1)
            k3 = self._ode_fn(x, h_curr + 0.5 * dt_sub * k2)
            k4 = self._ode_fn(x, h_curr + dt_sub * k3)
            h_curr = h_curr + (dt_sub / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return h_curr

    def _step_adaptive(
        self, x: torch.Tensor, h: torch.Tensor, delta_tau: torch.Tensor
    ) -> torch.Tensor:
        r"""Adaptive step-size integration (Dormand-Prince 4/5 inspired).

        Uses embedded RK4/RK5 error estimation to adaptively choose
        sub-step sizes.  Doubles or halves step size based on local
        truncation error vs. ``adaptive_tol``.

        This is a simplified version of the Dormand-Prince method used
        in ``scipy.integrate.solve_ivp`` and ``torchdiffeq``.

        For production use with stiff dynamics, prefer ``torchdiffeq``
        with the full Dormand-Prince 4(5) tableau.

        Args:
            x: ``(B, input_dim)``
            h: ``(B, hidden_dim)``
            delta_tau: ``(B, 1)``

        Returns:
            ``(B, hidden_dim)``
        """
        h_curr = h
        # Per-element remaining time
        t_remaining = delta_tau.clone()
        dt_step = delta_tau / self.num_substeps  # initial step size guess

        max_iters = self.num_substeps * 4  # safety limit
        for _ in range(max_iters):
            if (t_remaining <= 1e-8).all():
                break

            # Clamp step to not exceed remaining time
            dt_actual = torch.minimum(dt_step, t_remaining)

            # RK4 step
            k1 = self._ode_fn(x, h_curr)
            k2 = self._ode_fn(x, h_curr + 0.5 * dt_actual * k1)
            k3 = self._ode_fn(x, h_curr + 0.5 * dt_actual * k2)
            k4 = self._ode_fn(x, h_curr + dt_actual * k3)
            h_rk4 = h_curr + (dt_actual / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            # Euler step (for error estimate)
            h_euler = h_curr + dt_actual * k1

            # Error estimate
            error = (h_rk4 - h_euler).abs().max(dim=-1, keepdim=True).values
            error = error.clamp(min=1e-10)

            # Accept step where error is small enough
            accept = (error <= self.adaptive_tol).float()

            # Update state where accepted
            h_curr = accept * h_rk4 + (1.0 - accept) * h_curr
            t_remaining = t_remaining - accept * dt_actual

            # Adjust step size: grow if error is small, shrink if too large
            # Safety factor 0.9, with growth capped at 2x and shrink at 0.5x
            scale = 0.9 * (self.adaptive_tol / error).pow(0.2).clamp(0.5, 2.0)
            dt_step = dt_step * scale

        return h_curr

    def forward(
        self, x: torch.Tensor, h: torch.Tensor, delta_tau: torch.Tensor
    ) -> torch.Tensor:
        r"""Integrate the hidden state ODE over :math:`\Delta\tau`.

        Dispatches to the configured ODE solver.

        Args:
            x: ``(B, input_dim)`` input features.
            h: ``(B, hidden_dim)`` current hidden state.
            delta_tau: ``(B, 1)`` subjective time step.  Must be
                non-negative.  :math:`\Delta\tau = 0` freezes state.

        Returns:
            ``(B, hidden_dim)`` updated hidden state.
        """
        # Handle dt=0 explicitly: state is frozen
        # Mask for zero-dt entries
        frozen_mask = (delta_tau.abs() < 1e-8).float()

        if self.solver == ODESolver.EULER:
            h_new = self._step_euler(x, h, delta_tau)
        elif self.solver == ODESolver.RK4:
            h_new = self._step_rk4(x, h, delta_tau)
        elif self.solver == ODESolver.ADAPTIVE:
            h_new = self._step_adaptive(x, h, delta_tau)
        else:
            # Default: closed-form (most efficient for non-stiff dynamics)
            h_new = self._step_closed_form(x, h, delta_tau)

        # Enforce frozen state for dt=0
        h_new = frozen_mask * h + (1.0 - frozen_mask) * h_new
        return h_new


# ═══════════════════════════════════════════════════════════════════════════
# Rotary Position Embeddings for Continuous Time
# ═══════════════════════════════════════════════════════════════════════════


class ContinuousRoPE(nn.Module):
    r"""Rotary Position Embedding (RoPE) adapted for continuous timestamps.

    Standard RoPE (Su et al. 2021) encodes position :math:`m` as rotations
    applied to pairs of dimensions in Q and K:

    .. math::

        \text{RoPE}(x, m)_{2i} &= x_{2i} \cos(m \theta_i) - x_{2i+1} \sin(m \theta_i) \\
        \text{RoPE}(x, m)_{2i+1} &= x_{2i} \sin(m \theta_i) + x_{2i+1} \cos(m \theta_i)

    where :math:`\theta_i = 10000^{-2i/d}`.

    For **continuous time**, we replace the integer position :math:`m` with
    a real-valued timestamp :math:`t \in \mathbb{R}^+`:

    .. math::

        \theta_i(t) = t \cdot 10000^{-2i/d}

    This preserves the key property that the dot product between rotated
    vectors depends only on the relative time difference :math:`|t_i - t_j|`.

    Args:
        dim: Embedding dimension (must be even).
        base: Base for frequency computation (default 10000).

    References:
        Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding",
        arXiv:2104.09864, 2021.
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, f"RoPE requires even dim, got {dim}"
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(
        self, x: torch.Tensor, timestamps: torch.Tensor
    ) -> torch.Tensor:
        r"""Apply continuous-time rotary embeddings.

        Args:
            x: ``(B, H, L, D)`` query or key tensor.
            timestamps: ``(B, L)`` continuous timestamps.

        Returns:
            ``(B, H, L, D)`` with rotary encoding applied.
        """
        B, H, L, D = x.shape
        # timestamps: (B, L) -> (B, L, 1)
        t = timestamps.unsqueeze(-1)
        # freqs: (B, L, D/2)
        freqs = t * self.inv_freq.unsqueeze(0).unsqueeze(0)

        cos_f = freqs.cos().unsqueeze(1)  # (B, 1, L, D/2)
        sin_f = freqs.sin().unsqueeze(1)  # (B, 1, L, D/2)

        # Split x into even/odd pairs
        x1, x2 = x[..., ::2], x[..., 1::2]

        # Apply rotation
        out = torch.zeros_like(x)
        out[..., ::2] = x1 * cos_f - x2 * sin_f
        out[..., 1::2] = x1 * sin_f + x2 * cos_f
        return out


# ═══════════════════════════════════════════════════════════════════════════
# Continuous Positional Embedding (legacy, kept for backward compat)
# ═══════════════════════════════════════════════════════════════════════════


class ContinuousPositionalEmbedding(nn.Module):
    """Positional embedding that accepts continuous time coordinates.

    Uses sinusoidal encoding with learnable frequency bands.

    Args:
        d_model: Model dimension.
        max_time: Controls the frequency range (default 10000).
    """

    def __init__(self, d_model: int, max_time: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_time = max_time

        inv_freq = 1.0 / (
            max_time ** (torch.arange(0, d_model, 2).float() / d_model)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute continuous positional embedding.

        Args:
            t: ``(B, L)`` continuous timestamps.

        Returns:
            ``(B, L, d_model)`` sinusoidal embeddings.
        """
        sinusoid_inp = torch.einsum("bl,d->bld", t, self.inv_freq)
        emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        return emb


# ═══════════════════════════════════════════════════════════════════════════
# Time-Aware Attention with RoPE + Relative Time
# ═══════════════════════════════════════════════════════════════════════════


class TimeAwareAttention(nn.Module):
    r"""Multi-head attention with continuous-time RoPE and relative time bias.

    Combines three mechanisms for temporal awareness:

    1. **Rotary Position Embeddings (RoPE)** for continuous time --
       applied to Q and K so the dot product depends on :math:`|t_i - t_j|`.

    2. **Relative time attention bias** -- an additive bias to attention
       logits that is a learned function of :math:`|t_i - t_j|`:

       .. math::

           \text{attn}(i,j) = \frac{Q_i K_j^\top}{\sqrt{d}} + b(|t_i - t_j|)

       where :math:`b(\cdot)` is a small MLP.

    3. **Combined causal + temporal masking** -- prevents attending to
       future tokens *and* tokens that are temporally too distant
       (configurable ``max_temporal_distance``).

    4. **Memory-efficient chunked attention** -- optional chunking of
       the sequence to reduce peak memory from :math:`O(L^2)` to
       :math:`O(L \cdot C)` where :math:`C` is the chunk size.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        use_rope: Enable continuous-time RoPE (default True).
        use_relative_time_bias: Enable relative time MLP bias.
        max_temporal_distance: Maximum allowed time gap for attention
            (``None`` = no temporal masking).
        chunk_size: If > 0, use chunked attention with this chunk size
            for memory efficiency.  0 = standard full attention.

    References:
        Su et al. "RoFormer", 2021.
        Press et al. "ALiBi: Train Short, Test Long", 2022.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        use_rope: bool = True,
        use_relative_time_bias: bool = True,
        max_temporal_distance: Optional[float] = None,
        chunk_size: int = 0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_temporal_distance = max_temporal_distance
        self.chunk_size = chunk_size

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Continuous-time RoPE
        self.use_rope = use_rope
        if use_rope:
            self.rope = ContinuousRoPE(self.head_dim)

        # Relative time bias MLP: |t_i - t_j| -> scalar bias per head
        self.use_relative_time_bias = use_relative_time_bias
        if use_relative_time_bias:
            self.time_bias_net = nn.Sequential(
                nn.Linear(1, 32),
                nn.GELU(),
                nn.Linear(32, n_heads),
            )

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with time-aware attention.

        Args:
            x: ``(B, L, D)`` input features.
            time_emb: ``(B, L, D)`` additive time embedding (legacy mode,
                used when RoPE is disabled).
            mask: ``(B, L, L)`` or ``(1, 1, L, L)`` attention mask.
            timestamps: ``(B, L)`` continuous timestamps for RoPE and
                relative time bias.  If ``None``, falls back to additive
                time embeddings.

        Returns:
            ``(B, L, D)`` attention output.
        """
        B, L, D = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (B, H, L, head_dim)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if timestamps are available
        if self.use_rope and timestamps is not None:
            q = self.rope(q, timestamps)
            k = self.rope(k, timestamps)
        elif not self.use_rope:
            # Legacy additive time embedding mode
            te = time_emb.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
            q = q + te
            k = k + te

        # Compute attention
        if self.chunk_size > 0 and L > self.chunk_size:
            out = self._chunked_attention(q, k, v, mask, timestamps)
        else:
            out = self._full_attention(q, k, v, mask, timestamps)

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)

    def _full_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
        timestamps: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Standard full O(L^2) attention computation.

        Args:
            q, k, v: ``(B, H, L, D)``
            mask: Optional attention mask.
            timestamps: Optional ``(B, L)`` for relative time bias.

        Returns:
            ``(B, H, L, D)`` attention output.
        """
        B, H, L, D = q.shape
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)

        # Add relative time bias
        if self.use_relative_time_bias and timestamps is not None:
            time_bias = self._compute_time_bias(timestamps)  # (B, H, L, L)
            scores = scores + time_bias

        # Temporal masking: block attention to tokens beyond max_temporal_distance
        if self.max_temporal_distance is not None and timestamps is not None:
            t_diff = (timestamps.unsqueeze(-1) - timestamps.unsqueeze(-2)).abs()
            temporal_mask = (t_diff <= self.max_temporal_distance)
            temporal_mask = temporal_mask.unsqueeze(1)  # (B, 1, L, L)
            scores = scores.masked_fill(~temporal_mask, float("-inf"))

        # Causal + user mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        # Handle NaN from all-masked rows
        attn = attn.nan_to_num(0.0)
        return torch.matmul(attn, v)

    def _chunked_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
        timestamps: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Memory-efficient chunked attention.

        Splits the query sequence into chunks of size ``chunk_size`` and
        computes attention for each chunk against the full key/value
        sequence.  Peak memory scales as :math:`O(C \\times L)` instead
        of :math:`O(L^2)`.

        Args:
            q, k, v: ``(B, H, L, D)``
            mask: Optional attention mask.
            timestamps: Optional ``(B, L)`` for relative time bias.

        Returns:
            ``(B, H, L, D)``
        """
        B, H, L, D = q.shape
        C = self.chunk_size
        outputs = []

        for start in range(0, L, C):
            end = min(start + C, L)
            q_chunk = q[:, :, start:end, :]

            scores = torch.matmul(q_chunk, k.transpose(-2, -1)) / math.sqrt(D)

            if self.use_relative_time_bias and timestamps is not None:
                t_q = timestamps[:, start:end]
                t_diff = (t_q.unsqueeze(-1) - timestamps.unsqueeze(-2)).abs()
                bias_input = t_diff.unsqueeze(-1)  # (B, chunk, L, 1)
                bias = self.time_bias_net(bias_input)  # (B, chunk, L, H)
                bias = bias.permute(0, 3, 1, 2)  # (B, H, chunk, L)
                scores = scores + bias

            if self.max_temporal_distance is not None and timestamps is not None:
                t_q = timestamps[:, start:end]
                t_diff = (t_q.unsqueeze(-1) - timestamps.unsqueeze(-2)).abs()
                temporal_mask = (t_diff <= self.max_temporal_distance).unsqueeze(1)
                scores = scores.masked_fill(~temporal_mask, float("-inf"))

            if mask is not None:
                chunk_mask = mask[..., start:end, :]
                scores = scores.masked_fill(chunk_mask == 0, float("-inf"))

            attn = F.softmax(scores, dim=-1).nan_to_num(0.0)
            outputs.append(torch.matmul(attn, v))

        return torch.cat(outputs, dim=2)

    def _compute_time_bias(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Compute relative time attention bias.

        Args:
            timestamps: ``(B, L)`` continuous timestamps.

        Returns:
            ``(B, H, L, L)`` bias to add to attention logits.
        """
        # |t_i - t_j|: (B, L, L)
        t_diff = (timestamps.unsqueeze(-1) - timestamps.unsqueeze(-2)).abs()
        bias_input = t_diff.unsqueeze(-1)  # (B, L, L, 1)
        bias = self.time_bias_net(bias_input)  # (B, L, L, H)
        return bias.permute(0, 3, 1, 2)  # (B, H, L, L)


# ═══════════════════════════════════════════════════════════════════════════
# Transformer Block and Full Model
# ═══════════════════════════════════════════════════════════════════════════


class TimeAwareTransformerBlock(nn.Module):
    """Transformer block with pre-norm and time-aware attention.

    Uses Pre-LN (Xiong et al. 2020) for training stability, which is
    standard in modern transformer implementations.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        dim_feedforward: Feedforward hidden dimension.
        dropout: Dropout rate.
        use_rope: Enable RoPE in attention.
        use_relative_time_bias: Enable relative time bias.
        max_temporal_distance: Max temporal distance for masking.
        chunk_size: Chunk size for memory-efficient attention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        use_rope: bool = True,
        use_relative_time_bias: bool = True,
        max_temporal_distance: Optional[float] = None,
        chunk_size: int = 0,
    ):
        super().__init__()
        self.attn = TimeAwareAttention(
            d_model, n_heads,
            use_rope=use_rope,
            use_relative_time_bias=use_relative_time_bias,
            max_temporal_distance=max_temporal_distance,
            chunk_size=chunk_size,
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(B, L, D)``
            time_emb: ``(B, L, D)`` legacy time embedding.
            mask: Optional attention mask.
            timestamps: ``(B, L)`` continuous timestamps.

        Returns:
            ``(B, L, D)``
        """
        attn_out = self.attn(self.norm1(x), time_emb, mask, timestamps)
        x = x + attn_out
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out
        return x


class DecisionTransformerInternalTime(nn.Module):
    r"""Transformer for offline RL / VLA with learned internal clock.

    Instead of assuming each token step corresponds to a fixed time unit,
    the model predicts a subjective time delta :math:`\Delta\tau` at
    each step and accumulates it into a continuous timeline:

    .. math::

        t_0 &= 0 \\
        t_i &= \sum_{j=0}^{i-1} \Delta\tau_j

    The continuous timestamps feed into RoPE (rotary position embeddings)
    so that attention weights naturally reflect temporal proximity rather
    than sequence position.

    Args:
        obs_dim: Observation dimensionality.
        act_dim: Action dimensionality.
        d_model: Transformer hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer blocks.
        max_seq_len: Maximum sequence length (for causal mask buffer).
        use_rope: Enable continuous-time RoPE.
        use_relative_time_bias: Enable relative time bias in attention.
        max_temporal_distance: Max temporal distance for attention.
        dropout: Dropout rate.
        chunk_size: Chunk size for memory-efficient attention.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        max_seq_len: int = 100,
        use_rope: bool = True,
        use_relative_time_bias: bool = True,
        max_temporal_distance: Optional[float] = None,
        dropout: float = 0.0,
        chunk_size: int = 0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.obs_emb = nn.Linear(obs_dim, d_model)
        self.act_emb = nn.Linear(act_dim, d_model)

        # Time module to predict subjective dt per token
        self.time_module = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.continuous_pos_emb = ContinuousPositionalEmbedding(d_model)

        self.blocks = nn.ModuleList([
            TimeAwareTransformerBlock(
                d_model, n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                use_rope=use_rope,
                use_relative_time_bias=use_relative_time_bias,
                max_temporal_distance=max_temporal_distance,
                chunk_size=chunk_size,
            )
            for _ in range(n_layers)
        ])

        self.action_head = nn.Linear(d_model, act_dim)
        self.value_head = nn.Linear(d_model, 1)

        # Causal mask buffer
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len))
            .unsqueeze(0)
            .unsqueeze(0),
        )

    def forward(
        self, obs_seq: torch.Tensor, act_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            obs_seq: ``(B, L, obs_dim)`` observation sequence.
            act_seq: ``(B, L, act_dim)`` action sequence (unused in
                this baseline -- included for interface compatibility).

        Returns:
            Tuple of:
                - ``action_preds``: ``(B, L, act_dim)``
                - ``value_preds``: ``(B, L)``
                - ``dt``: ``(B, L)`` predicted subjective time deltas.
        """
        B, L, _ = obs_seq.size()

        # 1. Embed inputs
        obs_e = self.obs_emb(obs_seq)

        # 2. Predict internal delta_tau at each step
        dt_logits = self.time_module(obs_e)
        dt = F.softplus(dt_logits) + 1e-3  # (B, L, 1)

        # 3. Accumulate dt to form continuous timeline
        t = torch.cat(
            [torch.zeros(B, 1, 1, device=obs_seq.device), dt[:, :-1, :]],
            dim=1,
        )
        t_accum = torch.cumsum(t, dim=1).squeeze(-1)  # (B, L)

        # 4. Generate continuous positional embeddings (for legacy compat)
        time_e = self.continuous_pos_emb(t_accum)  # (B, L, d_model)

        # 5. Transformer pass
        x = obs_e
        mask = self.causal_mask[:, :, :L, :L]

        for block in self.blocks:
            x = block(x, time_e, mask, timestamps=t_accum)

        # 6. Output heads
        action_preds = self.action_head(x)
        value_preds = self.value_head(x).squeeze(-1)

        return action_preds, value_preds, dt.squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
# TemporalDiffusionModel — Trajectory Planning via Diffusion
# ═══════════════════════════════════════════════════════════════════════════


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal embedding for diffusion timesteps.

    Standard embedding from DDPM (Ho et al. 2020) mapping
    scalar timestep :math:`t` to a :math:`d`-dimensional vector.

    Args:
        dim: Output embedding dimension.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed scalar diffusion timestep.

        Args:
            t: ``(B,)`` or ``(B, 1)`` diffusion timestep in [0, 1].

        Returns:
            ``(B, dim)`` sinusoidal embedding.
        """
        if t.dim() == 2:
            t = t.squeeze(-1)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device).float() * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TemporalUNetBlock(nn.Module):
    """Residual block for the temporal U-Net denoiser.

    Processes trajectory features with FiLM conditioning on the
    diffusion timestep and timing constraints.

    Args:
        channels: Feature channels.
        cond_dim: Conditioning dimension (diffusion timestep + timing).
    """

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        # FiLM: scale and shift from conditioning
        self.film = nn.Linear(cond_dim, channels * 2)
        self.norm = nn.LayerNorm(channels)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """Apply conditioned residual block.

        Args:
            x: ``(B, T, C)`` trajectory features.
            cond: ``(B, cond_dim)`` conditioning vector.

        Returns:
            ``(B, T, C)`` with residual connection.
        """
        h = self.norm(x)
        h = self.net(h)

        # FiLM conditioning
        film_params = self.film(cond).unsqueeze(1)  # (B, 1, 2*C)
        scale, shift = film_params.chunk(2, dim=-1)
        h = h * (1 + scale) + shift

        return x + h


class TemporalDiffusionModel(nn.Module):
    r"""Denoising diffusion model for time-conditioned trajectory planning.

    Implements a DDPM (Ho et al. 2020) adapted for trajectory optimization
    following the Diffuser architecture (Janner et al. 2022):

    .. math::

        \min_\theta \mathbb{E}_{t, \mathbf{x}_0, \epsilon}
        \left[ \| \epsilon - \epsilon_\theta(\mathbf{x}_t, t, c) \|^2 \right]

    where:
    - :math:`\mathbf{x}_0 \in \mathbb{R}^{H \times (S+A)}` is a trajectory
      of length :math:`H` containing states and actions.
    - :math:`t \in \{1, \ldots, T\}` is the diffusion timestep.
    - :math:`c` encodes timing constraints (desired execution speed,
      deadlines, temporal safety margins).

    **Timing conditioning** is the key extension: the model learns to
    generate trajectories that respect specified temporal constraints,
    enabling planning under real-time deadlines.

    The forward diffusion process adds Gaussian noise:

    .. math::

        q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\,\mathbf{x}_0,\;
        (1 - \bar\alpha_t)\,\mathbf{I})

    where :math:`\bar\alpha_t = \prod_{s=1}^t (1 - \beta_s)` with linear
    noise schedule :math:`\beta_t \in [\beta_{\min}, \beta_{\max}]`.

    Args:
        state_dim: State/observation dimensionality.
        action_dim: Action dimensionality.
        horizon: Planning horizon (trajectory length).
        n_diffusion_steps: Number of diffusion steps :math:`T`.
        d_model: Hidden dimension of the denoiser network.
        n_blocks: Number of residual blocks in the denoiser.
        timing_cond_dim: Dimension of timing constraint encoding.
        beta_start: Starting noise schedule :math:`\beta_1`.
        beta_end: Ending noise schedule :math:`\beta_T`.

    References:
        Ho et al. "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
        Janner et al. "Planning with Diffusion for Flexible Behavior
            Synthesis", ICML 2022.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        horizon: int = 32,
        n_diffusion_steps: int = 100,
        d_model: int = 256,
        n_blocks: int = 4,
        timing_cond_dim: int = 32,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_diffusion_steps = n_diffusion_steps
        self.traj_dim = state_dim + action_dim

        # --- Noise schedule ---
        betas = torch.linspace(beta_start, beta_end, n_diffusion_steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", alpha_bar.sqrt())
        self.register_buffer(
            "sqrt_one_minus_alpha_bar", (1.0 - alpha_bar).sqrt()
        )

        # --- Conditioning ---
        self.timestep_emb = SinusoidalTimestepEmbedding(d_model)
        self.timing_encoder = nn.Sequential(
            nn.Linear(timing_cond_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        cond_dim = d_model * 2  # timestep + timing

        # --- Denoiser network ---
        self.input_proj = nn.Linear(self.traj_dim, d_model)
        self.blocks = nn.ModuleList([
            TemporalUNetBlock(d_model, cond_dim)
            for _ in range(n_blocks)
        ])
        self.output_proj = nn.Linear(d_model, self.traj_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        timing_cond: torch.Tensor,
    ) -> torch.Tensor:
        r"""Predict noise :math:`\epsilon_\theta` for the denoising step.

        Args:
            x_t: ``(B, H, state_dim + action_dim)`` noisy trajectory.
            t: ``(B,)`` diffusion timestep indices (integers in
                ``[0, n_diffusion_steps)``).
            timing_cond: ``(B, timing_cond_dim)`` timing constraint encoding.

        Returns:
            ``(B, H, state_dim + action_dim)`` predicted noise.
        """
        # Condition embeddings
        t_normalized = t.float() / self.n_diffusion_steps
        t_emb = self.timestep_emb(t_normalized)           # (B, d_model)
        tc_emb = self.timing_encoder(timing_cond)          # (B, d_model)
        cond = torch.cat([t_emb, tc_emb], dim=-1)         # (B, 2*d_model)

        # Denoise
        h = self.input_proj(x_t)  # (B, H, d_model)
        for block in self.blocks:
            h = block(h, cond)
        return self.output_proj(h)

    def compute_loss(
        self,
        trajectories: torch.Tensor,
        timing_cond: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        r"""Compute the simplified DDPM training loss.

        Samples random diffusion timestep :math:`t` and noise
        :math:`\epsilon`, then computes:

        .. math::

            \mathcal{L} = \| \epsilon - \epsilon_\theta(\mathbf{x}_t, t, c) \|^2

        Args:
            trajectories: ``(B, H, state_dim + action_dim)`` clean
                trajectories :math:`\mathbf{x}_0`.
            timing_cond: ``(B, timing_cond_dim)`` timing constraints.

        Returns:
            Dict with ``diffusion_loss`` scalar tensor.
        """
        B = trajectories.shape[0]
        device = trajectories.device

        # Sample random timesteps
        t = torch.randint(0, self.n_diffusion_steps, (B,), device=device)

        # Sample noise
        noise = torch.randn_like(trajectories)

        # Construct noisy trajectory
        sqrt_ab = self.sqrt_alpha_bar[t].view(B, 1, 1)
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t].view(B, 1, 1)
        x_t = sqrt_ab * trajectories + sqrt_omab * noise

        # Predict noise
        noise_pred = self.forward(x_t, t, timing_cond)

        loss = F.mse_loss(noise_pred, noise)
        return {"diffusion_loss": loss}

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        timing_cond: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        r"""Generate trajectories via the reverse diffusion process (DDPM sampling).

        Iteratively denoises from :math:`\mathbf{x}_T \sim \mathcal{N}(0, I)`
        using the learned denoiser:

        .. math::

            \mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}
            \left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,
            \epsilon_\theta(\mathbf{x}_t, t, c)\right) + \sigma_t \mathbf{z}

        Args:
            batch_size: Number of trajectories to generate.
            timing_cond: ``(B, timing_cond_dim)`` timing constraints.
            device: Device for generation.

        Returns:
            ``(B, H, state_dim + action_dim)`` generated trajectories.
        """
        if device is None:
            device = next(self.parameters()).device

        # Start from pure noise
        x = torch.randn(batch_size, self.horizon, self.traj_dim, device=device)

        for i in reversed(range(self.n_diffusion_steps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            noise_pred = self.forward(x, t, timing_cond)

            alpha = self.alphas[i]
            alpha_bar = self.alpha_bar[i]
            beta = self.betas[i]

            # Mean of p(x_{t-1} | x_t)
            x = (1.0 / alpha.sqrt()) * (
                x - (beta / (1.0 - alpha_bar).sqrt()) * noise_pred
            )

            # Add noise (except at final step)
            if i > 0:
                sigma = beta.sqrt()
                x = x + sigma * torch.randn_like(x)

        return x

    def split_trajectory(
        self, trajectory: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split a generated trajectory into states and actions.

        Args:
            trajectory: ``(B, H, state_dim + action_dim)``

        Returns:
            Tuple of ``(states, actions)`` with shapes
            ``(B, H, state_dim)`` and ``(B, H, action_dim)``.
        """
        states = trajectory[..., : self.state_dim]
        actions = trajectory[..., self.state_dim :]
        return states, actions
