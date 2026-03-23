"""Continuous-Time Agents and Neural ODE models.

Publication-quality implementations of continuous-time RL agents:

1. **LTCAgent** -- Agent using Liquid Time-Constant (LTC) dynamics for
   handling irregular time steps natively (Hasani et al. 2021).

2. **NeuralODEAgent** -- Agent with Neural ODE dynamics using the adjoint
   sensitivity method for memory-efficient training (Chen et al. 2018).

3. **ContinuousNormalizingFlowTiming** -- Continuous normalizing flow
   (CNF) for modeling the timing distribution p(dt) as an invertible
   transformation of a base distribution (Grathwohl et al. 2019).

References:
    [1] Hasani et al. "Liquid Time-constant Networks", AAAI 2021.
    [2] Chen et al. "Neural Ordinary Differential Equations", NeurIPS 2018.
    [3] Grathwohl et al. "FFJORD: Free-form Continuous Dynamics for
        Scalable Reversible Generative Models", ICLR 2019.
    [4] Pontryagin et al. "Mathematical Theory of Optimal Processes", 1962.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

from .encoder import ObservationEncoder
from .time_module import TimeModule
from .advanced import LiquidTimeCell


# ═══════════════════════════════════════════════════════════════════════════
# Neural ODE utilities
# ═══════════════════════════════════════════════════════════════════════════

# Check for torchdiffeq availability
_HAS_TORCHDIFFEQ = False
try:
    import torchdiffeq  # noqa: F401
    _HAS_TORCHDIFFEQ = True
except ImportError:
    pass


def _euler_integrate(
    func: nn.Module,
    y0: torch.Tensor,
    t_span: torch.Tensor,
    num_steps: int = 10,
) -> torch.Tensor:
    r"""Euler method fallback for ODE integration.

    Integrates :math:`\dot{y} = f(t, y)` from :math:`t_0` to :math:`t_1`
    using ``num_steps`` uniform Euler steps.

    .. math::

        y_{k+1} = y_k + \frac{t_1 - t_0}{N} \cdot f(t_k, y_k)

    Args:
        func: ODE dynamics module with ``forward(t, y) -> dy/dt``.
        y0: ``(B, D)`` initial state.
        t_span: ``(2,)`` tensor ``[t_0, t_1]``.
        num_steps: Number of Euler steps.

    Returns:
        ``(B, D)`` state at :math:`t_1`.
    """
    dt = (t_span[1] - t_span[0]) / num_steps
    y = y0
    t = t_span[0]
    for _ in range(num_steps):
        dy = func(t, y)
        y = y + dt * dy
        t = t + dt
    return y


def _rk4_integrate(
    func: nn.Module,
    y0: torch.Tensor,
    t_span: torch.Tensor,
    num_steps: int = 4,
) -> torch.Tensor:
    r"""Fourth-order Runge-Kutta integration.

    Integrates :math:`\dot{y} = f(t, y)` using the classical RK4 method
    with ``num_steps`` sub-steps for accuracy.

    Error per step: :math:`\mathcal{O}(h^5)`.

    Args:
        func: ODE dynamics module with ``forward(t, y) -> dy/dt``.
        y0: ``(B, D)`` initial state.
        t_span: ``(2,)`` tensor ``[t_0, t_1]``.
        num_steps: Number of RK4 steps.

    Returns:
        ``(B, D)`` state at :math:`t_1`.
    """
    h = (t_span[1] - t_span[0]) / num_steps
    y = y0
    t = t_span[0]
    for _ in range(num_steps):
        k1 = func(t, y)
        k2 = func(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = func(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = func(t + h, y + h * k3)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t = t + h
    return y


def odeint(
    func: nn.Module,
    y0: torch.Tensor,
    t_span: torch.Tensor,
    method: str = "rk4",
    num_steps: int = 10,
) -> torch.Tensor:
    """Unified ODE integration interface.

    Uses ``torchdiffeq.odeint`` if available, otherwise falls back to
    custom Euler or RK4 implementation.

    Args:
        func: ODE dynamics ``f(t, y) -> dy/dt``.
        y0: ``(B, D)`` initial condition.
        t_span: ``(2,)`` integration interval ``[t_0, t_1]``.
        method: Integration method (``"euler"``, ``"rk4"``, ``"dopri5"``).
        num_steps: Number of steps for Euler/RK4 fallback.

    Returns:
        ``(B, D)`` final state at ``t_span[1]``.
    """
    if _HAS_TORCHDIFFEQ and method in ("dopri5", "adams", "adaptive_heun"):
        import torchdiffeq as td
        # torchdiffeq returns (T, B, D); we want the final state
        result = td.odeint(func, y0, t_span, method=method)
        return result[-1]
    elif method == "euler":
        return _euler_integrate(func, y0, t_span, num_steps)
    else:
        return _rk4_integrate(func, y0, t_span, num_steps)


def odeint_adjoint(
    func: nn.Module,
    y0: torch.Tensor,
    t_span: torch.Tensor,
    method: str = "dopri5",
    num_steps: int = 10,
) -> torch.Tensor:
    r"""ODE integration with adjoint sensitivity method for memory-efficient training.

    The adjoint method (Pontryagin 1962, Chen et al. 2018) computes
    gradients by solving an augmented ODE *backward in time*:

    .. math::

        \frac{d\mathbf{a}}{dt} = -\mathbf{a}^\top \frac{\partial f}{\partial \mathbf{y}}

    where :math:`\mathbf{a}(t) = \partial L / \partial \mathbf{y}(t)` is the
    adjoint state.

    **Memory advantage**: The forward pass does not need to store
    intermediate activations, reducing memory from
    :math:`\mathcal{O}(L \times D)` to :math:`\mathcal{O}(D)`.

    Falls back to standard ``odeint`` (with gradient tape) if ``torchdiffeq``
    is not available.

    Args:
        func: ODE dynamics ``f(t, y) -> dy/dt``.
        y0: ``(B, D)`` initial condition.
        t_span: ``(2,)`` integration interval.
        method: Integration method (default ``"dopri5"``).
        num_steps: Steps for fallback integration.

    Returns:
        ``(B, D)`` final state with gradients computed via the adjoint method.

    References:
        Chen et al. "Neural Ordinary Differential Equations", NeurIPS 2018.
    """
    if _HAS_TORCHDIFFEQ:
        import torchdiffeq as td
        result = td.odeint_adjoint(func, y0, t_span, method=method)
        return result[-1]
    else:
        # Fallback: standard integration with autograd tape (less memory-efficient)
        return odeint(func, y0, t_span, method="rk4", num_steps=num_steps)


# ═══════════════════════════════════════════════════════════════════════════
# ODE Dynamics Networks
# ═══════════════════════════════════════════════════════════════════════════


class ODEFunc(nn.Module):
    r"""Neural network parameterizing ODE dynamics :math:`f(t, y)`.

    The dynamics function maps the current state :math:`y` (and optionally
    time :math:`t`) to the time derivative :math:`\dot{y}`:

    .. math::

        \frac{dy}{dt} = f_\theta(t, y)

    Uses Tanh activations to bound the Lipschitz constant of the dynamics,
    which is important for numerical stability and existence/uniqueness
    of solutions (Picard-Lindelof theorem).

    Optionally conditions on time :math:`t` via a sinusoidal embedding,
    allowing the dynamics to vary over the integration interval.

    Args:
        hidden_dim: State dimensionality.
        n_hidden: Width of hidden layers.
        time_dependent: Whether to condition on time.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_hidden: int = 128,
        time_dependent: bool = False,
    ):
        super().__init__()
        self.time_dependent = time_dependent
        input_dim = hidden_dim + (16 if time_dependent else 0)

        self.net = nn.Sequential(
            nn.Linear(input_dim, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, hidden_dim),
            nn.Tanh(),  # bound output for stability
        )

        if time_dependent:
            self.time_emb = nn.Linear(1, 16)

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute dy/dt.

        Args:
            t: Scalar time (may be unused if not time-dependent).
            y: ``(B, hidden_dim)`` current state.

        Returns:
            ``(B, hidden_dim)`` time derivative.
        """
        if self.time_dependent:
            if t.dim() == 0:
                t_input = t.unsqueeze(0).expand(y.shape[0], 1)
            else:
                t_input = t.view(-1, 1).expand(y.shape[0], 1)
            t_feat = self.time_emb(t_input)
            return self.net(torch.cat([y, t_feat], dim=-1))
        return self.net(y)


class ConditionedODEFunc(nn.Module):
    r"""ODE dynamics conditioned on external input (observation + action).

    .. math::

        \frac{dh}{dt} = f_\theta(h, x)

    where :math:`h` is the hidden state and :math:`x` is the conditioning
    input (e.g., encoded observation).  Time-invariant dynamics -- the
    integration interval itself provides temporal information.

    Args:
        hidden_dim: Hidden state dimension.
        cond_dim: Conditioning input dimension.
        n_hidden: Hidden layer width.
    """

    def __init__(self, hidden_dim: int, cond_dim: int, n_hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + cond_dim, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, hidden_dim),
            nn.Tanh(),
        )
        self._cond: Optional[torch.Tensor] = None

    def set_condition(self, cond: torch.Tensor) -> None:
        """Set the conditioning input (called before ODE integration).

        Args:
            cond: ``(B, cond_dim)`` conditioning input.
        """
        self._cond = cond

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Compute dh/dt conditioned on the stored input.

        Args:
            t: Scalar time (unused -- time-invariant dynamics).
            h: ``(B, hidden_dim)`` current hidden state.

        Returns:
            ``(B, hidden_dim)`` time derivative.
        """
        assert self._cond is not None, "Call set_condition() before forward()"
        return self.net(torch.cat([h, self._cond], dim=-1))


# ═══════════════════════════════════════════════════════════════════════════
# Continuous Normalizing Flow for Timing Distribution
# ═══════════════════════════════════════════════════════════════════════════


class CNFTimingDynamics(nn.Module):
    r"""Dynamics for continuous normalizing flow over timing :math:`\Delta\tau`.

    Models the flow ODE that transforms a base distribution (standard normal)
    into the timing distribution:

    .. math::

        \frac{dz}{dt} &= f_\theta(z, t, c) \\
        \frac{d\log p}{dt} &= -\text{tr}\left(\frac{\partial f}{\partial z}\right)

    where :math:`c` is a conditioning vector (e.g., hidden state features).

    The trace of the Jacobian is estimated using the Hutchinson estimator:

    .. math::

        \text{tr}(J) \approx \epsilon^\top J \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)

    Args:
        timing_dim: Dimensionality of timing variable (typically 1).
        cond_dim: Conditioning dimension.
        n_hidden: Hidden layer width.

    References:
        Grathwohl et al. "FFJORD", ICLR 2019.
        Chen et al. "Neural Ordinary Differential Equations", NeurIPS 2018.
    """

    def __init__(self, timing_dim: int = 1, cond_dim: int = 64, n_hidden: int = 64):
        super().__init__()
        self.timing_dim = timing_dim
        self.net = nn.Sequential(
            nn.Linear(timing_dim + cond_dim + 1, n_hidden),  # +1 for time
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, timing_dim),
        )
        self._cond: Optional[torch.Tensor] = None

    def set_condition(self, cond: torch.Tensor) -> None:
        """Set conditioning vector for the flow.

        Args:
            cond: ``(B, cond_dim)`` conditioning features.
        """
        self._cond = cond

    def forward(
        self, t: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        """Compute the augmented dynamics [dz/dt, d(log_p)/dt].

        Args:
            t: Scalar integration time.
            state: ``(B, timing_dim + 1)`` augmented state ``[z, log_p]``.

        Returns:
            ``(B, timing_dim + 1)`` time derivatives.
        """
        z = state[:, :self.timing_dim]
        B = z.shape[0]

        assert self._cond is not None, "Call set_condition() before forward()"

        t_input = t.unsqueeze(0).expand(B, 1) if t.dim() == 0 else t.view(-1, 1)
        net_input = torch.cat([z, self._cond, t_input], dim=-1)

        # Compute dz/dt
        dz = self.net(net_input)

        # Estimate trace of Jacobian via Hutchinson estimator
        if z.requires_grad:
            # Use autograd for exact Jacobian trace (timing_dim is small)
            trace = self._exact_trace(dz, z)
        else:
            trace = torch.zeros(B, 1, device=z.device)

        # d(log_p)/dt = -tr(df/dz)
        d_log_p = -trace

        return torch.cat([dz, d_log_p], dim=-1)

    @staticmethod
    def _exact_trace(dz: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute exact trace of Jacobian df/dz.

        For small timing_dim this is more efficient than the
        Hutchinson estimator.

        Args:
            dz: ``(B, D)`` output of dynamics.
            z: ``(B, D)`` input variable (requires grad).

        Returns:
            ``(B, 1)`` trace of Jacobian.
        """
        B, D = dz.shape
        trace = torch.zeros(B, 1, device=dz.device)
        for i in range(D):
            grad_i = torch.autograd.grad(
                dz[:, i].sum(), z,
                create_graph=True, retain_graph=True,
                allow_unused=True,
            )[0]
            if grad_i is not None:
                trace += grad_i[:, i:i+1]
        return trace


class ContinuousNormalizingFlowTiming(nn.Module):
    r"""Continuous normalizing flow for modeling timing distributions.

    Transforms a simple base distribution (standard normal) into a
    complex timing distribution via a learned invertible ODE:

    .. math::

        z(0) \sim \mathcal{N}(0, I), \quad
        \frac{dz}{dt} = f_\theta(z, t, c), \quad
        \Delta\tau = \text{softplus}(z(1))

    The log-probability of :math:`\Delta\tau` is computed via the
    instantaneous change-of-variables formula:

    .. math::

        \log p(\Delta\tau) = \log p(z(0)) - \int_0^1
        \text{tr}\left(\frac{\partial f}{\partial z}\right) dt
        - \log\left|\frac{d\,\text{softplus}}{d\,z}\right|

    Args:
        cond_dim: Dimension of conditioning features.
        n_hidden: Hidden layer width for the flow dynamics.
        integration_steps: Number of ODE integration steps.

    References:
        Grathwohl et al. "FFJORD", ICLR 2019.
    """

    def __init__(
        self,
        cond_dim: int = 64,
        n_hidden: int = 64,
        integration_steps: int = 10,
    ):
        super().__init__()
        self.dynamics = CNFTimingDynamics(
            timing_dim=1, cond_dim=cond_dim, n_hidden=n_hidden
        )
        self.integration_steps = integration_steps
        self.base_dist = Normal(0.0, 1.0)

    def forward(
        self, cond: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Sample :math:`\Delta\tau` and compute its log-probability.

        Args:
            cond: ``(B, cond_dim)`` conditioning features.

        Returns:
            Tuple of:
                - ``dt``: ``(B, 1)`` sampled timing values (positive).
                - ``log_prob``: ``(B, 1)`` log-probability of samples.
        """
        B = cond.shape[0]
        device = cond.device

        # Sample from base distribution
        z0 = self.base_dist.sample((B, 1)).to(device)
        log_p0 = self.base_dist.log_prob(z0)

        # Augmented state: [z, log_p]
        state0 = torch.cat([z0, log_p0], dim=-1)

        # Integrate forward
        self.dynamics.set_condition(cond)
        t_span = torch.tensor([0.0, 1.0], device=device)
        state1 = odeint(
            self.dynamics, state0, t_span,
            method="rk4", num_steps=self.integration_steps
        )

        z1 = state1[:, :1]
        log_p1 = state1[:, 1:]

        # Map to positive reals via softplus
        dt = F.softplus(z1) + 0.01  # ensure dt > 0

        # Jacobian correction for softplus transform
        # d/dz softplus(z) = sigmoid(z)
        log_det = F.logsigmoid(z1)
        log_prob = log_p1 - log_det

        return dt, log_prob

    def log_prob(
        self, dt: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        r"""Compute log-probability of observed :math:`\Delta\tau`.

        Inverts the flow to find the base-space representation, then
        computes the log-probability using the change-of-variables formula.

        Args:
            dt: ``(B, 1)`` observed timing values.
            cond: ``(B, cond_dim)`` conditioning features.

        Returns:
            ``(B, 1)`` log-probability.
        """
        # Invert softplus: z = log(exp(dt - 0.01) - 1)
        dt_shifted = (dt - 0.01).clamp(min=1e-6)
        z1 = torch.log(torch.exp(dt_shifted) - 1.0 + 1e-8)

        # Log-det of softplus inverse
        log_det = F.logsigmoid(z1)

        # Integrate backward to find z0
        state1 = torch.cat([z1, torch.zeros_like(z1)], dim=-1)
        self.dynamics.set_condition(cond)
        t_span = torch.tensor([1.0, 0.0], device=dt.device)
        state0 = odeint(
            self.dynamics, state1, t_span,
            method="rk4", num_steps=self.integration_steps
        )

        z0 = state0[:, :1]
        delta_log_p = state0[:, 1:]

        # Total log-prob
        log_p_base = self.base_dist.log_prob(z0)
        return log_p_base + delta_log_p - log_det


# ═══════════════════════════════════════════════════════════════════════════
# LTC Agent (updated)
# ═══════════════════════════════════════════════════════════════════════════


class LTCAgent(nn.Module):
    """Agent with Liquid Time-Constant (LTC) recurrent dynamics.

    Handles irregular time steps natively through the continuous-time
    ODE dynamics of :class:`LiquidTimeCell`.

    Supports both discrete and continuous action spaces.

    Args:
        obs_dim: Observation dimensionality.
        act_dim: Action dimensionality (num actions if discrete).
        hidden_dim: RNN hidden state size.
        latent_dim: Encoded observation size.
        time_hidden_dim: TimeModule hidden layer size.
        discrete_actions: Whether action space is discrete.
        ode_solver: ODE solver for LiquidTimeCell (``"euler"``, ``"rk4"``,
            ``"adaptive"``).
        num_substeps: Sub-steps for ODE integration.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        time_hidden_dim: int = 32,
        discrete_actions: bool = True,
        ode_solver: str = "euler",
        num_substeps: int = 4,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.discrete_actions = discrete_actions

        self.encoder = ObservationEncoder(obs_dim, latent_dim)
        self.time_module = TimeModule(hidden_dim, latent_dim, time_hidden_dim)
        self.rnn = LiquidTimeCell(
            latent_dim, hidden_dim,
            solver=ode_solver,
            num_substeps=num_substeps,
        )

        if discrete_actions:
            self.policy_head = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, act_dim),
            )
        else:
            self.policy_mean = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, act_dim),
            )
            self.policy_log_std = nn.Parameter(torch.zeros(1, act_dim))

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def get_initial_hidden(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Return zero-initialized hidden state.

        Args:
            batch_size: Batch size.
            device: Target device.

        Returns:
            ``(B, hidden_dim)`` zero tensor.
        """
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(
        self, obs: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple:
        """Forward pass producing action distribution, value, updated state, and dt.

        Args:
            obs: ``(B, obs_dim)`` raw observation.
            hidden: ``(B, hidden_dim)`` recurrent hidden state.

        Returns:
            Tuple of ``(dist, value, hidden_new, dt)``.
        """
        encoded = self.encoder(obs)
        dt = self.time_module(hidden, encoded)
        hidden_new = self.rnn(encoded, hidden, dt)

        if self.discrete_actions:
            logits = self.policy_head(hidden_new)
            dist = Categorical(logits=logits)
        else:
            mean = self.policy_mean(hidden_new)
            std = self.policy_log_std.exp().expand_as(mean)
            dist = Normal(mean, std)

        value = self.value_head(hidden_new).squeeze(-1)
        return dist, value, hidden_new, dt

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple:
        """Convenience method for PPO rollout and update.

        Args:
            obs: ``(B, obs_dim)`` observation.
            hidden: ``(B, hidden_dim)`` hidden state.
            action: Optional action (for log-prob computation).

        Returns:
            Tuple of ``(action, log_prob, entropy, value, hidden_new, dt)``.
        """
        dist, value, hidden_new, dt = self.forward(obs, hidden)
        if action is None:
            action = dist.sample()

        if self.discrete_actions:
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        else:
            log_prob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)

        return action, log_prob, entropy, value, hidden_new, dt


# ═══════════════════════════════════════════════════════════════════════════
# Neural ODE Agent
# ═══════════════════════════════════════════════════════════════════════════


class NeuralODEAgent(nn.Module):
    r"""RL Agent with Neural ODE state dynamics and adjoint training.

    The hidden state evolves via a Neural ODE:

    .. math::

        h(t + \Delta\tau) = h(t) + \int_t^{t+\Delta\tau} f_\theta(s, h(s), x) \, ds

    where :math:`f_\theta` is a neural network parameterizing the
    continuous dynamics and :math:`x` is the encoded observation
    (held constant during integration).

    **Memory-efficient training**: When ``use_adjoint=True`` (default),
    gradients are computed via the adjoint sensitivity method (Chen et al.
    2018), which does not store intermediate ODE solver states.

    Args:
        obs_dim: Observation dimensionality.
        act_dim: Action dimensionality.
        hidden_dim: ODE hidden state dimension.
        latent_dim: Encoded observation dimension.
        discrete_actions: Whether action space is discrete.
        use_adjoint: Use adjoint method for memory-efficient gradients.
        ode_method: ODE solver method (``"rk4"``, ``"euler"``, ``"dopri5"``).
        ode_steps: Number of integration steps (for Euler/RK4).
        time_dependent_dynamics: Whether ODE dynamics depend on time.
        cnf_timing: Use continuous normalizing flow for timing distribution.

    References:
        Chen et al. "Neural Ordinary Differential Equations", NeurIPS 2018.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        discrete_actions: bool = True,
        use_adjoint: bool = True,
        ode_method: str = "rk4",
        ode_steps: int = 10,
        time_dependent_dynamics: bool = False,
        cnf_timing: bool = False,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.discrete_actions = discrete_actions
        self.use_adjoint = use_adjoint
        self.ode_method = ode_method
        self.ode_steps = ode_steps

        # Encoder
        self.encoder = ObservationEncoder(obs_dim, latent_dim)

        # Time module: predicts delta_tau
        self.time_module = TimeModule(hidden_dim, latent_dim)

        # ODE dynamics
        self.ode_func = ConditionedODEFunc(hidden_dim, latent_dim)

        # Optional: CNF timing distribution
        self.cnf_timing = cnf_timing
        if cnf_timing:
            self.timing_flow = ContinuousNormalizingFlowTiming(
                cond_dim=hidden_dim + latent_dim
            )

        # Policy head
        if discrete_actions:
            self.policy_head = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, act_dim),
            )
        else:
            self.policy_mean = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, act_dim),
            )
            self.policy_log_std = nn.Parameter(torch.zeros(1, act_dim))

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def get_initial_hidden(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Return zero-initialized hidden state."""
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def _integrate_ode(
        self,
        h: torch.Tensor,
        encoded_obs: torch.Tensor,
        delta_tau: torch.Tensor,
    ) -> torch.Tensor:
        r"""Integrate the ODE from :math:`t=0` to :math:`t=\Delta\tau`.

        Uses the adjoint method when ``use_adjoint=True`` for
        memory-efficient backpropagation.

        Args:
            h: ``(B, hidden_dim)`` current hidden state.
            encoded_obs: ``(B, latent_dim)`` encoded observation.
            delta_tau: ``(B, 1)`` integration time.

        Returns:
            ``(B, hidden_dim)`` updated hidden state.
        """
        self.ode_func.set_condition(encoded_obs)

        # Use mean delta_tau for t_span (batch-uniform integration interval)
        dt_mean = delta_tau.mean().detach()
        t_span = torch.tensor(
            [0.0, dt_mean.item()], device=h.device
        )

        if self.use_adjoint:
            h_new = odeint_adjoint(
                self.ode_func, h, t_span,
                method=self.ode_method,
                num_steps=self.ode_steps,
            )
        else:
            h_new = odeint(
                self.ode_func, h, t_span,
                method=self.ode_method,
                num_steps=self.ode_steps,
            )

        return h_new

    def forward(
        self, obs: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple:
        """Forward pass.

        Args:
            obs: ``(B, obs_dim)`` raw observation.
            hidden: ``(B, hidden_dim)`` recurrent hidden state.

        Returns:
            Tuple of ``(dist, value, hidden_new, dt)``.
        """
        encoded = self.encoder(obs)
        dt = self.time_module(hidden, encoded)

        # Integrate ODE
        hidden_new = self._integrate_ode(hidden, encoded, dt)

        # Policy
        if self.discrete_actions:
            logits = self.policy_head(hidden_new)
            dist = Categorical(logits=logits)
        else:
            mean = self.policy_mean(hidden_new)
            std = self.policy_log_std.exp().expand_as(mean)
            dist = Normal(mean, std)

        value = self.value_head(hidden_new).squeeze(-1)
        return dist, value, hidden_new, dt

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple:
        """Convenience method for PPO rollout and update.

        Returns:
            Tuple of ``(action, log_prob, entropy, value, hidden_new, dt)``.
        """
        dist, value, hidden_new, dt = self.forward(obs, hidden)
        if action is None:
            action = dist.sample()

        if self.discrete_actions:
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        else:
            log_prob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)

        return action, log_prob, entropy, value, hidden_new, dt
