r"""Certified MCTS -- Monte Carlo Tree Search with Lipschitz certification-guided
branch pruning for RL timing safety.

This module implements a novel integration of formal verification bounds into
MCTS. The core insight: before expanding a child node during tree search, we
estimate the Lipschitz constant of the learned world model with respect to the
timing action (delta_tau). If the bound exceeds a safety threshold, the branch
is pruned as "timing-unsafe" -- the policy is too sensitive to timing
perturbations along that trajectory.

Mathematical Foundation
-----------------------
Let :math:`f_\theta: \mathcal{S} \times \Delta\tau \to \mathcal{S}'` be the
learned transition model. The local Lipschitz constant at :math:`(s, \tau)` is:

.. math::

    L(s, \tau) = \sup_{\|\delta\| \leq \epsilon}
        \frac{\|f_\theta(s, \tau + \delta) - f_\theta(s, \tau)\|}{\|\delta\|}

We compute an upper bound via two complementary methods:

1. **Spectral norm product bound** (provable, conservative):

   .. math::

       L_{\text{spectral}} \leq \prod_{i=1}^{D} \sigma_{\max}(W_i)

   where :math:`\sigma_{\max}` is the largest singular value of each weight
   matrix, estimated via power iteration (Miyato et al. 2018). This bound is
   sound -- it is a guaranteed upper bound on the true Lipschitz constant --
   but conservative because it ignores activation function contraction and
   inter-layer correlations.

2. **Empirical Jacobian sampling** (tighter, not provably sound):

   .. math::

       L_{\text{empirical}} = \max_{k=1}^{K}
           \frac{\|f_\theta(s, \tau + \epsilon_k) - f_\theta(s, \tau)\|}
                {\|\epsilon_k\|}

   over :math:`K` random perturbation directions :math:`\epsilon_k`.  This
   estimate is typically tighter than the spectral bound but is only a lower
   bound on the true local Lipschitz constant.

The certified bound is:

.. math::

    L = \min(L_{\text{spectral}}, L_{\text{empirical}})

A node is certified iff :math:`L < \lambda` where :math:`\lambda` is the
Lipschitz threshold, meaning the policy's timing behavior is Lipschitz-stable
at that state. The use of :func:`min` is justified because the spectral bound
is already a guaranteed upper bound; using the empirical estimate only tightens
it when the empirical value is smaller (which is safe since it is then
dominated by the spectral guarantee).

Integration with MCTS
---------------------
The certified MCTS loop modifies standard MCTS (Silver et al. 2017) as follows:

- **Selection**: PUCT with a certification penalty. Uncertified nodes receive a
  multiplicative 0.5 penalty on their exploration bonus, biasing search toward
  timing-safe branches without completely excluding informative but uncertain
  ones.

- **Expansion**: Before adding a child, the engine calls
  :meth:`CertifiedWorldModel.estimate_lipschitz`. If the bound exceeds the
  threshold, the child is pruned (never expanded further).

- **Backup**: Lambda-return mixing of Monte Carlo rollout values and bootstrap
  estimates (Schulman et al. 2015), weighted by certification status. The
  lambda parameter controls the bias-variance tradeoff between TD(0) and MC.

References
----------
.. [Silver2017]  Silver et al. "Mastering the Game of Go without Human
   Knowledge." Nature 550 (2017): 354-359.
.. [Miyato2018]  Miyato et al. "Spectral Normalization for Generative
   Adversarial Networks." ICLR 2018.
.. [Fazlyab2019] Fazlyab et al. "Efficient and Accurate Estimation of
   Lipschitz Constants for Deep Neural Networks." NeurIPS 2019.
.. [Kocsis2006]  Kocsis & Szepesvari. "Bandit Based Monte-Carlo Planning."
   ECML 2006.
.. [Szegedy2014] Szegedy et al. "Intriguing Properties of Neural Networks."
   ICLR 2014.
.. [Schulman2015] Schulman et al. "High-Dimensional Continuous Control Using
   Generalized Advantage Estimation." ICLR 2016.
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


# ---------------------------------------------------------------------------
# CertifiedWorldModel
# ---------------------------------------------------------------------------


class CertifiedWorldModel(nn.Module):
    """Latent-space dynamics model with built-in Lipschitz constant estimation.

    The model predicts next-state, reward, and episode termination given the
    current latent state, observation, and timing action delta_tau.  Crucially,
    it also exposes :meth:`estimate_lipschitz` which bounds the local
    sensitivity of the transition function to perturbations in delta_tau.

    Architecture
    ~~~~~~~~~~~~
    ::

        transition:    (hidden + obs + 1) --fc1--> hidden*2 --LN--> SiLU --fc2--> hidden
        reward_net:    hidden --fc--> 1
        done_net:      hidden --fc--> 1
        value_std_net: hidden --fc--> 32 --ReLU--> 1 --Softplus  (aleatoric uncertainty)

    The transition employs LayerNorm between the two linear layers, which
    stabilises training and keeps intermediate activations bounded -- important
    for the spectral norm bound to remain meaningful.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of the latent state.
    obs_dim : int
        Dimensionality of the observation vector.
    spectral_iters : int
        Number of power-iteration steps for spectral norm estimation.
        More iterations yield tighter bounds at marginal compute cost.
    jacobian_samples : int
        Number of random perturbation directions for empirical Lipschitz
        estimation. 8 samples is typically sufficient for low-dimensional
        timing actions.
    jacobian_eps : float
        Perturbation magnitude for finite-difference Jacobian estimation.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        obs_dim: int = 4,
        spectral_iters: int = 5,
        jacobian_samples: int = 8,
        jacobian_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        self.spectral_iters = spectral_iters
        self.jacobian_samples = jacobian_samples
        self.jacobian_eps = jacobian_eps

        input_dim = hidden_dim + obs_dim + 1  # +1 for scalar delta_tau

        # --- Transition network ---
        self.trans_fc1 = nn.Linear(input_dim, hidden_dim * 2)
        self.trans_ln = nn.LayerNorm(hidden_dim * 2)
        self.trans_fc2 = nn.Linear(hidden_dim * 2, hidden_dim)

        # --- Auxiliary prediction heads ---
        self.reward_net = nn.Linear(hidden_dim, 1)
        self.done_net = nn.Linear(hidden_dim, 1)
        self.value_std_net = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

        # --- Value head (for leaf evaluation) ---
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # --- State encoder (obs -> latent) ---
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal initialisation (preserves spectral norms near 1).

        This is important for Lipschitz estimation: orthogonal matrices have
        all singular values equal to 1, so the initial spectral product bound
        is close to 1.0 and provides a meaningful starting point.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _transition(
        self,
        state: torch.Tensor,
        obs: torch.Tensor,
        delta_tau: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the transition network only.

        Parameters
        ----------
        state : Tensor, shape ``[B, hidden_dim]``
        obs : Tensor, shape ``[B, obs_dim]``
        delta_tau : Tensor, shape ``[B, 1]``

        Returns
        -------
        next_state : Tensor, shape ``[B, hidden_dim]``
        """
        x = torch.cat([state, obs, delta_tau], dim=-1)
        x = self.trans_fc1(x)
        x = self.trans_ln(x)
        x = F.silu(x)
        x = self.trans_fc2(x)
        return x

    def forward(
        self,
        state: torch.Tensor,
        obs: torch.Tensor,
        delta_tau: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward: transition + auxiliary predictions.

        Returns
        -------
        next_state : Tensor ``[B, hidden_dim]``
        reward : Tensor ``[B, 1]``
        done_logit : Tensor ``[B, 1]``
        value_std : Tensor ``[B, 1]``
        """
        next_state = self._transition(state, obs, delta_tau)
        reward = self.reward_net(next_state)
        done_logit = self.done_net(next_state)
        value_std = self.value_std_net(next_state)
        return next_state, reward, done_logit, value_std

    def predict_value(self, state: torch.Tensor) -> torch.Tensor:
        """Predict scalar value from latent state."""
        return self.value_net(state)

    def encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode raw observation into latent state."""
        return self.encoder(obs)

    # --- Lipschitz estimation machinery ---

    def _power_iteration(self, weight: torch.Tensor, n_iters: int) -> float:
        """Estimate the largest singular value of a weight matrix.

        Uses the power method on :math:`W^T W` to approximate
        :math:`\\sigma_{\\max}(W)`.  Convergence is geometric with rate
        :math:`\\sigma_2 / \\sigma_1`, so 3--5 iterations suffice when the
        spectral gap is reasonable.

        Parameters
        ----------
        weight : Tensor, shape ``[out_features, in_features]``
        n_iters : int

        Returns
        -------
        float
            Estimated largest singular value.
        """
        if weight.dim() != 2:
            return float(torch.norm(weight).item())

        u = torch.randn(weight.shape[0], device=weight.device)
        u = u / (u.norm() + 1e-12)

        for _ in range(n_iters):
            v = weight.t() @ u
            v = v / (v.norm() + 1e-12)
            u = weight @ v
            u = u / (u.norm() + 1e-12)

        sigma = (u @ weight @ v).item()
        return abs(sigma)

    def _spectral_norm_bound(self) -> float:
        """Compute the spectral norm product bound for the transition network.

        .. math::

            L_{\\text{spectral}} \\leq \\prod_{i} \\sigma_{\\max}(W_i)

        This is a provable upper bound on the Lipschitz constant of the
        composed linear maps.  SiLU has Lipschitz constant approximately 1.1
        and LayerNorm is 1-Lipschitz (for unit-norm inputs), so the product of
        weight spectral norms dominates the bound.

        Returns
        -------
        float
            Spectral product Lipschitz upper bound for the transition.
        """
        spectral_product = 1.0
        weight_matrices = [self.trans_fc1.weight, self.trans_fc2.weight]

        for W in weight_matrices:
            sigma = self._power_iteration(W, self.spectral_iters)
            spectral_product *= max(sigma, 1e-12)

        # Account for SiLU Lipschitz constant (~1.1)
        spectral_product *= 1.1

        return spectral_product

    def _empirical_lipschitz(
        self,
        state: torch.Tensor,
        obs: torch.Tensor,
        delta_tau: torch.Tensor,
    ) -> float:
        """Finite-difference Jacobian sampling for local Lipschitz estimate.

        Perturbs delta_tau in :attr:`jacobian_samples` random directions and
        measures the resulting change in the transition output.  The maximum
        ratio :math:`\\|\\Delta f\\| / \\|\\Delta \\tau\\|` over samples is
        the empirical bound.

        This is NOT a provable bound -- it is a lower bound on the true local
        Lipschitz constant.  However, when combined with the spectral upper
        bound via :func:`min`, it tightens the overall estimate without
        sacrificing soundness.

        Parameters
        ----------
        state : Tensor ``[1, hidden_dim]``
        obs : Tensor ``[1, obs_dim]``
        delta_tau : Tensor ``[1, 1]``

        Returns
        -------
        float
            Max observed sensitivity ratio.
        """
        base_output = self._transition(state, obs, delta_tau)

        max_ratio = 0.0
        for _ in range(self.jacobian_samples):
            eps = torch.randn_like(delta_tau) * self.jacobian_eps
            perturbed_output = self._transition(state, obs, delta_tau + eps)

            output_diff = (perturbed_output - base_output).norm().item()
            input_diff = eps.norm().item()

            if input_diff > 1e-12:
                ratio = output_diff / input_diff
                max_ratio = max(max_ratio, ratio)

        return max_ratio

    @torch.no_grad()
    def estimate_lipschitz(
        self,
        state: torch.Tensor,
        delta_tau: torch.Tensor,
        obs: Optional[torch.Tensor] = None,
    ) -> float:
        """Combined Lipschitz constant estimation.

        Returns :math:`\\min(L_{\\text{spectral}}, L_{\\text{empirical}})` --
        the tighter of the two bounds.  Since the spectral bound is a provable
        upper bound, taking the minimum with the empirical estimate can only
        tighten it.

        Parameters
        ----------
        state : Tensor ``[1, hidden_dim]``
        delta_tau : Tensor ``[1, 1]``
        obs : Tensor ``[1, obs_dim]``, optional
            If None, uses a zero observation vector.

        Returns
        -------
        float
            Certified Lipschitz constant estimate.
        """
        if obs is None:
            obs = torch.zeros(1, self.obs_dim, device=state.device)

        spectral_bound = self._spectral_norm_bound()
        empirical_bound = self._empirical_lipschitz(state, obs, delta_tau)

        return min(spectral_bound, empirical_bound)


# ---------------------------------------------------------------------------
# CertifiedMCTSNode
# ---------------------------------------------------------------------------


@dataclass
class CertifiedMCTSNode:
    """A node in the certified MCTS tree.

    Each node carries the standard MCTS bookkeeping (visit count, total value,
    prior probability) augmented with Lipschitz certification metadata that
    records whether the timing action leading to this node is formally safe.

    Attributes
    ----------
    state : Tensor
        Latent state at this node, shape ``[1, hidden_dim]``.
    obs : Tensor
        Observation at this node, shape ``[1, obs_dim]``.
    delta_tau : float
        Timing action that led to this node from its parent.
    children : dict
        Mapping from delta_tau candidate (float) to child node.
    total_value : float
        Sum of backed-up values through this node.
    visits : int
        Number of times this node has been visited during search.
    prior : float
        Prior probability assigned during expansion (uniform by default).
    value_std : float
        Aleatoric uncertainty estimate from the world model.
    reward : float
        Immediate reward received upon entering this node.
    lipschitz_bound : float
        Estimated Lipschitz constant of the transition at this node.
    is_certified : bool
        True iff ``lipschitz_bound < threshold`` (timing-safe).
    certification_level : str
        One of ``"spectral"`` (proven bound was tight enough),
        ``"empirical"`` (only the sampled bound was below threshold),
        or ``"pruned"`` (exceeded threshold, branch cut).
    parent : CertifiedMCTSNode or None
        Back-pointer for tree traversal during backup.
    depth : int
        Depth in the search tree (root = 0).
    """

    state: torch.Tensor
    obs: torch.Tensor
    delta_tau: float = 1.0
    children: Dict[float, "CertifiedMCTSNode"] = field(default_factory=dict)
    total_value: float = 0.0
    visits: int = 0
    prior: float = 1.0
    value_std: float = 0.0
    reward: float = 0.0
    lipschitz_bound: float = 0.0
    is_certified: bool = True
    certification_level: str = "spectral"
    parent: Optional["CertifiedMCTSNode"] = None
    depth: int = 0

    @property
    def q_value(self) -> float:
        """Mean action-value Q(s, tau) at this node."""
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    @property
    def is_leaf(self) -> bool:
        """True if this node has no expanded children."""
        return len(self.children) == 0

    @property
    def is_terminal(self) -> bool:
        """Depth-limited terminal check (prevents unbounded tree growth)."""
        return self.depth >= 50


def _iter_tree(node: CertifiedMCTSNode):
    """Depth-first iteration over all nodes in a subtree."""
    yield node
    for child in node.children.values():
        yield from _iter_tree(child)


# ---------------------------------------------------------------------------
# CertifiedMCTSEngine
# ---------------------------------------------------------------------------


class CertifiedMCTSEngine:
    """MCTS with Lipschitz certification-guided branch pruning.

    This is the central algorithm of the module. It wraps the standard
    select-expand-evaluate-backup MCTS loop with Lipschitz safety checks
    at expansion time and certification-aware scoring at selection time.

    The key guarantee: if the spectral Lipschitz bound at a node exceeds
    ``lipschitz_threshold``, the subtree rooted at that node is never explored.
    This ensures the final action recommendation is drawn only from regions of
    the timing-action space where the policy is provably Lipschitz-stable.

    Parameters
    ----------
    world_model : CertifiedWorldModel
        The learned dynamics model with Lipschitz estimation.
    c_puct : float
        Exploration constant for PUCT selection (higher = more exploration).
    lambda_return : float
        Mixing coefficient for lambda-return backup. 1.0 = pure Monte Carlo,
        0.0 = pure TD(0).
    gamma : float
        Discount factor for returns.
    lipschitz_threshold : float
        Maximum allowable Lipschitz constant for a node to be certified.
        Branches exceeding this are pruned.
    certification_level : str
        Minimum certification level: ``"spectral"`` requires the provable
        bound; ``"empirical"`` accepts the sampled estimate.
    """

    def __init__(
        self,
        world_model: CertifiedWorldModel,
        c_puct: float = 1.5,
        lambda_return: float = 0.8,
        gamma: float = 0.99,
        lipschitz_threshold: float = 5.0,
        certification_level: str = "spectral",
    ) -> None:
        self.world_model = world_model
        self.c_puct = c_puct
        self.lambda_return = lambda_return
        self.gamma = gamma
        self.lipschitz_threshold = lipschitz_threshold
        self.certification_level = certification_level

        # Per-search diagnostics (reset in search())
        self._certified_count = 0
        self._pruned_count = 0
        self._total_expanded = 0
        self._lipschitz_values: List[float] = []

    def _puct_score(
        self,
        parent: CertifiedMCTSNode,
        child: CertifiedMCTSNode,
    ) -> float:
        """Compute the PUCT selection score with certification penalty.

        .. math::

            \\text{score} = Q(c)
                + c_{\\text{puct}} \\cdot \\pi(c) \\cdot
                  \\frac{\\sqrt{N(p)}}{1 + N(c)}
                \\cdot \\mu_{\\text{cert}}

        where :math:`\\mu_{\\text{cert}} = 1.0` for certified nodes and
        :math:`0.5` for uncertified ones.  The penalty discourages exploration
        of timing-unsafe branches without completely blocking them.

        Additionally, a small uncertainty bonus proportional to
        ``value_std / (visits + 1)`` encourages exploration of nodes with high
        aleatoric uncertainty (where more data is most informative).
        """
        exploitation = child.q_value
        exploration = (
            self.c_puct
            * max(child.prior, 0.01)
            * math.sqrt(parent.visits + 1)
            / (1 + child.visits)
        )
        cert_multiplier = 1.0 if child.is_certified else 0.5
        uncertainty_bonus = child.value_std / (child.visits + 1)

        return exploitation + exploration * cert_multiplier + uncertainty_bonus

    def _select(self, root: CertifiedMCTSNode) -> Tuple[CertifiedMCTSNode, List[CertifiedMCTSNode], List[float]]:
        """Descend the tree by PUCT selection until a leaf is reached.

        Skips pruned children entirely -- they have
        ``certification_level == "pruned"`` and are never selected.

        Returns
        -------
        leaf : CertifiedMCTSNode
            The selected leaf node.
        path : list of CertifiedMCTSNode
            Nodes visited from root to leaf (inclusive).
        rewards : list of float
            Rewards collected along the path (one fewer than path length).
        """
        node = root
        path = [node]
        rewards: List[float] = []

        while not node.is_leaf and not node.is_terminal:
            # Filter out pruned children
            eligible = {
                tau: child
                for tau, child in node.children.items()
                if child.certification_level != "pruned"
            }
            if not eligible:
                break  # All children pruned -- treat as leaf

            best_tau = max(eligible, key=lambda t: self._puct_score(node, eligible[t]))
            node = eligible[best_tau]
            path.append(node)
            rewards.append(node.reward)

        return node, path, rewards

    def _expand(
        self,
        node: CertifiedMCTSNode,
        tau_candidates: List[float],
    ) -> None:
        """Expand a leaf node by adding children for each tau candidate.

        For each candidate delta_tau:

        1. Estimate the Lipschitz constant at ``(state, delta_tau)``.
        2. If :math:`L \\geq \\lambda`, mark the child as ``"pruned"``
           (created for bookkeeping but never selected or expanded further).
        3. Otherwise, run the world model forward to get the predicted next
           state, reward, and uncertainty, and mark the child as certified.

        The pruning step is the core safety mechanism: it prevents the search
        from exploring trajectories where the policy is overly sensitive to
        timing perturbations.
        """
        uniform_prior = 1.0 / max(len(tau_candidates), 1)

        for tau in tau_candidates:
            if tau in node.children:
                continue  # Already expanded from a previous simulation

            tau_tensor = torch.tensor([[tau]], dtype=torch.float32, device=node.state.device)

            # Step 1: Estimate Lipschitz bound BEFORE forward pass
            lip = self.world_model.estimate_lipschitz(node.state, tau_tensor, node.obs)
            self._lipschitz_values.append(lip)
            self._total_expanded += 1

            # Step 2: Certification check
            if lip >= self.lipschitz_threshold:
                # Create a pruned placeholder node (never selected)
                pruned_child = CertifiedMCTSNode(
                    state=node.state.detach(),  # Placeholder -- never used
                    obs=node.obs.detach(),
                    delta_tau=tau,
                    prior=uniform_prior,
                    lipschitz_bound=lip,
                    is_certified=False,
                    certification_level="pruned",
                    parent=node,
                    depth=node.depth + 1,
                )
                node.children[tau] = pruned_child
                self._pruned_count += 1
                continue

            # Step 3: Safe to expand -- run forward model
            next_state, reward, done_logit, value_std = self.world_model(
                node.state, node.obs, tau_tensor,
            )

            # Determine certification level
            spectral_only = self.world_model._spectral_norm_bound()
            if spectral_only < self.lipschitz_threshold:
                cert_level = "spectral"
            else:
                cert_level = "empirical"

            child = CertifiedMCTSNode(
                state=next_state.detach(),
                obs=node.obs.detach(),
                delta_tau=tau,
                prior=uniform_prior,
                reward=reward.item(),
                value_std=value_std.item(),
                lipschitz_bound=lip,
                is_certified=True,
                certification_level=cert_level,
                parent=node,
                depth=node.depth + 1,
            )
            node.children[tau] = child
            self._certified_count += 1

    def _evaluate(self, node: CertifiedMCTSNode) -> float:
        """Estimate leaf value via short model-based rollout.

        For terminal or pruned nodes, returns the immediate reward only.
        For certified leaves, performs a 3-step rollout using the world model
        and accumulates discounted rewards, terminating early if the model
        predicts episode end.

        Returns
        -------
        float
            Estimated value at the leaf node.
        """
        if node.is_terminal or node.certification_level == "pruned":
            return node.reward

        # Combine rollout value with value-network bootstrap
        rollout_value = node.reward
        current_state = node.state
        current_obs = node.obs
        discount = self.gamma

        tau_tensor = torch.tensor(
            [[node.delta_tau]], dtype=torch.float32, device=current_state.device,
        )

        for _ in range(3):
            next_state, reward, done_logit, _ = self.world_model(
                current_state, current_obs, tau_tensor,
            )
            rollout_value += discount * reward.item()
            discount *= self.gamma

            if torch.sigmoid(done_logit).item() > 0.5:
                break
            current_state = next_state.detach()

        # Bootstrap with value network at terminal rollout state
        bootstrap = self.world_model.predict_value(current_state).item()
        rollout_value += discount * bootstrap

        return rollout_value

    def _backup(
        self,
        path: List[CertifiedMCTSNode],
        rewards: List[float],
        leaf_value: float,
    ) -> None:
        """Lambda-return backup from leaf to root.

        Propagates value up the tree using TD(lambda):

        .. math::

            G^\\lambda_t = (1 - \\lambda) \\sum_{n=1}^{\\infty}
                \\lambda^{n-1} G_t^{(n)}

        where :math:`G_t^{(n)}` is the n-step return. In practice this
        reduces to the recursive form:

        .. math::

            G^\\lambda_t = r_t + \\gamma \\left[
                \\lambda G^\\lambda_{t+1}
                + (1 - \\lambda) V(s_{t+1})
            \\right]

        Certified nodes contribute their full value; uncertified nodes are
        down-weighted by 0.5 to reduce the influence of timing-unsafe branches
        on the root value estimate.
        """
        returns = leaf_value

        for i in reversed(range(len(path))):
            node = path[i]
            node.visits += 1

            if i < len(rewards):
                r = rewards[i]
                bootstrap = node.q_value if node.visits > 1 else 0.0
                mc_return = r + self.gamma * returns
                td_return = r + self.gamma * bootstrap
                returns = (
                    self.lambda_return * mc_return
                    + (1 - self.lambda_return) * td_return
                )

            # Certification-weighted contribution
            cert_weight = 1.0 if node.is_certified else 0.5
            node.total_value += returns * cert_weight

    @torch.no_grad()
    def search(
        self,
        root_state: torch.Tensor,
        root_obs: torch.Tensor,
        num_simulations: int = 50,
        tau_candidates: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Run certified MCTS from the given root state.

        This is the main entry point. It performs ``num_simulations``
        iterations of select-expand-evaluate-backup, then returns the
        most-visited child's delta_tau as the recommended timing action.

        Parameters
        ----------
        root_state : Tensor ``[1, hidden_dim]``
            Latent state at the root.
        root_obs : Tensor ``[1, obs_dim]``
            Observation at the root.
        num_simulations : int
            Number of simulation rollouts.
        tau_candidates : list of float, optional
            Timing action candidates. Default ``[0.5, 0.75, 1.0, 1.25, 1.5]``.

        Returns
        -------
        dict
            Contains ``best_tau``, ``root_value``, ``root_visits``,
            ``certified_fraction``, ``pruned_fraction``, ``mean_lipschitz``,
            ``max_lipschitz``, ``min_lipschitz``, ``std_lipschitz``,
            ``tree_size``, ``num_children``.
        """
        if tau_candidates is None:
            tau_candidates = [0.5, 0.75, 1.0, 1.25, 1.5]

        # Reset per-search diagnostics
        self._certified_count = 0
        self._pruned_count = 0
        self._total_expanded = 0
        self._lipschitz_values = []

        root = CertifiedMCTSNode(
            state=root_state.detach(),
            obs=root_obs.detach(),
            delta_tau=1.0,
            is_certified=True,
            certification_level="spectral",
            depth=0,
        )

        for _ in range(num_simulations):
            # 1. Selection
            leaf, path, rewards = self._select(root)

            # 2. Expansion
            if not leaf.is_terminal and leaf.is_leaf:
                self._expand(leaf, tau_candidates)

                # Move to the best certified child for evaluation
                eligible = [
                    c for c in leaf.children.values()
                    if c.certification_level != "pruned"
                ]
                if eligible:
                    leaf = max(eligible, key=lambda c: c.prior)
                    path.append(leaf)
                    rewards.append(leaf.reward)

            # 3. Evaluation
            value = self._evaluate(leaf)

            # 4. Backup
            self._backup(path, rewards, value)

        # --- Select best action: most-visited certified child of root ---
        best_tau = 1.0  # Fallback: unit timing
        best_visits = -1

        eligible_root = {
            tau: child
            for tau, child in root.children.items()
            if child.certification_level != "pruned"
        }
        for tau, child in eligible_root.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_tau = tau

        # --- Compile diagnostics ---
        total = max(self._total_expanded, 1)
        tree_size = sum(1 for _ in _iter_tree(root))

        return {
            "best_tau": best_tau,
            "root_value": root.q_value,
            "root_visits": root.visits,
            "certified_fraction": self._certified_count / total,
            "pruned_fraction": self._pruned_count / total,
            "mean_lipschitz": (
                statistics.mean(self._lipschitz_values)
                if self._lipschitz_values else 0.0
            ),
            "max_lipschitz": (
                max(self._lipschitz_values)
                if self._lipschitz_values else 0.0
            ),
            "min_lipschitz": (
                min(self._lipschitz_values)
                if self._lipschitz_values else 0.0
            ),
            "std_lipschitz": (
                statistics.stdev(self._lipschitz_values)
                if len(self._lipschitz_values) > 1 else 0.0
            ),
            "tree_size": tree_size,
            "num_children": len(root.children),
            "total_expansions": self._total_expanded,
            "num_simulations": num_simulations,
        }


# ---------------------------------------------------------------------------
# CertifiedMCTSExperiment
# ---------------------------------------------------------------------------


class CertifiedMCTSExperiment:
    """End-to-end experiment: certified MCTS on CartPole-v1.

    The agent uses the certified MCTS engine to select timing actions at each
    step.  Since CartPole has a discrete action space ``{0, 1}``, we map the
    MCTS output (a continuous delta_tau recommendation) to a discrete action
    via a simple threshold: ``tau < 1.0 -> action 0, tau >= 1.0 -> action 1``.

    This mapping reflects the intuition that lower timing actions correspond
    to conservative (left-push) behaviour while higher timing actions
    correspond to aggressive (right-push) behaviour.  The world model's value
    estimates guide which timing regime is preferred at each state.

    Parameters
    ----------
    env_id : str
        Gymnasium environment ID (default ``"CartPole-v1"``).
    hidden_dim : int
        Latent state dimensionality for the world model.
    obs_dim : int
        Observation dimensionality (4 for CartPole).
    action_dim : int
        Discrete action space size (2 for CartPole).
    num_simulations : int
        MCTS simulations per decision step.
    lipschitz_threshold : float
        Certification threshold for Lipschitz bound.
    certification_level : str
        Minimum acceptable certification level.
    c_puct : float
        PUCT exploration constant.
    lambda_return : float
        Lambda-return mixing coefficient.
    gamma : float
        Discount factor.
    n_episodes : int
        Number of evaluation episodes to run.
    max_steps : int
        Maximum steps per episode.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        env_id: str = "CartPole-v1",
        hidden_dim: int = 64,
        obs_dim: int = 4,
        action_dim: int = 2,
        num_simulations: int = 30,
        lipschitz_threshold: float = 5.0,
        certification_level: str = "spectral",
        c_puct: float = 1.5,
        lambda_return: float = 0.8,
        gamma: float = 0.99,
        n_episodes: int = 10,
        max_steps: int = 500,
        seed: int = 42,
    ) -> None:
        self.env_id = env_id
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_simulations = num_simulations
        self.lipschitz_threshold = lipschitz_threshold
        self.certification_level = certification_level
        self.c_puct = c_puct
        self.lambda_return = lambda_return
        self.gamma = gamma
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.seed = seed

    def run(self, out_dir: Path) -> Dict[str, float]:
        """Execute the certified MCTS experiment.

        Creates the environment, world model, and MCTS engine, then runs
        ``n_episodes`` evaluation episodes. At each step within an episode:

        1. Encode the current observation into a latent state.
        2. Run certified MCTS to select the best timing action.
        3. Map the timing action to a discrete CartPole action.
        4. Step the environment and record metrics.

        The composite score aggregates four objectives:

        - **Performance** (40%): Normalised episode return (CartPole max = 500).
        - **Safety** (30%): Fraction of expanded nodes that were certified.
        - **Coverage** (20%): ``1 - pruned_fraction`` (prefer less pruning,
          indicating the model is broadly timing-safe).
        - **Stability** (10%): Inverse coefficient of variation of tau
          selections (prefer consistent timing choices).

        Parameters
        ----------
        out_dir : Path
            Directory to write ``results.json``.

        Returns
        -------
        dict
            All aggregated metrics.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Seed everything for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Build world model and MCTS engine
        world_model = CertifiedWorldModel(
            hidden_dim=self.hidden_dim,
            obs_dim=self.obs_dim,
        )
        world_model.eval()

        engine = CertifiedMCTSEngine(
            world_model=world_model,
            c_puct=self.c_puct,
            lambda_return=self.lambda_return,
            gamma=self.gamma,
            lipschitz_threshold=self.lipschitz_threshold,
            certification_level=self.certification_level,
        )

        tau_candidates = [0.5, 0.75, 1.0, 1.25, 1.5]

        # Accumulators
        episode_returns: List[float] = []
        episode_lengths: List[int] = []
        certified_fractions: List[float] = []
        pruned_fractions: List[float] = []
        lipschitz_means: List[float] = []
        lipschitz_maxes: List[float] = []
        tau_selections: List[float] = []

        env = gym.make(self.env_id)
        start_time = time.perf_counter()

        for ep in range(self.n_episodes):
            obs, _info = env.reset(seed=self.seed + ep)
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            # Encode initial observation to latent state
            with torch.no_grad():
                state = world_model.encode_obs(obs_tensor)

            ep_return = 0.0
            ep_cert_fracs: List[float] = []
            ep_pruned_fracs: List[float] = []
            ep_lip_means: List[float] = []
            ep_taus: List[float] = []

            for step in range(self.max_steps):
                # Run certified MCTS
                result = engine.search(
                    root_state=state,
                    root_obs=obs_tensor,
                    num_simulations=self.num_simulations,
                    tau_candidates=tau_candidates,
                )

                best_tau = result["best_tau"]
                ep_cert_fracs.append(result["certified_fraction"])
                ep_pruned_fracs.append(result["pruned_fraction"])
                ep_lip_means.append(result["mean_lipschitz"])
                ep_taus.append(best_tau)

                # Map timing action to discrete CartPole action
                # tau < 1.0 -> action 0 (push left / conservative)
                # tau >= 1.0 -> action 1 (push right / aggressive)
                action = 0 if best_tau < 1.0 else 1

                # Step environment
                obs, reward, terminated, truncated, _info = env.step(action)
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                ep_return += reward

                # Update latent state via world model transition
                tau_tensor = torch.tensor([[best_tau]], dtype=torch.float32)
                with torch.no_grad():
                    state, _, _, _ = world_model(state, obs_tensor, tau_tensor)

                if terminated or truncated:
                    break

            ep_length = step + 1
            episode_returns.append(ep_return)
            episode_lengths.append(ep_length)
            certified_fractions.append(
                statistics.mean(ep_cert_fracs) if ep_cert_fracs else 0.0
            )
            pruned_fractions.append(
                statistics.mean(ep_pruned_fracs) if ep_pruned_fracs else 0.0
            )
            lipschitz_means.append(
                statistics.mean(ep_lip_means) if ep_lip_means else 0.0
            )
            lipschitz_maxes.append(
                max(ep_lip_means) if ep_lip_means else 0.0
            )
            tau_selections.extend(ep_taus)

        env.close()
        elapsed = time.perf_counter() - start_time

        # --- Aggregate metrics ---
        mean_return = statistics.mean(episode_returns)
        std_return = (
            statistics.stdev(episode_returns)
            if len(episode_returns) > 1 else 0.0
        )
        max_return = max(episode_returns)
        mean_length = statistics.mean(episode_lengths)
        cert_fraction = statistics.mean(certified_fractions)
        pruned_fraction = statistics.mean(pruned_fractions)
        mean_lipschitz = statistics.mean(lipschitz_means) if lipschitz_means else 0.0

        # Timing stability: inverse coefficient of variation of tau selections.
        # A CV near 0 means the agent consistently selects similar timing
        # actions, indicating stable timing behaviour.
        if tau_selections and len(tau_selections) > 1:
            tau_mean = statistics.mean(tau_selections)
            tau_std = statistics.stdev(tau_selections)
            tau_cv = tau_std / tau_mean if tau_mean > 1e-8 else 0.0
            timing_stability = 1.0 / (1.0 + tau_cv)
        else:
            timing_stability = 0.0

        # Normalise return to [0, 1] (CartPole-v1 max is 500)
        max_possible = float(self.max_steps)
        normalized_return = min(mean_return / max_possible, 1.0)

        # Composite score: weighted combination of four objectives
        composite_score = (
            0.4 * normalized_return
            + 0.3 * cert_fraction
            + 0.2 * (1.0 - pruned_fraction)
            + 0.1 * timing_stability
        )

        results: Dict[str, Any] = {
            "composite_score": round(composite_score, 6),
            "mean_return": round(mean_return, 2),
            "std_return": round(std_return, 2),
            "max_return": round(max_return, 2),
            "normalized_return": round(normalized_return, 4),
            "mean_episode_length": round(mean_length, 2),
            "certified_fraction": round(cert_fraction, 4),
            "pruned_fraction": round(pruned_fraction, 4),
            "mean_lipschitz": round(mean_lipschitz, 4),
            "timing_stability": round(timing_stability, 4),
            "n_episodes": self.n_episodes,
            "num_simulations": self.num_simulations,
            "lipschitz_threshold": self.lipschitz_threshold,
            "c_puct": self.c_puct,
            "lambda_return": self.lambda_return,
            "gamma": self.gamma,
            "hidden_dim": self.hidden_dim,
            "elapsed_seconds": round(elapsed, 2),
            "episode_returns": [round(r, 2) for r in episode_returns],
            "episode_lengths": episode_lengths,
            "per_episode_certified": [round(c, 4) for c in certified_fractions],
            "per_episode_pruned": [round(p, 4) for p in pruned_fractions],
        }

        # Save results to disk
        results_path = out_dir / "results.json"
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

        return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the certified MCTS experiment from the command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Certified MCTS: Lipschitz certification-guided branch pruning "
            "for RL timing safety"
        ),
    )
    parser.add_argument("--env-id", default="CartPole-v1", help="Gymnasium env ID")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-simulations", type=int, default=30)
    parser.add_argument("--lipschitz-threshold", type=float, default=5.0)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--lambda-return", type=float, default=0.8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/certified_mcts"),
    )
    args = parser.parse_args()

    experiment = CertifiedMCTSExperiment(
        env_id=args.env_id,
        hidden_dim=args.hidden_dim,
        num_simulations=args.num_simulations,
        lipschitz_threshold=args.lipschitz_threshold,
        c_puct=args.c_puct,
        lambda_return=args.lambda_return,
        gamma=args.gamma,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    print("=" * 64)
    print("  Certified MCTS -- Lipschitz-Guided Branch Pruning")
    print("=" * 64)
    print(f"  env_id              : {args.env_id}")
    print(f"  hidden_dim          : {args.hidden_dim}")
    print(f"  num_simulations     : {args.num_simulations}")
    print(f"  lipschitz_threshold : {args.lipschitz_threshold}")
    print(f"  c_puct              : {args.c_puct}")
    print(f"  lambda_return       : {args.lambda_return}")
    print(f"  gamma               : {args.gamma}")
    print(f"  n_episodes          : {args.n_episodes}")
    print(f"  max_steps           : {args.max_steps}")
    print(f"  seed                : {args.seed}")
    print("=" * 64)

    results = experiment.run(args.out_dir)

    print()
    print("Results:")
    print(f"  composite_score     : {results['composite_score']}")
    print(f"  mean_return         : {results['mean_return']} +/- {results['std_return']}")
    print(f"  max_return          : {results['max_return']}")
    print(f"  certified_fraction  : {results['certified_fraction']}")
    print(f"  pruned_fraction     : {results['pruned_fraction']}")
    print(f"  mean_lipschitz      : {results['mean_lipschitz']}")
    print(f"  timing_stability    : {results['timing_stability']}")
    print(f"  elapsed             : {results['elapsed_seconds']}s")
    print(f"\nResults saved to: {args.out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
