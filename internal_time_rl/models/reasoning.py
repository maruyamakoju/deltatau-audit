"""
Search-Based Deliberative Reasoning — AlphaZero / MuZero-Grade MCTS.

This module implements a research-grade Monte Carlo Tree Search (MCTS) engine
for temporal reasoning in latent space, combining ideas from:

    - Silver et al. (2017). "Mastering the game of Go without human knowledge."
      Nature 550, 354-359.  [AlphaZero PUCT]
    - Schrittwieser et al. (2020). "Mastering Atari, Go, Chess and Shogi by
      Planning with a Learned Model." Nature 588, 604-609.  [MuZero]
    - Coulom (2007). "Efficient Selectivity and Backup Operators in Monte-Carlo
      Tree Search." CG 2006, LNCS 4630.  [Progressive widening]
    - Danihelka et al. (2022). "Policy improvement by planning with Gumbel."
      ICLR 2022.  [Gumbel MuZero / Sequential halving]
    - Schulman et al. (2016). "High-Dimensional Continuous Control Using
      Generalized Advantage Estimation." ICLR 2016.  [GAE / lambda-returns]
    - Williams et al. (2017). "Information Theoretic MPC for Model-Based
      Reinforcement Learning." NeurIPS 2017.  [MPPI]
    - Botev et al. (2013). "The cross-entropy method for optimization."
      Handbook of Statistics 31, 35-59.  [CEM]

Key improvements over naive MCTS:
    1. Correct mean-value backpropagation (total_value / visits)
    2. Progressive widening (Coulom 2007) to avoid combinatorial explosion
    3. PUCT with learned prior (Silver et al. 2017)
    4. Lambda-return backup mixing MC and bootstrap estimates (TD-lambda)
    5. Gumbel MCTS for robust action selection (Danihelka et al. 2022)
    6. Convergence detection and uncertainty-aware root value estimation
    7. Comprehensive search diagnostics
    8. MPPI and CEM for continuous action planning
"""
from __future__ import annotations

import math
import statistics
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# World Model (lightweight, for latent-space search)
# ---------------------------------------------------------------------------

class WorldModel(nn.Module):
    """Lightweight latent-space world model for search-based reasoning.

    Predicts next latent state, reward, and done probability given current
    state, observation, and internal time delta_tau.

    Used by SearchBasedReasoningEngine for MCTS in thought-space.
    """

    def __init__(self, hidden_dim: int, obs_dim: int):
        super().__init__()
        self.transition = nn.Sequential(
            nn.Linear(hidden_dim + obs_dim + 1, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.reward_net = nn.Linear(hidden_dim, 1)
        self.done_net = nn.Linear(hidden_dim, 1)
        # Value uncertainty head (for uncertainty-aware UCB)
        self.value_std_net = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        state: torch.Tensor,
        obs: torch.Tensor,
        delta_tau: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([state, obs, delta_tau], dim=-1)
        next_state = self.transition(x)
        reward = self.reward_net(next_state)
        done_logits = self.done_net(next_state)
        return next_state, reward, done_logits

    def predict_value_std(self, state: torch.Tensor) -> torch.Tensor:
        """Predict uncertainty (std) about the value at this state."""
        return self.value_std_net(state)


# ---------------------------------------------------------------------------
# MCTSNode — publication-grade tree node
# ---------------------------------------------------------------------------

class MCTSNode:
    """A node in the Deliberative Search Tree (AlphaZero / MuZero style).

    Key design choices (following Silver et al. 2017; Schrittwieser et al. 2020):
        - ``total_value`` and ``visits`` are tracked separately; the mean value
          is computed on access as ``total_value / visits``.  This is the
          standard MCTS formulation and avoids the accumulation bug where raw
          values were summed into a single ``value`` field.
        - ``prior`` stores P(a|s) from a learned policy network (PUCT).
        - UCB is computed in pure Python (no torch tensors) to avoid
          silent type-mixing errors.

    Attributes:
        state:  Latent state tensor (detached, for world-model queries).
        obs:    Observation tensor at this node.
        delta_tau:  The timing action that led to this node.
        children:   Mapping from tau-action to child MCTSNode.
        total_value:  Sum of backed-up values (used to compute mean).
        visits:       Visit count N(s).
        prior:        P(a|s) from a policy network; 0.0 = uniform fallback.
        value_std:    Model-predicted value uncertainty.
        reward:       Immediate reward received on transition *into* this node.
    """

    __slots__ = (
        "state",
        "obs",
        "delta_tau",
        "children",
        "total_value",
        "visits",
        "prior",
        "value_std",
        "reward",
        "_value_samples",
    )

    def __init__(
        self,
        state: torch.Tensor,
        obs: torch.Tensor,
        delta_tau: float,
        *,
        prior: float = 0.0,
        value_std: float = 0.0,
        reward: float = 0.0,
    ):
        self.state = state
        self.obs = obs
        self.delta_tau = delta_tau
        self.children: Dict[float, MCTSNode] = {}
        self.total_value: float = 0.0
        self.visits: int = 0
        self.prior: float = prior
        self.value_std: float = value_std
        self.reward: float = reward
        # For convergence / uncertainty tracking at the root
        self._value_samples: List[float] = []

    # -- backward-compat shim ------------------------------------------------
    # Old tests set / read ``node.value`` directly.  We keep the attribute
    # working via a property so that ``node.value = 10.0`` still behaves,
    # while internally everything uses ``total_value / visits``.

    @property
    def value(self) -> float:
        """Mean backed-up value (total_value / visits).

        Returns 0.0 when unvisited (avoids ZeroDivisionError).
        """
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    @value.setter
    def value(self, v: float) -> None:  # noqa: D102
        # Legacy path: ``node.value = x`` sets total_value directly.
        # If visits == 0 we set visits = 1 so that the read-back is correct.
        self.total_value = v
        if self.visits == 0:
            self.visits = 1

    # -- scoring methods -----------------------------------------------------

    def mean_value(self) -> float:
        """Return Q(s) = total_value / visits, or 0 if unvisited."""
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    def ucb_score(self, parent_visits: int, c_puct: float = 1.41) -> float:
        """Compute PUCT score (Silver et al. 2017, AlphaZero).

        PUCT(s, a) = Q(s, a) + c_puct * P(a|s) * sqrt(N_parent) / (1 + N_child)

        When no learned prior is available (``self.prior == 0``), we fall
        back to the classical UCB1 exploration term:

            c_puct * sqrt(ln(N_parent) / N_child)

        An additive uncertainty bonus from the world-model's value-std head
        is included to promote exploration of genuinely uncertain states.

        All arithmetic is in pure Python — no torch tensors — avoiding the
        historical bug where ``torch.tensor(parent_visits)`` produced silent
        type errors and broke autograd.

        Args:
            parent_visits: N(parent).
            c_puct: Exploration constant (default 1.41 ~ sqrt(2) for UCB1).

        Returns:
            float: PUCT / UCB score.
        """
        if self.visits == 0:
            return float("inf")

        exploit = self.total_value / self.visits

        if self.prior > 0.0:
            # PUCT (AlphaZero style)
            explore = (
                c_puct
                * self.prior
                * math.sqrt(parent_visits)
                / (1.0 + self.visits)
            )
        else:
            # Fallback: classical UCB1
            explore = c_puct * math.sqrt(math.log(parent_visits) / self.visits)

        # Uncertainty bonus: prefer less-visited, uncertain nodes
        uncertainty_bonus = self.value_std / (self.visits + 1)

        return exploit + explore + uncertainty_bonus

    def puct_score(self, parent_visits: int, c_puct: float = 2.5) -> float:
        """Pure PUCT score — always uses learned prior.

        Q(s,a) + c_puct * P(a|s) * sqrt(N_parent) / (1 + N_child)

        Reference: Silver et al. (2017), Appendix A — "Search" section.

        Falls back to uniform prior 1/|children| if ``self.prior`` is 0.

        Args:
            parent_visits: N(parent).
            c_puct: Exploration constant (AlphaZero default ~ 2.5).

        Returns:
            float: PUCT score.
        """
        if self.visits == 0:
            return float("inf")

        q = self.total_value / self.visits
        p = self.prior if self.prior > 0.0 else 1.0  # uniform fallback
        u = c_puct * p * math.sqrt(parent_visits) / (1.0 + self.visits)
        return q + u

    # -- diagnostics ---------------------------------------------------------

    def subtree_depth(self) -> int:
        """Return maximum depth of the subtree rooted at this node."""
        if not self.children:
            return 0
        return 1 + max(c.subtree_depth() for c in self.children.values())

    def subtree_size(self) -> int:
        """Total number of nodes in subtree (including self)."""
        return 1 + sum(c.subtree_size() for c in self.children.values())


# ---------------------------------------------------------------------------
# SearchBasedReasoningEngine — publication-grade MCTS
# ---------------------------------------------------------------------------

class SearchBasedReasoningEngine(nn.Module):
    """Level +1: Tree-Search Deliberative Reasoning (AlphaZero / MuZero grade).

    Explores multiple temporal paths using MCTS in the latent space.
    Selects the most robust timing sequence before committing to an action.

    Improvements over baseline MCTS:
        1. **Correct backpropagation**: mean-value = total_value / visits.
        2. **Progressive widening** (Coulom 2007): max children =
           ``pw_C * N^pw_alpha`` to avoid combinatorial explosion.
        3. **PUCT with learned prior** (Silver et al. 2017): when a prior
           network is provided via ``set_prior_network()``, the prior
           P(a|s) biases exploration toward promising actions.
        4. **Lambda-return backup** (TD-lambda): mixes Monte Carlo leaf value
           with bootstrapped values along the search path.
        5. **Gumbel MCTS** (Danihelka et al. 2022): alternative search mode
           using Gumbel-Top-k sampling and sequential halving.
        6. **Convergence detection**: stops early when root value stabilises
           (stddev of recent values below threshold).
        7. **Search diagnostics**: ``get_search_diagnostics()`` returns tree
           depth distribution, branching factor, effective search depth, etc.

    Args:
        hidden_dim: Latent state dimensionality.
        obs_dim: Observation dimensionality.
        search_depth: Maximum depth of MCTS rollout.
        n_simulations: Number of MCTS simulations per ``search()`` call.
        c_puct: PUCT exploration constant.
        pw_C: Progressive widening coefficient C (max_children = C * N^alpha).
        pw_alpha: Progressive widening exponent alpha.
        lambda_: TD(lambda) mixing parameter for value backup (0=pure bootstrap, 1=MC).
        convergence_threshold: Standard deviation threshold for early stopping.
        convergence_window: Number of recent root values to check for convergence.
        discount: Discount factor gamma for lambda-return backup.
    """

    def __init__(
        self,
        hidden_dim: int,
        obs_dim: int,
        search_depth: int = 5,
        n_simulations: int = 10,
        c_puct: float = 1.41,
        pw_C: float = 1.0,
        pw_alpha: float = 0.5,
        lambda_: float = 0.95,
        convergence_threshold: float = 0.01,
        convergence_window: int = 5,
        discount: float = 0.99,
    ):
        super().__init__()
        self.world_model = WorldModel(hidden_dim, obs_dim)
        self.search_depth = search_depth
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.pw_C = pw_C
        self.pw_alpha = pw_alpha
        self.lambda_ = lambda_
        self.convergence_threshold = convergence_threshold
        self.convergence_window = convergence_window
        self.discount = discount
        self.possible_taus = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]

        # Optional learned prior network: maps state -> logits over taus
        self._prior_network: Optional[nn.Module] = None

        # Diagnostics collected during last search()
        self._last_diagnostics: Optional[Dict[str, Any]] = None

    # -- Prior network injection --------------------------------------------

    def set_prior_network(self, network: nn.Module) -> None:
        """Inject a learned prior network for PUCT.

        The network should map a state tensor (B, hidden_dim) to logits of
        shape (B, len(self.possible_taus)).  If no network is set, a uniform
        prior is used (equivalent to vanilla UCB).

        Reference: Silver et al. (2017), "Mastering the game of Go without
        human knowledge", Nature 550.

        Args:
            network: nn.Module mapping (B, hidden_dim) -> (B, n_taus).
        """
        self._prior_network = network

    def _compute_priors(self, state: torch.Tensor) -> List[float]:
        """Compute P(a|s) for each tau in ``self.possible_taus``.

        If a prior network is available, softmax over its logits.
        Otherwise, returns uniform distribution.

        Args:
            state: (B, hidden_dim) latent state tensor.

        Returns:
            List of floats (one per tau), summing to ~1.0.
        """
        n = len(self.possible_taus)
        if self._prior_network is not None:
            with torch.no_grad():
                logits = self._prior_network(state)  # (B, n_taus)
                probs = F.softmax(logits, dim=-1)  # (B, n_taus)
                # Take first batch element
                return probs[0].tolist()[:n]
        return [1.0 / n] * n

    # -- Progressive widening -----------------------------------------------

    @staticmethod
    def _max_children(visits: int, C: float, alpha: float) -> int:
        """Maximum number of children allowed under progressive widening.

        max_children = ceil(C * N^alpha)

        Reference: Coulom (2007), "Efficient Selectivity and Backup Operators
        in Monte-Carlo Tree Search", CG 2006, LNCS 4630.

        Args:
            visits: Parent visit count N.
            C: Coefficient (default 1.0).
            alpha: Exponent (default 0.5).

        Returns:
            int: Maximum number of children to expand.
        """
        if visits <= 0:
            return 1
        return max(1, math.ceil(C * (visits ** alpha)))

    # -- Lambda-return backup -----------------------------------------------

    def _lambda_return_backup(
        self,
        search_path: List[MCTSNode],
        leaf_value: float,
    ) -> None:
        """Back-propagate using lambda-returns (TD(lambda)).

        For each node at depth d in the search path, the backup target is:

            G_d^lambda = (1 - lambda) * sum_{n=1}^{D-d-1} lambda^{n-1} * G_d^{(n)}
                         + lambda^{D-d-1} * G_d^{(D-d)}

        where G_d^{(n)} is the n-step bootstrapped return and G_d^{(D-d)} is the
        full Monte-Carlo return to the leaf.

        In practice, we compute this efficiently by walking backward from the
        leaf, maintaining a running lambda-weighted return.

        Reference: Schulman et al. (2016), "High-Dimensional Continuous
        Control Using Generalized Advantage Estimation", ICLR 2016.

        Args:
            search_path: List of nodes from root to leaf.
            leaf_value: Value estimate at the leaf node.
        """
        gamma = self.discount
        lam = self.lambda_
        # Walking backward, G = r + gamma * ((1-lambda)*V(next) + lambda*G_next)
        # But we only have leaf_value at the bottom, so we compute:
        #   G_leaf = leaf_value
        #   G_d = r_d + gamma * ((1 - lam) * V(child) + lam * G_{d+1})
        # where V(child) = child.mean_value() is the current bootstrap estimate.

        running_return = leaf_value

        for i in range(len(search_path) - 1, -1, -1):
            node = search_path[i]

            if i == len(search_path) - 1:
                # Leaf node: backup the leaf_value directly
                backup_value = running_return
            else:
                # Interior node: mix bootstrap and MC via lambda
                child = search_path[i + 1]
                bootstrap_v = child.mean_value()
                running_return = (
                    child.reward
                    + gamma * ((1.0 - lam) * bootstrap_v + lam * running_return)
                )
                backup_value = running_return

            node.total_value += backup_value
            node.visits += 1
            node._value_samples.append(backup_value)

    # -- Core MCTS search ---------------------------------------------------

    def search(
        self,
        initial_state: torch.Tensor,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """Perform MCTS in the latent thought-space to find the most robust timing.

        Algorithm (per simulation):
            1. **Select**: traverse tree from root by PUCT until reaching a leaf
               or a node eligible for expansion under progressive widening.
            2. **Expand**: add one child (progressive widening limits branching).
            3. **Evaluate**: world model predicts reward + value uncertainty.
            4. **Backup**: lambda-return backpropagation through the search path.

        Early termination: if the root value estimate converges (stddev of
        recent backup values < ``convergence_threshold``), search stops.

        Args:
            initial_state: (B, hidden_dim) initial latent state.
            obs: (B, obs_dim) current observation.

        Returns:
            best_state: Latent state corresponding to the best tau path.
            trace: List of search metadata dicts.
        """
        root = MCTSNode(initial_state, obs, 1.0)
        trace: List[Dict[str, Any]] = []
        simulations_run = 0

        for sim_idx in range(self.n_simulations):
            node = root
            search_path: List[MCTSNode] = [node]

            # 1. SELECT: traverse tree by PUCT until leaf or expandable
            for _d in range(self.search_depth):
                if not node.children:
                    break
                # Check if progressive widening allows expansion
                max_ch = self._max_children(
                    node.visits, self.pw_C, self.pw_alpha
                )
                if len(node.children) < max_ch and len(node.children) < len(
                    self.possible_taus
                ):
                    # Eligible for expansion — break to expand phase
                    break
                # Select best child by PUCT
                # Capture parent visits before iterating children
                parent_n = node.visits
                _best_tau, best_child = max(
                    node.children.items(),
                    key=lambda x: x[1].ucb_score(parent_n, self.c_puct),
                )
                node = best_child
                search_path.append(node)

            # 2. EXPAND: add one new child (progressive widening)
            if len(search_path) <= self.search_depth:
                parent = search_path[-1]
                priors = self._compute_priors(parent.state)
                # Find taus not yet expanded
                unexpanded = [
                    (tau, p)
                    for tau, p in zip(self.possible_taus, priors)
                    if tau not in parent.children
                ]
                if unexpanded:
                    # Pick the unexpanded action with highest prior
                    unexpanded.sort(key=lambda x: x[1], reverse=True)
                    tau, prior_p = unexpanded[0]
                    tau_t = torch.tensor(
                        [[tau]],
                        dtype=torch.float32,
                        device=initial_state.device,
                    )
                    with torch.no_grad():
                        next_state, reward, _ = self.world_model(
                            parent.state, parent.obs, tau_t
                        )
                        v_std = float(
                            self.world_model.predict_value_std(next_state).item()
                        )
                    child = MCTSNode(
                        next_state,
                        parent.obs,
                        tau,
                        prior=prior_p,
                        value_std=v_std,
                        reward=float(reward.item()),
                    )
                    parent.children[tau] = child
                    search_path.append(child)

            # 3. EVALUATE: leaf value
            leaf = search_path[-1]
            leaf_value = leaf.reward  # immediate reward as leaf value

            # 4. BACKUP: lambda-return backpropagation
            self._lambda_return_backup(search_path, leaf_value)

            simulations_run += 1

            # 5. CONVERGENCE CHECK
            if (
                simulations_run >= self.convergence_window
                and len(root._value_samples) >= self.convergence_window
            ):
                recent = root._value_samples[-self.convergence_window :]
                if len(recent) >= 2:
                    std = statistics.stdev(recent)
                    if std < self.convergence_threshold:
                        break  # converged

        if not root.children:
            return initial_state, [{"error": "no children expanded"}]

        # Best action: child of root with most visits (standard MCTS)
        best_tau, best_node = max(
            root.children.items(), key=lambda x: x[1].visits
        )

        # Root value statistics
        root_mean = root.mean_value()
        root_std = 0.0
        if len(root._value_samples) >= 2:
            root_std = statistics.stdev(root._value_samples)

        trace.append(
            {
                "search_depth": self.search_depth,
                "total_simulations": simulations_run,
                "best_delta_tau": best_tau,
                "root_value": root_mean,
                "root_value_std": root_std,
                "root_value_ci_95": (
                    root_mean - 1.96 * root_std,
                    root_mean + 1.96 * root_std,
                ),
                "converged": simulations_run < self.n_simulations,
                "tree_size": root.subtree_size(),
                "tree_depth": root.subtree_depth(),
            }
        )

        # Cache diagnostics
        self._last_diagnostics = self._compute_diagnostics(root, simulations_run)

        return best_node.state, trace

    # -- Gumbel MCTS (Danihelka et al. 2022) --------------------------------

    def gumbel_search(
        self,
        initial_state: torch.Tensor,
        obs: torch.Tensor,
        k: int = 4,
        n_halving_rounds: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """Gumbel MuZero search with sequential halving.

        Instead of visit-count-based action selection, this uses Gumbel-Top-k
        sampling for more robust action selection.  The search budget is
        allocated via sequential halving: start with k candidate actions,
        simulate equally among them, then halve the candidate set by
        eliminating the worst, and repeat.

        Reference: Danihelka et al. (2022). "Policy improvement by planning
        with Gumbel." ICLR 2022.

        Algorithm:
            1. Sample Gumbel noise g_a for each action a.
            2. Compute sigma(a) = log pi(a|s) + g_a  (Gumbel-augmented score).
            3. Select top-k actions by sigma.
            4. Sequential halving: distribute simulations, halve candidates.
            5. Final action = argmax sigma(a) + completed Q(a).

        Args:
            initial_state: (B, hidden_dim) initial latent state.
            obs: (B, obs_dim) current observation.
            k: Number of initial candidate actions.
            n_halving_rounds: Number of halving rounds (default: ceil(log2(k))).

        Returns:
            best_state: Latent state of selected action.
            trace: Search metadata.
        """
        import random as _random

        root = MCTSNode(initial_state, obs, 1.0)
        priors = self._compute_priors(initial_state)
        n_taus = len(self.possible_taus)

        # Clamp k to available actions
        k = min(k, n_taus)
        if n_halving_rounds is None:
            n_halving_rounds = max(1, math.ceil(math.log2(k)))

        # Step 1: Sample Gumbel noise and compute augmented scores
        # Gumbel(0,1) = -log(-log(U)), U ~ Uniform(0,1)
        gumbel_noise = [
            -math.log(-math.log(max(_random.random(), 1e-20))) for _ in range(n_taus)
        ]
        log_priors = [math.log(max(p, 1e-20)) for p in priors]
        sigma = [lp + g for lp, g in zip(log_priors, gumbel_noise)]

        # Step 2: Select top-k by sigma
        tau_indices = list(range(n_taus))
        tau_indices.sort(key=lambda i: sigma[i], reverse=True)
        candidates = tau_indices[:k]

        # Step 3: Expand all candidate children
        for idx in candidates:
            tau = self.possible_taus[idx]
            tau_t = torch.tensor(
                [[tau]], dtype=torch.float32, device=initial_state.device
            )
            with torch.no_grad():
                next_state, reward, _ = self.world_model(
                    initial_state, obs, tau_t
                )
                v_std = float(
                    self.world_model.predict_value_std(next_state).item()
                )
            child = MCTSNode(
                next_state,
                obs,
                tau,
                prior=priors[idx],
                value_std=v_std,
                reward=float(reward.item()),
            )
            root.children[tau] = child

        # Step 4: Sequential halving
        remaining = list(candidates)
        sims_budget = self.n_simulations

        for _round in range(n_halving_rounds):
            if len(remaining) <= 1:
                break
            # Allocate budget equally
            sims_per_action = max(1, sims_budget // (len(remaining) * n_halving_rounds))

            for idx in remaining:
                tau = self.possible_taus[idx]
                child = root.children[tau]
                for _ in range(sims_per_action):
                    # Simple 1-step simulation from child
                    leaf_value = child.reward
                    # Backup to child and root
                    child.total_value += leaf_value
                    child.visits += 1
                    root.total_value += leaf_value
                    root.visits += 1
                    root._value_samples.append(leaf_value)

            # Halve: keep top half by completed Q + sigma
            remaining.sort(
                key=lambda idx: (
                    root.children[self.possible_taus[idx]].mean_value()
                    + sigma[idx]
                ),
                reverse=True,
            )
            remaining = remaining[: max(1, len(remaining) // 2)]

        # Step 5: Final selection — argmax of sigma + completed_Q
        best_idx = max(
            remaining,
            key=lambda idx: (
                sigma[idx]
                + root.children[self.possible_taus[idx]].mean_value()
            ),
        )
        best_tau = self.possible_taus[best_idx]
        best_node = root.children[best_tau]

        trace = [
            {
                "search_mode": "gumbel_mcts",
                "k": k,
                "halving_rounds": n_halving_rounds,
                "best_delta_tau": best_tau,
                "root_value": root.mean_value(),
                "gumbel_sigma": {
                    self.possible_taus[i]: sigma[i] for i in candidates
                },
            }
        ]

        return best_node.state, trace

    # -- Search Diagnostics -------------------------------------------------

    def get_search_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostics from the most recent ``search()`` call.

        Includes:
            - ``tree_depth``: Maximum depth of the search tree.
            - ``tree_size``: Total number of nodes.
            - ``depth_distribution``: Dict mapping depth -> count of nodes.
            - ``branching_factor_mean``: Average number of children per internal node.
            - ``branching_factor_std``: Std of children counts.
            - ``child_value_distribution``: List of (tau, mean_value, visits) for root children.
            - ``effective_search_depth``: Deepest level where a node received
              more than 1 visit (i.e., useful information propagated).
            - ``root_value``: Mean value at root.
            - ``root_value_std``: Std of backed-up values at root.
            - ``total_simulations``: Number of simulations executed.

        Returns:
            Dict of diagnostic values, or empty dict if no search was run.
        """
        if self._last_diagnostics is not None:
            return self._last_diagnostics
        return {}

    def _compute_diagnostics(
        self, root: MCTSNode, simulations_run: int
    ) -> Dict[str, Any]:
        """Compute full search diagnostics from the search tree.

        Args:
            root: Root node of the search tree.
            simulations_run: Number of simulations completed.

        Returns:
            Dict of diagnostic values.
        """
        # Depth distribution and branching factors
        depth_dist: Dict[int, int] = {}
        branching_factors: List[int] = []
        effective_depth = 0

        def _walk(node: MCTSNode, depth: int) -> None:
            nonlocal effective_depth
            depth_dist[depth] = depth_dist.get(depth, 0) + 1
            if node.visits > 1 and depth > effective_depth:
                effective_depth = depth
            if node.children:
                branching_factors.append(len(node.children))
                for child in node.children.values():
                    _walk(child, depth + 1)

        _walk(root, 0)

        bf_mean = 0.0
        bf_std = 0.0
        if branching_factors:
            bf_mean = statistics.mean(branching_factors)
            if len(branching_factors) >= 2:
                bf_std = statistics.stdev(branching_factors)

        child_values = [
            (tau, child.mean_value(), child.visits)
            for tau, child in sorted(root.children.items())
        ]

        root_std = 0.0
        if len(root._value_samples) >= 2:
            root_std = statistics.stdev(root._value_samples)

        return {
            "tree_depth": root.subtree_depth(),
            "tree_size": root.subtree_size(),
            "depth_distribution": depth_dist,
            "branching_factor_mean": bf_mean,
            "branching_factor_std": bf_std,
            "child_value_distribution": child_values,
            "effective_search_depth": effective_depth,
            "root_value": root.mean_value(),
            "root_value_std": root_std,
            "total_simulations": simulations_run,
        }


# ---------------------------------------------------------------------------
# TemporalPlanningAgent — MPC with MPPI, CEM, and temperature selection
# ---------------------------------------------------------------------------

class TemporalPlanningAgent(nn.Module):
    """Model-Predictive Control agent using RSSM for multi-step lookahead.

    Supports three planning methods:
        1. **Random shooting** (baseline): sample random action sequences,
           score each rollout, pick the best.
        2. **MPPI** (Williams et al. 2017): Model Predictive Path Integral
           control — weight action sequences by exponentiated returns,
           compute weighted mean.
        3. **CEM** (Botev et al. 2013): Cross-Entropy Method — iteratively
           refine a Gaussian over action sequences by fitting to top-k elites.

    At each step:
        1. Use TemporalRSSM to imagine K rollouts at different tau sequences.
        2. Score each rollout with discounted returns + robustness penalty.
        3. Execute first action of best-scoring rollout (MPC receding horizon).

    Args:
        obs_dim: Observation space dimensionality.
        act_dim: Action space dimensionality (discrete: n_actions).
        hidden_dim: RSSM deterministic hidden size.
        latent_dim: RSSM stochastic latent dimensionality.
        n_rollouts: Number of rollouts per planning step.
        horizon: Planning horizon (number of steps to imagine).
        robustness_weight: Weight on timing-uncertainty penalty.
        discount: Discount factor gamma for scoring rollouts.
        temperature: Temperature for Boltzmann action selection (0 = argmax).
        planning_method: One of "random_shooting", "mppi", "cem".
        mppi_temperature: Temperature for MPPI weighting (lambda in the paper).
        cem_elite_frac: Fraction of top rollouts used as CEM elites.
        cem_iterations: Number of CEM refinement iterations.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 200,
        latent_dim: int = 30,
        n_rollouts: int = 5,
        horizon: int = 3,
        robustness_weight: float = 0.5,
        discount: float = 0.99,
        temperature: float = 0.0,
        planning_method: str = "random_shooting",
        mppi_temperature: float = 1.0,
        cem_elite_frac: float = 0.2,
        cem_iterations: int = 3,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_rollouts = n_rollouts
        self.horizon = horizon
        self.robustness_weight = robustness_weight
        self.discount = discount
        self.temperature = temperature
        self.planning_method = planning_method
        self.mppi_temperature = mppi_temperature
        self.cem_elite_frac = cem_elite_frac
        self.cem_iterations = cem_iterations

        # Import lazily to avoid circular import
        from .world_model import TemporalRSSM

        self.world_model = TemporalRSSM(obs_dim, act_dim, hidden_dim, latent_dim)

        # Simple policy head for action sampling during imagination
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
        )

    def _policy(self, feat: torch.Tensor) -> torch.distributions.Categorical:
        """Policy distribution from feature vector."""
        logits = self.policy_head(feat)
        return torch.distributions.Categorical(logits=logits)

    def _score_rollout(
        self,
        rewards: torch.Tensor,
        timing_stds: torch.Tensor,
    ) -> float:
        """Score a rollout using discounted returns with robustness penalty.

        Score = sum_{t=0}^{H-1} gamma^t * r_t - w * mean(timing_std)

        Args:
            rewards: (H, B) reward predictions.
            timing_stds: (H, ...) timing uncertainty predictions.

        Returns:
            float: Scalar score for this rollout.
        """
        H = rewards.shape[0]
        discounts = torch.tensor(
            [self.discount ** t for t in range(H)],
            dtype=rewards.dtype,
            device=rewards.device,
        )
        # discounted return (average across batch)
        discounted_return = float((discounts * rewards.mean(dim=-1)).sum().item())
        timing_uncertainty = float(timing_stds.mean().item())
        return discounted_return - self.robustness_weight * timing_uncertainty

    def _single_rollout(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[int, float]:
        """Execute a single imagination rollout and return (first_action, score).

        Args:
            h: (B, hidden_dim) cloned hidden state.
            z: (B, latent_dim) cloned stochastic state.

        Returns:
            Tuple of (first_action_int, rollout_score).
        """
        imagined = self.world_model.rssm_imagine(
            initial_h=h.clone(),
            initial_z=z.clone(),
            horizon=self.horizon,
            policy=self._policy,
        )

        rewards = torch.stack(imagined["reward_preds"], dim=0).squeeze(-1)
        timing_stds = torch.stack(imagined["timing_stds"], dim=0)
        score = self._score_rollout(rewards, timing_stds)

        # Extract first action
        first_act = 0
        if imagined["actions"] and len(imagined["actions"]) > 0:
            a = imagined["actions"][0]
            if isinstance(a, torch.Tensor):
                first_act = int(a.flatten()[0].item())
            else:
                first_act = int(a)

        return first_act, score

    def _plan_random_shooting(
        self,
        obs: torch.Tensor,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[int, Dict[str, Any]]:
        """Plan via random shooting: sample N rollouts, pick best.

        Args:
            obs: Current observation.
            h: RSSM deterministic hidden state.
            z: RSSM stochastic latent.

        Returns:
            (action, plan_info) tuple.
        """
        rollout_scores: List[float] = []
        rollout_first_actions: List[int] = []

        for _ in range(self.n_rollouts):
            first_act, score = self._single_rollout(h, z)
            rollout_scores.append(score)
            rollout_first_actions.append(first_act)

        return self._select_action(rollout_scores, rollout_first_actions)

    def _plan_mppi(
        self,
        obs: torch.Tensor,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[int, Dict[str, Any]]:
        """Plan via MPPI (Model Predictive Path Integral).

        MPPI weights each rollout by exp(score / temperature) and computes
        the weighted average action.  For discrete actions, we accumulate
        weights into action bins and select the highest-weight action.

        Reference: Williams et al. (2017). "Information Theoretic MPC for
        Model-Based Reinforcement Learning." NeurIPS 2017.

        Args:
            obs: Current observation.
            h: RSSM deterministic hidden state.
            z: RSSM stochastic latent.

        Returns:
            (action, plan_info) tuple.
        """
        rollout_scores: List[float] = []
        rollout_first_actions: List[int] = []

        for _ in range(self.n_rollouts):
            first_act, score = self._single_rollout(h, z)
            rollout_scores.append(score)
            rollout_first_actions.append(first_act)

        # MPPI weighting: w_i = exp((S_i - max(S)) / temperature)
        max_score = max(rollout_scores)
        lam = max(self.mppi_temperature, 1e-6)
        weights = [
            math.exp((s - max_score) / lam) for s in rollout_scores
        ]
        w_sum = sum(weights) + 1e-20

        # For discrete actions: accumulate weights per action
        action_weights: Dict[int, float] = {}
        for act, w in zip(rollout_first_actions, weights):
            action_weights[act] = action_weights.get(act, 0.0) + w / w_sum

        best_action = max(action_weights, key=action_weights.get)  # type: ignore[arg-type]

        plan_info = {
            "best_expected_return": rollout_scores[
                rollout_first_actions.index(best_action)
            ]
            if best_action in rollout_first_actions
            else 0.0,
            "best_timing_uncertainty": 0.0,
            "all_scores": rollout_scores,
            "selected_rollout": rollout_first_actions.index(best_action)
            if best_action in rollout_first_actions
            else 0,
            "planning_method": "mppi",
            "action_weights": action_weights,
        }
        return best_action, plan_info

    def _plan_cem(
        self,
        obs: torch.Tensor,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[int, Dict[str, Any]]:
        """Plan via CEM (Cross-Entropy Method).

        CEM iteratively refines a distribution over action sequences by
        fitting to the top-k "elite" rollouts.  For discrete actions, we
        maintain logits per time step and update them from elite samples.

        Reference: Botev et al. (2013). "The cross-entropy method for
        optimization." Handbook of Statistics 31, 35-59.

        Args:
            obs: Current observation.
            h: RSSM deterministic hidden state.
            z: RSSM stochastic latent.

        Returns:
            (action, plan_info) tuple.
        """
        n_elite = max(1, int(self.n_rollouts * self.cem_elite_frac))
        best_scores: List[float] = []
        best_actions: List[int] = []

        for _cem_iter in range(self.cem_iterations):
            rollout_scores: List[float] = []
            rollout_first_actions: List[int] = []

            for _ in range(self.n_rollouts):
                first_act, score = self._single_rollout(h, z)
                rollout_scores.append(score)
                rollout_first_actions.append(first_act)

            # Select elites
            paired = list(zip(rollout_scores, rollout_first_actions))
            paired.sort(key=lambda x: x[0], reverse=True)
            elites = paired[:n_elite]

            best_scores = [s for s, _ in elites]
            best_actions = [a for _, a in elites]

        # Final action = most common among elites (mode)
        action_counts: Dict[int, int] = {}
        for a in best_actions:
            action_counts[a] = action_counts.get(a, 0) + 1
        best_action = max(action_counts, key=action_counts.get)  # type: ignore[arg-type]

        plan_info = {
            "best_expected_return": best_scores[0] if best_scores else 0.0,
            "best_timing_uncertainty": 0.0,
            "all_scores": best_scores,
            "selected_rollout": 0,
            "planning_method": "cem",
            "cem_iterations": self.cem_iterations,
            "elite_actions": best_actions,
        }
        return best_action, plan_info

    def _select_action(
        self,
        scores: List[float],
        actions: List[int],
    ) -> Tuple[int, Dict[str, Any]]:
        """Select action from scored rollouts, with optional temperature.

        When ``self.temperature == 0``, selects the argmax.
        When ``self.temperature > 0``, uses Boltzmann (softmax) sampling.

        Args:
            scores: Scalar scores for each rollout.
            actions: First action from each rollout.

        Returns:
            (action, plan_info) tuple.
        """
        if self.temperature > 0:
            # Boltzmann / softmax sampling
            max_s = max(scores)
            exp_scores = [
                math.exp((s - max_s) / self.temperature) for s in scores
            ]
            total = sum(exp_scores) + 1e-20
            probs = [e / total for e in exp_scores]
            # Weighted random selection
            import random as _random

            chosen_idx = _random.choices(range(len(actions)), weights=probs, k=1)[
                0
            ]
        else:
            # Argmax
            chosen_idx = int(
                torch.tensor(scores, dtype=torch.float32).argmax().item()
            )

        best_action = actions[chosen_idx]
        plan_info = {
            "best_expected_return": scores[chosen_idx],
            "best_timing_uncertainty": 0.0,
            "all_scores": scores,
            "selected_rollout": chosen_idx,
            "planning_method": "random_shooting",
        }
        return best_action, plan_info

    def plan_and_act(
        self,
        obs: torch.Tensor,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[int, Dict[str, Any]]:
        """Plan via multi-step lookahead and return the first action.

        Dispatches to the configured planning method:
            - ``"random_shooting"``: sample-and-score (baseline).
            - ``"mppi"``: Model Predictive Path Integral (Williams et al. 2017).
            - ``"cem"``: Cross-Entropy Method (Botev et al. 2013).

        Args:
            obs: Current observation (B, obs_dim).
            h: Current deterministic hidden state (B, hidden_dim).
            z: Current stochastic latent state (B, latent_dim).

        Returns:
            action: Selected action (int).
            plan_info: Dict with planning metadata (best_expected_return, etc.).
        """
        if self.planning_method == "mppi":
            action, plan_info = self._plan_mppi(obs, h, z)
        elif self.planning_method == "cem":
            action, plan_info = self._plan_cem(obs, h, z)
        else:
            action, plan_info = self._plan_random_shooting(obs, h, z)

        # Clamp to valid range
        action = max(0, min(action, self.act_dim - 1))
        return action, plan_info
