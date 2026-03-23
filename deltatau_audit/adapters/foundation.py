"""Foundation Robot Model Adapters for RL Audit.

Provides research-grade adapters for auditing large, multi-modal foundation
models that operate over high-dimensional observations (images, language,
proprioception) and use sequence-based context windows as implicit state.

Three adapter classes are provided:

1. **FoundationModelAdapter** (abstract base): Defines the canonical interface
   for any foundation-model-based policy. Maintains a sliding context window,
   tracks token counts, and delegates observation preprocessing to subclasses.

2. **OctoAdapter**: Wraps Octo-style Vision-Language-Action (VLA) generalist
   models (Team et al., "Octo: An Open-Source Generalist Robot Policy", 2024).
   Accepts a pre-loaded ``octo.model.OctoModel`` with a ``sample_actions``
   interface and supports language-conditioned and goal-image-conditioned tasks.

3. **TransformerPolicyAdapter**: Generic adapter for autoregressive
   transformer policies such as Decision Transformer (Chen et al., "Decision
   Transformer: Reinforcement Learning via Sequence Modeling", NeurIPS 2021)
   and Gato-style multi-task agents (Reed et al., "A Generalist Agent", TMLR
   2022). Maintains a context window of (observation, action, reward/RTG)
   tuples and performs autoregressive action prediction via the model's
   forward pass.

All adapters conform to the ``AgentAdapter`` ABC defined in
``deltatau_audit.adapters.base``, so they integrate directly with
``run_full_audit()`` and the rest of the deltatau audit pipeline.

References:
    - Octo: Team et al. (2024). "Octo: An Open-Source Generalist Robot
      Policy." arXiv:2405.12213.
    - Decision Transformer: Chen et al. (2021). "Decision Transformer:
      Reinforcement Learning via Sequence Modeling." NeurIPS 2021.
    - Gato: Reed et al. (2022). "A Generalist Agent." TMLR 2022.
"""

from __future__ import annotations

import abc
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .base import AgentAdapter


# ---------------------------------------------------------------------------
# Diagnostics dataclass
# ---------------------------------------------------------------------------

@dataclass
class FoundationDiagnostics:
    """Accumulated diagnostics for a foundation model adapter.

    Tracks token throughput, context utilisation, and latency statistics
    across the lifetime of an adapter instance (reset via ``reset_diagnostics``).
    """

    total_tokens_processed: int = 0
    total_forward_passes: int = 0
    total_latency_s: float = 0.0
    context_fill_history: List[float] = field(default_factory=list)

    @property
    def mean_tokens_per_step(self) -> float:
        if self.total_forward_passes == 0:
            return 0.0
        return self.total_tokens_processed / self.total_forward_passes

    @property
    def mean_latency_s(self) -> float:
        if self.total_forward_passes == 0:
            return 0.0
        return self.total_latency_s / self.total_forward_passes

    @property
    def mean_context_fill(self) -> float:
        if not self.context_fill_history:
            return 0.0
        return float(np.mean(self.context_fill_history))

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_tokens_processed": self.total_tokens_processed,
            "total_forward_passes": self.total_forward_passes,
            "total_latency_s": self.total_latency_s,
            "mean_tokens_per_step": self.mean_tokens_per_step,
            "mean_latency_s": self.mean_latency_s,
            "mean_context_fill": self.mean_context_fill,
        }


# ---------------------------------------------------------------------------
# Hidden state for foundation models = context window + diagnostics handle
# ---------------------------------------------------------------------------

@dataclass
class ContextWindowState:
    """Hidden state representation for foundation model adapters.

    Instead of a recurrent hidden vector, foundation models condition on a
    sliding window of past observations (and optionally past actions and
    rewards).  This dataclass stores that window per batch element.
    """

    history: Deque[torch.Tensor] = field(default_factory=lambda: deque())
    max_len: int = 10

    def append(self, token: torch.Tensor) -> None:
        """Append a token tensor and evict the oldest if window is full."""
        self.history.append(token)
        if len(self.history) > self.max_len:
            self.history.popleft()

    @property
    def length(self) -> int:
        return len(self.history)

    @property
    def fill_ratio(self) -> float:
        return self.length / self.max_len if self.max_len > 0 else 0.0

    def to_sequence_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Stack history into ``(1, L, D)`` sequence tensor.

        Returns:
            Tensor of shape ``(1, L, D)`` where *L* is the current context
            length and *D* is the token dimension.
        """
        if not self.history:
            raise RuntimeError("Context window is empty; call act() first.")
        return torch.stack(list(self.history)).unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Abstract base: FoundationModelAdapter
# ---------------------------------------------------------------------------

class FoundationModelAdapter(AgentAdapter):
    """Abstract base adapter for large foundation-model-based RL policies.

    Subclass this for any model that:
    * Accepts **dict observations** (multimodal: images, language embeddings,
      proprioception vectors).
    * Maintains a **sliding context window** rather than a compact hidden
      state.
    * Produces continuous actions (typical for robotic manipulation).

    Subclasses **must** implement:
        ``_preprocess_obs(obs)`` -- convert a raw environment observation
        (dict or tensor) into a single 1-D token tensor suitable for the
        model.

    Subclasses **may** override:
        ``_forward(sequence_tensor)`` -- run the model forward pass on a
        ``(1, L, D)`` sequence tensor and return ``(action, value)``.
        The default implementation calls ``self.model(sequence_tensor)``.

    Args:
        model: The underlying model (any object with a callable forward).
        context_len: Maximum number of tokens retained in the sliding window.
        action_dim: Dimensionality of the continuous action space. Used as
            a fallback when the context is empty and the model cannot be
            queried.
        device: Torch device string.

    Diagnostics:
        Access ``self.diagnostics`` (a ``FoundationDiagnostics`` instance)
        for token throughput, latency, and context utilisation stats.
    """

    def __init__(
        self,
        model: Any,
        context_len: int = 10,
        action_dim: int = 7,
        device: str = "cpu",
    ):
        self.model = model
        self.context_len = context_len
        self.action_dim = action_dim
        self.device = device
        self.diagnostics = FoundationDiagnostics()

        # Move model to device if it is an nn.Module
        if isinstance(self.model, nn.Module):
            self.model = self.model.to(self.device)
            self.model.eval()

    # -- AgentAdapter interface ------------------------------------------------

    def reset_hidden(
        self,
        batch: int = 1,
        device: str = "cpu",
    ) -> List[ContextWindowState]:
        """Return fresh context windows for each batch element.

        Returns:
            List of ``ContextWindowState`` objects (one per batch element).
        """
        return [
            ContextWindowState(max_len=self.context_len) for _ in range(batch)
        ]

    @torch.no_grad()
    def act(
        self,
        obs: Any,
        hidden: Any,
    ) -> Tuple[np.ndarray, float, Any, Optional[float]]:
        """Single-step forward pass through the foundation model.

        1. Preprocess the observation into a token via ``_preprocess_obs``.
        2. Append the token to the context window (first batch element).
        3. Construct the sequence tensor and call ``_forward``.
        4. Record diagnostics (tokens, latency, context fill).

        Args:
            obs: Raw environment observation (dict, np.ndarray, or Tensor).
            hidden: List of ``ContextWindowState`` (output of ``reset_hidden``).

        Returns:
            action: ``np.ndarray`` of shape ``(action_dim,)``.
            value: Scalar value estimate (float).
            hidden_new: Updated context window states.
            dt: ``None`` (foundation models do not expose an internal clock
                by default; subclasses may override).
        """
        if hidden is None:
            hidden = self.reset_hidden(batch=1, device=self.device)

        # Ensure hidden is a list of ContextWindowState
        if isinstance(hidden, list) and hidden and isinstance(hidden[0], list):
            # Backward compat: old-style list-of-lists -> upgrade
            hidden = self._upgrade_legacy_hidden(hidden)

        ctx: ContextWindowState = hidden[0]

        # Tokenise observation
        token = self._preprocess_obs(obs)
        if not isinstance(token, torch.Tensor):
            token = torch.as_tensor(token, dtype=torch.float32)
        token = token.to(self.device)
        ctx.append(token)

        # Build sequence tensor (1, L, D)
        seq = ctx.to_sequence_tensor(device=self.device)

        # Timed forward pass
        t0 = time.perf_counter()
        action_t, value_t = self._forward(seq)
        elapsed = time.perf_counter() - t0

        # Record diagnostics
        token_count = seq.shape[1] * seq.shape[2]  # L * D
        self.diagnostics.total_tokens_processed += token_count
        self.diagnostics.total_forward_passes += 1
        self.diagnostics.total_latency_s += elapsed
        self.diagnostics.context_fill_history.append(ctx.fill_ratio)

        # Convert outputs to numpy / float
        if isinstance(action_t, torch.Tensor):
            action = action_t.detach().cpu().numpy().flatten()
        else:
            action = np.asarray(action_t, dtype=np.float32).flatten()

        if action.shape[0] != self.action_dim:
            action = action[: self.action_dim]  # truncate to expected dim

        if isinstance(value_t, torch.Tensor):
            value = float(value_t.detach().cpu().item())
        else:
            value = float(value_t)

        return action, value, hidden, None

    # -- Extension points for subclasses ---------------------------------------

    @abc.abstractmethod
    def _preprocess_obs(self, obs: Any) -> torch.Tensor:
        """Convert a raw environment observation into a 1-D token tensor.

        Must be overridden by every concrete subclass.

        Args:
            obs: Observation from the environment. Typically a dict with keys
                like ``"pixels"``, ``"instruction"``, ``"proprio"``.

        Returns:
            1-D ``torch.Tensor`` of shape ``(token_dim,)``.
        """

    def _forward(
        self,
        seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the model forward pass on a sequence tensor.

        Default implementation calls ``self.model(seq)`` and expects the model
        to return ``(action, value)`` tensors.  Subclasses should override
        this when the model's call signature differs (e.g. Octo's
        ``sample_actions``).

        Args:
            seq: ``(1, L, D)`` float tensor — the context window.

        Returns:
            action: Tensor broadcastable to ``(action_dim,)``.
            value: Scalar tensor.
        """
        out = self.model(seq)
        if isinstance(out, tuple) and len(out) >= 2:
            return out[0], out[1]
        # If model only returns actions, synthesise a zero value
        return out, torch.tensor(0.0, device=seq.device)

    # -- Diagnostics API -------------------------------------------------------

    def reset_diagnostics(self) -> None:
        """Clear all accumulated diagnostics counters."""
        self.diagnostics = FoundationDiagnostics()

    def get_diagnostics(self) -> Dict[str, float]:
        """Return current diagnostics as a flat dict."""
        return self.diagnostics.to_dict()

    # -- Internal helpers -------------------------------------------------------

    def _upgrade_legacy_hidden(
        self,
        old_hidden: List[List[Any]],
    ) -> List[ContextWindowState]:
        """Convert old list-of-lists hidden state to ContextWindowState."""
        result = []
        for h in old_hidden:
            ctx = ContextWindowState(max_len=self.context_len)
            for item in h:
                if isinstance(item, torch.Tensor):
                    ctx.append(item)
                elif isinstance(item, np.ndarray):
                    ctx.append(torch.as_tensor(item, dtype=torch.float32))
                # dict items from old API are dropped (cannot be stacked)
            result.append(ctx)
        return result


# ---------------------------------------------------------------------------
# OctoAdapter
# ---------------------------------------------------------------------------

class OctoAdapter(FoundationModelAdapter):
    """Adapter for Octo-style Vision-Language-Action generalist models.

    Octo (Team et al., 2024) is an open-source generalist robot policy that
    takes dict observations (images, language, proprioception) and produces
    continuous actions via a diffusion or MLP action head.  Its Python API
    exposes::

        actions = model.sample_actions(
            jax_observations,
            task,
            rng=jax.random.PRNGKey(0),
        )

    This adapter bridges that interface to ``AgentAdapter.act()``.

    Args:
        model: A loaded Octo model (``octo.model.OctoModel``).
        task: Task specification — either a language string (e.g.
            ``"pick up the red block"``) or a dict with ``"language_instruction"``
            and/or ``"goal"`` keys, following the Octo task API.
        context_len: Number of past observations to retain.
        action_dim: Expected dimensionality of the action vector.
        image_key: Observation dict key for the primary image.
        proprio_key: Observation dict key for the proprioception vector.
        device: Torch device string (Octo runs in JAX, but we convert I/O).

    Raises:
        ImportError: If ``jax`` or ``octo`` are not installed, with
            install instructions.

    Example::

        from octo.model.octo_model import OctoModel
        model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base")

        from deltatau_audit.adapters.foundation import OctoAdapter
        adapter = OctoAdapter(model, task="pick up the blue cup")

        from deltatau_audit.auditor import run_full_audit
        result = run_full_audit(adapter, env_factory, ...)

    References:
        Team et al. (2024). "Octo: An Open-Source Generalist Robot Policy."
        arXiv:2405.12213.
    """

    def __init__(
        self,
        model: Any,
        task: Union[str, Dict[str, Any]] = "",
        context_len: int = 2,
        action_dim: int = 7,
        image_key: str = "image_primary",
        proprio_key: str = "proprio",
        device: str = "cpu",
    ):
        # Validate JAX availability eagerly so users get a clear error
        self._jnp = self._import_jax()

        super().__init__(
            model=model,
            context_len=context_len,
            action_dim=action_dim,
            device=device,
        )

        self.image_key = image_key
        self.proprio_key = proprio_key

        # Normalise task to Octo's expected dict format
        if isinstance(task, str):
            self._task = self._create_language_task(task)
        else:
            self._task = task

        # Pre-allocate JAX RNG
        self._rng = self._jnp.array([0, 42], dtype=self._jnp.uint32)

        # Observation accumulation buffer (Octo expects a window of obs)
        self._obs_window: Deque[Dict[str, Any]] = deque(maxlen=context_len)

    # -- JAX / Octo dependency management ------------------------------------

    @staticmethod
    def _import_jax():
        """Import and return jax.numpy, raising helpful errors if missing."""
        try:
            import jax.numpy as jnp  # noqa: F811
            return jnp
        except ImportError:
            raise ImportError(
                "OctoAdapter requires JAX and Octo. Install with:\n"
                "  pip install --upgrade jax jaxlib\n"
                "  pip install octo\n"
                "See https://github.com/octo-models/octo for details."
            )

    def _create_language_task(self, instruction: str) -> Dict[str, Any]:
        """Build Octo task dict from a language instruction string.

        Uses the Octo model's own ``create_tasks`` helper if available;
        otherwise constructs the dict manually.
        """
        if hasattr(self.model, "create_tasks"):
            return self.model.create_tasks(texts=[instruction])
        # Manual fallback
        return {"language_instruction": [instruction]}

    # -- Observation preprocessing -------------------------------------------

    def _preprocess_obs(self, obs: Any) -> torch.Tensor:
        """Convert dict observation into a flat token tensor.

        For diagnostics and context tracking we flatten the observation into a
        single 1-D tensor. The actual Octo forward pass uses the raw dict
        observations stored in ``self._obs_window``.

        Args:
            obs: Dict with image and proprioception keys, or a flat array.

        Returns:
            1-D ``torch.Tensor``.
        """
        if isinstance(obs, dict):
            parts = []

            # Image
            if self.image_key in obs:
                img = obs[self.image_key]
                if isinstance(img, np.ndarray):
                    img_t = torch.as_tensor(img, dtype=torch.float32).flatten()
                    # Normalise uint8 images to [0, 1]
                    if img.dtype == np.uint8:
                        img_t = img_t / 255.0
                else:
                    img_t = torch.as_tensor(img, dtype=torch.float32).flatten()
                parts.append(img_t)
            elif "pixels" in obs:
                img = obs["pixels"]
                img_t = torch.as_tensor(img, dtype=torch.float32).flatten()
                if isinstance(img, np.ndarray) and img.dtype == np.uint8:
                    img_t = img_t / 255.0
                parts.append(img_t)

            # Proprioception
            if self.proprio_key in obs:
                prop = torch.as_tensor(
                    obs[self.proprio_key], dtype=torch.float32
                ).flatten()
                parts.append(prop)

            # Language / instruction embedding (if pre-embedded)
            if "instruction" in obs:
                instr = torch.as_tensor(
                    obs["instruction"], dtype=torch.float32
                ).flatten()
                parts.append(instr)

            if not parts:
                raise ValueError(
                    f"Observation dict has no recognised keys. "
                    f"Expected at least one of: {self.image_key!r}, 'pixels', "
                    f"{self.proprio_key!r}, 'instruction'. "
                    f"Got keys: {list(obs.keys())}"
                )

            return torch.cat(parts)
        else:
            # Flat observation (np.ndarray or tensor)
            return torch.as_tensor(obs, dtype=torch.float32).flatten()

    # -- Forward pass via Octo API -------------------------------------------

    def _forward(
        self,
        seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run Octo's ``sample_actions`` on the accumulated observation window.

        The ``seq`` tensor (used by the base class for diagnostics) is
        constructed from the flat token representations. The actual model
        call uses the structured observation dicts stored in
        ``self._obs_window``.

        Returns:
            action: ``torch.Tensor`` of shape ``(action_dim,)``.
            value: ``torch.tensor(0.0)`` (Octo does not produce value
                estimates; auditor skips value-dependent metrics).
        """
        jnp = self._jnp

        # Build Octo-format observation dict from the window
        octo_obs = self._build_octo_observation()

        # Advance JAX RNG
        import jax
        self._rng, sample_rng = jax.random.split(self._rng)

        # Call the model
        raw_actions = self.model.sample_actions(
            octo_obs,
            self._task,
            rng=sample_rng,
        )

        # raw_actions is a JAX array; typically (1, horizon, action_dim) or
        # (1, action_dim).  We take the first timestep of the first batch.
        action_np = np.asarray(raw_actions)
        while action_np.ndim > 1:
            action_np = action_np[0]
        action_np = action_np[: self.action_dim].astype(np.float32)

        action_t = torch.as_tensor(action_np, dtype=torch.float32)
        value_t = torch.tensor(0.0)

        return action_t, value_t

    @torch.no_grad()
    def act(
        self,
        obs: Any,
        hidden: Any,
    ) -> Tuple[np.ndarray, float, Any, Optional[float]]:
        """Override to accumulate raw observations for Octo's native API.

        In addition to the base-class context-window bookkeeping (used for
        diagnostics), we maintain ``self._obs_window`` with the original dict
        observations so that ``_forward`` can build Octo-format inputs.
        """
        # Accumulate raw obs for the Octo model
        if isinstance(obs, dict):
            self._obs_window.append(obs)
        else:
            self._obs_window.append({"flat": obs})

        return super().act(obs, hidden)

    def _build_octo_observation(self) -> Dict[str, Any]:
        """Assemble the Octo-format observation dict from the obs window.

        Octo expects observations batched with a leading batch dim and a
        time/window dim, e.g.::

            {
                "image_primary": jnp.array of shape (1, T, H, W, C),
                "proprio": jnp.array of shape (1, T, D),
                "pad_mask": jnp.array of shape (1, T),
            }

        This method stacks the accumulated observations into that format.
        """
        jnp = self._jnp
        obs_list = list(self._obs_window)
        T = len(obs_list)

        result: Dict[str, Any] = {}

        # Image
        img_key = self.image_key if any(
            self.image_key in o for o in obs_list if isinstance(o, dict)
        ) else "pixels"

        images = []
        for o in obs_list:
            if isinstance(o, dict) and img_key in o:
                img = np.asarray(o[img_key])
                # Ensure HWC format
                if img.ndim == 3 and img.shape[0] in (1, 3, 4):
                    img = np.transpose(img, (1, 2, 0))
                images.append(img)

        if images:
            img_stack = np.stack(images, axis=0)  # (T, H, W, C)
            result[self.image_key] = jnp.array(img_stack[np.newaxis])  # (1, T, H, W, C)

        # Proprioception
        proprios = []
        for o in obs_list:
            if isinstance(o, dict) and self.proprio_key in o:
                proprios.append(np.asarray(o[self.proprio_key]).flatten())

        if proprios:
            prop_stack = np.stack(proprios, axis=0)  # (T, D)
            result[self.proprio_key] = jnp.array(prop_stack[np.newaxis])  # (1, T, D)

        # Pad mask — all ones (no padding in our window)
        result["pad_mask"] = jnp.ones((1, T))

        return result

    def reset_hidden(
        self,
        batch: int = 1,
        device: str = "cpu",
    ) -> List[ContextWindowState]:
        """Reset context windows and clear the Octo observation buffer."""
        self._obs_window.clear()
        return super().reset_hidden(batch=batch, device=device)


# ---------------------------------------------------------------------------
# TransformerPolicyAdapter
# ---------------------------------------------------------------------------

class TransformerPolicyAdapter(FoundationModelAdapter):
    """Adapter for autoregressive transformer RL policies.

    Supports Decision-Transformer-style models (Chen et al., 2021) and
    Gato-style multi-task sequence models (Reed et al., 2022).  These models
    condition on a window of ``(observation, action, return-to-go)`` tuples
    and autoregressively predict the next action.

    The adapter manages the context window of historical transitions and
    constructs the input sequence expected by the transformer.

    Model contract:
        The ``model`` must be callable as::

            action_tensor, value_tensor = model(sequence_tensor)

        where ``sequence_tensor`` has shape ``(1, L, D)`` and:
        * ``L`` is the current sequence length (variable, up to
          ``context_len``).
        * ``D`` is the per-timestep token dimension (obs + action + 1 for
          RTG).
        * ``action_tensor`` has shape ``(1, action_dim)`` or ``(action_dim,)``.
        * ``value_tensor`` is a scalar tensor (or the model returns a single
          tensor, in which case value defaults to 0).

    If your model returns only actions, override ``_forward`` for custom
    unpacking.

    Args:
        model: A callable transformer policy (``nn.Module`` or similar).
        context_len: Number of past transitions retained.
        obs_dim: Dimensionality of a flat observation vector. If ``None``,
            inferred from the first observation.
        action_dim: Dimensionality of the action space.
        target_rtg: The return-to-go conditioning value. For Decision
            Transformer, this is the desired episode return. Set to 0 if the
            model does not use RTG conditioning.
        use_rtg: Whether to include return-to-go tokens in the sequence.
        device: Torch device string.

    Example::

        # Decision Transformer
        from my_dt import DecisionTransformer
        dt = DecisionTransformer.load("dt_hopper.pt")

        from deltatau_audit.adapters.foundation import TransformerPolicyAdapter
        adapter = TransformerPolicyAdapter(
            model=dt,
            context_len=20,
            obs_dim=11,
            action_dim=3,
            target_rtg=3600.0,
        )

        from deltatau_audit.auditor import run_full_audit
        result = run_full_audit(adapter, env_factory, ...)

    References:
        Chen et al. (2021). "Decision Transformer: Reinforcement Learning via
        Sequence Modeling." NeurIPS 2021.

        Reed et al. (2022). "A Generalist Agent." TMLR 2022.
    """

    def __init__(
        self,
        model: Any,
        context_len: int = 20,
        obs_dim: Optional[int] = None,
        action_dim: int = 7,
        target_rtg: float = 0.0,
        use_rtg: bool = True,
        device: str = "cpu",
    ):
        super().__init__(
            model=model,
            context_len=context_len,
            action_dim=action_dim,
            device=device,
        )
        self.obs_dim = obs_dim
        self.target_rtg = target_rtg
        self.use_rtg = use_rtg

        # Transition history: each entry is (obs_flat, action, rtg/reward)
        self._transition_history: Deque[Tuple[torch.Tensor, torch.Tensor, float]] = deque(
            maxlen=context_len,
        )
        self._current_rtg: float = target_rtg
        self._inferred_token_dim: Optional[int] = None
        self._last_action: Optional[torch.Tensor] = None

    # -- Observation preprocessing -------------------------------------------

    def _preprocess_obs(self, obs: Any) -> torch.Tensor:
        """Flatten observation to a 1-D tensor.

        Handles dict observations (by concatenating all numeric values) and
        flat arrays. Infers ``obs_dim`` from the first call if not set
        explicitly.

        Args:
            obs: Raw observation (dict, np.ndarray, or Tensor).

        Returns:
            1-D ``torch.Tensor`` of shape ``(obs_dim,)``.
        """
        if isinstance(obs, dict):
            parts = []
            for key in sorted(obs.keys()):
                val = obs[key]
                t = torch.as_tensor(val, dtype=torch.float32).flatten()
                # Normalise uint8 images
                if isinstance(val, np.ndarray) and val.dtype == np.uint8:
                    t = t / 255.0
                parts.append(t)
            flat = torch.cat(parts)
        elif isinstance(obs, torch.Tensor):
            flat = obs.float().flatten()
        else:
            flat = torch.as_tensor(obs, dtype=torch.float32).flatten()

        # Infer obs_dim on first call
        if self.obs_dim is None:
            self.obs_dim = flat.shape[0]
        return flat

    # -- Context window management -------------------------------------------

    def _build_sequence_tensor(
        self,
        current_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Build the transformer input sequence from transition history.

        For Decision Transformer, each timestep token is the concatenation of
        ``[RTG, obs, action]``.  The current timestep has the observation but
        a zero-padded action (to be predicted).

        For models without RTG conditioning (``use_rtg=False``), tokens are
        ``[obs, action]``.

        Returns:
            ``(1, L, token_dim)`` tensor.
        """
        tokens = []

        # Historical transitions
        for obs_h, act_h, rtg_h in self._transition_history:
            token = self._make_token(obs_h, act_h, rtg_h)
            tokens.append(token)

        # Current timestep: obs is known, action is zero-padded
        dummy_action = torch.zeros(self.action_dim, device=self.device)
        token = self._make_token(current_obs, dummy_action, self._current_rtg)
        tokens.append(token)

        seq = torch.stack(tokens).unsqueeze(0).to(self.device)  # (1, L, D)

        if self._inferred_token_dim is None:
            self._inferred_token_dim = seq.shape[-1]

        return seq

    def _make_token(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        rtg: float,
    ) -> torch.Tensor:
        """Concatenate obs, action, and (optionally) RTG into a single token."""
        obs = obs.to(self.device)
        action = action.to(self.device)
        if self.use_rtg:
            rtg_t = torch.tensor([rtg], dtype=torch.float32, device=self.device)
            return torch.cat([rtg_t, obs, action])
        else:
            return torch.cat([obs, action])

    # -- Forward pass --------------------------------------------------------

    @torch.no_grad()
    def act(
        self,
        obs: Any,
        hidden: Any,
    ) -> Tuple[np.ndarray, float, Any, Optional[float]]:
        """Autoregressive action prediction from the transition context.

        Steps:
        1. Preprocess the observation to a flat token.
        2. Build the full sequence from history + current obs.
        3. Forward through the transformer to get action prediction.
        4. Record the transition (obs, action, rtg) into history.
        5. Update return-to-go for the next step.

        Args:
            obs: Raw environment observation.
            hidden: List of ``ContextWindowState`` (from ``reset_hidden``),
                used by the base class for diagnostics. Transition history
                is managed internally via ``self._transition_history``.

        Returns:
            action: ``np.ndarray`` of shape ``(action_dim,)``.
            value: Scalar value estimate.
            hidden_new: Updated context window state.
            dt: ``None``.
        """
        if hidden is None:
            hidden = self.reset_hidden(batch=1, device=self.device)

        # Ensure hidden is list of ContextWindowState
        if isinstance(hidden, list) and hidden and isinstance(hidden[0], list):
            hidden = self._upgrade_legacy_hidden(hidden)

        ctx: ContextWindowState = hidden[0]

        # Preprocess
        obs_flat = self._preprocess_obs(obs)
        obs_flat = obs_flat.to(self.device)

        # Build sequence for the transformer
        seq = self._build_sequence_tensor(obs_flat)

        # Diagnostics: token into context window for base-class tracking
        ctx.append(obs_flat)

        # Timed forward pass
        t0 = time.perf_counter()
        action_t, value_t = self._forward(seq)
        elapsed = time.perf_counter() - t0

        # Record diagnostics
        token_count = seq.shape[1] * seq.shape[2]
        self.diagnostics.total_tokens_processed += token_count
        self.diagnostics.total_forward_passes += 1
        self.diagnostics.total_latency_s += elapsed
        self.diagnostics.context_fill_history.append(ctx.fill_ratio)

        # Extract action
        if isinstance(action_t, torch.Tensor):
            action = action_t.detach().cpu().numpy().flatten()
        else:
            action = np.asarray(action_t, dtype=np.float32).flatten()
        action = action[: self.action_dim]

        # Extract value
        if isinstance(value_t, torch.Tensor):
            value = float(value_t.detach().cpu().item())
        else:
            value = float(value_t)

        # Record transition
        self._last_action = torch.as_tensor(
            action, dtype=torch.float32, device=self.device
        )
        self._transition_history.append(
            (obs_flat.detach(), self._last_action.detach(), self._current_rtg)
        )

        return action, value, hidden, None

    def update_rtg(self, reward: float) -> None:
        """Update return-to-go after receiving a reward.

        For Decision Transformer, this should be called after each env step
        to decrement the RTG by the received reward.

        Args:
            reward: The scalar reward from the environment step.
        """
        self._current_rtg -= reward

    def reset_hidden(
        self,
        batch: int = 1,
        device: str = "cpu",
    ) -> List[ContextWindowState]:
        """Reset context windows and transition history."""
        self._transition_history.clear()
        self._current_rtg = self.target_rtg
        self._last_action = None
        return super().reset_hidden(batch=batch, device=device)

    @property
    def current_context_length(self) -> int:
        """Number of transitions currently in the context window."""
        return len(self._transition_history)

    @property
    def token_dim(self) -> Optional[int]:
        """Per-timestep token dimension (inferred after the first forward pass)."""
        return self._inferred_token_dim
