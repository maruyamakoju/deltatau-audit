"""World Model Trainer for TemporalRSSM — Publication-Grade Training Infrastructure.

Trains the RSSM with ELBO objective:
    total_loss = reconstruction_loss + kl_loss + timing_nll

Features beyond the standard Dreamer trainer:
    - Cosine annealing with warm restarts (Loshchilov & Hutter, 2017)
    - Configurable linear warmup period
    - Per-parameter-group gradient norm monitoring
    - Automatic Mixed Precision (AMP) for large world models
    - Best-model / periodic checkpointing with resume support
    - Open-loop rollout evaluation with per-timestep error decomposition
    - KL decomposition (temporal vs. spatial), posterior entropy tracking
    - Timing prediction calibration diagnostics

References:
    [1] Hafner et al. "Dream to Control: Learning Behaviors by Latent Imagination"
        (DreamerV1), ICLR 2020.
    [2] Hafner et al. "Mastering Atari with Discrete World Models"
        (DreamerV2), ICLR 2021.
    [3] Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts",
        ICLR 2017.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainerConfig:
    """Full configuration for :class:`WorldModelTrainer`.

    Attributes:
        lr: Peak learning rate (default 6e-4 as in DreamerV2).
        grad_clip: Max gradient norm for clipping (default 100.0).
        kl_weight: Beta weighting on KL divergence (default 1.0).
        free_nats: KL free-nats threshold to prevent posterior collapse (default 3.0).
        device: Torch device string.

        # --- Learning-rate scheduling ---
        lr_schedule: One of ``"cosine"``, ``"linear"``, ``"constant"`` (default).
        warmup_steps: Linear warmup period in optimizer steps (default 1000).
        cosine_t_0: Period of first cosine cycle (steps).  Only used when
            ``lr_schedule="cosine"``.  Default 10000.
        cosine_t_mult: Multiplicative factor for subsequent cosine cycle lengths.
            ``T_{i+1} = cosine_t_mult * T_i``.  Default 1 (no growth).
        cosine_eta_min: Minimum LR at the end of each cosine cycle (default 1e-6).
        total_steps: Total training steps (used for linear decay). Default 100000.

        # --- Gradient monitoring ---
        grad_log_interval: Log gradient statistics every N optimizer steps (0=off).
        grad_explosion_threshold: Warn if any group norm exceeds this value.
        grad_vanishing_threshold: Warn if all group norms are below this value.

        # --- Mixed precision ---
        use_amp: Enable automatic mixed precision via ``torch.cuda.amp``.
        amp_dtype: AMP dtype (``torch.float16`` or ``torch.bfloat16``).

        # --- Checkpointing ---
        checkpoint_dir: Directory for saving checkpoints.  ``None`` disables.
        checkpoint_interval: Save a periodic checkpoint every N steps.
        keep_last_k: Number of most recent periodic checkpoints to keep (0=keep all).
    """

    lr: float = 6e-4
    grad_clip: float = 100.0
    kl_weight: float = 1.0
    free_nats: float = 3.0
    device: str = "cpu"

    # LR schedule
    lr_schedule: str = "constant"
    warmup_steps: int = 1000
    cosine_t_0: int = 10_000
    cosine_t_mult: int = 1
    cosine_eta_min: float = 1e-6
    total_steps: int = 100_000

    # Gradient monitoring
    grad_log_interval: int = 0
    grad_explosion_threshold: float = 1000.0
    grad_vanishing_threshold: float = 1e-7

    # AMP
    use_amp: bool = False
    amp_dtype: Any = None  # set to torch.float16 / bfloat16

    # Checkpointing
    checkpoint_dir: Optional[str] = None
    checkpoint_interval: int = 5000
    keep_last_k: int = 3


# ---------------------------------------------------------------------------
# Gradient statistics helper
# ---------------------------------------------------------------------------


@dataclass
class GradStats:
    """Gradient statistics for a single parameter group."""

    group_name: str
    norm: float
    max_abs: float
    mean_abs: float
    num_params: int


def _compute_grad_stats(
    model: nn.Module,
    group_names: Optional[Dict[int, str]] = None,
) -> List[GradStats]:
    """Compute per-parameter-group gradient statistics.

    Args:
        model: The model whose ``.grad`` attributes will be inspected.
        group_names: Optional mapping ``param_group_index -> name``.

    Returns:
        List of :class:`GradStats`, one per group with at least one gradient.
    """
    stats: Dict[str, List[torch.Tensor]] = {}

    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        # Assign to a group by top-level module name (e.g. "recurrent", "prior_net")
        group = name.split(".")[0]
        stats.setdefault(group, []).append(p.grad.detach())

    results: List[GradStats] = []
    for gname, grads in stats.items():
        flat = torch.cat([g.reshape(-1) for g in grads])
        results.append(
            GradStats(
                group_name=gname,
                norm=float(flat.norm(2).item()),
                max_abs=float(flat.abs().max().item()),
                mean_abs=float(flat.abs().mean().item()),
                num_params=flat.numel(),
            )
        )
    return results


# ---------------------------------------------------------------------------
# WorldModelTrainer
# ---------------------------------------------------------------------------


class WorldModelTrainer:
    """Publication-grade trainer for TemporalRSSM (RSSM with temporal uncertainty).

    Compared to a minimal training loop this adds:

    1. **LR scheduling** -- cosine annealing with warm restarts
       (Loshchilov & Hutter 2017) or linear decay, both with configurable
       linear warmup.

    2. **Gradient monitoring** -- per-module gradient norms logged every
       ``grad_log_interval`` steps with explosion / vanishing detection.

    3. **Mixed precision** -- optional ``torch.cuda.amp`` for 2x memory
       savings on large world models.

    4. **Checkpointing** -- best-model tracking (by validation loss),
       periodic snapshots, and full resume support.

    5. **Open-loop evaluation** -- multi-step rollout measuring per-horizon
       prediction error plus optional SSIM for image observations.

    6. **Diagnostics** -- KL decomposition along latent dimensions,
       posterior entropy tracking, and timing calibration metrics.

    Args:
        model: TemporalRSSM instance.
        config: :class:`TrainerConfig` with all hyperparameters.  When
            ``None``, sensible defaults are used.
        lr: Learning rate shortcut (overrides ``config.lr`` if both given).
        grad_clip: Gradient clipping shortcut.
        kl_weight: KL weight shortcut.
        free_nats: Free-nats shortcut.
        device: Device shortcut.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainerConfig] = None,
        *,
        lr: float = 6e-4,
        grad_clip: float = 100.0,
        kl_weight: float = 1.0,
        free_nats: float = 3.0,
        device: str = "cpu",
    ):
        if config is None:
            config = TrainerConfig(
                lr=lr,
                grad_clip=grad_clip,
                kl_weight=kl_weight,
                free_nats=free_nats,
                device=device,
            )

        self.config = config
        self.device = config.device
        self.kl_weight = config.kl_weight
        self.free_nats = config.free_nats
        self.grad_clip = config.grad_clip

        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        # ---- LR scheduler ----
        self._scheduler = self._build_scheduler()

        # ---- AMP scaler ----
        self._use_amp = config.use_amp and self.device != "cpu"
        amp_dtype = config.amp_dtype
        if amp_dtype is None:
            amp_dtype = torch.float16
        self._amp_dtype = amp_dtype
        self._scaler: Optional[torch.amp.GradScaler] = None
        if self._use_amp:
            self._scaler = torch.amp.GradScaler("cuda")

        # ---- State tracking ----
        self._train_history: List[Dict[str, float]] = []
        self._global_step: int = 0
        self._best_val_loss: float = float("inf")
        self._grad_log_buffer: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # LR scheduling
    # ------------------------------------------------------------------

    def _build_scheduler(self) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        """Build the LR scheduler according to ``config.lr_schedule``.

        Supported schedules:

        * ``"constant"`` -- no scheduling (returns ``None``).
        * ``"cosine"``   -- CosineAnnealingWarmRestarts (SGDR).
        * ``"linear"``   -- Linear decay from peak LR to 0.

        All schedules apply *after* the linear warmup, which is handled
        separately in :meth:`_apply_warmup`.

        Returns:
            An ``LRScheduler`` or ``None`` for constant LR.
        """
        sched = self.config.lr_schedule
        if sched == "constant":
            return None
        elif sched == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.cosine_t_0,
                T_mult=self.config.cosine_t_mult,
                eta_min=self.config.cosine_eta_min,
            )
        elif sched == "linear":
            def _linear_factor(step: int) -> float:
                if step >= self.config.total_steps:
                    return 0.0
                return 1.0 - step / self.config.total_steps

            return torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=_linear_factor
            )
        else:
            raise ValueError(
                f"Unknown lr_schedule={sched!r}. "
                "Choose from 'constant', 'cosine', 'linear'."
            )

    def _apply_warmup(self) -> None:
        """Apply linear warmup scaling to the current learning rate.

        During the first ``config.warmup_steps`` optimizer steps the LR is
        linearly ramped from 0 to the base LR.  After warmup completes
        the scheduler (if any) takes over fully.

        Mathematical formulation::

            lr(t) = lr_base * min(1, t / warmup_steps)   for t < warmup_steps
            lr(t) = scheduler(t - warmup_steps)           otherwise
        """
        warmup = self.config.warmup_steps
        if warmup <= 0 or self._global_step >= warmup:
            return

        scale = self._global_step / max(warmup, 1)
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.config.lr * scale

    def current_lr(self) -> float:
        """Return the current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    # ------------------------------------------------------------------
    # Gradient monitoring
    # ------------------------------------------------------------------

    def _log_gradients(self) -> Optional[Dict[str, Any]]:
        """Compute and log gradient statistics.

        Called after ``backward()`` but before ``optimizer.step()`` every
        ``config.grad_log_interval`` steps.

        Returns:
            Dict with per-group gradient norms, or ``None`` when logging
            is skipped this step.
        """
        interval = self.config.grad_log_interval
        if interval <= 0 or self._global_step % interval != 0:
            return None

        stats = _compute_grad_stats(self.model)
        record: Dict[str, Any] = {
            "step": self._global_step,
            "groups": {},
        }

        total_norm = 0.0
        for s in stats:
            record["groups"][s.group_name] = {
                "norm": s.norm,
                "max_abs": s.max_abs,
                "mean_abs": s.mean_abs,
                "num_params": s.num_params,
            }
            total_norm += s.norm ** 2

        total_norm = math.sqrt(total_norm)
        record["total_norm"] = total_norm

        # Explosion / vanishing detection
        cfg = self.config
        if total_norm > cfg.grad_explosion_threshold:
            logger.warning(
                "Gradient EXPLOSION detected at step %d: total_norm=%.4f > %.4f",
                self._global_step,
                total_norm,
                cfg.grad_explosion_threshold,
            )
            record["alert"] = "explosion"

        if total_norm < cfg.grad_vanishing_threshold and self._global_step > 0:
            logger.warning(
                "Gradient VANISHING detected at step %d: total_norm=%.8f < %.8f",
                self._global_step,
                total_norm,
                cfg.grad_vanishing_threshold,
            )
            record["alert"] = "vanishing"

        self._grad_log_buffer.append(record)
        return record

    @property
    def gradient_logs(self) -> List[Dict[str, Any]]:
        """Access recorded gradient statistics."""
        return self._grad_log_buffer

    # ------------------------------------------------------------------
    # Core training
    # ------------------------------------------------------------------

    def train_step(
        self,
        obs_seq: torch.Tensor,
        act_seq: torch.Tensor,
        dt_seq: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Single gradient update step.

        Supports optional AMP, gradient monitoring, LR scheduling,
        and periodic checkpointing.

        Args:
            obs_seq: ``(T, B, obs_dim)`` observation sequence.
            act_seq: ``(T, B, act_dim)`` action sequence.
            dt_seq:  ``(T, B, 1)`` timing sequence; ``None`` to skip timing loss.

        Returns:
            Dict with loss values (``total_loss``, ``reconstruction_loss``,
            ``kl_loss``, ``timing_loss``, ``lr``, ``grad_norm``) as Python floats.
        """
        obs_seq = obs_seq.to(self.device)
        act_seq = act_seq.to(self.device)
        if dt_seq is not None:
            dt_seq = dt_seq.to(self.device)

        self.model.train()
        self.optimizer.zero_grad()

        # Apply warmup before the step
        self._apply_warmup()

        # ---- Forward (optionally in AMP context) ----
        if self._use_amp:
            with torch.amp.autocast("cuda", dtype=self._amp_dtype):
                losses = self.model.compute_loss(
                    obs_seq, act_seq, dt_seq,
                    kl_weight=self.kl_weight,
                    free_nats=self.free_nats,
                )
            self._scaler.scale(losses["total_loss"]).backward()

            # Gradient monitoring (before unscale for logging, after for clip)
            self._scaler.unscale_(self.optimizer)
            grad_info = self._log_gradients()
            grad_norm = float(
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip).item()
            )
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            losses = self.model.compute_loss(
                obs_seq, act_seq, dt_seq,
                kl_weight=self.kl_weight,
                free_nats=self.free_nats,
            )
            losses["total_loss"].backward()

            grad_info = self._log_gradients()
            grad_norm = float(
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip).item()
            )
            self.optimizer.step()

        # ---- LR scheduler step (after warmup) ----
        if self._global_step >= self.config.warmup_steps and self._scheduler is not None:
            self._scheduler.step()

        self._global_step += 1

        # ---- Periodic checkpoint ----
        if (
            self.config.checkpoint_dir is not None
            and self.config.checkpoint_interval > 0
            and self._global_step % self.config.checkpoint_interval == 0
        ):
            self.save_checkpoint(tag=f"step_{self._global_step}")

        step_result = {k: float(v.item()) for k, v in losses.items()}
        step_result["lr"] = self.current_lr()
        step_result["grad_norm"] = grad_norm
        self._train_history.append(step_result)
        return step_result

    def train_epoch(
        self,
        dataloader: DataLoader,
        dt_available: bool = False,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """Train for one full epoch over the dataloader.

        Args:
            dataloader: DataLoader yielding ``(obs_seq, act_seq)`` or
                ``(obs_seq, act_seq, dt_seq)`` batches.
            dt_available: Whether the dataloader yields ``dt_seq``.
            verbose: Print per-batch loss.

        Returns:
            Dict with mean losses over the epoch.
        """
        epoch_losses: Dict[str, List[float]] = {
            "total_loss": [], "reconstruction_loss": [],
            "kl_loss": [], "timing_loss": [],
        }

        for batch in dataloader:
            if dt_available and len(batch) == 3:
                obs_seq, act_seq, dt_seq = batch
            else:
                obs_seq, act_seq = batch[0], batch[1]
                dt_seq = None

            step = self.train_step(obs_seq, act_seq, dt_seq)
            for k in epoch_losses:
                epoch_losses[k].append(step.get(k, 0.0))

            if verbose:
                print(f"  loss={step['total_loss']:.4f} "
                      f"recon={step['reconstruction_loss']:.4f} "
                      f"kl={step['kl_loss']:.4f} "
                      f"timing={step['timing_loss']:.4f} "
                      f"lr={step['lr']:.2e}")

        return {k: float(np.mean(v)) for k, v in epoch_losses.items() if v}

    # ------------------------------------------------------------------
    # One-step evaluation (backward compatible)
    # ------------------------------------------------------------------

    def evaluate_one_step(
        self,
        obs_seq: torch.Tensor,
        act_seq: torch.Tensor,
        dt_seq: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Evaluate one-step prediction accuracy.

        Args:
            obs_seq: ``(T, B, obs_dim)``
            act_seq: ``(T, B, act_dim)``
            dt_seq:  ``(T, B, 1)`` optional timing targets

        Returns:
            Dict with ``obs_mse`` and optionally ``timing_mae``.
        """
        self.model.eval()
        obs_seq = obs_seq.to(self.device)
        act_seq = act_seq.to(self.device)
        if dt_seq is not None:
            dt_seq = dt_seq.to(self.device)

        with torch.no_grad():
            out = self.model.rssm_observe(obs_seq, act_seq)

        obs_recon = torch.stack(out["obs_recon"], dim=0)
        obs_mse = float(F.mse_loss(obs_recon, obs_seq).item())

        metrics: Dict[str, float] = {"obs_mse": obs_mse}

        if dt_seq is not None:
            timing_mus = torch.stack(
                [d.loc for d in out["timing_dists"]], dim=0
            )
            timing_mae = float(
                (timing_mus - dt_seq).abs().mean().item()
            )
            metrics["timing_mae"] = timing_mae

        return metrics

    # ------------------------------------------------------------------
    # Multi-step timing evaluation (backward compatible)
    # ------------------------------------------------------------------

    def evaluate_multistep_timing(
        self,
        obs_seq: torch.Tensor,
        act_seq: torch.Tensor,
        dt_seq: torch.Tensor,
        rollout_steps: int = 5,
    ) -> Dict[str, Any]:
        """Evaluate multi-step timing prediction accuracy.

        Seeds the model from the first observation using the posterior,
        then unrolls ``rollout_steps`` steps using only the prior
        (imagination mode).

        Args:
            obs_seq: ``(T, B, obs_dim)`` -- only first step seeds the state.
            act_seq: ``(T, B, act_dim)``
            dt_seq:  ``(T, B, 1)`` ground-truth timing.
            rollout_steps: Steps to imagine ahead.

        Returns:
            Dict with ``timing_mae_by_step``: ``{step: mae_value}``.
        """
        self.model.eval()
        T, B = obs_seq.shape[:2]
        device = self.device

        obs_seq = obs_seq.to(device)
        act_seq = act_seq.to(device)
        dt_seq = dt_seq.to(device)

        # Initialize from first obs using posterior
        with torch.no_grad():
            h, z = self.model.initial_state(B, device)
            out = self.model.rssm_observe(obs_seq[:1], act_seq[:1])
            h = out["h_dets"][0]
            z = out["z_posts"][0]

        # Rollout using prior (no future observations)
        timing_errors_by_step: Dict[int, float] = {}

        with torch.no_grad():
            h_curr, z_curr = h, z
            for step in range(min(rollout_steps, T - 1)):
                act_t = act_seq[step + 1]
                h_curr = self.model.recurrent(
                    torch.cat([z_curr, act_t], dim=-1), h_curr
                )
                prior = self.model._prior(h_curr)
                z_curr = prior.mean

                dt_dist = self.model._timing_dist(h_curr, z_curr)
                dt_pred = dt_dist.loc
                dt_true = dt_seq[step + 1]
                mae = float((dt_pred - dt_true).abs().mean().item())
                timing_errors_by_step[step + 1] = mae

        return {"timing_mae_by_step": timing_errors_by_step}

    # ------------------------------------------------------------------
    # Open-loop evaluation (full episode rollout)
    # ------------------------------------------------------------------

    def evaluate_open_loop(
        self,
        obs_seq: torch.Tensor,
        act_seq: torch.Tensor,
        dt_seq: Optional[torch.Tensor] = None,
        context_steps: int = 5,
    ) -> Dict[str, Any]:
        """Open-loop rollout evaluation with per-timestep error decomposition.

        Seeds the model with ``context_steps`` observations using the
        posterior, then predicts the remainder of the sequence using only
        the learned prior (no further observations).

        This is the standard evaluation protocol for world models
        (Hafner et al. 2019, 2020).

        For image-like observations (``obs_dim >= 64``), computes a
        simplified SSIM-proxy metric in addition to MSE.

        Args:
            obs_seq: ``(T, B, obs_dim)`` full observation sequence.
            act_seq: ``(T, B, act_dim)`` full action sequence.
            dt_seq:  ``(T, B, 1)`` optional ground-truth timing.
            context_steps: Number of initial steps to condition on (posterior).

        Returns:
            Dict containing:
                - ``obs_mse_by_step``: ``{horizon: float}`` MSE at each
                  prediction horizon.
                - ``obs_mse_mean``: Mean MSE across all prediction horizons.
                - ``timing_mae_by_step``: ``{horizon: float}`` if ``dt_seq``
                  is provided.
                - ``ssim_by_step``: ``{horizon: float}`` approximate SSIM
                  for image-like observations (``obs_dim >= 64``).
        """
        self.model.eval()
        T, B = obs_seq.shape[:2]
        obs_dim = obs_seq.shape[2]
        device = self.device

        obs_seq = obs_seq.to(device)
        act_seq = act_seq.to(device)
        if dt_seq is not None:
            dt_seq = dt_seq.to(device)

        context_steps = min(context_steps, T - 1)

        # --- Phase 1: Context (posterior conditioning) ---
        with torch.no_grad():
            h, z = self.model.initial_state(B, device)
            out_ctx = self.model.rssm_observe(
                obs_seq[:context_steps], act_seq[:context_steps]
            )
            h = out_ctx["h_dets"][-1]
            z = out_ctx["z_posts"][-1]

        # --- Phase 2: Open-loop prediction (prior only) ---
        pred_horizon = T - context_steps
        obs_mse_by_step: Dict[int, float] = {}
        timing_mae_by_step: Dict[int, float] = {}
        ssim_by_step: Dict[int, float] = {}

        with torch.no_grad():
            h_curr, z_curr = h, z
            for step in range(pred_horizon):
                t_idx = context_steps + step
                act_t = act_seq[t_idx]

                # Prior transition
                h_curr = self.model.recurrent(
                    torch.cat([z_curr, act_t], dim=-1), h_curr
                )
                prior = self.model._prior(h_curr)
                z_curr = prior.mean

                # Decode observation
                feat = torch.cat([h_curr, z_curr], dim=-1)
                obs_pred = self.model.obs_decoder(feat)
                obs_true = obs_seq[t_idx]

                mse = float(F.mse_loss(obs_pred, obs_true).item())
                obs_mse_by_step[step + 1] = mse

                # SSIM proxy for image-like observations
                if obs_dim >= 64:
                    ssim_val = self._compute_ssim_proxy(obs_pred, obs_true)
                    ssim_by_step[step + 1] = ssim_val

                # Timing prediction
                if dt_seq is not None:
                    dt_dist = self.model._timing_dist(h_curr, z_curr)
                    dt_pred = dt_dist.loc
                    dt_true = dt_seq[t_idx]
                    timing_mae = float((dt_pred - dt_true).abs().mean().item())
                    timing_mae_by_step[step + 1] = timing_mae

        result: Dict[str, Any] = {
            "obs_mse_by_step": obs_mse_by_step,
            "obs_mse_mean": float(np.mean(list(obs_mse_by_step.values())))
            if obs_mse_by_step
            else 0.0,
        }
        if timing_mae_by_step:
            result["timing_mae_by_step"] = timing_mae_by_step
        if ssim_by_step:
            result["ssim_by_step"] = ssim_by_step

        return result

    @staticmethod
    def _compute_ssim_proxy(pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute a simplified SSIM-like metric for 1D observation vectors.

        This is a structural similarity proxy that captures the correlation
        structure between predicted and target vectors.  For true image SSIM,
        reshape to spatial dimensions and use ``torchmetrics.SSIM``.

        The formula is based on Wang et al. (2004) simplified for 1D:

        .. math::

            \\text{SSIM}(x, y) = \\frac{(2\\mu_x \\mu_y + C_1)(2\\sigma_{xy} + C_2)}
                                       {(\\mu_x^2 + \\mu_y^2 + C_1)(\\sigma_x^2 + \\sigma_y^2 + C_2)}

        where :math:`C_1 = (0.01 L)^2, C_2 = (0.03 L)^2` and
        :math:`L` is the dynamic range.

        Args:
            pred:   ``(B, D)`` predicted observation.
            target: ``(B, D)`` ground-truth observation.

        Returns:
            Scalar SSIM value averaged over the batch.
        """
        # Estimate dynamic range from target
        L = float((target.max() - target.min()).item()) + 1e-8
        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        mu_x = pred.mean(dim=-1)
        mu_y = target.mean(dim=-1)
        sigma_x_sq = pred.var(dim=-1)
        sigma_y_sq = target.var(dim=-1)
        sigma_xy = ((pred - pred.mean(dim=-1, keepdim=True))
                     * (target - target.mean(dim=-1, keepdim=True))).mean(dim=-1)

        numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x_sq + sigma_y_sq + C2)

        ssim = numerator / (denominator + 1e-8)
        return float(ssim.mean().item())

    # ------------------------------------------------------------------
    # Training diagnostics
    # ------------------------------------------------------------------

    def compute_diagnostics(
        self,
        obs_seq: torch.Tensor,
        act_seq: torch.Tensor,
        dt_seq: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Compute detailed training diagnostics.

        Provides insights beyond scalar losses to diagnose common world
        model training pathologies:

        1. **KL decomposition** -- breaks down KL divergence by latent
           dimension to identify which dimensions carry information vs.
           which are collapsed (posterior = prior).

        2. **Posterior entropy** -- tracks whether the stochastic latent
           is actually being used.  Low entropy across all dimensions
           indicates the posterior has collapsed to a point estimate.

        3. **Timing calibration** -- measures whether the predicted timing
           distribution is well-calibrated by computing the fraction of
           ground-truth timings falling within predicted confidence intervals.

        Args:
            obs_seq: ``(T, B, obs_dim)``
            act_seq: ``(T, B, act_dim)``
            dt_seq:  ``(T, B, 1)`` optional timing targets.

        Returns:
            Dict with:
                - ``kl_per_dim``: ``(latent_dim,)`` mean KL per latent dimension.
                - ``kl_temporal``: list of mean KL per timestep (temporal pattern).
                - ``posterior_entropy_per_dim``: ``(latent_dim,)`` mean posterior
                  entropy per latent dim.
                - ``posterior_entropy_mean``: Scalar mean entropy.
                - ``active_latent_dims``: Number of dims with KL > 0.1 nats.
                - ``timing_calibration``: Dict with coverage at 50%/90%/95% CI
                  (only if ``dt_seq`` provided).
        """
        self.model.eval()
        obs_seq = obs_seq.to(self.device)
        act_seq = act_seq.to(self.device)
        if dt_seq is not None:
            dt_seq = dt_seq.to(self.device)

        with torch.no_grad():
            out = self.model.rssm_observe(obs_seq, act_seq)

        T = len(out["priors"])
        latent_dim = out["z_posts"][0].shape[-1]

        # --- KL decomposition ---
        # Per-dimension KL: kl_divergence returns (B, latent_dim) before sum
        kl_per_dim_accum = torch.zeros(latent_dim, device=self.device)
        kl_temporal: List[float] = []

        for prior, posterior in zip(out["priors"], out["posteriors"]):
            kl_per_dim = torch.distributions.kl_divergence(posterior, prior)  # (B, latent_dim)
            kl_per_dim_accum += kl_per_dim.mean(dim=0)  # average over batch
            kl_temporal.append(float(kl_per_dim.sum(dim=-1).mean().item()))

        kl_per_dim_mean = kl_per_dim_accum / T

        # --- Posterior entropy ---
        entropy_accum = torch.zeros(latent_dim, device=self.device)
        for posterior in out["posteriors"]:
            # Normal entropy = 0.5 * ln(2 * pi * e * sigma^2)
            ent = posterior.entropy()  # (B, latent_dim)
            entropy_accum += ent.mean(dim=0)
        entropy_per_dim = entropy_accum / T

        active_dims = int((kl_per_dim_mean > 0.1).sum().item())

        result: Dict[str, Any] = {
            "kl_per_dim": kl_per_dim_mean.cpu().tolist(),
            "kl_temporal": kl_temporal,
            "posterior_entropy_per_dim": entropy_per_dim.cpu().tolist(),
            "posterior_entropy_mean": float(entropy_per_dim.mean().item()),
            "active_latent_dims": active_dims,
            "total_latent_dims": latent_dim,
        }

        # --- Timing calibration ---
        if dt_seq is not None:
            calibration = self._compute_timing_calibration(out, dt_seq)
            result["timing_calibration"] = calibration

        return result

    def _compute_timing_calibration(
        self,
        out: Dict[str, Any],
        dt_seq: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute timing prediction calibration.

        For a well-calibrated model, X% of ground truth values should
        fall within the X% confidence interval of the predicted distribution.

        Checks 50%, 90%, and 95% confidence intervals.

        Args:
            out: Output from ``rssm_observe``.
            dt_seq: ``(T, B, 1)`` ground-truth timing.

        Returns:
            Dict mapping CI level to observed coverage fraction.
        """
        ci_levels = {
            "coverage_50": 0.6745,  # z-score for 50% CI
            "coverage_90": 1.6449,  # z-score for 90% CI
            "coverage_95": 1.9600,  # z-score for 95% CI
        }

        counts = {k: 0 for k in ci_levels}
        total = 0

        for t, dt_dist in enumerate(out["timing_dists"]):
            mu = dt_dist.loc       # (B, 1)
            sigma = dt_dist.scale  # (B, 1)
            dt_true = dt_seq[t]    # (B, 1)

            z_score = ((dt_true - mu) / (sigma + 1e-8)).abs()
            total += z_score.numel()

            for name, threshold in ci_levels.items():
                counts[name] += int((z_score <= threshold).sum().item())

        return {name: count / max(total, 1) for name, count in counts.items()}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        tag: str = "latest",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Save a training checkpoint.

        The checkpoint includes model weights, optimizer state, scheduler
        state, scaler state, global step, best validation loss, and
        training history.

        Args:
            tag: Filename tag (e.g. ``"best"``, ``"step_5000"``).
            extra: Additional data to store in the checkpoint.

        Returns:
            Path to the saved checkpoint file, or ``None`` if
            ``checkpoint_dir`` is not configured.
        """
        ckpt_dir = self.config.checkpoint_dir
        if ckpt_dir is None:
            return None

        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f"checkpoint_{tag}.pt")

        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self._global_step,
            "best_val_loss": self._best_val_loss,
            "config": {
                "lr": self.config.lr,
                "grad_clip": self.config.grad_clip,
                "kl_weight": self.config.kl_weight,
                "free_nats": self.config.free_nats,
                "lr_schedule": self.config.lr_schedule,
                "warmup_steps": self.config.warmup_steps,
            },
        }

        if self._scheduler is not None:
            state["scheduler_state_dict"] = self._scheduler.state_dict()
        if self._scaler is not None:
            state["scaler_state_dict"] = self._scaler.state_dict()
        if extra is not None:
            state["extra"] = extra

        torch.save(state, path)
        logger.info("Saved checkpoint to %s (step %d)", path, self._global_step)

        # Cleanup old periodic checkpoints
        self._cleanup_old_checkpoints()

        return path

    def save_best(self, val_loss: float) -> Optional[str]:
        """Save checkpoint if ``val_loss`` is the best seen so far.

        Args:
            val_loss: Current validation loss.

        Returns:
            Path to saved checkpoint if this was the best, else ``None``.
        """
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            return self.save_checkpoint(tag="best")
        return None

    def load_checkpoint(self, path: str, strict: bool = True) -> Dict[str, Any]:
        """Resume training from a checkpoint.

        Restores model weights, optimizer state, LR scheduler state,
        AMP scaler state, and training metadata.

        Args:
            path: Path to the checkpoint ``.pt`` file.
            strict: Whether to require an exact match of model keys.

        Returns:
            The full checkpoint dict (including any ``extra`` data).
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._global_step = checkpoint.get("global_step", 0)
        self._best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        if "scheduler_state_dict" in checkpoint and self._scheduler is not None:
            self._scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint and self._scaler is not None:
            self._scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(
            "Loaded checkpoint from %s (step %d, best_val_loss=%.6f)",
            path, self._global_step, self._best_val_loss,
        )
        return checkpoint

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old periodic checkpoints, keeping only the last K.

        Never removes ``checkpoint_best.pt``.
        """
        keep = self.config.keep_last_k
        ckpt_dir = self.config.checkpoint_dir
        if keep <= 0 or ckpt_dir is None:
            return

        ckpt_path = Path(ckpt_dir)
        if not ckpt_path.exists():
            return

        # Find step-based checkpoints
        step_files = sorted(
            ckpt_path.glob("checkpoint_step_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )

        # Remove oldest, keep last K
        to_remove = step_files[:-keep] if len(step_files) > keep else []
        for f in to_remove:
            f.unlink(missing_ok=True)
            logger.debug("Removed old checkpoint: %s", f)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def global_step(self) -> int:
        """Current global optimizer step count."""
        return self._global_step

    @property
    def train_history(self) -> List[Dict[str, float]]:
        """Access training loss history."""
        return self._train_history
