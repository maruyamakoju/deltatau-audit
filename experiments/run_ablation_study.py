#!/usr/bin/env python3
"""Ablation study for ACT and RSSM components of the deltatau-audit framework.

This script systematically disables each extension of the
DeliberativeInternalTimeAgent (ACT) and TemporalRSSM (Dreamer v3) models,
measuring the impact on key metrics. The results are written to
``results/ablation/`` as Markdown tables, LaTeX tables, plots, and a
machine-readable JSON summary.

ACT ablation (Section 1):
    Tests six configurations by toggling geometric halting prior,
    information-theoretic halting, adaptive max steps, multi-head
    deliberation, and a no-ACT baseline.

RSSM ablation (Section 2):
    Tests seven configurations covering categorical vs Gaussian latent,
    symlog transform, KL balancing vs free nats, continue predictor,
    multi-step prediction loss, and LogNormal vs Normal timing.

Usage::

    python experiments/run_ablation_study.py               # full study
    python experiments/run_ablation_study.py --section act  # ACT only
    python experiments/run_ablation_study.py --section rssm # RSSM only
    python experiments/run_ablation_study.py --quick         # fast mode

All experiments run on CPU and complete in under 5 minutes.

References:
    [1] Graves, A. (2016). Adaptive Computation Time for RNNs.
    [2] Hafner et al. (2023). Mastering Diverse Domains through World Models.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Project root on sys.path so that internal_time_rl is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from internal_time_rl.models.deliberative import (
    DeliberativeInternalTimeAgent,
    PonderingDiagnostics,
)
from internal_time_rl.models.world_model import TemporalRSSM

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ablation")

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
RESULTS_DIR = _PROJECT_ROOT / "results" / "ablation"

# ---------------------------------------------------------------------------
# Experiment constants (overridden in --quick mode)
# ---------------------------------------------------------------------------
ACT_NUM_PASSES = 100          # forward passes per config
ACT_BATCH_SIZE = 32
ACT_OBS_DIM = 16
ACT_ACT_DIM = 4

RSSM_NUM_STEPS = 10           # training steps per config
RSSM_SEQ_LEN = 20             # sequence length
RSSM_BATCH_SIZE = 16
RSSM_OBS_DIM = 16
RSSM_ACT_DIM = 4


# ═══════════════════════════════════════════════════════════════════════════════
# Data classes for results
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ACTResult:
    """Results from a single ACT ablation configuration."""

    name: str
    mean_ponder_steps: float = 0.0
    ponder_cost: float = 0.0
    halt_entropy: float = 0.0
    weight_sum_error: float = 0.0
    forward_time_ms: float = 0.0
    halting_distribution: List[float] = field(default_factory=list)


@dataclass
class RSSMResult:
    """Results from a single RSSM ablation configuration."""

    name: str
    total_loss: float = 0.0
    kl_loss: float = 0.0
    reconstruction_loss: float = 0.0
    timing_loss: float = 0.0
    timing_mae: float = 0.0
    loss_curve: List[float] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: ACT Ablation
# ═══════════════════════════════════════════════════════════════════════════════


def _build_act_configs() -> List[Tuple[str, Dict[str, Any]]]:
    """Return the six ACT ablation configurations.

    Each entry is (display_name, kwargs_for_DeliberativeInternalTimeAgent).
    """
    base = dict(
        obs_dim=ACT_OBS_DIM,
        act_dim=ACT_ACT_DIM,
        hidden_dim=64,
        latent_dim=32,
        time_hidden_dim=16,
        max_thinking_steps=5,
        use_internal_time=True,
        transition_type="gru",
        lambda_geo=0.5,
        info_gain_threshold=0.01,
        use_adaptive_steps=False,
        min_steps=1,
        hard_max_steps=10,
        num_heads=1,
    )

    configs: List[Tuple[str, Dict[str, Any]]] = []

    # 1. Full model (all extensions)
    full = {**base, "use_adaptive_steps": True, "num_heads": 4, "head_dim": 16}
    configs.append(("Full model", full))

    # 2. No geometric halting prior (use naive ponder cost)
    #    We keep the prior object but set lambda_geo very close to 0 so it
    #    becomes a near-uniform prior, effectively eliminating the geometric
    #    regularisation signal.
    no_geo = {**base, "use_adaptive_steps": True, "num_heads": 4,
              "head_dim": 16, "lambda_geo": 1e-6}
    configs.append(("No geometric prior", no_geo))

    # 3. No information-theoretic halting (set threshold to 0 so it never fires)
    no_info = {**base, "use_adaptive_steps": True, "num_heads": 4,
               "head_dim": 16, "info_gain_threshold": 0.0}
    configs.append(("No info-gain halt", no_info))

    # 4. No adaptive max steps (fixed max_thinking_steps)
    no_adaptive = {**base, "use_adaptive_steps": False, "num_heads": 4,
                   "head_dim": 16}
    configs.append(("No adaptive steps", no_adaptive))

    # 5. No multi-head deliberation (single head)
    no_multi = {**base, "use_adaptive_steps": True, "num_heads": 1}
    configs.append(("Single head", no_multi))

    # 6. Baseline: no ACT (single forward pass)
    baseline = {**base, "max_thinking_steps": 1, "use_adaptive_steps": False,
                "num_heads": 1}
    configs.append(("No ACT (baseline)", baseline))

    return configs


def _run_act_experiment(
    name: str,
    kwargs: Dict[str, Any],
    num_passes: int,
) -> ACTResult:
    """Run forward passes for one ACT configuration and collect metrics."""
    log.info("  ACT config: %s", name)

    torch.manual_seed(42)
    agent = DeliberativeInternalTimeAgent(**kwargs)
    agent.eval()

    all_ponder_steps: List[float] = []
    all_ponder_costs: List[float] = []
    all_halt_entropies: List[float] = []
    all_weight_sum_errors: List[float] = []
    all_halt_dists: List[torch.Tensor] = []

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_passes):
            obs = torch.randn(ACT_BATCH_SIZE, ACT_OBS_DIM)
            hidden = agent.get_initial_hidden(ACT_BATCH_SIZE, obs.device)
            _dist, _value, _wh, _cum, ponder_cost = agent.forward(obs, hidden)

            diag = agent.get_pondering_diagnostics()
            all_ponder_steps.append(diag.mean_ponder_steps)
            all_ponder_costs.append(float(ponder_cost.mean().item()))
            all_halt_entropies.append(diag.halt_entropy)
            all_weight_sum_errors.append(diag.weight_sum_error)

            # Collect halting distribution (per-step mean across batch)
            hw = agent._last_halt_weights  # (B, N)
            if hw is not None and hw.numel() > 0:
                all_halt_dists.append(hw.mean(dim=0))

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Aggregate halting distribution: zero-pad to same length, then average
    max_len = max((d.shape[0] for d in all_halt_dists), default=0)
    if max_len > 0:
        padded = []
        for d in all_halt_dists:
            if d.shape[0] < max_len:
                d = torch.cat([d, torch.zeros(max_len - d.shape[0])])
            padded.append(d)
        avg_dist = torch.stack(padded).mean(dim=0).tolist()
    else:
        avg_dist = []

    n = max(len(all_ponder_steps), 1)
    return ACTResult(
        name=name,
        mean_ponder_steps=sum(all_ponder_steps) / n,
        ponder_cost=sum(all_ponder_costs) / n,
        halt_entropy=sum(all_halt_entropies) / n,
        weight_sum_error=sum(all_weight_sum_errors) / n,
        forward_time_ms=elapsed_ms / n,
        halting_distribution=avg_dist,
    )


def run_act_ablation(num_passes: int) -> List[ACTResult]:
    """Run the full ACT ablation study."""
    log.info("=" * 60)
    log.info("Section 1: ACT Ablation Study")
    log.info("=" * 60)

    configs = _build_act_configs()
    results: List[ACTResult] = []
    for name, kwargs in configs:
        result = _run_act_experiment(name, kwargs, num_passes)
        log.info(
            "    ponder=%.2f  entropy=%.4f  wt_err=%.6f  time=%.1fms",
            result.mean_ponder_steps,
            result.halt_entropy,
            result.weight_sum_error,
            result.forward_time_ms,
        )
        results.append(result)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: RSSM Ablation
# ═══════════════════════════════════════════════════════════════════════════════


def _build_rssm_configs() -> List[Tuple[str, Dict[str, Any]]]:
    """Return the seven RSSM ablation configurations.

    Each entry is (display_name, kwargs_for_TemporalRSSM).
    """
    base_cat = dict(
        obs_dim=RSSM_OBS_DIM,
        act_dim=RSSM_ACT_DIM,
        hidden_dim=64,
        latent_dim=30,
        num_categories=8,
        category_dim=4,
        kl_balance_alpha=0.8,
        multistep_horizon=5,
        multistep_weight=0.5,
        continue_weight=1.0,
        timing_consistency_weight=0.1,
        timing_concentration=1.0,
        use_symlog=True,
    )

    configs: List[Tuple[str, Dict[str, Any]]] = []

    # 1. Full Dreamer v3
    configs.append(("Full Dreamer v3", {**base_cat}))

    # 2. Gaussian latent (no categorical)
    gauss = {**base_cat}
    del gauss["num_categories"]
    del gauss["category_dim"]
    configs.append(("Gaussian latent", gauss))

    # 3. No symlog
    configs.append(("No symlog", {**base_cat, "use_symlog": False}))

    # 4. Free nats instead of KL balancing (set alpha=0 so both terms
    #    use the standard KL; in Gaussian mode this triggers free-nats
    #    path. For categorical mode we simulate by setting alpha to 0.5
    #    which gives equal weight -- closest analog to non-balanced KL.)
    configs.append(("No KL balancing", {**base_cat, "kl_balance_alpha": 0.5}))

    # 5. No continue predictor
    configs.append(("No continue pred.", {**base_cat, "continue_weight": 0.0}))

    # 6. No multi-step prediction loss
    configs.append(("No multistep loss", {**base_cat, "multistep_horizon": 0}))

    # 7. Normal timing instead of LogNormal (very small concentration -> tiny
    #    sigma in log-space, so LogNormal collapses toward a point mass and
    #    timing predictions lose the heavy-tailed structure)
    configs.append(("Normal-like timing", {**base_cat, "timing_concentration": 0.01}))

    return configs


def _generate_synthetic_data(
    seq_len: int, batch_size: int, obs_dim: int, act_dim: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate synthetic training data for RSSM ablation.

    Returns:
        obs_seq:      (T, B, obs_dim)
        act_seq:      (T, B, act_dim) -- one-hot actions
        dt_seq:       (T, B, 1) -- positive timing values
        continue_seq: (T, B, 1) -- binary continuation flags
        reward_seq:   (T, B, 1) -- reward targets
    """
    obs_seq = torch.randn(seq_len, batch_size, obs_dim)
    # Random one-hot actions
    act_indices = torch.randint(0, act_dim, (seq_len, batch_size))
    act_seq = torch.nn.functional.one_hot(act_indices, act_dim).float()
    # Positive timing values (LogNormal-like)
    dt_seq = torch.exp(torch.randn(seq_len, batch_size, 1) * 0.3 + 0.0).clamp(0.01, 10.0)
    # Episode continues for most steps, terminates at the end
    continue_seq = torch.ones(seq_len, batch_size, 1)
    continue_seq[-1] = 0.0  # terminal
    # Rewards: small signal
    reward_seq = torch.randn(seq_len, batch_size, 1) * 0.5

    return obs_seq, act_seq, dt_seq, continue_seq, reward_seq


def _run_rssm_experiment(
    name: str,
    kwargs: Dict[str, Any],
    num_steps: int,
    seq_len: int,
) -> RSSMResult:
    """Run training steps for one RSSM configuration and collect metrics."""
    log.info("  RSSM config: %s", name)

    torch.manual_seed(42)
    model = TemporalRSSM(**kwargs)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    obs_seq, act_seq, dt_seq, continue_seq, reward_seq = _generate_synthetic_data(
        seq_len, RSSM_BATCH_SIZE, RSSM_OBS_DIM, RSSM_ACT_DIM,
    )

    loss_curve: List[float] = []
    final_losses: Dict[str, float] = {}

    for step_idx in range(num_steps):
        optimizer.zero_grad()
        losses = model.compute_loss(
            obs_seq, act_seq, dt_seq,
            kl_weight=1.0,
            free_nats=3.0,
            continue_seq=continue_seq,
            reward_seq=reward_seq,
        )
        total = losses["total_loss"]
        total.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
        optimizer.step()
        loss_curve.append(float(total.item()))
        final_losses = {k: float(v.item()) for k, v in losses.items()}

    # Compute timing MAE on the final step
    with torch.no_grad():
        out = model.rssm_observe(obs_seq, act_seq)
        timing_preds = []
        for t_idx in range(len(out["h_dets"])):
            mu_dt, _ = model._timing_mean_std(out["h_dets"][t_idx], out["z_posts"][t_idx])
            timing_preds.append(mu_dt)
        timing_pred_stack = torch.stack(timing_preds, dim=0)  # (T, B, 1)
        timing_mae = float((timing_pred_stack - dt_seq).abs().mean().item())

    return RSSMResult(
        name=name,
        total_loss=final_losses.get("total_loss", 0.0),
        kl_loss=final_losses.get("kl_loss", 0.0),
        reconstruction_loss=final_losses.get("reconstruction_loss", 0.0),
        timing_loss=final_losses.get("timing_loss", 0.0),
        timing_mae=timing_mae,
        loss_curve=loss_curve,
    )


def run_rssm_ablation(num_steps: int, seq_len: int) -> List[RSSMResult]:
    """Run the full RSSM ablation study."""
    log.info("=" * 60)
    log.info("Section 2: RSSM Ablation Study")
    log.info("=" * 60)

    configs = _build_rssm_configs()
    results: List[RSSMResult] = []
    for name, kwargs in configs:
        result = _run_rssm_experiment(name, kwargs, num_steps, seq_len)
        log.info(
            "    total=%.4f  kl=%.4f  recon=%.4f  timing=%.4f  MAE=%.4f",
            result.total_loss,
            result.kl_loss,
            result.reconstruction_loss,
            result.timing_loss,
            result.timing_mae,
        )
        results.append(result)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Output generation
# ═══════════════════════════════════════════════════════════════════════════════


def _write_act_markdown(results: List[ACTResult], path: Path) -> None:
    """Write the ACT ablation results as a Markdown table."""
    lines = [
        "# ACT Ablation Results",
        "",
        "| Configuration | Ponder Steps | Ponder Cost | Halt Entropy | Wt-Sum Error | Time (ms) |",
        "|:---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.name} | {r.mean_ponder_steps:.2f} | {r.ponder_cost:.2f} "
            f"| {r.halt_entropy:.4f} | {r.weight_sum_error:.6f} | {r.forward_time_ms:.1f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote %s", path)


def _write_act_latex(results: List[ACTResult], path: Path) -> None:
    """Write the ACT ablation results as a LaTeX table."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{ACT Ablation Study: Impact of disabling individual extensions.}",
        r"\label{tab:act-ablation}",
        r"\begin{tabular}{l r r r r r}",
        r"\toprule",
        r"Configuration & Ponder Steps & Ponder Cost & Halt Entropy & Wt-Sum Err & Time (ms) \\",
        r"\midrule",
    ]
    for r in results:
        safe_name = r.name.replace("&", r"\&")
        lines.append(
            f"{safe_name} & {r.mean_ponder_steps:.2f} & {r.ponder_cost:.2f} "
            f"& {r.halt_entropy:.4f} & {r.weight_sum_error:.6f} & {r.forward_time_ms:.1f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote %s", path)


def _write_rssm_markdown(results: List[RSSMResult], path: Path) -> None:
    """Write the RSSM ablation results as a Markdown table."""
    lines = [
        "# RSSM Ablation Results",
        "",
        "| Configuration | Total Loss | KL Loss | Recon Loss | Timing Loss | Timing MAE |",
        "|:---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.name} | {r.total_loss:.4f} | {r.kl_loss:.4f} "
            f"| {r.reconstruction_loss:.4f} | {r.timing_loss:.4f} | {r.timing_mae:.4f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote %s", path)


def _write_rssm_latex(results: List[RSSMResult], path: Path) -> None:
    """Write the RSSM ablation results as a LaTeX table."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{RSSM Ablation Study: Impact of disabling Dreamer v3 innovations.}",
        r"\label{tab:rssm-ablation}",
        r"\begin{tabular}{l r r r r r}",
        r"\toprule",
        r"Configuration & Total & KL & Recon & Timing & Timing MAE \\",
        r"\midrule",
    ]
    for r in results:
        safe_name = r.name.replace("&", r"\&")
        lines.append(
            f"{safe_name} & {r.total_loss:.4f} & {r.kl_loss:.4f} "
            f"& {r.reconstruction_loss:.4f} & {r.timing_loss:.4f} & {r.timing_mae:.4f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote %s", path)


def _plot_act_halting(results: List[ACTResult], path: Path) -> None:
    """Plot halting distributions across ACT ablation configs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available -- skipping ACT halting plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    max_steps = max((len(r.halting_distribution) for r in results), default=1)

    for r in results:
        dist = r.halting_distribution
        if not dist:
            continue
        steps = list(range(1, len(dist) + 1))
        ax.plot(steps, dist, marker="o", markersize=4, label=r.name, linewidth=1.5)

    ax.set_xlabel("Pondering Step")
    ax.set_ylabel("Mean Halting Weight")
    ax.set_title("ACT Halting Distribution by Configuration")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0.5, max_steps + 0.5)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    log.info("Wrote %s", path)


def _plot_rssm_loss_curves(results: List[RSSMResult], path: Path) -> None:
    """Plot loss convergence curves for RSSM ablation configs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available -- skipping RSSM loss curve plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    for r in results:
        if not r.loss_curve:
            continue
        steps = list(range(1, len(r.loss_curve) + 1))
        ax.plot(steps, r.loss_curve, marker="s", markersize=3, label=r.name,
                linewidth=1.5)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Total Loss")
    ax.set_title("RSSM Loss Convergence by Configuration")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0.5, max((len(r.loss_curve) for r in results), default=1) + 0.5)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    log.info("Wrote %s", path)


def _write_summary_json(
    act_results: Optional[List[ACTResult]],
    rssm_results: Optional[List[RSSMResult]],
    path: Path,
) -> None:
    """Write machine-readable JSON summary of all results."""
    summary: Dict[str, Any] = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    if act_results is not None:
        summary["act_ablation"] = [asdict(r) for r in act_results]

    if rssm_results is not None:
        summary["rssm_ablation"] = [asdict(r) for r in rssm_results]

    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Wrote %s", path)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Entry point for the ablation study."""
    parser = argparse.ArgumentParser(
        description="Run ACT / RSSM ablation study for deltatau-audit."
    )
    parser.add_argument(
        "--section",
        choices=["act", "rssm"],
        default=None,
        help="Run only one section (default: run both).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast mode: fewer forward passes and training steps.",
    )
    args = parser.parse_args()

    # Quick-mode overrides
    if args.quick:
        global ACT_NUM_PASSES, RSSM_NUM_STEPS, RSSM_SEQ_LEN
        ACT_NUM_PASSES = 10
        RSSM_NUM_STEPS = 3
        RSSM_SEQ_LEN = 8

    # Create output directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Results directory: %s", RESULTS_DIR)

    run_act = args.section in (None, "act")
    run_rssm = args.section in (None, "rssm")

    act_results: Optional[List[ACTResult]] = None
    rssm_results: Optional[List[RSSMResult]] = None

    t_start = time.perf_counter()

    # --- ACT ablation ---
    if run_act:
        act_results = run_act_ablation(num_passes=ACT_NUM_PASSES)
        _write_act_markdown(act_results, RESULTS_DIR / "act_ablation.md")
        _write_act_latex(act_results, RESULTS_DIR / "act_ablation.tex")
        _plot_act_halting(act_results, RESULTS_DIR / "act_halting_distribution.png")

    # --- RSSM ablation ---
    if run_rssm:
        rssm_results = run_rssm_ablation(
            num_steps=RSSM_NUM_STEPS, seq_len=RSSM_SEQ_LEN,
        )
        _write_rssm_markdown(rssm_results, RESULTS_DIR / "rssm_ablation.md")
        _write_rssm_latex(rssm_results, RESULTS_DIR / "rssm_ablation.tex")
        _plot_rssm_loss_curves(rssm_results, RESULTS_DIR / "rssm_loss_curves.png")

    # --- JSON summary ---
    _write_summary_json(act_results, rssm_results, RESULTS_DIR / "summary.json")

    elapsed = time.perf_counter() - t_start
    log.info("=" * 60)
    log.info("Ablation study complete in %.1f seconds.", elapsed)
    log.info("Results written to %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
