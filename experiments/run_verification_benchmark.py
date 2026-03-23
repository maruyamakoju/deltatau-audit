#!/usr/bin/env python3
"""Formal Verification Comparison Benchmark for deltatau-audit.

This script creates feedforward neural networks of varying depth and width,
then runs every verification method implemented in
``deltatau_audit.verification.formal`` on each network.  It measures bound
tightness, computation time, and certification rates, and produces
publication-quality comparison tables (LaTeX) and figures (PNG, 300 DPI).

Usage
-----
Full benchmark (all configurations)::

    python experiments/run_verification_benchmark.py

Quick mode (small subset for CI / smoke testing)::

    python experiments/run_verification_benchmark.py --quick

Output
------
All artefacts are written to ``results/verification/``:

- ``tables/``   -- LaTeX tables (method x network size)
- ``figures/``  -- PNG figures (bound tightness, timing, certification rate)
- ``summary.json`` -- machine-readable aggregate results

Theory
------
Each verification method produces an upper bound on the network's Lipschitz
constant.  A *tighter* bound (closer to the true empirical estimate) is
better.  The benchmark quantifies this via the **tightness ratio**:

    tightness = bound / empirical_estimate

A ratio of 1.0 means the bound equals the best empirical estimate (perfect).
Larger ratios indicate looser (more conservative) bounds.

References
----------
See ``deltatau_audit/verification/formal.py`` for full citations:
- Miyato et al. 2018 (spectral norms)
- Gowal et al. 2019 (IBP)
- Zhang et al. 2018 (CROWN)
- Clopper & Pearson 1934 (statistical bounds)
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

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Imports from the deltatau-audit verification module
# ---------------------------------------------------------------------------
from deltatau_audit.verification.formal import (
    CertificationLevel,
    LipschitzCertificate,
    compute_spectral_lipschitz_bound,
    compute_spectral_norms,
    estimate_holder_exponent,
    propagate_crown,
    propagate_ibp,
    safe_clopper_pearson,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("verification_benchmark")

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "verification"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"


# ============================================================================
# Network factory
# ============================================================================

@dataclass
class NetworkConfig:
    """Configuration for a test network.

    Attributes
    ----------
    name : str
        Human-readable label (e.g. "small-relu").
    n_layers : int
        Number of hidden layers.
    hidden_dim : int
        Width of each hidden layer.
    activation : str
        Activation function: "relu" or "tanh".
    input_dim : int
        Input dimensionality.
    output_dim : int
        Output dimensionality.
    """

    name: str
    n_layers: int
    hidden_dim: int
    activation: str = "relu"
    input_dim: int = 8
    output_dim: int = 4


def build_network(cfg: NetworkConfig) -> nn.Sequential:
    """Construct an ``nn.Sequential`` feedforward network from *cfg*.

    The architecture is::

        Linear(input_dim, hidden_dim) -> Act -> ... -> Linear(hidden_dim, output_dim)

    with *n_layers* hidden layers.  Weights are initialised with Kaiming
    uniform (default PyTorch init) for reproducibility.

    Parameters
    ----------
    cfg : NetworkConfig
        Network specification.

    Returns
    -------
    nn.Sequential
        The constructed network in eval mode.
    """
    act_cls = nn.ReLU if cfg.activation == "relu" else nn.Tanh

    layers: list[nn.Module] = []

    # Input layer
    layers.append(nn.Linear(cfg.input_dim, cfg.hidden_dim))
    layers.append(act_cls())

    # Hidden layers
    for _ in range(cfg.n_layers - 1):
        layers.append(nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
        layers.append(act_cls())

    # Output layer (no activation)
    layers.append(nn.Linear(cfg.hidden_dim, cfg.output_dim))

    model = nn.Sequential(*layers)
    model.eval()
    return model


# ============================================================================
# Benchmark configurations
# ============================================================================

def _full_configs() -> List[NetworkConfig]:
    """Return the full set of network configurations for benchmarking."""
    sizes = [
        ("small",  2,  32),
        ("medium", 3,  64),
        ("large",  4, 128),
        ("deep",   6, 256),
    ]
    activations = ["relu", "tanh"]

    configs: List[NetworkConfig] = []
    for size_name, n_layers, hidden_dim in sizes:
        for act in activations:
            configs.append(NetworkConfig(
                name=f"{size_name}-{act}",
                n_layers=n_layers,
                hidden_dim=hidden_dim,
                activation=act,
            ))
    return configs


def _quick_configs() -> List[NetworkConfig]:
    """Return a minimal set of configs for quick smoke testing."""
    return [
        NetworkConfig(name="small-relu", n_layers=2, hidden_dim=32, activation="relu"),
        NetworkConfig(name="medium-tanh", n_layers=3, hidden_dim=64, activation="tanh"),
    ]


# ============================================================================
# Individual verification runners
# ============================================================================

@dataclass
class MethodResult:
    """Result of running one verification method on one network.

    Attributes
    ----------
    method : str
        Human-readable method name.
    bound : float
        Lipschitz upper bound (or empirical estimate).
    elapsed_s : float
        Wall-clock time in seconds.
    certified : bool
        Whether the network passes the default threshold (L < 5.0).
    extra : Dict[str, Any]
        Method-specific extra information.
    """

    method: str
    bound: float
    elapsed_s: float
    certified: bool
    extra: Dict[str, Any] = field(default_factory=dict)


def _run_empirical_jacobian(
    model: nn.Sequential,
    x: torch.Tensor,
    n_samples: int,
) -> MethodResult:
    """Level 1: Empirical Jacobian sampling.

    Evaluates the network at *n_samples* input points perturbed around *x*
    and computes the Jacobian norm via autograd.  The maximum over all
    samples serves as the empirical Lipschitz estimate.

    Parameters
    ----------
    model : nn.Sequential
        Network to verify.
    x : torch.Tensor
        Centre input point, shape ``(1, input_dim)``.
    n_samples : int
        Number of perturbation samples.

    Returns
    -------
    MethodResult
    """
    t0 = time.perf_counter()

    lipschitz_vals: List[float] = []
    perturbations = torch.randn(n_samples, x.shape[1]) * 0.5 + x

    for i in range(n_samples):
        xi = perturbations[i : i + 1].clone().detach().requires_grad_(True)
        out = model(xi)
        # Compute full Jacobian norm
        jac_rows = []
        for j in range(out.shape[1]):
            grad = torch.autograd.grad(
                out[0, j], xi, retain_graph=True, create_graph=False
            )[0]
            jac_rows.append(grad)
        J = torch.stack(jac_rows, dim=0).squeeze(1)  # (out_dim, in_dim)
        # Spectral norm of the Jacobian = largest singular value
        sv = torch.linalg.svdvals(J)
        lipschitz_vals.append(float(sv[0].item()))

    elapsed = time.perf_counter() - t0
    l_max = max(lipschitz_vals) if lipschitz_vals else 0.0
    l_mean = float(np.mean(lipschitz_vals)) if lipschitz_vals else 0.0

    return MethodResult(
        method=f"Empirical (n={n_samples})",
        bound=l_max,
        elapsed_s=elapsed,
        certified=l_max < 5.0,
        extra={"n_samples": n_samples, "L_mean": l_mean},
    )


def _run_clopper_pearson(
    model: nn.Sequential,
    x: torch.Tensor,
    n_samples: int,
    threshold: float = 5.0,
) -> MethodResult:
    """Level 2: Monte Carlo + Clopper-Pearson statistical bound.

    Counts the fraction of sampled points where the local Lipschitz constant
    exceeds *threshold*, then computes a 95% Clopper-Pearson confidence
    interval on the true violation rate.

    Parameters
    ----------
    model : nn.Sequential
        Network to verify.
    x : torch.Tensor
        Centre input point.
    n_samples : int
        Number of Monte Carlo samples.
    threshold : float
        Lipschitz violation threshold.

    Returns
    -------
    MethodResult
    """
    t0 = time.perf_counter()

    violations = 0
    l_vals: List[float] = []
    perturbations = torch.randn(n_samples, x.shape[1]) * 0.5 + x

    for i in range(n_samples):
        xi = perturbations[i : i + 1].clone().detach().requires_grad_(True)
        out = model(xi)
        jac_rows = []
        for j in range(out.shape[1]):
            grad = torch.autograd.grad(
                out[0, j], xi, retain_graph=True, create_graph=False
            )[0]
            jac_rows.append(grad)
        J = torch.stack(jac_rows, dim=0).squeeze(1)
        sv = torch.linalg.svdvals(J)
        lc = float(sv[0].item())
        l_vals.append(lc)
        if lc > threshold:
            violations += 1

    ci_lo, ci_hi = safe_clopper_pearson(violations, n_samples, alpha=0.05)
    elapsed = time.perf_counter() - t0

    l_max = max(l_vals) if l_vals else 0.0

    return MethodResult(
        method="Clopper-Pearson",
        bound=l_max,
        elapsed_s=elapsed,
        certified=(ci_hi < 0.01),  # certified if upper CI on violation < 1%
        extra={
            "violations": violations,
            "violation_rate": violations / max(n_samples, 1),
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "n_samples": n_samples,
        },
    )


def _run_ibp(
    model: nn.Sequential,
    x: torch.Tensor,
    epsilon: float,
) -> MethodResult:
    """Level 3: Interval Bound Propagation.

    Propagates the interval ``[x - eps, x + eps]`` through the network and
    derives a Lipschitz bound from the output spread.

    Parameters
    ----------
    model : nn.Sequential
        Network to verify.
    x : torch.Tensor
        Centre input point.
    epsilon : float
        L-infinity perturbation radius.

    Returns
    -------
    MethodResult
    """
    t0 = time.perf_counter()
    bounds = propagate_ibp(model, x, epsilon, threshold=0.1)
    elapsed = time.perf_counter() - t0

    # Derive Lipschitz bound: max output change / (2 * epsilon)
    l_ibp = bounds.max_spread / (2 * epsilon) if epsilon > 1e-12 else float("inf")

    return MethodResult(
        method=f"IBP (eps={epsilon})",
        bound=l_ibp,
        elapsed_s=elapsed,
        certified=bounds.certified_robust,
        extra={
            "epsilon": epsilon,
            "max_spread": bounds.max_spread,
            "certified_robust": bounds.certified_robust,
        },
    )


def _run_spectral(
    model: nn.Sequential,
) -> MethodResult:
    """Level 4: Spectral norm product bound.

    Computes the product of per-layer spectral norms, which is a provable
    upper bound on the network Lipschitz constant for networks with
    1-Lipschitz activations.

    Parameters
    ----------
    model : nn.Sequential
        Network to verify.

    Returns
    -------
    MethodResult
    """
    t0 = time.perf_counter()
    l_spectral, layer_norms = compute_spectral_lipschitz_bound(model, n_iters=20)
    elapsed = time.perf_counter() - t0

    return MethodResult(
        method="Spectral",
        bound=l_spectral,
        elapsed_s=elapsed,
        certified=l_spectral < 5.0,
        extra={
            "layer_norms": layer_norms,
            "n_layers": len(layer_norms),
        },
    )


def _run_crown(
    model: nn.Sequential,
    x: torch.Tensor,
    epsilon: float,
) -> MethodResult:
    """Level 5: CROWN linear relaxation.

    Computes tighter-than-IBP bounds via backward linear relaxation, then
    derives a Lipschitz bound from the output spread.

    Parameters
    ----------
    model : nn.Sequential
        Network to verify.
    x : torch.Tensor
        Centre input point.
    epsilon : float
        L-infinity perturbation radius.

    Returns
    -------
    MethodResult
    """
    t0 = time.perf_counter()
    bounds = propagate_crown(model, x, epsilon, threshold=0.1)
    elapsed = time.perf_counter() - t0

    l_crown = bounds.max_spread / (2 * epsilon) if epsilon > 1e-12 else float("inf")

    return MethodResult(
        method=f"CROWN (eps={epsilon})",
        bound=l_crown,
        elapsed_s=elapsed,
        certified=bounds.certified_robust,
        extra={
            "epsilon": epsilon,
            "max_spread": bounds.max_spread,
            "certified_robust": bounds.certified_robust,
        },
    )


def _run_holder(
    model: nn.Sequential,
    x: torch.Tensor,
    n_samples: int = 500,
) -> MethodResult:
    """Holder exponent estimation.

    Evaluates the network at *n_samples* nearby points and estimates the
    Holder exponent alpha and constant C via log-log regression on pairwise
    output differences.

    Parameters
    ----------
    model : nn.Sequential
        Network to verify.
    x : torch.Tensor
        Centre input point.
    n_samples : int
        Number of sample points.

    Returns
    -------
    MethodResult
    """
    t0 = time.perf_counter()

    # Generate sample inputs along random directions from x
    directions = torch.randn(n_samples, x.shape[1])
    directions = directions / (directions.norm(dim=1, keepdim=True) + 1e-12)
    scales = torch.linspace(0.001, 1.0, n_samples).unsqueeze(1)
    inputs = x + directions * scales  # (n_samples, input_dim)

    with torch.no_grad():
        outputs = model(inputs).numpy()
        inputs_np = inputs.numpy()

    # Pairwise differences (consecutive pairs for efficiency)
    diffs_input = np.linalg.norm(inputs_np[1:] - inputs_np[:-1], axis=1)
    diffs_output = np.linalg.norm(outputs[1:] - outputs[:-1], axis=1)

    alpha, C = estimate_holder_exponent(diffs_input, diffs_output)
    elapsed = time.perf_counter() - t0

    # Derive Lipschitz bound from Holder: if alpha >= 1, then L = C
    # (for alpha < 1, the Lipschitz constant is technically infinite)
    if math.isinf(alpha):
        l_holder = 0.0
    elif alpha >= 1.0:
        l_holder = C
    else:
        l_holder = float("inf")

    return MethodResult(
        method="Holder",
        bound=l_holder,
        elapsed_s=elapsed,
        certified=l_holder < 5.0 and alpha >= 1.0,
        extra={
            "alpha": alpha,
            "C": C,
            "interpretation": (
                "constant" if math.isinf(alpha) else
                "smoother than Lipschitz" if alpha > 1.0 else
                "Lipschitz" if abs(alpha - 1.0) < 0.1 else
                "rough (sub-Lipschitz)"
            ),
        },
    )


# ============================================================================
# Full benchmark runner
# ============================================================================

@dataclass
class NetworkBenchmark:
    """Complete benchmark results for one network configuration.

    Attributes
    ----------
    config : NetworkConfig
        Network specification.
    empirical_baseline : float
        Best empirical Lipschitz estimate (from 1000-sample Jacobian).
    results : List[MethodResult]
        Results from every verification method.
    """

    config: NetworkConfig
    empirical_baseline: float
    results: List[MethodResult]


def run_benchmark_for_network(
    cfg: NetworkConfig,
    quick: bool = False,
) -> NetworkBenchmark:
    """Run all verification methods on a single network.

    Parameters
    ----------
    cfg : NetworkConfig
        Network to benchmark.
    quick : bool
        If True, use smaller sample counts and fewer epsilon values.

    Returns
    -------
    NetworkBenchmark
        Aggregate results.
    """
    log.info("=" * 60)
    log.info("Network: %s  (layers=%d, hidden=%d, act=%s)",
             cfg.name, cfg.n_layers, cfg.hidden_dim, cfg.activation)
    log.info("=" * 60)

    torch.manual_seed(42)
    model = build_network(cfg)
    x = torch.randn(1, cfg.input_dim)

    results: List[MethodResult] = []

    # ── Level 1: Empirical Jacobian sampling ──
    empirical_samples = [100, 500, 1000] if not quick else [100]
    for ns in empirical_samples:
        log.info("  Running Empirical Jacobian (n=%d) ...", ns)
        r = _run_empirical_jacobian(model, x, ns)
        results.append(r)
        log.info("    -> bound=%.4f  time=%.3fs", r.bound, r.elapsed_s)

    # Use the largest-sample empirical result as baseline
    empirical_baseline = max(
        (r.bound for r in results if r.method.startswith("Empirical")),
        default=0.0,
    )

    # ── Level 2: Clopper-Pearson ──
    cp_samples = 500 if not quick else 100
    log.info("  Running Clopper-Pearson (n=%d) ...", cp_samples)
    r = _run_clopper_pearson(model, x, cp_samples)
    results.append(r)
    log.info("    -> bound=%.4f  viol_rate=%.4f  CI=[%.4f, %.4f]  time=%.3fs",
             r.bound, r.extra["violation_rate"],
             r.extra["ci_lower"], r.extra["ci_upper"], r.elapsed_s)

    # ── Level 3: IBP at multiple epsilon values ──
    ibp_epsilons = [0.01, 0.05, 0.1, 0.5] if not quick else [0.01, 0.1]
    for eps in ibp_epsilons:
        log.info("  Running IBP (eps=%.3f) ...", eps)
        r = _run_ibp(model, x, eps)
        results.append(r)
        log.info("    -> bound=%.4f  certified=%s  time=%.3fs",
                 r.bound, r.certified, r.elapsed_s)

    # ── Level 4: Spectral norm product ──
    log.info("  Running Spectral norm product ...")
    r = _run_spectral(model)
    results.append(r)
    log.info("    -> bound=%.4f  n_layers=%d  time=%.3fs",
             r.bound, r.extra["n_layers"], r.elapsed_s)

    # ── Level 5: CROWN ──
    crown_epsilons = [0.01, 0.05, 0.1, 0.5] if not quick else [0.01, 0.1]
    for eps in crown_epsilons:
        log.info("  Running CROWN (eps=%.3f) ...", eps)
        r = _run_crown(model, x, eps)
        results.append(r)
        log.info("    -> bound=%.4f  certified=%s  time=%.3fs",
                 r.bound, r.certified, r.elapsed_s)

    # ── Holder exponent ──
    holder_n = 500 if not quick else 100
    log.info("  Running Holder estimation (n=%d) ...", holder_n)
    r = _run_holder(model, x, n_samples=holder_n)
    results.append(r)
    log.info("    -> alpha=%.4f  C=%.4f  time=%.3fs",
             r.extra["alpha"], r.extra["C"], r.elapsed_s)

    return NetworkBenchmark(
        config=cfg,
        empirical_baseline=empirical_baseline,
        results=results,
    )


# ============================================================================
# Table generation (LaTeX)
# ============================================================================

def _tightness_ratio(bound: float, baseline: float) -> float:
    """Compute bound tightness: ratio of formal bound to empirical estimate.

    A ratio of 1.0 means the bound is as tight as the best empirical
    estimate.  Larger is looser.
    """
    if baseline < 1e-12 or math.isinf(bound) or math.isnan(bound):
        return float("inf")
    return bound / baseline


def generate_comparison_table(benchmarks: List[NetworkBenchmark]) -> str:
    """Generate a LaTeX table: Method x Network -> (bound, time, certified).

    Highlights the tightest (non-empirical) bound per network with bold
    formatting.

    Parameters
    ----------
    benchmarks : List[NetworkBenchmark]
        Results from all network configurations.

    Returns
    -------
    str
        LaTeX source for a ``tabular`` environment.
    """
    # Collect all unique method names, preserving order
    all_methods: List[str] = []
    for bm in benchmarks:
        for r in bm.results:
            if r.method not in all_methods:
                all_methods.append(r.method)

    net_names = [bm.config.name for bm in benchmarks]
    n_nets = len(net_names)

    # Header
    col_spec = "l" + "rrr" * n_nets
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Verification Benchmark: Method $\times$ Network Size.  "
        r"Columns show (Lipschitz bound, time [s], certified?).  "
        r"\textbf{Bold} marks the tightest provable bound per network.}",
        r"\label{tab:verification-benchmark}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    # Multi-column header for each network
    header1_parts = [r"\textbf{Method}"]
    header2_parts = [""]
    for name in net_names:
        safe_name = name.replace("_", r"\_")
        header1_parts.append(rf"\multicolumn{{3}}{{c}}{{\textbf{{{safe_name}}}}}")
        header2_parts.append(r"$L$ & $t$ [s] & Cert.")
    lines.append(" & ".join(header1_parts) + r" \\")
    lines.append(r"\cmidrule(lr){2-" + str(1 + 3 * n_nets) + "}")
    lines.append(" & ".join(header2_parts) + r" \\")
    lines.append(r"\midrule")

    # Find the tightest provable (non-empirical) bound per network
    tightest_provable: Dict[str, Tuple[float, str]] = {}
    provable_methods = {"Spectral"}  # always provable
    for bm in benchmarks:
        best_bound = float("inf")
        best_method = ""
        for r in bm.results:
            is_provable = (
                r.method.startswith("IBP")
                or r.method.startswith("CROWN")
                or r.method == "Spectral"
            )
            if is_provable and r.bound < best_bound:
                best_bound = r.bound
                best_method = r.method
        tightest_provable[bm.config.name] = (best_bound, best_method)

    # Data rows
    for method_name in all_methods:
        safe_method = method_name.replace("_", r"\_")
        row_parts = [safe_method]
        for bm in benchmarks:
            # Find this method's result for this network
            matched = [r for r in bm.results if r.method == method_name]
            if matched:
                r = matched[0]
                # Check if this is the tightest provable bound
                is_best = (
                    tightest_provable[bm.config.name][1] == method_name
                    and method_name != ""
                )
                bound_str = f"{r.bound:.4f}" if not math.isinf(r.bound) else r"$\infty$"
                if is_best:
                    bound_str = r"\textbf{" + bound_str + "}"
                time_str = f"{r.elapsed_s:.3f}"
                cert_str = r"\checkmark" if r.certified else r"--"
                row_parts.append(f"{bound_str} & {time_str} & {cert_str}")
            else:
                row_parts.append(r"-- & -- & --")
        lines.append(" & ".join(row_parts) + r" \\")

    # Baseline row
    lines.append(r"\midrule")
    baseline_parts = [r"\textit{Empirical baseline}"]
    for bm in benchmarks:
        baseline_parts.append(
            f"{bm.empirical_baseline:.4f} & -- & --"
        )
    lines.append(" & ".join(baseline_parts) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_tightness_table(benchmarks: List[NetworkBenchmark]) -> str:
    """Generate a LaTeX table of bound tightness ratios.

    Each cell shows ``bound / empirical_baseline`` so that 1.0 is
    optimal (bound equals the empirical estimate) and larger values
    indicate looser bounds.

    Parameters
    ----------
    benchmarks : List[NetworkBenchmark]
        Benchmark results.

    Returns
    -------
    str
        LaTeX source.
    """
    all_methods: List[str] = []
    for bm in benchmarks:
        for r in bm.results:
            if r.method not in all_methods:
                all_methods.append(r.method)

    net_names = [bm.config.name for bm in benchmarks]
    n_nets = len(net_names)
    col_spec = "l" + "r" * n_nets

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Bound Tightness Ratios (bound / empirical baseline).  "
        r"Lower is tighter; 1.0 is optimal.  \textbf{Bold} marks the tightest "
        r"provable bound per column.}",
        r"\label{tab:tightness-ratios}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    header_parts = [r"\textbf{Method}"]
    for name in net_names:
        header_parts.append(r"\textbf{" + name.replace("_", r"\_") + "}")
    lines.append(" & ".join(header_parts) + r" \\")
    lines.append(r"\midrule")

    # Find best provable tightness per network
    best_provable_ratio: Dict[str, float] = {}
    for bm in benchmarks:
        best = float("inf")
        for r in bm.results:
            is_provable = (
                r.method.startswith("IBP")
                or r.method.startswith("CROWN")
                or r.method == "Spectral"
            )
            if is_provable:
                ratio = _tightness_ratio(r.bound, bm.empirical_baseline)
                if ratio < best:
                    best = ratio
        best_provable_ratio[bm.config.name] = best

    for method_name in all_methods:
        safe_method = method_name.replace("_", r"\_")
        row_parts = [safe_method]
        for bm in benchmarks:
            matched = [r for r in bm.results if r.method == method_name]
            if matched:
                ratio = _tightness_ratio(matched[0].bound, bm.empirical_baseline)
                if math.isinf(ratio):
                    cell = r"$\infty$"
                else:
                    cell = f"{ratio:.2f}"
                    # Bold if this is the best provable for this network
                    is_provable = (
                        method_name.startswith("IBP")
                        or method_name.startswith("CROWN")
                        or method_name == "Spectral"
                    )
                    if (is_provable and
                            abs(ratio - best_provable_ratio[bm.config.name]) < 1e-6):
                        cell = r"\textbf{" + cell + "}"
                row_parts.append(cell)
            else:
                row_parts.append("--")
        lines.append(" & ".join(row_parts) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


# ============================================================================
# Figure generation
# ============================================================================

def generate_figures(benchmarks: List[NetworkBenchmark]) -> None:
    """Generate publication-quality PNG figures.

    Creates three figures:
    1. Bound tightness vs. network depth (log scale).
    2. Computation time vs. network size.
    3. Certification rate vs. epsilon threshold.

    All figures are saved to ``FIGURES_DIR`` at 300 DPI.

    Parameters
    ----------
    benchmarks : List[NetworkBenchmark]
        Benchmark results.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        log.warning("matplotlib not available -- skipping figure generation")
        return

    # Publication style
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    # ── Figure 1: Bound tightness vs network depth (log scale) ──
    fig1, ax1 = plt.subplots(figsize=(8, 5))

    # Group by depth
    depth_to_benchmarks: Dict[int, List[NetworkBenchmark]] = {}
    for bm in benchmarks:
        d = bm.config.n_layers
        depth_to_benchmarks.setdefault(d, []).append(bm)

    # Collect method categories
    method_categories = {
        "Empirical (best)": lambda r: r.method.startswith("Empirical"),
        "IBP (eps=0.01)": lambda r: r.method == "IBP (eps=0.01)",
        "IBP (eps=0.1)": lambda r: r.method == "IBP (eps=0.1)",
        "CROWN (eps=0.01)": lambda r: r.method == "CROWN (eps=0.01)",
        "CROWN (eps=0.1)": lambda r: r.method == "CROWN (eps=0.1)",
        "Spectral": lambda r: r.method == "Spectral",
    }
    colours = ["#2ca02c", "#1f77b4", "#1f77b4", "#ff7f0e", "#ff7f0e", "#d62728"]
    markers = ["o", "s", "D", "^", "v", "P"]
    linestyles = ["-", "-", "--", "-", "--", "-"]

    depths = sorted(depth_to_benchmarks.keys())
    for idx, (cat_name, cat_filter) in enumerate(method_categories.items()):
        y_vals = []
        for d in depths:
            bms = depth_to_benchmarks[d]
            bounds = []
            for bm in bms:
                matching = [r for r in bm.results if cat_filter(r)]
                if matching:
                    # For empirical, take the max bound across sample sizes
                    if cat_name.startswith("Empirical"):
                        bounds.append(max(r.bound for r in matching))
                    else:
                        bounds.append(matching[0].bound)
            y_vals.append(np.mean(bounds) if bounds else float("nan"))

        valid = [(d, y) for d, y in zip(depths, y_vals)
                 if not (math.isnan(y) or math.isinf(y))]
        if valid:
            xv, yv = zip(*valid)
            ax1.semilogy(
                xv, yv,
                marker=markers[idx],
                color=colours[idx],
                linestyle=linestyles[idx],
                label=cat_name,
                linewidth=1.5,
                markersize=6,
            )

    ax1.set_xlabel("Network Depth (number of hidden layers)")
    ax1.set_ylabel("Lipschitz Bound (log scale)")
    ax1.set_title("Bound Tightness vs. Network Depth")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.set_xticks(depths)
    fig1.savefig(FIGURES_DIR / "bound_vs_depth.png")
    plt.close(fig1)
    log.info("  Saved figures/bound_vs_depth.png")

    # ── Figure 2: Computation time vs network size ──
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    # x-axis: total network parameters
    time_categories = {
        "Empirical (n=100)": lambda r: r.method == "Empirical (n=100)",
        "Clopper-Pearson": lambda r: r.method == "Clopper-Pearson",
        "IBP (eps=0.1)": lambda r: r.method == "IBP (eps=0.1)",
        "Spectral": lambda r: r.method == "Spectral",
        "CROWN (eps=0.1)": lambda r: r.method == "CROWN (eps=0.1)",
        "Holder": lambda r: r.method == "Holder",
    }
    time_colours = ["#2ca02c", "#9467bd", "#1f77b4", "#d62728", "#ff7f0e", "#8c564b"]
    time_markers = ["o", "s", "D", "P", "^", "X"]

    net_sizes = []
    for bm in benchmarks:
        model = build_network(bm.config)
        n_params = sum(p.numel() for p in model.parameters())
        net_sizes.append(n_params)

    for idx, (cat_name, cat_filter) in enumerate(time_categories.items()):
        x_vals = []
        y_vals = []
        for i, bm in enumerate(benchmarks):
            matching = [r for r in bm.results if cat_filter(r)]
            if matching:
                x_vals.append(net_sizes[i])
                y_vals.append(matching[0].elapsed_s)
        if x_vals:
            ax2.semilogy(
                x_vals, y_vals,
                marker=time_markers[idx],
                color=time_colours[idx],
                label=cat_name,
                linewidth=1.5,
                markersize=6,
            )

    ax2.set_xlabel("Network Size (number of parameters)")
    ax2.set_ylabel("Computation Time [s] (log scale)")
    ax2.set_title("Verification Time vs. Network Size")
    ax2.legend(loc="upper left", framealpha=0.9)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x / 1000:.0f}k" if x >= 1000 else f"{x:.0f}"
    ))
    fig2.savefig(FIGURES_DIR / "time_vs_size.png")
    plt.close(fig2)
    log.info("  Saved figures/time_vs_size.png")

    # ── Figure 3: Certification rate vs epsilon threshold ──
    fig3, ax3 = plt.subplots(figsize=(8, 5))

    # For each epsilon, compute the fraction of networks certified by IBP / CROWN
    eps_values = sorted(set(
        r.extra.get("epsilon", None)
        for bm in benchmarks
        for r in bm.results
        if r.extra.get("epsilon") is not None
    ))

    ibp_cert_rates = []
    crown_cert_rates = []
    for eps in eps_values:
        ibp_certified = 0
        ibp_total = 0
        crown_certified = 0
        crown_total = 0
        for bm in benchmarks:
            for r in bm.results:
                if r.method == f"IBP (eps={eps})":
                    ibp_total += 1
                    if r.certified:
                        ibp_certified += 1
                if r.method == f"CROWN (eps={eps})":
                    crown_total += 1
                    if r.certified:
                        crown_certified += 1
        ibp_cert_rates.append(ibp_certified / max(ibp_total, 1))
        crown_cert_rates.append(crown_certified / max(crown_total, 1))

    if eps_values:
        ax3.plot(
            eps_values, ibp_cert_rates,
            marker="s", color="#1f77b4", label="IBP",
            linewidth=2, markersize=8,
        )
        ax3.plot(
            eps_values, crown_cert_rates,
            marker="^", color="#ff7f0e", label="CROWN",
            linewidth=2, markersize=8,
        )
        ax3.set_xlabel(r"Perturbation Radius $\epsilon$")
        ax3.set_ylabel("Certification Rate")
        ax3.set_title(r"Certification Rate vs. $\epsilon$ Threshold")
        ax3.set_ylim(-0.05, 1.05)
        ax3.legend(loc="upper right", framealpha=0.9)

    fig3.savefig(FIGURES_DIR / "cert_rate_vs_epsilon.png")
    plt.close(fig3)
    log.info("  Saved figures/cert_rate_vs_epsilon.png")


# ============================================================================
# Summary JSON
# ============================================================================

def generate_summary(benchmarks: List[NetworkBenchmark]) -> Dict[str, Any]:
    """Produce a machine-readable summary of all benchmark results.

    Parameters
    ----------
    benchmarks : List[NetworkBenchmark]
        Benchmark results.

    Returns
    -------
    Dict[str, Any]
        Nested dict suitable for JSON serialization.
    """
    summary: Dict[str, Any] = {
        "n_networks": len(benchmarks),
        "networks": {},
    }

    for bm in benchmarks:
        net_entry: Dict[str, Any] = {
            "config": {
                "name": bm.config.name,
                "n_layers": bm.config.n_layers,
                "hidden_dim": bm.config.hidden_dim,
                "activation": bm.config.activation,
                "input_dim": bm.config.input_dim,
                "output_dim": bm.config.output_dim,
            },
            "empirical_baseline": bm.empirical_baseline,
            "methods": {},
        }

        for r in bm.results:
            method_entry: Dict[str, Any] = {
                "bound": r.bound if not math.isinf(r.bound) else "inf",
                "elapsed_s": round(r.elapsed_s, 6),
                "certified": r.certified,
                "tightness_ratio": (
                    round(_tightness_ratio(r.bound, bm.empirical_baseline), 4)
                    if not math.isinf(_tightness_ratio(r.bound, bm.empirical_baseline))
                    else "inf"
                ),
            }
            # Include meaningful extras (filter out large tensors)
            for k, v in r.extra.items():
                if k == "layer_norms":
                    method_entry[k] = [round(x, 6) for x in v]
                elif isinstance(v, (int, float, str, bool)):
                    method_entry[k] = v
            net_entry["methods"][r.method] = method_entry

        summary["networks"][bm.config.name] = net_entry

    # Aggregate statistics
    all_spectral_ratios = []
    all_ibp_ratios = []
    all_crown_ratios = []
    for bm in benchmarks:
        for r in bm.results:
            ratio = _tightness_ratio(r.bound, bm.empirical_baseline)
            if not math.isinf(ratio):
                if r.method == "Spectral":
                    all_spectral_ratios.append(ratio)
                elif r.method.startswith("IBP"):
                    all_ibp_ratios.append(ratio)
                elif r.method.startswith("CROWN"):
                    all_crown_ratios.append(ratio)

    summary["aggregate"] = {
        "spectral_tightness": {
            "mean": round(float(np.mean(all_spectral_ratios)), 4) if all_spectral_ratios else None,
            "median": round(float(np.median(all_spectral_ratios)), 4) if all_spectral_ratios else None,
        },
        "ibp_tightness": {
            "mean": round(float(np.mean(all_ibp_ratios)), 4) if all_ibp_ratios else None,
            "median": round(float(np.median(all_ibp_ratios)), 4) if all_ibp_ratios else None,
        },
        "crown_tightness": {
            "mean": round(float(np.mean(all_crown_ratios)), 4) if all_crown_ratios else None,
            "median": round(float(np.median(all_crown_ratios)), 4) if all_crown_ratios else None,
        },
    }

    return summary


# ============================================================================
# Main entry point
# ============================================================================

def main() -> None:
    """Run the verification comparison benchmark.

    Parses CLI arguments, runs all verification methods on each network
    configuration, and writes tables, figures, and a JSON summary to
    ``results/verification/``.
    """
    parser = argparse.ArgumentParser(
        description="Formal verification comparison benchmark for deltatau-audit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: use a small subset of networks and fewer samples.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"

    # Update global dirs for figure generation
    global TABLES_DIR, FIGURES_DIR
    TABLES_DIR = tables_dir
    FIGURES_DIR = figures_dir

    # Create output directories
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    log.info("Verification Benchmark")
    log.info("  Mode: %s", "quick" if args.quick else "full")
    log.info("  Output: %s", output_dir)

    # Select configurations
    configs = _quick_configs() if args.quick else _full_configs()
    log.info("  Networks: %d configurations", len(configs))

    # Run benchmarks
    t_total_start = time.perf_counter()
    benchmarks: List[NetworkBenchmark] = []
    for cfg in configs:
        bm = run_benchmark_for_network(cfg, quick=args.quick)
        benchmarks.append(bm)

    t_total = time.perf_counter() - t_total_start
    log.info("=" * 60)
    log.info("All benchmarks complete.  Total time: %.1fs", t_total)

    # ── Generate tables ──
    log.info("Generating LaTeX tables ...")
    comparison_table = generate_comparison_table(benchmarks)
    tightness_table = generate_tightness_table(benchmarks)

    with open(tables_dir / "comparison_table.tex", "w") as f:
        f.write(comparison_table)
    log.info("  Saved tables/comparison_table.tex")

    with open(tables_dir / "tightness_table.tex", "w") as f:
        f.write(tightness_table)
    log.info("  Saved tables/tightness_table.tex")

    # ── Generate figures ──
    log.info("Generating figures ...")
    generate_figures(benchmarks)

    # ── Generate summary JSON ──
    log.info("Generating summary.json ...")
    summary = generate_summary(benchmarks)
    summary["total_time_s"] = round(t_total, 3)
    summary["mode"] = "quick" if args.quick else "full"

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("  Saved summary.json")

    # ── Print summary to console ──
    print("\n" + "=" * 70)
    print("VERIFICATION BENCHMARK SUMMARY")
    print("=" * 70)
    for bm in benchmarks:
        print(f"\n  {bm.config.name}  "
              f"(depth={bm.config.n_layers}, width={bm.config.hidden_dim}, "
              f"act={bm.config.activation})")
        print(f"    Empirical baseline: L = {bm.empirical_baseline:.4f}")
        for r in bm.results:
            ratio = _tightness_ratio(r.bound, bm.empirical_baseline)
            ratio_str = f"{ratio:.2f}x" if not math.isinf(ratio) else "inf"
            cert_mark = "PASS" if r.certified else "FAIL"
            bound_str = f"{r.bound:.4f}" if not math.isinf(r.bound) else "inf"
            print(f"    {r.method:30s}  L={bound_str:>10s}  "
                  f"ratio={ratio_str:>8s}  {r.elapsed_s:6.3f}s  [{cert_mark}]")

    print(f"\nTotal time: {t_total:.1f}s")
    print(f"Output: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
