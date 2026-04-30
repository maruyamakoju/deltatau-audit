"""Lipschitz / spectral-norm utilities shared across certified frontiers.

Used by ``certified_mcts``, ``certified_multiscale_deliberation``, and
``temporal_consistency_distillation``. Each had its own slightly different
implementation; this module unifies them.

The two routines answer different questions:

    - :func:`spectral_norm_estimate` — analytic upper bound for a *linear*
      operator, computed by power iteration on the weight matrix. Cheap.
    - :func:`empirical_lipschitz` — sampling-based estimate for an arbitrary
      callable (e.g. a non-linear network), computed by finite differences
      around given input samples. More expensive, more general.
"""

from __future__ import annotations

from typing import Callable

import torch

__all__ = ["spectral_norm_estimate", "empirical_lipschitz"]


_EPS = 1e-12


def spectral_norm_estimate(weight: torch.Tensor, n_iter: int = 10) -> float:
    """Estimate the largest singular value of ``weight`` via power iteration.

    For a 1-D tensor the abs-max is returned. Higher-order tensors are
    flattened to ``(out, -1)`` (the ``nn.Conv2d`` convention). The result is
    detached and returned as a Python float.
    """
    if weight.numel() == 0:
        return 0.0
    if weight.dim() < 2:
        return float(weight.detach().abs().max().item())

    W = weight.detach().reshape(weight.shape[0], -1)
    out_dim, in_dim = W.shape
    u = torch.randn(out_dim, device=W.device, dtype=W.dtype)
    u = u / (u.norm() + _EPS)
    v = torch.zeros(in_dim, device=W.device, dtype=W.dtype)
    for _ in range(max(1, n_iter)):
        v = W.t() @ u
        v = v / (v.norm() + _EPS)
        u = W @ v
        u = u / (u.norm() + _EPS)
    sigma = (u @ W @ v).abs().item()
    return float(sigma)


def empirical_lipschitz(
    fn: Callable[[torch.Tensor], torch.Tensor],
    samples: torch.Tensor,
    *,
    eps: float = 1e-3,
    n_perturbations: int = 4,
) -> float:
    """Estimate the empirical Lipschitz constant of ``fn`` near ``samples``.

    For each of ``n_perturbations`` random Gaussian deltas of magnitude
    ``eps``, computes ``|fn(x+δ) - fn(x)| / |δ|`` per sample and takes the
    max ratio across all perturbations and samples. ``fn`` is called inside
    a ``no_grad`` block.
    """
    samples = samples.detach()
    with torch.no_grad():
        base = fn(samples)
        max_ratio = 0.0
        for _ in range(max(1, n_perturbations)):
            delta = torch.randn_like(samples) * eps
            perturbed = fn(samples + delta)
            num = (perturbed - base).reshape(samples.shape[0], -1).norm(dim=-1)
            den = delta.reshape(samples.shape[0], -1).norm(dim=-1).clamp(min=_EPS)
            ratio = (num / den).max().item()
            if ratio > max_ratio:
                max_ratio = float(ratio)
    return max_ratio
