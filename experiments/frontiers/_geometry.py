"""Hyperbolic / Poincaré-ball geometry primitives shared by frontier experiments.

Five frontier modules independently re-defined ``mobius_add`` and ``exp_map0``
with identical formulas (up to ``+1e-8`` epsilons in slightly different places).
The duplication is mathematically interesting but operationally annoying: a
formula tweak in one file leaves the other four wrong, and there is nothing
in the type signatures that hints at the relationship.

This module is the single source of truth. Existing files can either import
from here or keep their local copies; new files should import from here.

References:
    Ganea et al., "Hyperbolic Neural Networks" (NeurIPS 2018) — Möbius
    operations on the Poincaré ball model.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "mobius_add",
    "exp_map0",
    "log_map0",
    "poincare_distance",
]


_EPS = 1e-8


def mobius_add(u: torch.Tensor, v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Möbius addition on the Poincaré ball of curvature ``-c``.

    ``u`` and ``v`` must have the same trailing dimension. Reduces along
    ``dim=-1`` for the inner-product / squared-norm terms; broadcasting on
    leading dims is preserved.
    """
    v = v + _EPS
    u_norm2 = torch.sum(u * u, dim=-1, keepdim=True)
    v_norm2 = torch.sum(v * v, dim=-1, keepdim=True)
    uv = torch.sum(u * v, dim=-1, keepdim=True)
    num = (1 + 2 * c * uv + c * v_norm2) * u + (1 - c * u_norm2) * v
    den = 1 + 2 * c * uv + (c**2) * u_norm2 * v_norm2
    return num / (den + _EPS)


def exp_map0(u: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Exponential map at the origin of the Poincaré ball of curvature ``-c``.

    Maps a tangent vector ``u`` to the manifold point that lies a distance
    ``|u|`` along the geodesic in direction ``u/|u|``.
    """
    u_norm = torch.norm(u, p=2, dim=-1, keepdim=True)
    sqrt_c = math.sqrt(c)
    return torch.tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm + _EPS)


def log_map0(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Logarithmic map at the origin — inverse of :func:`exp_map0`.

    Provided for completeness; not currently used by the frontier code but
    cheap to keep alongside its inverse.
    """
    x_norm = torch.norm(x, p=2, dim=-1, keepdim=True).clamp(min=_EPS, max=1.0 - _EPS)
    sqrt_c = math.sqrt(c)
    return torch.atanh(sqrt_c * x_norm) * x / (sqrt_c * x_norm + _EPS)


def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Geodesic (hyperbolic) distance between points on the Poincaré ball."""
    sqrt_c = math.sqrt(c)
    diff = mobius_add(-x, y, c=c)
    diff_norm = torch.norm(diff, p=2, dim=-1).clamp(min=_EPS, max=1.0 - _EPS)
    return (2.0 / sqrt_c) * torch.atanh(sqrt_c * diff_norm)
