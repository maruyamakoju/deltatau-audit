"""Shared ACT bookkeeping utilities.

These helpers keep the halting semantics consistent across the core
deliberative agent and frontier experiments.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch


def _as_column(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 1:
        return tensor.unsqueeze(-1)
    return tensor


def apply_act_step(
    cumulative_halt: torch.Tensor,
    remainder: torch.Tensor,
    p_halt: torch.Tensor,
    still_running: torch.Tensor,
    force_halt: torch.Tensor | None = None,
    halt_eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one ACT halting step and return the effective halt weight.

    All inputs are broadcast as ``(batch, 1)`` tensors. The returned
    ``lambda_n`` is the actual mass assigned to the current step after
    accounting for ACT remainder handling.
    """

    cumulative_halt = _as_column(cumulative_halt)
    remainder = _as_column(remainder)
    p_halt = _as_column(p_halt)
    still_running = _as_column(still_running).float()
    if force_halt is None:
        force_halt = torch.zeros_like(still_running)
    else:
        force_halt = _as_column(force_halt).float()

    proposed_halt = cumulative_halt + p_halt * still_running
    use_remainder = (
        ((proposed_halt >= 1.0 - halt_eps) | (force_halt > 0.0)).float()
        * still_running
    )
    lambda_n = torch.where(use_remainder.bool(), remainder.clamp(min=0.0), p_halt)
    lambda_n = (lambda_n * still_running).clamp(min=0.0, max=1.0)

    cumulative_halt = (cumulative_halt + lambda_n).clamp(0.0, 1.0)
    remainder = (remainder - lambda_n).clamp(min=0.0)
    used_remainder = use_remainder.squeeze(-1)
    return lambda_n, cumulative_halt, remainder, used_remainder


def stack_halt_weights(
    step_weights: Sequence[torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    """Stack per-step halt weights into a ``(batch, steps)`` matrix."""

    if not step_weights:
        return torch.zeros(batch_size, 1, device=device), 0.0

    weight_matrix = torch.stack(step_weights, dim=1).squeeze(-1)
    weight_sum_error = (weight_matrix.sum(dim=1) - 1.0).abs().mean().item()
    return weight_matrix, weight_sum_error


def halt_distribution_stats(weight_matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute expected depth and summary statistics from halt weights."""

    if weight_matrix.dim() == 1:
        weight_matrix = weight_matrix.unsqueeze(0)

    batch_size, steps = weight_matrix.shape
    if batch_size == 0 or steps == 0:
        zeros = torch.zeros(0, device=weight_matrix.device)
        return {
            "expected_steps": zeros,
            "halt_entropy": zeros,
            "halt_mode": zeros,
            "halt_variance": zeros,
        }

    safe_weights = weight_matrix.clamp(min=1e-10)
    step_index = torch.arange(
        1,
        steps + 1,
        dtype=weight_matrix.dtype,
        device=weight_matrix.device,
    )
    expected_steps = (weight_matrix * step_index.unsqueeze(0)).sum(dim=1)
    halt_entropy = -(safe_weights * torch.log(safe_weights)).sum(dim=1)
    halt_mode = weight_matrix.argmax(dim=1).to(weight_matrix.dtype) + 1.0
    halt_variance = (
        weight_matrix
        * (step_index.unsqueeze(0) - expected_steps.unsqueeze(1)).pow(2)
    ).sum(dim=1)
    return {
        "expected_steps": expected_steps,
        "halt_entropy": halt_entropy,
        "halt_mode": halt_mode,
        "halt_variance": halt_variance,
    }
