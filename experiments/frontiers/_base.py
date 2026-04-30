"""Shared scaffolding for frontier research experiments.

Most frontier modules in this directory follow the same shape:

    1. Build env + agent in __init__.
    2. Seed torch / numpy / random.
    3. Train for N episodes, accumulate metrics.
    4. Optionally evaluate under perturbations.
    5. Combine numbers into a single ``composite_score``.
    6. Save the summary as ``results.json``.
    7. Return the metrics dict.

Before this module, each frontier reinvented every step. The duplication was
visible enough that any change to e.g. seeding behaviour required touching
~13 files. This module gives the *implicit* contract a name and a place.

Public surface:

    - ``seed_all(seed)``: torch/numpy/random/CUDA seeding in one call.
    - ``FrontierConfig``: dataclass of fields shared across nearly every frontier.
    - ``FrontierExperiment``: optional ABC that orchestrates the lifecycle.
    - ``save_summary(out_dir, summary)``: persists results.json consistently.
    - ``make_frontier_parser(...)``: argparse factory with the standard flags.
    - ``run_default_cli(experiment_factory, parser)``: glue for ``__main__`` blocks.

The ABC is opt-in. Existing experiments that already work continue to work
without changes; new (or migrated) experiments can either subclass the ABC
or call the helpers directly.
"""

from __future__ import annotations

import argparse
import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import torch

__all__ = [
    "seed_all",
    "FrontierConfig",
    "FrontierExperiment",
    "save_summary",
    "make_frontier_parser",
    "run_default_cli",
]


def seed_all(seed: int) -> None:
    """Seed torch, numpy, and Python's ``random`` from a single integer.

    Also seeds CUDA when available. Idempotent and safe to call multiple times
    (e.g. once at config-time and again at run-time).
    """
    seed = int(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class FrontierConfig:
    """Fields common to virtually every frontier experiment.

    Subclass and add experiment-specific fields. The field names are the ones
    that already exist in the codebase, so existing call-sites that pass
    ``params={"env": ..., "device": ..., ...}`` translate cleanly.
    """

    env_id: str = "CartPole-v1"
    device: str = "cpu"
    seed: int = 42
    n_episodes: int = 30
    max_steps: int = 500
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "FrontierConfig":
        """Build a config from a loose params dict (back-compat helper).

        Accepts both ``env`` and ``env_id`` keys. Unknown keys are stored in
        ``extra`` so subclasses can pull them out without losing them.
        """
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs: Dict[str, Any] = {}
        extras: Dict[str, Any] = {}
        for key, value in params.items():
            target = "env_id" if key == "env" else key
            if target in known and target != "extra":
                kwargs[target] = value
            else:
                extras[key] = value
        if extras:
            kwargs["extra"] = extras
        return cls(**kwargs)


def save_summary(
    out_dir: Path,
    summary: Dict[str, Any],
    *,
    filename: str = "results.json",
    extras: Optional[Dict[str, Any]] = None,
) -> Path:
    """Persist a summary dict to ``out_dir/filename`` as indented JSON.

    Creates ``out_dir`` if missing. ``extras`` are merged under an
    ``"extras"`` key (without clobbering top-level metric names). Returns
    the file path written.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = dict(summary)
    if extras:
        merged_extras = dict(payload.get("extras", {}))
        merged_extras.update(extras)
        payload["extras"] = merged_extras
    target = out_path / filename
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return target


class FrontierExperiment(ABC):
    """Optional ABC that captures the implicit frontier contract.

    Subclasses must implement ``train()``, ``evaluate()``, and
    ``compute_composite()``. ``run(out_dir)`` orchestrates seeding,
    setup, training, evaluation, composite-score computation, and JSON save.

    Migration policy: existing experiments that maintain a custom ``run()``
    continue to work — the orchestration layer in ``autonomous_research.py``
    only depends on ``run(out_dir) -> Dict[str, float]`` returning a dict
    that contains ``composite_score``.
    """

    config: FrontierConfig

    def __init__(self, config: FrontierConfig):
        self.config = config

    def setup(self) -> None:
        """One-time setup hook called after seeding. Default: no-op."""
        return None

    @abstractmethod
    def train(self) -> Dict[str, float]:
        """Run training. Return a metrics dict (composite_score not required)."""

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation/robustness. Return a metrics dict.

        Implementations that don't need an evaluation phase should return
        an empty dict.
        """

    @abstractmethod
    def compute_composite(self, metrics: Dict[str, float]) -> float:
        """Combine the merged train+eval metrics into a single score."""

    def run(self, out_dir: Path) -> Dict[str, float]:
        out_path = Path(out_dir)
        seed_all(self.config.seed)
        self.setup()
        train_metrics = dict(self.train())
        eval_metrics = dict(self.evaluate())
        merged: Dict[str, float] = {**train_metrics, **eval_metrics}
        merged["composite_score"] = float(self.compute_composite(merged))
        save_summary(out_path, merged)
        return merged


def make_frontier_parser(
    *,
    name: str,
    description: str,
    default_out: str = "results/frontier_run",
    extra_args: Optional[Sequence[Callable[[argparse.ArgumentParser], None]]] = None,
) -> argparse.ArgumentParser:
    """Argparse factory with the flags every frontier CLI tends to expose.

    The factory adds: ``--out-dir``, ``--seed``, ``--device``, ``--env``,
    ``--n-episodes``, ``--max-steps``. Pass callables in ``extra_args`` to
    register experiment-specific flags without re-implementing the parser.
    """
    parser = argparse.ArgumentParser(prog=name, description=description)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(default_out),
        help="Directory to write results.json into.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--env", dest="env_id", type=str, default="CartPole-v1")
    parser.add_argument("--n-episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=500)
    if extra_args:
        for register in extra_args:
            register(parser)
    return parser


def run_default_cli(
    experiment_factory: Callable[[argparse.Namespace], "FrontierExperiment"],
    parser: argparse.ArgumentParser,
) -> Dict[str, float]:
    """Standard ``__main__`` glue: parse args, build experiment, run, return."""
    args = parser.parse_args()
    experiment = experiment_factory(args)
    return experiment.run(args.out_dir)
