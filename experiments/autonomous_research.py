#!/usr/bin/env python3
"""
Autonomous Research Orchestrator — 24/7 frontier exploration engine.

This orchestrator continuously:
  1. Selects the most promising research frontier based on past results
  2. Runs the experiment with the current best hyperparameters
  3. Analyzes results and updates the research journal
  4. Mutates hyperparameters toward the frontier (Bayesian-inspired)
  5. Loops forever, pushing into uncharted territory

Usage:
    python experiments/autonomous_research.py --cycles 0   # infinite
    python experiments/autonomous_research.py --cycles 10  # 10 cycles
    python experiments/autonomous_research.py --frontier certified_mcts  # single frontier
    python experiments/autonomous_research.py --stop-file research_runs/STOP
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════════════
# Research Journal — persistent log of all experiments and findings
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ExperimentRecord:
    """Single experiment run record."""
    frontier: str
    cycle: int
    timestamp: str
    hyperparams: Dict[str, Any]
    metrics: Dict[str, float]
    duration_sec: float
    status: str  # "success", "failed", "timeout"
    finding: str  # one-line summary of what was learned
    error: Optional[str] = None


@dataclass
class ResearchJournal:
    """Persistent research journal tracking all experiments."""
    records: List[ExperimentRecord] = field(default_factory=list)
    frontier_scores: Dict[str, List[float]] = field(default_factory=dict)
    best_per_frontier: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    total_cycles: int = 0
    breakthroughs: List[str] = field(default_factory=list)

    def add(self, record: ExperimentRecord) -> None:
        self.records.append(record)
        self.total_cycles += 1
        if record.status == "success" and record.metrics:
            key = record.frontier
            score = record.metrics.get("composite_score", 0.0)
            self.frontier_scores.setdefault(key, []).append(score)
            best = self.best_per_frontier.get(key, {})
            if score > best.get("score", -float("inf")):
                self.best_per_frontier[key] = {
                    "score": score,
                    "hyperparams": record.hyperparams,
                    "cycle": record.cycle,
                    "metrics": record.metrics,
                }

    def recent_frontier_records(self, frontier_name: str, limit: Optional[int] = None) -> List[ExperimentRecord]:
        """Return the most recent records for a frontier."""
        matches = [record for record in self.records if record.frontier == frontier_name]
        if limit is None:
            return matches
        return matches[-limit:]

    def tail_failure_streak(self) -> int:
        """Count consecutive failed records from the end of the journal."""
        streak = 0
        for record in reversed(self.records):
            if record.status == "success":
                break
            streak += 1
        return streak

    def tail_resource_exhaustion_streak(self) -> int:
        """Count trailing failures caused by OOM/resource exhaustion."""
        streak = 0
        for record in reversed(self.records):
            if record.status == "success":
                break
            if is_resource_exhaustion_error(record.error or record.finding):
                streak += 1
                continue
            break
        return streak

    def frontier_failure_rate(self, frontier_name: str, window: int = 8) -> float:
        """Estimate the recent failure rate for a frontier."""
        recent = self.recent_frontier_records(frontier_name, limit=window)
        if not recent:
            return 0.0
        failures = sum(1 for record in recent if record.status != "success")
        return failures / len(recent)

    def frontier_resource_exhaustion_streak(self, frontier_name: str) -> int:
        """Count trailing OOM/resource failures for a frontier."""
        streak = 0
        for record in reversed(self.recent_frontier_records(frontier_name)):
            if record.status == "success":
                break
            if is_resource_exhaustion_error(record.error or record.finding):
                streak += 1
                continue
            break
        return streak

    def get_frontier_priority(self) -> Dict[str, float]:
        """UCB1-style priority: exploit best + explore undersampled."""
        priorities = {}
        total_n = max(self.total_cycles, 1)
        for name in FRONTIER_REGISTRY:
            scores = self.frontier_scores.get(name, [])
            n = max(len(scores), 1)
            mean = np.mean(scores) if scores else 0.5
            ucb = mean + math.sqrt(2.0 * math.log(total_n) / n)
            # Bonus for frontiers with high improvement rate
            if len(scores) >= 2:
                improvement = scores[-1] - scores[-2]
                ucb += max(improvement, 0) * 0.5
            failure_rate = self.frontier_failure_rate(name, window=8)
            if failure_rate > 0:
                ucb -= 0.75 * failure_rate
            oom_streak = self.frontier_resource_exhaustion_streak(name)
            if oom_streak > 0:
                ucb -= min(oom_streak, 4) * 0.35
            priorities[name] = float(ucb)
        return priorities

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "total_cycles": self.total_cycles,
            "breakthroughs": self.breakthroughs,
            "best_per_frontier": self.best_per_frontier,
            "frontier_scores": {k: v[-50:] for k, v in self.frontier_scores.items()},
            "recent_records": [asdict(r) for r in self.records[-100:]],
        }
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ResearchJournal":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            journal = cls()
            journal.total_cycles = data.get("total_cycles", 0)
            journal.breakthroughs = data.get("breakthroughs", [])
            journal.best_per_frontier = data.get("best_per_frontier", {})
            journal.frontier_scores = data.get("frontier_scores", {})
            for item in data.get("recent_records", []):
                try:
                    journal.records.append(ExperimentRecord(
                        frontier=item.get("frontier", "unknown"),
                        cycle=int(item.get("cycle", 0)),
                        timestamp=item.get("timestamp", ""),
                        hyperparams=dict(item.get("hyperparams", {})),
                        metrics=dict(item.get("metrics", {})),
                        duration_sec=float(item.get("duration_sec", 0.0)),
                        status=item.get("status", "unknown"),
                        finding=item.get("finding", ""),
                        error=item.get("error"),
                    ))
                except Exception:
                    continue
            return journal
        except Exception:
            return cls()


# ═══════════════════════════════════════════════════════════════════════════════
# Frontier Registry — each frontier is a research direction
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class FrontierSpec:
    """Specification for a research frontier."""
    name: str
    description: str
    runner: Callable[[Dict[str, Any], Path], Dict[str, float]]
    default_hyperparams: Dict[str, Any]
    hyperparam_ranges: Dict[str, Tuple[float, float]]
    sanitize_hyperparams: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    mutation_sigma: float = 0.15


FRONTIER_REGISTRY: Dict[str, FrontierSpec] = {}
CONSOLE_TRANSLATION_TABLE = str.maketrans({
    "—": "--",
    "–": "-",
    "→": "->",
    "×": "x",
    "±": "+/-",
    "Δ": "Delta",
})


def register_frontier(spec: FrontierSpec) -> None:
    FRONTIER_REGISTRY[spec.name] = spec


def console_safe(text: Any) -> str:
    """Normalize text for Windows consoles that still use legacy encodings."""
    normalized = str(text).translate(CONSOLE_TRANSLATION_TABLE)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return normalized.encode(encoding, errors="replace").decode(encoding)


def is_resource_exhaustion_error(text: Optional[str]) -> bool:
    """Return True when an error message looks like memory/resource exhaustion."""
    if not text:
        return False
    lowered = str(text).lower()
    markers = (
        "out of memory",
        "cuda error: out of memory",
        "cublas_status_alloc_failed",
        "cuda out of memory",
        "insufficient memory",
    )
    return any(marker in lowered for marker in markers)


# ═══════════════════════════════════════════════════════════════════════════════
# Hyperparameter mutation — Bayesian-inspired exploration
# ═══════════════════════════════════════════════════════════════════════════════


def mutate_hyperparams(
    base: Dict[str, Any],
    ranges: Dict[str, Tuple[float, float]],
    sigma: float = 0.15,
    journal: Optional[ResearchJournal] = None,
    frontier_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Mutate hyperparameters with Gaussian noise, respecting ranges.

    If journal has a best config for this frontier, interpolate toward it
    with probability 0.5 (exploitation), otherwise explore randomly.
    """
    result = dict(base)
    best = None
    if journal and frontier_name:
        best_info = journal.best_per_frontier.get(frontier_name, {})
        best = best_info.get("hyperparams")

    for key, (lo, hi) in ranges.items():
        if key not in result:
            continue
        val = float(result[key])

        # Exploitation: interpolate toward best with prob 0.4
        if best and key in best and random.random() < 0.4:
            target = float(best[key])
            val = val + 0.3 * (target - val)

        # Exploration: Gaussian perturbation
        noise = random.gauss(0, sigma * (hi - lo))
        val = np.clip(val + noise, lo, hi)

        # Preserve int types
        if isinstance(result[key], int):
            result[key] = int(round(val))
        else:
            result[key] = float(round(val, 6))

    return result


def snap_to_multiple(value: int, step: int, lo: int, hi: int) -> int:
    """Round an integer into the nearest valid multiple inside [lo, hi]."""
    if step <= 1:
        return int(np.clip(value, lo, hi))

    candidates = []
    lower = (value // step) * step
    upper = lower + step
    for candidate in (lower, upper):
        if lo <= candidate <= hi:
            candidates.append(candidate)

    if not candidates:
        minimum = math.ceil(lo / step) * step
        maximum = (hi // step) * step
        if minimum <= hi:
            candidates.append(minimum)
        if maximum >= lo:
            candidates.append(maximum)

    if not candidates:
        return int(np.clip(value, lo, hi))

    return min(candidates, key=lambda candidate: (abs(candidate - value), candidate))


# ═══════════════════════════════════════════════════════════════════════════════
# Frontier 1: Certified MCTS
# ═══════════════════════════════════════════════════════════════════════════════


def _run_certified_mcts(params: Dict[str, Any], out_dir: Path) -> Dict[str, float]:
    """Run Certified MCTS experiment — MCTS with Lipschitz-bounded branch pruning."""
    from frontiers.certified_mcts import CertifiedMCTSExperiment

    exp = CertifiedMCTSExperiment(
        env_id=params.get("env", "CartPole-v1"),
        hidden_dim=params.get("hidden_dim", 128),
        obs_dim=params.get("obs_dim", 4),
        action_dim=params.get("action_dim", 2),
        num_simulations=params.get("num_simulations", 64),
        lipschitz_threshold=params.get("lipschitz_threshold", 2.0),
        certification_level=params.get("certification_level", "spectral"),
        c_puct=params.get("c_puct", 1.5),
        lambda_return=params.get("lambda_return", 0.8),
        gamma=params.get("gamma", 0.99),
        n_episodes=params.get("n_episodes", 20),
        max_steps=params.get("max_steps", 500),
    )
    results = exp.run(out_dir)
    return results


register_frontier(FrontierSpec(
    name="certified_mcts",
    description="MCTS with Lipschitz certification — prune timing-unsafe branches",
    runner=_run_certified_mcts,
    default_hyperparams={
        "env": "CartPole-v1",
        "hidden_dim": 128,
        "obs_dim": 4,
        "action_dim": 2,
        "num_simulations": 32,
        "lipschitz_threshold": 2.0,
        "c_puct": 1.5,
        "lambda_return": 0.8,
        "gamma": 0.99,
        "n_episodes": 10,
        "max_steps": 200,
    },
    hyperparam_ranges={
        "num_simulations": (16, 128),
        "lipschitz_threshold": (0.5, 10.0),
        "c_puct": (0.1, 5.0),
        "lambda_return": (0.5, 1.0),
        "hidden_dim": (64, 512),
    },
))


# ═══════════════════════════════════════════════════════════════════════════════
# Frontier 2: World Model-Guided Deliberation
# ═══════════════════════════════════════════════════════════════════════════════


def _run_wm_deliberation(params: Dict[str, Any], out_dir: Path) -> Dict[str, float]:
    """World model imagination guides ACT halting depth."""
    from frontiers.world_model_guided_deliberation import (
        ExperimentConfig,
        WMGuidedDeliberationExperiment,
    )

    cfg = ExperimentConfig(
        env_id=params.get("env", "CartPole-v1"),
        obs_dim=params.get("obs_dim", 4),
        action_dim=params.get("action_dim", 2),
        hidden_dim=params.get("hidden_dim", 128),
        rssm_stoch_dim=params.get("rssm_stoch_dim", 32),
        rssm_num_classes=params.get("rssm_num_classes", 32),
        max_thinking_steps=params.get("max_thinking_steps", 10),
        imagination_horizon=params.get("imagination_horizon", 5),
        uncertainty_threshold=params.get("uncertainty_threshold", 0.3),
        lambda_geo=params.get("lambda_geo", 0.5),
        n_episodes=params.get("n_episodes", 20),
        max_steps=params.get("max_steps", 500),
        seed=params.get("seed", random.randint(0, 2**31 - 1)),
    )
    exp = WMGuidedDeliberationExperiment(cfg)
    return exp.run(out_dir)


register_frontier(FrontierSpec(
    name="wm_guided_deliberation",
    description="World model imagination guides ACT halting — ponder more when uncertain",
    runner=_run_wm_deliberation,
    mutation_sigma=0.30,  # aggressive exploration to escape local optimum
    default_hyperparams={
        "env": "CartPole-v1",
        "obs_dim": 4,
        "action_dim": 2,
        "hidden_dim": 128,
        "rssm_stoch_dim": 32,
        "rssm_num_classes": 32,
        "max_thinking_steps": 10,
        "imagination_horizon": 5,
        "uncertainty_threshold": 0.3,
        "lambda_geo": 0.5,
        "n_episodes": 20,
        "max_steps": 500,
    },
    hyperparam_ranges={
        "max_thinking_steps": (3, 30),
        "imagination_horizon": (1, 15),
        "uncertainty_threshold": (0.05, 0.8),
        "lambda_geo": (0.1, 0.9),
        "hidden_dim": (64, 512),
        "rssm_stoch_dim": (16, 64),
    },
))


# ═══════════════════════════════════════════════════════════════════════════════
# Frontier 3: Multi-Scale Temporal World Model
# ═══════════════════════════════════════════════════════════════════════════════


def _run_multiscale_wm(params: Dict[str, Any], out_dir: Path) -> Dict[str, float]:
    """Hierarchical RSSM with slow/fast latent variables."""
    from frontiers.multiscale_temporal_world_model import MultiScaleWMExperiment

    exp = MultiScaleWMExperiment()
    # Set attributes from params (MultiScaleWMExperiment uses class attributes)
    for key in ("obs_dim", "action_dim", "fast_hidden_dim", "slow_hidden_dim",
                "fast_stoch_dim", "slow_stoch_dim", "num_classes",
                "cross_scale_heads", "slow_tick_every", "sequence_length",
                "batch_size", "train_steps", "lr"):
        if key in params:
            setattr(exp, key, params[key])
    return exp.run(out_dir)


def _sanitize_multiscale_hyperparams(params: Dict[str, Any]) -> Dict[str, Any]:
    """Enforce architectural divisibility constraints for the multiscale model."""
    result = dict(params)
    heads = max(1, int(result.get("cross_scale_heads", 4)))
    result["cross_scale_heads"] = heads

    fast_hidden_dim = int(result.get("fast_hidden_dim", 64))
    required_multiple = math.lcm(4, heads)
    result["fast_hidden_dim"] = snap_to_multiple(
        fast_hidden_dim,
        required_multiple,
        lo=32,
        hi=256,
    )
    return result


register_frontier(FrontierSpec(
    name="multiscale_temporal_wm",
    description="Hierarchical RSSM — slow/fast latent variables with cross-scale attention",
    runner=_run_multiscale_wm,
    sanitize_hyperparams=_sanitize_multiscale_hyperparams,
    default_hyperparams={
        "obs_dim": 4,
        "action_dim": 2,
        "fast_hidden_dim": 64,
        "slow_hidden_dim": 128,
        "fast_stoch_dim": 16,
        "slow_stoch_dim": 32,
        "num_classes": 16,
        "cross_scale_heads": 4,
        "slow_tick_every": 4,
        "sequence_length": 50,
        "batch_size": 32,
        "train_steps": 500,
        "lr": 3e-4,
    },
    hyperparam_ranges={
        "fast_hidden_dim": (32, 256),
        "slow_hidden_dim": (64, 512),
        "fast_stoch_dim": (8, 64),
        "slow_stoch_dim": (16, 128),
        "cross_scale_heads": (1, 8),
        "slow_tick_every": (2, 16),
        "train_steps": (200, 2000),
        "lr": (1e-5, 1e-2),
    },
))


# ═══════════════════════════════════════════════════════════════════════════════
# Frontier 4: Temporal Consistency Distillation
# ═══════════════════════════════════════════════════════════════════════════════


def _run_consistency_distillation(params: Dict[str, Any], out_dir: Path) -> Dict[str, float]:
    """Distill MCTS search policy into fast network preserving Lipschitz bounds."""
    from frontiers.temporal_consistency_distillation import ConsistencyDistillationExperiment

    # Filter to only valid ExperimentConfig fields
    valid_keys = {
        "obs_dim", "action_dim", "teacher_hidden_dim", "student_hidden_dim",
        "num_simulations", "distill_steps", "lipschitz_penalty", "temperature",
        "lr", "batch_size", "n_eval_episodes", "max_steps",
        "lip_margin", "delta_tau_nominal", "seed", "env_id", "use_spectral_norm",
    }
    kwargs = {k: v for k, v in params.items() if k in valid_keys}
    exp = ConsistencyDistillationExperiment(**kwargs)
    return exp.run(out_dir)


register_frontier(FrontierSpec(
    name="consistency_distillation",
    description="Distill MCTS into fast network with Lipschitz-preserving consistency loss",
    runner=_run_consistency_distillation,
    default_hyperparams={
        "obs_dim": 4,
        "action_dim": 2,
        "teacher_hidden_dim": 256,
        "student_hidden_dim": 64,
        "num_simulations": 64,
        "distill_steps": 1000,
        "lipschitz_penalty": 0.1,
        "temperature": 1.0,
        "lr": 1e-3,
        "batch_size": 64,
        "n_eval_episodes": 20,
        "max_steps": 500,
    },
    hyperparam_ranges={
        "teacher_hidden_dim": (128, 512),
        "student_hidden_dim": (32, 256),
        "num_simulations": (16, 256),
        "distill_steps": (500, 5000),
        "lipschitz_penalty": (0.001, 1.0),
        "temperature": (0.1, 5.0),
        "lr": (1e-5, 1e-2),
    },
))


# ═══════════════════════════════════════════════════════════════════════════════
# Frontier 5: Certified Multi-Scale Deliberation (novel fusion)
# ═══════════════════════════════════════════════════════════════════════════════


def _run_cms_deliberation(params: Dict[str, Any], out_dir: Path) -> Dict[str, float]:
    """Certified Multi-Scale Deliberation — fuses multi-scale WM + Lipschitz + ACT."""
    from frontiers.certified_multiscale_deliberation import (
        CMSDExperimentConfig,
        CMSDExperiment,
    )

    cfg = CMSDExperimentConfig(
        env_id=params.get("env", "CartPole-v1"),
        obs_dim=params.get("obs_dim", 4),
        action_dim=params.get("action_dim", 2),
        fast_hidden_dim=params.get("fast_hidden_dim", 64),
        slow_hidden_dim=params.get("slow_hidden_dim", 48),
        slow_tick_every=params.get("slow_tick_every", 3),
        max_thinking_steps=params.get("max_thinking_steps", 12),
        imagination_horizon=params.get("imagination_horizon", 5),
        n_train_episodes=params.get("n_train_episodes", 40),
        n_eval_episodes=params.get("n_eval_episodes", 20),
        max_steps=params.get("max_steps", 500),
        train_epochs=params.get("train_epochs", 8),
        lr=params.get("lr", 3e-4),
        gamma=params.get("gamma", 0.99),
        seed=params.get("seed", 42),
    )
    exp = CMSDExperiment(cfg)
    return exp.run(out_dir)


register_frontier(FrontierSpec(
    name="cms_deliberation",
    description="Certified Multi-Scale Deliberation -- triple-signal ACT with fast/slow WM + Lipschitz",
    runner=_run_cms_deliberation,
    default_hyperparams={
        "env": "CartPole-v1",
        "obs_dim": 4,
        "action_dim": 2,
        "fast_hidden_dim": 64,
        "slow_hidden_dim": 48,
        "slow_tick_every": 3,
        "max_thinking_steps": 12,
        "imagination_horizon": 5,
        "n_train_episodes": 40,
        "n_eval_episodes": 20,
        "max_steps": 500,
        "train_epochs": 8,
        "lr": 3e-4,
        "seed": 42,
    },
    hyperparam_ranges={
        "fast_hidden_dim": (32, 256),
        "slow_hidden_dim": (24, 192),
        "slow_tick_every": (2, 8),
        "max_thinking_steps": (4, 30),
        "imagination_horizon": (2, 15),
        "train_epochs": (4, 20),
        "lr": (1e-5, 1e-2),
    },
))


# ═══════════════════════════════════════════════════════════════════════════════
# Frontier 6: Adversarial Audit Synthesis (Discovery & Patching)
# ═══════════════════════════════════════════════════════════════════════════════


def _run_adversarial_audit_synthesis(params: Dict[str, Any], out_dir: Path) -> Dict[str, float]:
    """Learns adversarial timing patterns and identifies critical vulnerabilities."""
    from frontiers.adversarial_audit_synthesis import (
        AdversarialAuditSynthesisExperiment,
    )

    exp = AdversarialAuditSynthesisExperiment(params)
    return exp.run(out_dir)


register_frontier(FrontierSpec(
    name="adversarial_audit_synthesis",
    description="Learned timing adversary policy -- discover sequential vulnerabilities and synthesize patches",
    runner=_run_adversarial_audit_synthesis,
    default_hyperparams={
        "env": "CartPole-v1",
        "obs_dim": 4,
        "attacker_hidden": 96,
        "attack_train_episodes": 60,
        "victim_timesteps": 15000,
        "lr": 1e-3,
    },
    hyperparam_ranges={
        "attacker_hidden": (32, 256),
        "attack_train_episodes": (20, 200),
        "lr": (1e-4, 1e-2),
    },
))


# ═══════════════════════════════════════════════════════════════════════════════
# Frontier 7: Meta-Policy Distillation (Universal Adversary)
# ═══════════════════════════════════════════════════════════════════════════════


def _run_meta_policy_distillation(params: Dict[str, Any], out_dir: Path) -> Dict[str, float]:
    """Trains a cross-environment meta-adversary to discover universal timing flaws."""
    from frontiers.meta_policy_distillation import (
        MetaPolicyDistillationExperiment,
    )

    exp = MetaPolicyDistillationExperiment(params)
    return exp.run(out_dir)


register_frontier(FrontierSpec(
    name="meta_policy_distillation",
    description="Cross-environment transformer adversary -- zero-shot timing vulnerability discovery",
    runner=_run_meta_policy_distillation,
    default_hyperparams={
        "device_policy": "auto",
    },
    hyperparam_ranges={}, # Fixed architecture for now
))


# ═══════════════════════════════════════════════════════════════════════════════
# Frontier 8: Temporal Subjectification (Subjective Clock)
# ═══════════════════════════════════════════════════════════════════════════════


def _run_temporal_subjectification(params: Dict[str, Any], out_dir: Path) -> Dict[str, float]:
    """Trains an agent that operates in its own subjective continuous timeline."""
    from frontiers.temporal_subjectification import (
        TemporalSubjectificationExperiment,
    )

    exp = TemporalSubjectificationExperiment(params)
    return exp.run(out_dir)


register_frontier(FrontierSpec(
    name="temporal_subjectification",
    description="Subjective continuous-time clock via Neural ODE -- decoupling cognition from env time",
    runner=_run_temporal_subjectification,
    default_hyperparams={
        "env": "CartPole-v1",
        "obs_dim": 4,
        "act_dim": 2,
        "hidden_dim": 64,
        "lr": 1e-3,
        "n_episodes": 50,
        "nominal_return": 200.0,
    },
    hyperparam_ranges={
        "hidden_dim": (32, 256),
        "lr": (1e-4, 1e-2),
        "n_episodes": (20, 200),
    },
))


# ═══════════════════════════════════════════════════════════════════════════════
# Frontier 9: Recursive Self-Architecture (Adaptive Complexity)
# ═══════════════════════════════════════════════════════════════════════════════


def _run_recursive_self_architecture(params: Dict[str, Any], out_dir: Path) -> Dict[str, float]:
    """Evolves agents that dynamically reconfigure their own computational depth."""
    from frontiers.recursive_self_architecture import (
        RecursiveSelfArchitectureExperiment,
    )

    exp = RecursiveSelfArchitectureExperiment(params)
    return exp.run(out_dir)


register_frontier(FrontierSpec(
    name="recursive_self_architecture",
    description="Dynamically scaling ODE resolution and model depth based on stress",
    runner=_run_recursive_self_architecture,
    default_hyperparams={
        "env": "CartPole-v1",
        "obs_dim": 4,
        "act_dim": 2,
        "hidden_dim": 64,
        "lr": 1e-3,
        "n_episodes": 40,
        "nominal_return": 200.0,
    },
    hyperparam_ranges={
        "hidden_dim": (32, 256),
        "lr": (1e-4, 1e-2),
    },
))


# ═══════════════════════════════════════════════════════════════════════════════
# Frontier 10: Causal Temporal Reasoning (Counterfactual Timing)
# ═══════════════════════════════════════════════════════════════════════════════


def _run_causal_temporal_reasoning(params: Dict[str, Any], out_dir: Path) -> Dict[str, float]:
    """Evolves agents that use latent imagination to optimize action timing."""
    from frontiers.causal_temporal_reasoning import (
        CausalTemporalReasoningExperiment,
    )

    exp = CausalTemporalReasoningExperiment(params)
    return exp.run(out_dir)


register_frontier(FrontierSpec(
    name="causal_temporal_reasoning",
    description="Causal counterfactual timing simulation via Temporal World Model",
    runner=_run_causal_temporal_reasoning,
    default_hyperparams={
        "env": "CartPole-v1",
        "obs_dim": 4,
        "act_dim": 2,
        "hidden_dim": 128,
        "lr": 1e-3,
        "n_episodes": 30,
    },
    hyperparam_ranges={
        "hidden_dim": (64, 512),
        "lr": (1e-4, 1e-2),
    },
))


# ═══════════════════════════════════════════════════════════════════════════════
# Frontier 11: Adaptive-dt Policy Gradient (Claude-proposed 2026-04-18)
# Tests whether learning per-step dt improves RETURNS on variable-speed envs.
# ═══════════════════════════════════════════════════════════════════════════════


def _run_adaptive_dt_policy_gradient(params: Dict[str, Any], out_dir: Path) -> Dict[str, float]:
    """REINFORCE agent with learned dt vs fixed-dt baseline on VariableFrequencyChainEnv."""
    from frontiers.adaptive_dt_policy_gradient import AdaptiveDTExperiment

    exp = AdaptiveDTExperiment(params)
    return exp.run(out_dir)


register_frontier(FrontierSpec(
    name="adaptive_dt_policy_gradient",
    description="Adaptive-dt policy head on variable-speed chain; measures whether learned dt correlates with env speed and beats fixed-dt baseline",
    runner=_run_adaptive_dt_policy_gradient,
    default_hyperparams={
        "hidden_dim": 64,
        "lr": 3e-3,
        "n_episodes": 80,
        "chain_length": 20,
        "noise": 0.05,
        "eval_per_speed": 20,
        "seed": 0,
        "device_policy": "cpu",
    },
    hyperparam_ranges={
        "hidden_dim": (32, 256),
        "lr": (1e-4, 1e-2),
        "n_episodes": (40, 200),
        "noise": (0.0, 0.2),
        "eval_per_speed": (10, 40),
    },
))


# ═══════════════════════════════════════════════════════════════════════════════
# Frontier 13: Causal-Relativistic World Model (CRWM)
# ═══════════════════════════════════════════════════════════════════════════════


def _run_causal_relativistic_wm(params: Dict[str, Any], out_dir: Path) -> Dict[str, float]:
    """Relativistic temporal field via Hyperbolic latent ODE."""
    from frontiers.causal_relativistic_world_model import CRWMExperiment

    exp = CRWMExperiment(params)
    return exp.run(out_dir)


def _sanitize_relativistic_hyperparams(params: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure hidden_dim is a multiple of n_groups."""
    result = dict(params)
    n_groups = max(1, int(result.get("n_groups", 4)))
    hidden_dim = int(result.get("hidden_dim", 128))
    # Round hidden_dim up to the nearest multiple of n_groups
    result["hidden_dim"] = ((hidden_dim + n_groups - 1) // n_groups) * n_groups
    result["n_groups"] = n_groups
    return result


register_frontier(FrontierSpec(
    name="causal_relativistic_wm",
    description="Proper Time Field over Hyperbolic latent space -- relativistic action timing",
    runner=_run_causal_relativistic_wm,
    sanitize_hyperparams=_sanitize_relativistic_hyperparams,
    default_hyperparams={
        "env": "CartPole-v1",
        "obs_dim": 4,
        "act_dim": 2,
        "hidden_dim": 128,
        "n_groups": 4,
        "lr": 1e-3,
        "n_episodes": 40,
    },
    hyperparam_ranges={
        "hidden_dim": (64, 256),
        "n_groups": (2, 8),
        "lr": (1e-4, 5e-3),
        "n_episodes": (20, 100),
    },
))


# ═══════════════════════════════════════════════════════════════════════════════
# Frontier 14: Quantum-Tunneling Relativistic World Model (QTRWM)
# ═══════════════════════════════════════════════════════════════════════════════


def _run_quantum_tunneling_wm(params: Dict[str, Any], out_dir: Path) -> Dict[str, float]:
    """Quantum-tunneling latent jumps over relativistic manifold."""
    from frontiers.frontier_14 import QTRWMExperiment

    exp = QTRWMExperiment(params)
    return exp.run(out_dir)


register_frontier(FrontierSpec(
    name="quantum_tunneling_wm",
    description="Relativistic latent ODE with stochastic quantum-tunneling jumps for extreme timing jitter",
    runner=_run_quantum_tunneling_wm,
    sanitize_hyperparams=_sanitize_relativistic_hyperparams,
    default_hyperparams={
        "env": "CartPole-v1",
        "obs_dim": 4,
        "act_dim": 2,
        "hidden_dim": 128,
        "n_groups": 4,
        "lr": 1e-3,
        "n_episodes": 48,
    },
    hyperparam_ranges={
        "hidden_dim": (64, 256),
        "n_groups": (2, 8),
        "lr": (1e-4, 5e-3),
        "n_episodes": (30, 150),
    },
))


# ═══════════════════════════════════════════════════════════════════════════════
# Frontier 15: Entropic Causal Manifold Alignment (ECMA)
# ═══════════════════════════════════════════════════════════════════════════════


def _run_entropic_causal_manifold_alignment(params: Dict[str, Any], out_dir: Path) -> Dict[str, float]:
    """Riemannian manifold ODE with entropic information bottleneck."""
    from frontiers.entropic_causal_manifold_alignment import ECMAExperiment

    exp = ECMAExperiment(params)
    return exp.run(out_dir)


def _sanitize_ecma_hyperparams(params: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure environment-specific dimensions and robust defaults."""
    result = dict(params)
    env_id = result.get("env", "CartPole-v1")
    
    # Auto-detect dimensions if not provided
    if "obs_dim" not in result or "act_dim" not in result:
        try:
            temp_env = gym.make(env_id)
            result["obs_dim"] = temp_env.observation_space.shape[0]
            if isinstance(temp_env.action_space, gym.spaces.Discrete):
                result["act_dim"] = temp_env.action_space.n
            else:
                result["act_dim"] = temp_env.action_space.shape[0]
            temp_env.close()
        except Exception:
            result.setdefault("obs_dim", 4)
            result.setdefault("act_dim", 2)

    # Set nominal returns based on env
    if "nominal_return" not in result:
        if "CartPole" in env_id: result["nominal_return"] = 500.0
        elif "HalfCheetah" in env_id: result["nominal_return"] = 3000.0
        elif "Hopper" in env_id: result["nominal_return"] = 2000.0
        else: result["nominal_return"] = 100.0
        
    return result


register_frontier(FrontierSpec(
    name="entropic_causal_manifold_alignment",
    description="Riemannian geodesic flow in latent space with entropic regularization for scale-invariant causal discovery",
    runner=_run_entropic_causal_manifold_alignment,
    sanitize_hyperparams=_sanitize_ecma_hyperparams,
    default_hyperparams={
        "env": "CartPole-v1",
        "hidden_dim": 128,
        "lr": 1e-3,
        "n_episodes": 50,
    },
    hyperparam_ranges={
        "hidden_dim": (64, 512),
        "lr": (1e-4, 5e-3),
        "n_episodes": (20, 200),
    },
))


# ═══════════════════════════════════════════════════════════════════════════════
# Frontier 16: Spatiotemporal Contrastive Foundation Transformer (SCFT)
# ═══════════════════════════════════════════════════════════════════════════════


def _run_scft_foundation_transformer(params: Dict[str, Any], out_dir: Path) -> Dict[str, float]:
    """Continuous-time foundation transformer with Rotary Positional Embeddings."""
    from frontiers.scft_foundation_transformer import SCFTExperiment

    exp = SCFTExperiment(params)
    return exp.run(out_dir)


def _sanitize_scft_hyperparams(params: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure latent_dim is divisible by num_heads (4)."""
    result = dict(params)
    latent_dim = int(result.get("latent_dim", 64))
    # Round up to nearest multiple of 4
    result["latent_dim"] = ((latent_dim + 3) // 4) * 4
    return result


register_frontier(FrontierSpec(
    name="scft_foundation_transformer",
    description="Continuous-time foundation transformer with Rotary Positional Embeddings (RoPE) and latent time-query head",
    runner=_run_scft_foundation_transformer,
    sanitize_hyperparams=_sanitize_scft_hyperparams,
    default_hyperparams={
        "env": "CartPole-v1",
        "latent_dim": 64,
        "lr": 1e-3,
        "n_episodes": 30,
    },
    hyperparam_ranges={
        "latent_dim": (32, 256),
        "lr": (1e-4, 5e-3),
        "n_episodes": (20, 100),
    },
))


# ═══════════════════════════════════════════════════════════════════════════════
# Frontier 17: Fractal-Temporal World Model (FTWM)
# ═══════════════════════════════════════════════════════════════════════════════


def _run_fractal_temporal_world_model(params: Dict[str, Any], out_dir: Path) -> Dict[str, float]:
    """Scale-indexed Neural ODE with fractal scale-attention."""
    from frontiers.fractal_temporal_world_model import FTWMExperiment

    exp = FTWMExperiment(params)
    return exp.run(out_dir)


register_frontier(FrontierSpec(
    name="fractal_temporal_world_model",
    description="Time as a fractal manifold; continuous scale-indexed Neural ODE with scale-attention for infinite-resolution cognition",
    runner=_run_fractal_temporal_world_model,
    default_hyperparams={
        "env": "CartPole-v1",
        "hidden_dim": 128,
        "n_scales": 4,
        "lr": 1e-3,
        "n_episodes": 40,
    },
    hyperparam_ranges={
        "hidden_dim": (64, 256),
        "n_scales": (2, 8),
        "lr": (1e-4, 5e-3),
        "n_episodes": (20, 100),
    },
))


# ═══════════════════════════════════════════════════════════════════════════════
# Frontier 18: Meta-Temporal Evolution (MTE)
# ═══════════════════════════════════════════════════════════════════════════════


def _run_meta_temporal_evolution(params: Dict[str, Any], out_dir: Path) -> Dict[str, float]:
    """Online meta-learning of temporal clock parameters."""
    from frontiers.meta_temporal_evolution import MTEExperiment

    exp = MTEExperiment(params)
    return exp.run(out_dir)


register_frontier(FrontierSpec(
    name="meta_temporal_evolution",
    description="Online meta-learning loop for dynamic clock (dt) and resolution adaptation",
    runner=_run_meta_temporal_evolution,
    default_hyperparams={
        "env": "CartPole-v1",
        "hidden_dim": 128,
        "lr": 1e-3,
        "n_episodes": 50,
    },
    hyperparam_ranges={
        "hidden_dim": (64, 256),
        "lr": (1e-4, 5e-3),
        "n_episodes": (20, 100),
    },
))


# ═══════════════════════════════════════════════════════════════════════════════
# Frontier 19: Causal-Entangled Message Passing Transformer (CEMPT)
# ═══════════════════════════════════════════════════════════════════════════════


def _run_cempt_causal_transformer(params: Dict[str, Any], out_dir: Path) -> Dict[str, float]:
    """Spatiotemporal causal graph with message-passing transformer."""
    from frontiers.cempt_causal_transformer import CEMPTExperiment

    exp = CEMPTExperiment(params)
    return exp.run(out_dir)


register_frontier(FrontierSpec(
    name="cempt_causal_transformer",
    description="Environment as a causal graph; message-passing transformer for explicit structural reasoning",
    runner=_run_cempt_causal_transformer,
    default_hyperparams={
        "env": "CartPole-v1",
        "hidden_dim": 128,
        "lr": 1e-3,
        "n_episodes": 30,
    },
    hyperparam_ranges={
        "hidden_dim": (64, 256),
        "lr": (1e-4, 5e-3),
        "n_episodes": (20, 100),
    },
))


# ═══════════════════════════════════════════════════════════════════════════════
# Frontier 20: Temporal Singularity World Model (TSWM)
# ═══════════════════════════════════════════════════════════════════════════════


def _run_temporal_singularity_world_model(params: Dict[str, Any], out_dir: Path) -> Dict[str, float]:
    """Inverse-causality singularity with Hawking radiation regularization."""
    from frontiers.temporal_singularity_world_model import TSWMExperiment

    exp = TSWMExperiment(params)
    return exp.run(out_dir)


register_frontier(FrontierSpec(
    name="temporal_singularity_wm",
    description="Time as a singularity; inverse-causality decoding from an Event Horizon latent",
    runner=_run_temporal_singularity_world_model,
    default_hyperparams={
        "env": "CartPole-v1",
        "hidden_dim": 128,
        "lr": 1e-3,
        "n_episodes": 40,
    },
    hyperparam_ranges={
        "hidden_dim": (64, 256),
        "lr": (1e-4, 5e-3),
        "n_episodes": (20, 100),
    },
))


# ═══════════════════════════════════════════════════════════════════════════════
# Main orchestrator loop
# ═══════════════════════════════════════════════════════════════════════════════


def select_frontier(journal: ResearchJournal, forced: Optional[str] = None) -> str:
    """Select next frontier using UCB1 bandit strategy."""
    if forced:
        if forced not in FRONTIER_REGISTRY:
            raise ValueError(f"Unknown frontier: {forced}")
        return forced
    priorities = journal.get_frontier_priority()
    # Softmax selection with temperature
    names = list(priorities.keys())
    scores = np.array([priorities[n] for n in names])
    temp = 0.5
    probs = np.exp((scores - scores.max()) / temp)
    probs /= probs.sum()
    return np.random.choice(names, p=probs)


def prepare_frontier_params(frontier_name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Merge caller params with defaults and apply frontier-specific sanitization."""
    if frontier_name not in FRONTIER_REGISTRY:
        raise ValueError(f"Unknown frontier: {frontier_name}")

    spec = FRONTIER_REGISTRY[frontier_name]
    merged = dict(spec.default_hyperparams)
    if params:
        merged.update(params)
    if spec.sanitize_hyperparams is not None:
        merged = spec.sanitize_hyperparams(merged)
    return merged


@dataclass(frozen=True)
class CycleRuntimeConfig:
    """Execution settings for one orchestrator cycle."""

    journal_path: Optional[Path] = None
    child_timeout_seconds: int = 7200
    device_policy: str = "auto"
    cpu_fallback_after_failures: int = 3
    consecutive_failures: int = 0


CURRENT_CYCLE_RUNTIME_CONFIG: Optional[CycleRuntimeConfig] = None


CPU_SAFE_PARAM_CAPS: Dict[str, Dict[str, Any]] = {
    "certified_mcts": {
        "hidden_dim": 128,
        "num_simulations": 32,
        "n_episodes": 8,
        "max_steps": 200,
    },
    "wm_guided_deliberation": {
        "hidden_dim": 128,
        "rssm_stoch_dim": 24,
        "max_thinking_steps": 12,
        "imagination_horizon": 4,
        "n_episodes": 10,
        "max_steps": 250,
    },
    "multiscale_temporal_wm": {
        "fast_hidden_dim": 64,
        "slow_hidden_dim": 128,
        "fast_stoch_dim": 16,
        "slow_stoch_dim": 32,
        "cross_scale_heads": 4,
        "batch_size": 16,
        "train_steps": 200,
        "sequence_length": 40,
    },
    "consistency_distillation": {
        "teacher_hidden_dim": 192,
        "student_hidden_dim": 64,
        "num_simulations": 32,
        "distill_steps": 500,
        "batch_size": 32,
        "n_eval_episodes": 8,
        "max_steps": 250,
    },
    "cms_deliberation": {
        "fast_hidden_dim": 64,
        "slow_hidden_dim": 48,
        "max_thinking_steps": 10,
        "imagination_horizon": 4,
        "n_train_episodes": 20,
        "n_eval_episodes": 8,
        "max_steps": 250,
        "train_epochs": 4,
    },
}


def normalize_device_policy(value: Optional[str]) -> str:
    """Normalize device policy tokens used by the orchestrator."""
    policy = str(value or "auto").strip().lower()
    if policy not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unknown device policy: {value}")
    return policy


def resolve_cycle_device_policy(
    journal: ResearchJournal,
    frontier_name: str,
    runtime_config: Optional[CycleRuntimeConfig],
) -> str:
    """Decide whether the next cycle should run on CPU or leave device choice automatic."""
    configured = normalize_device_policy(runtime_config.device_policy if runtime_config else "auto")
    if configured != "auto":
        return configured
    if runtime_config is None:
        return "auto"

    recovered_failure_streak = max(journal.tail_failure_streak(), int(runtime_config.consecutive_failures))
    if recovered_failure_streak >= runtime_config.cpu_fallback_after_failures:
        return "cpu"
    if journal.tail_resource_exhaustion_streak() >= 2:
        return "cpu"
    if journal.frontier_resource_exhaustion_streak(frontier_name) >= 2:
        return "cpu"
    return "auto"


def apply_resource_profile(frontier_name: str, params: Dict[str, Any], device_policy: str) -> Dict[str, Any]:
    """Clamp expensive hyperparameters when running in CPU-safe mode."""
    normalized = normalize_device_policy(device_policy)
    result = dict(params)
    result["device_policy"] = normalized

    if normalized != "cpu":
        return result

    result["device"] = "cpu"
    for key, cap in CPU_SAFE_PARAM_CAPS.get(frontier_name, {}).items():
        if key not in result:
            continue
        current = result[key]
        try:
            if isinstance(cap, int):
                result[key] = min(int(current), int(cap))
            else:
                result[key] = min(float(current), float(cap))
        except Exception:
            result[key] = cap
    return result


def build_child_environment(device_policy: str) -> Dict[str, str]:
    """Build the child process environment for one isolated experiment."""
    normalized = normalize_device_policy(device_policy)
    env = os.environ.copy()
    env["DELTA_TAU_DEVICE_POLICY"] = normalized
    if normalized == "cpu":
        # `-1` reliably hides CUDA on Windows; empty string is treated like unset.
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    elif normalized == "cuda" and env.get("CUDA_VISIBLE_DEVICES") in {"", "-1"}:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def build_experiment_command(
    *,
    cycle: int,
    frontier_name: str,
    out_root: Path,
    journal_path: Path,
    params_path: Path,
    result_path: Path,
) -> List[str]:
    """Build the child process command used to isolate one frontier execution."""
    helper = PROJECT_ROOT / "experiments" / "run_frontier_once.py"
    return [
        sys.executable,
        str(helper),
        "--cycle",
        str(cycle),
        "--frontier",
        frontier_name,
        "--out",
        str(out_root),
        "--journal",
        str(journal_path),
        "--params-json",
        str(params_path),
        "--result-json",
        str(result_path),
    ]


def experiment_record_from_dict(payload: Dict[str, Any]) -> ExperimentRecord:
    """Reconstruct an ExperimentRecord from JSON data."""
    return ExperimentRecord(
        frontier=str(payload.get("frontier", "unknown")),
        cycle=int(payload.get("cycle", 0)),
        timestamp=str(payload.get("timestamp", "")),
        hyperparams=dict(payload.get("hyperparams", {})),
        metrics=dict(payload.get("metrics", {})),
        duration_sec=float(payload.get("duration_sec", 0.0)),
        status=str(payload.get("status", "failed")),
        finding=str(payload.get("finding", "")),
        error=payload.get("error"),
    )


def run_frontier_once_isolated(
    *,
    cycle: int,
    frontier_name: str,
    params: Dict[str, Any],
    journal_path: Path,
    out_root: Path,
    timeout_seconds: int,
    device_policy: str,
) -> ExperimentRecord:
    """Run one frontier in a child process so failures do not poison the host loop."""
    cycle_dir = out_root / f"cycle_{cycle:05d}_{frontier_name}"
    cycle_dir.mkdir(parents=True, exist_ok=True)

    params_path = cycle_dir / "isolated_params.json"
    result_path = cycle_dir / "isolated_record.json"
    stdout_path = cycle_dir / "isolated_stdout.log"
    stderr_path = cycle_dir / "isolated_stderr.log"
    params_path.write_text(json.dumps(params, indent=2, default=str), encoding="utf-8")

    command = build_experiment_command(
        cycle=cycle,
        frontier_name=frontier_name,
        out_root=out_root,
        journal_path=journal_path,
        params_path=params_path,
        result_path=result_path,
    )
    start = time.perf_counter()

    try:
        completed = subprocess.run(
            command,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            cwd=PROJECT_ROOT,
            env=build_child_environment(device_policy),
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - start
        stdout_path.write_text(exc.stdout or "", encoding="utf-8")
        stderr_path.write_text(exc.stderr or "", encoding="utf-8")
        return ExperimentRecord(
            frontier=frontier_name,
            cycle=cycle,
            timestamp=datetime.now(timezone.utc).isoformat(),
            hyperparams=params,
            metrics={},
            duration_sec=duration,
            status="failed",
            finding=f"FAILED: TimeoutExpired after {timeout_seconds}s",
            error=f"Child experiment timed out after {timeout_seconds} seconds",
        )

    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")

    if completed.returncode != 0:
        duration = time.perf_counter() - start
        detail = (completed.stderr or completed.stdout).strip()[:4000]
        return ExperimentRecord(
            frontier=frontier_name,
            cycle=cycle,
            timestamp=datetime.now(timezone.utc).isoformat(),
            hyperparams=params,
            metrics={},
            duration_sec=duration,
            status="failed",
            finding=f"FAILED: ChildProcessError exit={completed.returncode}",
            error=detail or f"Child experiment exited with code {completed.returncode}",
        )

    if not result_path.exists():
        duration = time.perf_counter() - start
        return ExperimentRecord(
            frontier=frontier_name,
            cycle=cycle,
            timestamp=datetime.now(timezone.utc).isoformat(),
            hyperparams=params,
            metrics={},
            duration_sec=duration,
            status="failed",
            finding="FAILED: Child process produced no experiment record",
            error="Missing isolated_record.json from child process",
        )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    return experiment_record_from_dict(payload)


def analyze_result(
    frontier_name: str,
    metrics: Dict[str, float],
    journal: ResearchJournal,
) -> str:
    """Generate a one-line finding from experiment results."""
    best = journal.best_per_frontier.get(frontier_name, {})
    if "score" not in best:
        return (
            f"Baseline established: {frontier_name} "
            f"composite={metrics.get('composite_score', 0.0):.4f}"
        )
    best_score = best.get("score", 0.0)
    current_score = metrics.get("composite_score", 0.0)

    if current_score > best_score * 1.1:
        return f"BREAKTHROUGH: {frontier_name} composite={current_score:.4f} (+{(current_score-best_score)/max(best_score,1e-8)*100:.1f}% over previous best)"
    elif current_score > best_score:
        return f"Improvement: {frontier_name} composite={current_score:.4f} (marginal gain)"
    else:
        return f"No improvement: {frontier_name} composite={current_score:.4f} (best={best_score:.4f})"


def run_cycle(
    cycle: int,
    journal: ResearchJournal,
    out_root: Path,
    forced_frontier: Optional[str] = None,
    runtime_config: Optional[CycleRuntimeConfig] = None,
    env_override: Optional[str] = None,
) -> ExperimentRecord:
    """Execute one research cycle."""
    runtime_config = runtime_config or CURRENT_CYCLE_RUNTIME_CONFIG
    frontier_name = select_frontier(journal, forced_frontier)

    # Get hyperparams: mutate from best or use defaults
    spec = FRONTIER_REGISTRY[frontier_name]
    best_info = journal.best_per_frontier.get(frontier_name, {})
    base_params = best_info.get("hyperparams", spec.default_hyperparams)
    params = mutate_hyperparams(
        base_params, spec.hyperparam_ranges, spec.mutation_sigma,
        journal, frontier_name,
    )
    if env_override:
        params["env"] = env_override
    params = prepare_frontier_params(frontier_name, params)
    device_policy = resolve_cycle_device_policy(journal, frontier_name, runtime_config)
    params = apply_resource_profile(frontier_name, params, device_policy)
    params = prepare_frontier_params(frontier_name, params)

    # Inject a unique seed per cycle to prevent deterministic repetition
    params["seed"] = random.randint(0, 2**31 - 1)

    if runtime_config and runtime_config.journal_path is not None:
        return run_frontier_once_isolated(
            cycle=cycle,
            frontier_name=frontier_name,
            params=params,
            journal_path=runtime_config.journal_path,
            out_root=out_root,
            timeout_seconds=runtime_config.child_timeout_seconds,
            device_policy=device_policy,
        )

    return run_frontier_once(
        cycle=cycle,
        frontier_name=frontier_name,
        params=params,
        journal=journal,
        out_root=out_root,
    )


def run_frontier_once(
    cycle: int,
    frontier_name: str,
    params: Dict[str, Any],
    journal: ResearchJournal,
    out_root: Path,
) -> ExperimentRecord:
    """Execute a specific frontier with caller-supplied hyperparameters."""
    spec = FRONTIER_REGISTRY[frontier_name]
    params = prepare_frontier_params(frontier_name, params)

    cycle_dir = out_root / f"cycle_{cycle:05d}_{frontier_name}"
    cycle_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    config_path = cycle_dir / "config.json"
    config_path.write_text(json.dumps({
        "frontier": frontier_name,
        "cycle": cycle,
        "hyperparams": params,
    }, indent=2, default=str), encoding="utf-8")

    timestamp = datetime.now(timezone.utc).isoformat()
    print(f"\n{'='*72}")
    print(f"  CYCLE {cycle} | Frontier: {frontier_name}")
    print(f"  {console_safe(spec.description)}")
    print(f"  Time: {timestamp}")
    print(f"  Output: {cycle_dir}")
    print(f"{'='*72}")

    start = time.perf_counter()
    try:
        metrics = spec.runner(params, cycle_dir)
        duration = time.perf_counter() - start
        finding = analyze_result(frontier_name, metrics, journal)

        record = ExperimentRecord(
            frontier=frontier_name,
            cycle=cycle,
            timestamp=timestamp,
            hyperparams=params,
            metrics=metrics,
            duration_sec=duration,
            status="success",
            finding=finding,
        )

        # Check for breakthrough
        if "BREAKTHROUGH" in finding:
            journal.breakthroughs.append(f"Cycle {cycle}: {finding}")
            print(f"\n  *** {finding} ***\n")
        else:
            print(f"\n  {finding}")

        print(f"  Duration: {duration:.1f}s")
        print(f"  Metrics: {json.dumps(metrics, indent=2)}")

    except Exception as exc:
        duration = time.perf_counter() - start
        error_text = traceback.format_exc()
        print(f"\n  FAILED: {exc}")
        print(f"  {error_text}")

        record = ExperimentRecord(
            frontier=frontier_name,
            cycle=cycle,
            timestamp=timestamp,
            hyperparams=params,
            metrics={},
            duration_sec=duration,
            status="failed",
            finding=f"FAILED: {exc.__class__.__name__}: {exc}",
            error=error_text,
        )

    # Save results
    results_path = cycle_dir / "results.json"
    results_path.write_text(
        json.dumps(asdict(record), indent=2, default=str), encoding="utf-8"
    )

    return record


def print_dashboard(journal: ResearchJournal) -> None:
    """Print current research status dashboard."""
    print(f"\n{'='*72}")
    print(f"  RESEARCH DASHBOARD | Total cycles: {journal.total_cycles}")
    print(f"{'='*72}")

    priorities = journal.get_frontier_priority()
    print(f"\n  Frontier Priorities (UCB1):")
    for name, priority in sorted(priorities.items(), key=lambda x: -x[1]):
        n_runs = len(journal.frontier_scores.get(name, []))
        best = journal.best_per_frontier.get(name, {})
        best_score = best.get("score", 0.0)
        print(f"    {name:35s} | UCB={priority:.3f} | runs={n_runs:3d} | best={best_score:.4f}")

    if journal.breakthroughs:
        print(f"\n  Breakthroughs ({len(journal.breakthroughs)}):")
        for b in journal.breakthroughs[-5:]:
            print(f"    {b}")

    print()


def should_stop(stop_path: Optional[Path]) -> bool:
    """Return True when a cooperative stop file is present."""
    return stop_path is not None and stop_path.exists()


def generate_dashboard_safely(journal_path: Path, dashboard_path: Path) -> None:
    """Refresh the HTML dashboard without crashing the orchestrator."""
    if not journal_path.exists():
        return
    try:
        from frontiers.research_dashboard import generate_dashboard

        generate_dashboard(journal_path, dashboard_path)
    except Exception as exc:
        print(f"  WARNING: dashboard refresh failed ({exc})")


def build_status_snapshot(
    *,
    state: str,
    started_at: str,
    out_root: Path,
    journal_path: Path,
    dashboard_path: Path,
    status_path: Path,
    stop_path: Path,
    journal: ResearchJournal,
    start_cycle: int,
    next_cycle: int,
    target_cycles: int,
    forced_frontier: Optional[str],
    consecutive_failures: int,
    last_record: Optional[ExperimentRecord],
    configured_device_policy: str = "auto",
    active_device_policy: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a runtime status payload for external monitoring."""
    best_frontier = None
    if journal.best_per_frontier:
        frontier_name, payload = max(
            journal.best_per_frontier.items(),
            key=lambda item: float(item[1].get("score", -float("inf"))),
        )
        best_frontier = {
            "name": frontier_name,
            "score": float(payload.get("score", 0.0)),
            "cycle": payload.get("cycle"),
        }

    completed_in_session = max(next_cycle - start_cycle, 0)
    remaining_cycles = None if target_cycles == 0 else max(target_cycles - completed_in_session, 0)

    return {
        "state": state,
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "output_root": str(out_root),
        "journal_path": str(journal_path),
        "dashboard_path": str(dashboard_path),
        "status_path": str(status_path),
        "stop_file": str(stop_path),
        "forced_frontier": forced_frontier,
        "configured_device_policy": configured_device_policy,
        "active_device_policy": active_device_policy or configured_device_policy,
        "start_cycle": start_cycle,
        "next_cycle": next_cycle,
        "session_target_cycles": None if target_cycles == 0 else target_cycles,
        "session_completed_cycles": completed_in_session,
        "session_remaining_cycles": remaining_cycles,
        "total_cycles": journal.total_cycles,
        "consecutive_failures": consecutive_failures,
        "recent_failure_streak": journal.tail_failure_streak(),
        "recent_resource_exhaustion_streak": journal.tail_resource_exhaustion_streak(),
        "breakthrough_count": len(journal.breakthroughs),
        "frontier_priorities": journal.get_frontier_priority(),
        "best_frontier": best_frontier,
        "last_record": asdict(last_record) if last_record else None,
    }


def write_status(path: Path, payload: Dict[str, Any]) -> None:
    """Persist the runtime status JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def persist_runtime_artifacts(
    *,
    state: str,
    started_at: str,
    out_root: Path,
    journal_path: Path,
    dashboard_path: Path,
    status_path: Path,
    stop_path: Path,
    journal: ResearchJournal,
    start_cycle: int,
    next_cycle: int,
    target_cycles: int,
    forced_frontier: Optional[str],
    consecutive_failures: int,
    last_record: Optional[ExperimentRecord],
    configured_device_policy: str = "auto",
    active_device_policy: Optional[str] = None,
) -> None:
    """Save journal + dashboard + status snapshot."""
    journal.save(journal_path)
    generate_dashboard_safely(journal_path, dashboard_path)
    write_status(
        status_path,
        build_status_snapshot(
            state=state,
            started_at=started_at,
            out_root=out_root,
            journal_path=journal_path,
            dashboard_path=dashboard_path,
            status_path=status_path,
            stop_path=stop_path,
            journal=journal,
            start_cycle=start_cycle,
            next_cycle=next_cycle,
            target_cycles=target_cycles,
            forced_frontier=forced_frontier,
            consecutive_failures=consecutive_failures,
            last_record=last_record,
            configured_device_policy=configured_device_policy,
            active_device_policy=active_device_policy,
        ),
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Autonomous Research Orchestrator")
    parser.add_argument("--env", type=str, default=None, help="Override default environment")
    parser.add_argument("--cycles", type=int, default=0, help="Number of cycles (0=infinite)")
    parser.add_argument("--frontier", type=str, default=None, help="Force specific frontier")
    parser.add_argument("--out", type=str, default="research_runs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--journal", type=str, default=None, help="Path to journal file")
    parser.add_argument("--dashboard", type=str, default=None, help="Path to dashboard HTML")
    parser.add_argument("--status", type=str, default=None, help="Path to runtime status JSON")
    parser.add_argument("--stop-file", type=str, default=None, help="Cooperative stop file path")
    parser.add_argument(
        "--cycle-delay-seconds",
        type=float,
        default=1.0,
        help="Delay between successful cycles",
    )
    parser.add_argument(
        "--failure-backoff-seconds",
        type=float,
        default=15.0,
        help="Delay after a failed cycle before retrying",
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=0,
        help="Abort after N failed cycles in a row (0 disables the limit)",
    )
    parser.add_argument(
        "--experiment-timeout-seconds",
        type=int,
        default=7200,
        help="Timeout per isolated frontier experiment subprocess",
    )
    parser.add_argument(
        "--device-policy",
        type=str,
        default="auto",
        help="One of: auto, cpu, cuda",
    )
    parser.add_argument(
        "--cpu-fallback-after-failures",
        type=int,
        default=3,
        help="Switch to CPU-safe mode after N consecutive failures",
    )
    args = parser.parse_args(argv)

    if args.cycles < 0:
        parser.error("--cycles must be >= 0")
    if args.frontier and args.frontier not in FRONTIER_REGISTRY:
        parser.error(
            f"--frontier must be one of: {', '.join(sorted(FRONTIER_REGISTRY))}"
        )
    if args.cycle_delay_seconds < 0:
        parser.error("--cycle-delay-seconds must be >= 0")
    if args.failure_backoff_seconds < 0:
        parser.error("--failure-backoff-seconds must be >= 0")
    if args.max_consecutive_failures < 0:
        parser.error("--max-consecutive-failures must be >= 0")
    if args.experiment_timeout_seconds <= 0:
        parser.error("--experiment-timeout-seconds must be > 0")
    if args.cpu_fallback_after_failures < 0:
        parser.error("--cpu-fallback-after-failures must be >= 0")
    device_policy = normalize_device_policy(args.device_policy)

    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        if device_policy != "cpu":
            torch.manual_seed(args.seed)
    except RuntimeError as exc:
        if is_resource_exhaustion_error(str(exc)):
            print(
                "WARNING: host torch.manual_seed skipped because CUDA was already in a "
                "resource-exhausted state; isolated child runs will handle seeding."
            )
        else:
            raise

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    journal_path = Path(args.journal).resolve() if args.journal else out_root / "journal.json"
    dashboard_path = Path(args.dashboard).resolve() if args.dashboard else out_root / "dashboard.html"
    status_path = Path(args.status).resolve() if args.status else out_root / "status.json"
    stop_path = Path(args.stop_file).resolve() if args.stop_file else out_root / "STOP"
    journal = ResearchJournal.load(journal_path)
    start_cycle = journal.total_cycles
    cycle = start_cycle
    stop_after_cycle = None if args.cycles == 0 else start_cycle + args.cycles
    started_at = datetime.now(timezone.utc).isoformat()
    last_record: Optional[ExperimentRecord] = journal.records[-1] if journal.records else None
    consecutive_failures = journal.tail_failure_streak()
    last_device_policy = device_policy
    exit_state = "starting"

    print(f"\n{'#'*72}")
    print(f"#  AUTONOMOUS RESEARCH ORCHESTRATOR")
    print(f"#  Pushing into uncharted territory -- 24/7")
    print(f"#  Output: {out_root}")
    print(f"#  Journal: {journal_path}")
    print(f"#  Dashboard: {dashboard_path}")
    print(f"#  Status: {status_path}")
    print(f"#  Stop file: {stop_path}")
    print(f"#  Cycles this session: {'infinite' if args.cycles == 0 else args.cycles}")
    print(f"#  Device policy: {device_policy}")
    print(f"#  Frontiers: {', '.join(FRONTIER_REGISTRY.keys())}")
    print(f"{'#'*72}")

    persist_runtime_artifacts(
        state=exit_state,
        started_at=started_at,
        out_root=out_root,
        journal_path=journal_path,
        dashboard_path=dashboard_path,
        status_path=status_path,
        stop_path=stop_path,
        journal=journal,
        start_cycle=start_cycle,
        next_cycle=cycle,
        target_cycles=args.cycles,
        forced_frontier=args.frontier,
        consecutive_failures=consecutive_failures,
        last_record=last_record,
        configured_device_policy=device_policy,
        active_device_policy=last_device_policy,
    )

    try:
        while stop_after_cycle is None or cycle < stop_after_cycle:
            if should_stop(stop_path):
                print(f"\nStop signal detected at {stop_path}. Exiting cleanly.")
                exit_state = "stopped"
                break

            exit_state = "running"
            print_dashboard(journal)
            persist_runtime_artifacts(
                state=exit_state,
                started_at=started_at,
                out_root=out_root,
                journal_path=journal_path,
                dashboard_path=dashboard_path,
                status_path=status_path,
                stop_path=stop_path,
                journal=journal,
                start_cycle=start_cycle,
                next_cycle=cycle,
                target_cycles=args.cycles,
                forced_frontier=args.frontier,
                consecutive_failures=consecutive_failures,
                last_record=last_record,
                configured_device_policy=device_policy,
                active_device_policy=last_device_policy,
            )

            runtime_config = CycleRuntimeConfig(
                journal_path=journal_path,
                child_timeout_seconds=args.experiment_timeout_seconds,
                device_policy=device_policy,
                cpu_fallback_after_failures=args.cpu_fallback_after_failures,
                consecutive_failures=consecutive_failures,
            )
            target_frontier = args.frontier or select_frontier(journal)
            last_device_policy = resolve_cycle_device_policy(journal, target_frontier, runtime_config)
            global CURRENT_CYCLE_RUNTIME_CONFIG
            CURRENT_CYCLE_RUNTIME_CONFIG = runtime_config
            try:
                record = run_cycle(cycle, journal, out_root, target_frontier, env_override=args.env)
            finally:
                CURRENT_CYCLE_RUNTIME_CONFIG = None
            last_record = record
            journal.add(record)
            consecutive_failures = 0 if record.status == "success" else consecutive_failures + 1

            cycle += 1
            persist_runtime_artifacts(
                state=exit_state,
                started_at=started_at,
                out_root=out_root,
                journal_path=journal_path,
                dashboard_path=dashboard_path,
                status_path=status_path,
                stop_path=stop_path,
                journal=journal,
                start_cycle=start_cycle,
                next_cycle=cycle,
                target_cycles=args.cycles,
                forced_frontier=args.frontier,
                consecutive_failures=consecutive_failures,
                last_record=last_record,
                configured_device_policy=device_policy,
                active_device_policy=last_device_policy,
            )

            if (
                args.max_consecutive_failures > 0
                and consecutive_failures >= args.max_consecutive_failures
            ):
                print(
                    "\nReached the consecutive failure limit "
                    f"({consecutive_failures}/{args.max_consecutive_failures})."
                )
                exit_state = "failed_limit"
                break

            delay_seconds = (
                args.cycle_delay_seconds
                if record.status == "success"
                else args.failure_backoff_seconds
            )
            if delay_seconds > 0 and (stop_after_cycle is None or cycle < stop_after_cycle):
                time.sleep(delay_seconds)

    except KeyboardInterrupt:
        print("\n\nResearch paused by operator. Runtime state saved.")
        exit_state = "paused"

    if exit_state == "running":
        exit_state = "completed"

    persist_runtime_artifacts(
        state=exit_state,
        started_at=started_at,
        out_root=out_root,
        journal_path=journal_path,
        dashboard_path=dashboard_path,
        status_path=status_path,
        stop_path=stop_path,
        journal=journal,
        start_cycle=start_cycle,
        next_cycle=cycle,
        target_cycles=args.cycles,
        forced_frontier=args.frontier,
        consecutive_failures=consecutive_failures,
        last_record=last_record,
        configured_device_policy=device_policy,
        active_device_policy=last_device_policy,
    )
    print_dashboard(journal)
    print(f"Final journal saved to: {journal_path}")
    print(f"Dashboard: {dashboard_path}")
    print(f"Status: {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
