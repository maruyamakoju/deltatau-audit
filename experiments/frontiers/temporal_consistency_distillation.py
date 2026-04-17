"""
Temporal Consistency Distillation
=================================
Distills a slow MCTS search policy into a fast neural network while preserving
timing safety guarantees via a Lipschitz-preserving consistency loss.

    L_distill = L_KL(teacher, student)
              + lambda_lip * max(0, L_student - L_teacher * margin)

L_student is the student's spectral norm product bound; L_teacher is estimated
from the MCTS value landscape's sensitivity to delta_tau perturbations.

Self-contained -- no imports from the parent package.
"""
from __future__ import annotations

import json, math, random, statistics, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _spectral_norm_estimate(weight: torch.Tensor, n_iters: int = 3) -> torch.Tensor:
    """Largest singular value via power iteration (no forward-hook mutation)."""
    if weight.ndim < 2:
        return weight.abs().max()
    mat = weight.reshape(weight.shape[0], -1)
    u = F.normalize(torch.randn(mat.shape[0], device=weight.device), dim=0)
    with torch.no_grad():
        for _ in range(n_iters):
            v = F.normalize(mat.t() @ u, dim=0)
            u = F.normalize(mat @ v, dim=0)
    return (u @ mat @ v).abs()


# ---------------------------------------------------------------------------
# 1. TeacherMCTS
# ---------------------------------------------------------------------------

class WorldModel(nn.Module):
    """(hidden, obs, delta_tau) -> (next_hidden, reward, done_logit, value)."""

    def __init__(self, obs_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim + obs_dim + 1, hidden_dim * 2), nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.done_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.hidden_dim = hidden_dim

    def forward(self, hidden: torch.Tensor, obs: torch.Tensor,
                delta_tau: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x = torch.cat([hidden, obs, delta_tau.unsqueeze(-1)], dim=-1)
        h = self.trunk(x)
        return h, self.reward_head(h), self.done_head(h), self.value_head(h)

    def initial_hidden(self, batch: int = 1, device: torch.device | None = None) -> torch.Tensor:
        return torch.zeros(batch, self.hidden_dim,
                           device=device or next(self.parameters()).device)


@dataclass
class MCTSNode:
    hidden: torch.Tensor
    reward: float = 0.0
    value_sum: float = 0.0
    visit_count: int = 0
    prior: float = 1.0
    children: Dict[int, "MCTSNode"] = field(default_factory=dict)
    done: bool = False

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0


class TeacherMCTS:
    """MCTS teacher with PUCT selection, lambda-return backup, and empirical
    Lipschitz estimation of value w.r.t. delta_tau."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128,
                 c_puct: float = 1.5, gamma: float = 0.99, lam: float = 0.8,
                 delta_tau_nominal: float = 0.05, delta_tau_perturb: float = 0.01,
                 device: torch.device | None = None) -> None:
        self.action_dim, self.c_puct, self.gamma, self.lam = action_dim, c_puct, gamma, lam
        self.delta_tau_nominal = delta_tau_nominal
        self.delta_tau_perturb = delta_tau_perturb
        self.device = device or torch.device("cpu")

        self.world_model = WorldModel(obs_dim, hidden_dim).to(self.device)
        self.prior_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim),
        ).to(self.device)

        for net in (self.world_model, self.prior_net):
            for p in net.parameters():
                if p.ndim >= 2:
                    nn.init.orthogonal_(p, gain=0.5)

    def _puct_score(self, parent: MCTSNode, child: MCTSNode) -> float:
        return child.value + self.c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)

    def _select_action(self, node: MCTSNode) -> int:
        best_a, best_s = 0, -float("inf")
        for a, ch in node.children.items():
            s = self._puct_score(node, ch)
            if s > best_s:
                best_a, best_s = a, s
        return best_a

    @torch.no_grad()
    def _expand(self, node: MCTSNode, obs: torch.Tensor, delta_tau: float) -> None:
        dt = torch.tensor([delta_tau], device=self.device)
        priors = F.softmax(self.prior_net(obs), dim=-1).squeeze(0)
        for a in range(self.action_dim):
            obs_a = obs.clone()
            obs_a[0, a % obs_a.shape[-1]] += 0.1 * (a + 1)
            h_next, r, d, v = self.world_model(node.hidden, obs_a, dt)
            ch = MCTSNode(hidden=h_next, reward=r.item(), prior=priors[a].item(),
                          done=torch.sigmoid(d).item() > 0.5)
            ch.value_sum, ch.visit_count = v.item(), 1
            node.children[a] = ch

    def _backup(self, path: List[Tuple[MCTSNode, int]], leaf_value: float) -> None:
        g = leaf_value
        for node, act in reversed(path):
            ch = node.children[act]
            g = ch.reward + self.gamma * ((1 - self.lam) * ch.value + self.lam * g)
            ch.value_sum += g
            ch.visit_count += 1

    @torch.no_grad()
    def _search(self, obs: torch.Tensor, n_sims: int, delta_tau: float) -> MCTSNode:
        root = MCTSNode(hidden=self.world_model.initial_hidden(1, self.device))
        self._expand(root, obs, delta_tau)
        for _ in range(n_sims):
            node, path = root, []
            while node.children and not node.done:
                a = self._select_action(node); path.append((node, a)); node = node.children[a]
            if not node.done and node.visit_count > 0:
                self._expand(node, obs, delta_tau)
                if node.children:
                    a = self._select_action(node); path.append((node, a)); node = node.children[a]
            self._backup(path, node.value)
        return root

    @staticmethod
    def _estimate_root_value(root: MCTSNode) -> float:
        if not root.children:
            return 0.0
        total_visits = sum(max(ch.visit_count, 0) for ch in root.children.values())
        if total_visits <= 0:
            return 0.0
        weighted_value = sum(ch.value * ch.visit_count for ch in root.children.values())
        return float(weighted_value / total_visits)

    def generate_targets(self, obs: np.ndarray, n_sims: int = 64
                         ) -> Tuple[np.ndarray, float, float]:
        """Return (action_probs, value_estimate, lipschitz_estimate)."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        root_nom = self._search(obs_t, n_sims, self.delta_tau_nominal)
        root_pert = self._search(obs_t, n_sims, self.delta_tau_nominal + self.delta_tau_perturb)
        root_nom_value = self._estimate_root_value(root_nom)
        root_pert_value = self._estimate_root_value(root_pert)
        visits = np.array([root_nom.children[a].visit_count if a in root_nom.children else 0
                           for a in range(self.action_dim)], dtype=np.float32)
        action_probs = visits / max(visits.sum(), 1.0)
        lip = abs(root_pert_value - root_nom_value) / max(abs(self.delta_tau_perturb), 1e-8)
        return action_probs, root_nom_value, lip

    def act(self, obs: np.ndarray, n_sims: int = 64) -> int:
        return int(np.argmax(self.generate_targets(obs, n_sims)[0]))


# ---------------------------------------------------------------------------
# 2. StudentNetwork
# ---------------------------------------------------------------------------

class StudentNetwork(nn.Module):
    """Fast feedforward: encoder -> policy_head | value_head | timing_head."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64,
                 use_spectral_norm: bool = False) -> None:
        super().__init__()
        sn = nn.utils.spectral_norm if use_spectral_norm else (lambda m: m)
        self.encoder = nn.Sequential(sn(nn.Linear(obs_dim, hidden_dim)), nn.ReLU(),
                                     sn(nn.Linear(hidden_dim, hidden_dim)))
        self.policy_head = sn(nn.Linear(hidden_dim, action_dim))
        self.value_head = sn(nn.Linear(hidden_dim, 1))
        self.timing_head = nn.Sequential(sn(nn.Linear(hidden_dim, 1)), nn.Softplus())

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (policy_logits, value, dt)."""
        h = self.encoder(obs)
        return self.policy_head(h), self.value_head(h), self.timing_head(h)


# ---------------------------------------------------------------------------
# 3. LipschitzConsistencyLoss
# ---------------------------------------------------------------------------

class LipschitzConsistencyLoss(nn.Module):
    """L_lip = ReLU(student_lip - teacher_lip * margin) + w * MSE(dt)."""

    def __init__(self, margin: float = 1.2, timing_weight: float = 0.1,
                 power_iters: int = 5) -> None:
        super().__init__()
        self.margin, self.timing_weight, self.power_iters = margin, timing_weight, power_iters

    def _lipschitz_product_bound(self, model: nn.Module) -> torch.Tensor:
        lip = torch.tensor(1.0, device=next(model.parameters()).device)
        for m in model.modules():
            if isinstance(m, nn.Linear):
                lip = lip * _spectral_norm_estimate(m.weight, self.power_iters)
        return lip

    def forward(self, student: StudentNetwork, teacher_lip: torch.Tensor,
                student_dt: torch.Tensor, target_dt: torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        s_lip = self._lipschitz_product_bound(student)
        t_lip = teacher_lip.mean() if teacher_lip.ndim > 0 else teacher_lip
        lip_viol = F.relu(s_lip - t_lip * self.margin)
        timing_mse = F.mse_loss(student_dt, target_dt)
        total = lip_viol + self.timing_weight * timing_mse
        info = {"student_lip": s_lip.item(), "teacher_lip_mean": t_lip.item(),
                "lip_violation": lip_viol.item(), "timing_mse": timing_mse.item()}
        return total, info


# ---------------------------------------------------------------------------
# 4. DistillationTrainer
# ---------------------------------------------------------------------------

@dataclass
class DistillationBuffer:
    observations: List[np.ndarray] = field(default_factory=list)
    action_probs: List[np.ndarray] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    lipschitz_estimates: List[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.observations)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        idx = np.random.choice(len(self), size=min(batch_size, len(self)), replace=False)
        return (np.stack([self.observations[i] for i in idx]),
                np.stack([self.action_probs[i] for i in idx]),
                np.array([self.values[i] for i in idx], dtype=np.float32),
                np.array([self.lipschitz_estimates[i] for i in idx], dtype=np.float32))


class DistillationTrainer:
    """Phase 1: generate MCTS targets. Phase 2: train student. Phase 3: evaluate."""

    def __init__(self, teacher: TeacherMCTS, student: StudentNetwork,
                 env_id: str = "CartPole-v1", lr: float = 3e-4,
                 lipschitz_penalty: float = 0.5, temperature: float = 1.0,
                 lip_margin: float = 1.2, delta_tau_nominal: float = 0.05,
                 seed: int = 42, device: torch.device | None = None) -> None:
        self.teacher, self.student = teacher, student
        self.env_id, self.lipschitz_penalty = env_id, lipschitz_penalty
        self.temperature, self.delta_tau_nominal = temperature, delta_tau_nominal
        self.seed = seed
        self.device = device or torch.device("cpu")
        self.optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=lr * 0.01)
        self.consistency_loss = LipschitzConsistencyLoss(margin=lip_margin)
        self.buffer = DistillationBuffer()
        self.train_log: List[Dict[str, float]] = []

    def _episode_seed(self, episode_index: int, phase_offset: int = 0) -> int:
        return int(self.seed + phase_offset + episode_index)

    # Phase 1 ----------------------------------------------------------------

    def generate_teacher_data(self, n_samples: int = 512, n_sims: int = 64,
                              max_episode_steps: int = 200) -> int:
        env = gym.make(self.env_id)
        collected = 0
        episode_index = 0
        while collected < n_samples:
            obs, _ = env.reset(seed=self._episode_seed(episode_index))
            episode_index += 1
            for _ in range(max_episode_steps):
                probs, val, lip = self.teacher.generate_targets(obs, n_sims)
                self.buffer.observations.append(obs.copy())
                self.buffer.action_probs.append(probs)
                self.buffer.values.append(val)
                self.buffer.lipschitz_estimates.append(lip)
                collected += 1
                if collected >= n_samples:
                    break
                obs, _, term, trunc, _ = env.step(int(np.argmax(probs)))
                if term or trunc:
                    break
        env.close()
        return collected

    # Phase 2 ----------------------------------------------------------------

    def train_step(self, batch_size: int = 64) -> Dict[str, float]:
        obs_np, probs_np, vals_np, lips_np = self.buffer.sample(batch_size)
        obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
        target_probs = torch.tensor(probs_np, dtype=torch.float32, device=self.device)
        target_vals = torch.tensor(vals_np, dtype=torch.float32, device=self.device).unsqueeze(-1)
        teacher_lip = torch.tensor(lips_np, dtype=torch.float32, device=self.device)

        logits, pred_val, pred_dt = self.student(obs)
        kl_loss = F.kl_div(F.log_softmax(logits / self.temperature, dim=-1),
                           target_probs.clamp(min=1e-8), reduction="batchmean")
        value_loss = F.mse_loss(pred_val, target_vals)
        target_dt = torch.full_like(pred_dt, self.delta_tau_nominal)
        consistency, c_info = self.consistency_loss(self.student, teacher_lip, pred_dt, target_dt)
        total = kl_loss + value_loss + self.lipschitz_penalty * consistency

        self.optimizer.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        m = {"total_loss": total.item(), "kl_loss": kl_loss.item(),
             "value_loss": value_loss.item(), "consistency_loss": consistency.item(),
             "lr": self.optimizer.param_groups[0]["lr"], **c_info}
        self.train_log.append(m)
        return m

    def train(self, steps: int = 500, batch_size: int = 64, log_every: int = 50) -> None:
        self.student.train()
        for step in range(1, steps + 1):
            m = self.train_step(batch_size)
            if step % log_every == 0 or step == 1:
                print(f"  [step {step:>5d}]  loss={m['total_loss']:.4f}  kl={m['kl_loss']:.4f}  "
                      f"val={m['value_loss']:.4f}  lip_viol={m['lip_violation']:.4f}  "
                      f"s_lip={m['student_lip']:.3f}")

    # Phase 3 ----------------------------------------------------------------

    def evaluate(self, n_episodes: int = 20, max_steps: int = 500,
                 n_sims: int = 32) -> Dict[str, float]:
        self.student.eval()
        env = gym.make(self.env_id)
        t_rets, s_rets, agrees, t_times, s_times = [], [], [], [], []

        for episode_index in range(n_episodes):
            episode_seed = self._episode_seed(episode_index, phase_offset=10_000)
            # Teacher episode
            obs, _ = env.reset(seed=episode_seed); ret = 0.0; t0 = time.perf_counter()
            for __ in range(max_steps):
                obs, r, term, trunc, _ = env.step(self.teacher.act(obs, n_sims)); ret += r
                if term or trunc: break
            t_times.append(time.perf_counter() - t0); t_rets.append(ret)

            # Student episode
            obs, _ = env.reset(seed=episode_seed); ret = 0.0; t0 = time.perf_counter()
            for __ in range(max_steps):
                with torch.no_grad():
                    logits, _, _ = self.student(
                        torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
                obs, r, term, trunc, _ = env.step(int(logits.argmax(-1).item())); ret += r
                if term or trunc: break
            s_times.append(time.perf_counter() - t0); s_rets.append(ret)

            # Policy agreement
            obs, _ = env.reset(seed=episode_seed); agree, total = 0, 0
            for __ in range(max_steps):
                ta = self.teacher.act(obs, n_sims)
                with torch.no_grad():
                    logits, _, _ = self.student(
                        torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
                agree += int(ta == int(logits.argmax(-1).item())); total += 1
                obs, _, term, trunc, _ = env.step(ta)
                if term or trunc: break
            agrees.append(agree / max(total, 1))

        env.close()
        _std = lambda xs: statistics.stdev(xs) if len(xs) > 1 else 0.0
        return {"teacher_return_mean": statistics.mean(t_rets),
                "teacher_return_std": _std(t_rets),
                "student_return_mean": statistics.mean(s_rets),
                "student_return_std": _std(s_rets),
                "policy_agreement": statistics.mean(agrees),
                "teacher_time_mean": statistics.mean(t_times),
                "student_time_mean": statistics.mean(s_times)}


# ---------------------------------------------------------------------------
# 5. ConsistencyDistillationExperiment
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    obs_dim: int = 4
    action_dim: int = 2
    teacher_hidden_dim: int = 128
    student_hidden_dim: int = 64
    num_simulations: int = 32
    distill_steps: int = 500
    lipschitz_penalty: float = 0.5
    temperature: float = 1.0
    lr: float = 3e-4
    batch_size: int = 64
    n_eval_episodes: int = 10
    max_steps: int = 200
    lip_margin: float = 1.2
    delta_tau_nominal: float = 0.05
    seed: int = 42
    env_id: str = "CartPole-v1"
    use_spectral_norm: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class ConsistencyDistillationExperiment:
    """Full pipeline: build -> generate targets -> distill -> evaluate -> score.

    composite = 0.30 * return_ratio + 0.25 * agreement + 0.25 * lip_preservation + 0.20 * speedup
    """

    def __init__(self, **kwargs: Any) -> None:
        self.cfg = ExperimentConfig(**kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_teacher(self) -> TeacherMCTS:
        return TeacherMCTS(self.cfg.obs_dim, self.cfg.action_dim, self.cfg.teacher_hidden_dim,
                           delta_tau_nominal=self.cfg.delta_tau_nominal, device=self.device)

    def _build_student(self) -> StudentNetwork:
        return StudentNetwork(self.cfg.obs_dim, self.cfg.action_dim, self.cfg.student_hidden_dim,
                              self.cfg.use_spectral_norm).to(self.device)

    @staticmethod
    def _final_lipschitz(student: StudentNetwork) -> float:
        lip = 1.0
        for m in student.modules():
            if isinstance(m, nn.Linear):
                lip *= _spectral_norm_estimate(m.weight, n_iters=10).item()
        return lip

    def run(self, out_dir: Path) -> Dict[str, float]:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        _set_seed(self.cfg.seed)
        c = self.cfg
        print(f"=== Temporal Consistency Distillation ===\nDevice: {self.device}")
        print(f"Config: {json.dumps(c.to_dict(), indent=2)}")

        teacher, student = self._build_teacher(), self._build_student()
        trainer = DistillationTrainer(
            teacher, student, env_id=c.env_id, lr=c.lr,
            lipschitz_penalty=c.lipschitz_penalty, temperature=c.temperature,
            lip_margin=c.lip_margin, delta_tau_nominal=c.delta_tau_nominal,
            seed=c.seed, device=self.device)

        # Phase 1: generate targets
        n_target = c.batch_size * 10
        print(f"\n--- Phase 1: {n_target} teacher targets (n_sims={c.num_simulations}) ---")
        t0 = time.perf_counter()
        n_col = trainer.generate_teacher_data(n_target, c.num_simulations, c.max_steps)
        gen_time = time.perf_counter() - t0
        lips = trainer.buffer.lipschitz_estimates
        mean_t_lip = statistics.mean(lips) if lips else 1.0
        print(f"  {n_col} samples in {gen_time:.1f}s  |  teacher Lip: "
              f"mean={mean_t_lip:.4f} max={max(lips):.4f} min={min(lips):.4f}")

        # Phase 2: train
        print(f"\n--- Phase 2: Training ({c.distill_steps} steps) ---")
        t0 = time.perf_counter()
        trainer.train(c.distill_steps, c.batch_size, max(1, c.distill_steps // 10))
        train_time = time.perf_counter() - t0
        print(f"  Done in {train_time:.1f}s")

        # Phase 3: evaluate
        print(f"\n--- Phase 3: Evaluation ({c.n_eval_episodes} episodes) ---")
        t0 = time.perf_counter()
        ev = trainer.evaluate(c.n_eval_episodes, c.max_steps, c.num_simulations)
        eval_time = time.perf_counter() - t0
        print(f"  Done in {eval_time:.1f}s")

        # Derived metrics
        s_lip = self._final_lipschitz(student)
        t_ret = max(ev["teacher_return_mean"], 1e-8)
        s_ret = ev["student_return_mean"]
        ret_ratio = s_ret / t_ret
        # When both teacher and student have near-zero Lipschitz, preservation is perfect
        if mean_t_lip < 0.01 and s_lip < 0.01:
            lip_pres = 1.0
        elif mean_t_lip < 0.01:
            lip_pres = max(0.0, 1.0 - s_lip)  # penalise student Lipschitz directly
        else:
            lip_pres = max(0.0, 1.0 - abs(s_lip - mean_t_lip) / mean_t_lip)
        spd_raw = max(ev["teacher_time_mean"], 1e-8) / max(ev["student_time_mean"], 1e-8)
        spd_norm = min(spd_raw, 100.0) / 100.0
        composite = (0.30 * np.clip(ret_ratio, 0.0, 1.5) + 0.25 * ev["policy_agreement"]
                     + 0.25 * np.clip(lip_pres, 0.0, 1.0) + 0.20 * spd_norm)

        results: Dict[str, float] = {
            "teacher_return_mean": ev["teacher_return_mean"],
            "teacher_return_std": ev["teacher_return_std"],
            "student_return_mean": s_ret, "student_return_std": ev["student_return_std"],
            "student_return_ratio": float(ret_ratio),
            "policy_agreement": ev["policy_agreement"],
            "teacher_lipschitz_mean": mean_t_lip, "student_lipschitz": s_lip,
            "lipschitz_preservation": float(lip_pres),
            "teacher_time_mean": ev["teacher_time_mean"],
            "student_time_mean": ev["student_time_mean"],
            "speedup_raw": spd_raw, "speedup_factor": float(spd_norm),
            "composite_score": float(composite),
            "generation_time": gen_time, "training_time": train_time,
            "evaluation_time": eval_time, "total_time": gen_time + train_time + eval_time,
            "n_samples_collected": n_col,
            "final_kl_loss": trainer.train_log[-1]["kl_loss"] if trainer.train_log else 0.0,
            "final_value_loss": trainer.train_log[-1]["value_loss"] if trainer.train_log else 0.0,
        }

        print(f"\n=== Results ===")
        for k in ("teacher_return_mean", "student_return_mean", "student_return_ratio",
                   "policy_agreement", "teacher_lipschitz_mean", "student_lipschitz",
                   "lipschitz_preservation", "speedup_raw", "composite_score"):
            print(f"  {k:>26s}: {results[k]:.4f}")

        # Save artifacts
        with open(out_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        with open(out_dir / "config.json", "w") as f:
            json.dump(c.to_dict(), f, indent=2)
        if trainer.train_log:
            with open(out_dir / "training_curve.json", "w") as f:
                json.dump(trainer.train_log, f, indent=2)
        torch.save({"model_state_dict": student.state_dict(), "config": c.to_dict(),
                     "results": results}, out_dir / "student_checkpoint.pt")
        print(f"  Artifacts saved to {out_dir}")
        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Temporal Consistency Distillation")
    p.add_argument("--out-dir", default="results/temporal_consistency_distillation")
    p.add_argument("--distill-steps", type=int, default=500)
    p.add_argument("--num-sims", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-eval", type=int, default=10)
    p.add_argument("--lipschitz-penalty", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--spectral-norm", action="store_true")
    a = p.parse_args()
    ConsistencyDistillationExperiment(
        distill_steps=a.distill_steps, num_simulations=a.num_sims,
        batch_size=a.batch_size, n_eval_episodes=a.n_eval,
        lipschitz_penalty=a.lipschitz_penalty, temperature=a.temperature,
        lr=a.lr, seed=a.seed, use_spectral_norm=a.spectral_norm,
    ).run(Path(a.out_dir))


if __name__ == "__main__":
    main()
