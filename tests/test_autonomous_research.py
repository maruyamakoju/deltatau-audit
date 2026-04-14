"""Tests for the autonomous research orchestrator and frontier modules."""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Add experiments to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrator unit tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestResearchJournal:
    """Test the research journal persistence and UCB1 frontier selection."""

    def test_journal_save_load_roundtrip(self, tmp_path):
        from autonomous_research import ResearchJournal, ExperimentRecord

        journal = ResearchJournal()
        record = ExperimentRecord(
            frontier="test_frontier",
            cycle=0,
            timestamp="2026-04-05T00:00:00Z",
            hyperparams={"lr": 0.001},
            metrics={"composite_score": 0.75},
            duration_sec=10.0,
            status="success",
            finding="Test finding",
        )
        journal.add(record)

        journal_path = tmp_path / "journal.json"
        journal.save(journal_path)

        loaded = ResearchJournal.load(journal_path)
        assert loaded.total_cycles == 1
        assert "test_frontier" in loaded.frontier_scores
        assert loaded.best_per_frontier["test_frontier"]["score"] == 0.75

    def test_journal_load_restores_recent_records(self, tmp_path):
        from autonomous_research import ResearchJournal, ExperimentRecord

        journal = ResearchJournal()
        record = ExperimentRecord(
            frontier="test_frontier",
            cycle=3,
            timestamp="2026-04-05T00:00:00Z",
            hyperparams={"lr": 0.001},
            metrics={"composite_score": 0.75},
            duration_sec=10.0,
            status="success",
            finding="Recovered from disk",
        )
        journal.add(record)

        journal_path = tmp_path / "journal.json"
        journal.save(journal_path)

        loaded = ResearchJournal.load(journal_path)
        assert len(loaded.records) == 1
        assert loaded.records[0].cycle == 3
        assert loaded.records[0].finding == "Recovered from disk"

    def test_ucb1_priority_explores_unvisited(self):
        from autonomous_research import ResearchJournal, ExperimentRecord, FRONTIER_REGISTRY

        journal = ResearchJournal()
        for i in range(5):
            record = ExperimentRecord(
                frontier="certified_mcts",
                cycle=i,
                timestamp="2026-04-05T00:00:00Z",
                hyperparams={},
                metrics={"composite_score": 0.5},
                duration_sec=1.0,
                status="success",
                finding="test",
            )
            journal.add(record)

        priorities = journal.get_frontier_priority()
        visited_score = priorities.get("certified_mcts", 0)
        for name in FRONTIER_REGISTRY:
            if name != "certified_mcts":
                assert priorities.get(name, 0) >= visited_score, \
                    f"Unvisited frontier {name} should have higher priority"

    def test_breakthrough_detection(self):
        from autonomous_research import ResearchJournal, analyze_result

        journal = ResearchJournal()
        journal.best_per_frontier["test"] = {"score": 0.5}

        metrics = {"composite_score": 0.7}
        finding = analyze_result("test", metrics, journal)
        assert "BREAKTHROUGH" in finding

    def test_no_breakthrough_on_marginal_gain(self):
        from autonomous_research import ResearchJournal, analyze_result

        journal = ResearchJournal()
        journal.best_per_frontier["test"] = {"score": 0.5}

        metrics = {"composite_score": 0.52}
        finding = analyze_result("test", metrics, journal)
        assert "BREAKTHROUGH" not in finding

    def test_first_success_establishes_baseline(self):
        from autonomous_research import ResearchJournal, analyze_result

        journal = ResearchJournal()
        metrics = {"composite_score": 0.61}
        finding = analyze_result("test", metrics, journal)
        assert finding.startswith("Baseline established")

    def test_tail_failure_streak_restored_from_disk(self, tmp_path):
        from autonomous_research import ResearchJournal, ExperimentRecord

        journal = ResearchJournal()
        journal.add(ExperimentRecord(
            frontier="test_frontier",
            cycle=0,
            timestamp="2026-04-05T00:00:00Z",
            hyperparams={},
            metrics={"composite_score": 0.5},
            duration_sec=1.0,
            status="success",
            finding="baseline",
        ))
        for cycle in (1, 2):
            journal.add(ExperimentRecord(
                frontier="test_frontier",
                cycle=cycle,
                timestamp="2026-04-05T00:00:00Z",
                hyperparams={},
                metrics={},
                duration_sec=1.0,
                status="failed",
                finding="FAILED: RuntimeError: CUDA error: out of memory",
                error="CUDA error: out of memory",
            ))

        journal_path = tmp_path / "journal.json"
        journal.save(journal_path)

        loaded = ResearchJournal.load(journal_path)
        assert loaded.tail_failure_streak() == 2
        assert loaded.tail_resource_exhaustion_streak() == 2


class TestHyperparamMutation:
    """Test hyperparameter mutation logic."""

    def test_mutation_stays_in_range(self):
        from autonomous_research import mutate_hyperparams

        base = {"lr": 0.001, "hidden_dim": 128}
        ranges = {"lr": (1e-5, 1e-2), "hidden_dim": (32, 512)}

        for _ in range(100):
            result = mutate_hyperparams(base, ranges, sigma=0.5)
            assert 1e-5 <= result["lr"] <= 1e-2
            assert 32 <= result["hidden_dim"] <= 512

    def test_int_params_stay_int(self):
        from autonomous_research import mutate_hyperparams

        base = {"hidden_dim": 128}
        ranges = {"hidden_dim": (32, 512)}
        result = mutate_hyperparams(base, ranges)
        assert isinstance(result["hidden_dim"], int)

    def test_multiscale_sanitizer_enforces_attention_divisibility(self):
        from autonomous_research import _sanitize_multiscale_hyperparams

        sanitized = _sanitize_multiscale_hyperparams({
            "fast_hidden_dim": 59,
            "cross_scale_heads": 4,
        })

        assert sanitized["fast_hidden_dim"] % 4 == 0
        assert sanitized["fast_hidden_dim"] >= 32


class TestConsoleSafety:
    """Test console-safe normalization for Windows shells."""

    def test_console_safe_normalizes_legacy_host_characters(self):
        from autonomous_research import console_safe

        normalized = console_safe("slow/fast latent variables — 1→5→1 with ± jitter")
        assert normalized == "slow/fast latent variables -- 1->5->1 with +/- jitter"


class TestResourceRecovery:
    """Test failure recovery and isolated child execution."""

    def test_run_cycle_switches_to_cpu_safe_mode_after_oom_streak(self, tmp_path, monkeypatch):
        import autonomous_research as m

        journal = m.ResearchJournal()
        for cycle in range(3):
            journal.add(m.ExperimentRecord(
                frontier="consistency_distillation",
                cycle=cycle,
                timestamp="2026-04-05T00:00:00Z",
                hyperparams={"distill_steps": 1000},
                metrics={},
                duration_sec=0.1,
                status="failed",
                finding="FAILED: RuntimeError: CUDA error: out of memory",
                error="CUDA error: out of memory",
            ))

        captured = {}

        def fake_run_frontier_once_isolated(**kwargs):
            captured.update(kwargs)
            return m.ExperimentRecord(
                frontier=kwargs["frontier_name"],
                cycle=kwargs["cycle"],
                timestamp="2026-04-05T00:00:00Z",
                hyperparams=kwargs["params"],
                metrics={"composite_score": 0.55},
                duration_sec=0.2,
                status="success",
                finding="recovered on cpu",
            )

        monkeypatch.setattr(m, "mutate_hyperparams", lambda base, *args, **kwargs: dict(base))
        monkeypatch.setattr(m, "run_frontier_once_isolated", fake_run_frontier_once_isolated)

        runtime_config = m.CycleRuntimeConfig(
            journal_path=tmp_path / "journal.json",
            child_timeout_seconds=60,
            device_policy="auto",
            cpu_fallback_after_failures=2,
            consecutive_failures=journal.tail_failure_streak(),
        )

        record = m.run_cycle(
            cycle=3,
            journal=journal,
            out_root=tmp_path,
            forced_frontier="consistency_distillation",
            runtime_config=runtime_config,
        )

        assert record.status == "success"
        assert captured["device_policy"] == "cpu"
        assert captured["params"]["device"] == "cpu"
        assert captured["params"]["distill_steps"] <= 500
        assert captured["params"]["num_simulations"] <= 32

    def test_run_frontier_once_isolated_hides_cuda_for_cpu_policy(self, tmp_path, monkeypatch):
        import autonomous_research as m

        seen_env = {}

        def fake_run(command, **kwargs):
            seen_env.update(kwargs["env"])
            result_path = Path(command[command.index("--result-json") + 1])
            result_path.write_text(json.dumps({
                "frontier": "certified_mcts",
                "cycle": 4,
                "timestamp": "2026-04-05T00:00:00Z",
                "hyperparams": {"num_simulations": 32, "device_policy": "cpu"},
                "metrics": {"composite_score": 0.42},
                "duration_sec": 0.5,
                "status": "success",
                "finding": "child ok",
                "error": None,
            }), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="child ok", stderr="")

        monkeypatch.setattr(m.subprocess, "run", fake_run)

        record = m.run_frontier_once_isolated(
            cycle=4,
            frontier_name="certified_mcts",
            params={"num_simulations": 32, "device_policy": "cpu"},
            journal_path=tmp_path / "journal.json",
            out_root=tmp_path / "runs",
            timeout_seconds=60,
            device_policy="cpu",
        )

        assert record.status == "success"
        assert seen_env["DELTA_TAU_DEVICE_POLICY"] == "cpu"
        assert seen_env["CUDA_VISIBLE_DEVICES"] == "-1"


class TestAutonomousResearchMain:
    """Operational tests for the autonomous research entrypoint."""

    @staticmethod
    def _record(module, cycle: int, status: str = "success"):
        metrics = {"composite_score": 0.7} if status == "success" else {}
        return module.ExperimentRecord(
            frontier="certified_mcts",
            cycle=cycle,
            timestamp="2026-04-05T00:00:00Z",
            hyperparams={"num_simulations": 32},
            metrics=metrics,
            duration_sec=0.01,
            status=status,
            finding=f"{status} cycle {cycle}",
            error=None if status == "success" else "boom",
        )

    def test_main_respects_finite_cycle_count_and_writes_runtime_artifacts(self, tmp_path, monkeypatch):
        import autonomous_research as m

        seen_cycles = []

        def fake_run_cycle(cycle, journal, out_root, forced_frontier=None):
            seen_cycles.append(cycle)
            return self._record(m, cycle, status="success")

        monkeypatch.setattr(m, "run_cycle", fake_run_cycle)
        monkeypatch.setattr(m.time, "sleep", lambda _: None)

        out_dir = tmp_path / "runs"
        exit_code = m.main(["--cycles", "2", "--out", str(out_dir), "--cycle-delay-seconds", "0"])

        assert exit_code == 0
        assert seen_cycles == [0, 1]

        journal_data = json.loads((out_dir / "journal.json").read_text(encoding="utf-8"))
        status_data = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
        dashboard_html = (out_dir / "dashboard.html").read_text(encoding="utf-8")

        assert journal_data["total_cycles"] == 2
        assert status_data["state"] == "completed"
        assert status_data["session_completed_cycles"] == 2
        assert "Autonomous Research Orchestrator" in dashboard_html

    def test_stop_file_prevents_cycle_execution(self, tmp_path, monkeypatch):
        import autonomous_research as m

        out_dir = tmp_path / "runs"
        stop_file = out_dir / "STOP"
        out_dir.mkdir(parents=True)
        stop_file.write_text("", encoding="utf-8")

        def fake_run_cycle(*args, **kwargs):
            pytest.fail("run_cycle should not execute when the stop file already exists")

        monkeypatch.setattr(m, "run_cycle", fake_run_cycle)

        exit_code = m.main(["--cycles", "5", "--out", str(out_dir), "--stop-file", str(stop_file)])
        assert exit_code == 0

        journal_data = json.loads((out_dir / "journal.json").read_text(encoding="utf-8"))
        status_data = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))

        assert journal_data["total_cycles"] == 0
        assert status_data["state"] == "stopped"
        assert status_data["session_completed_cycles"] == 0

    def test_failure_limit_stops_infinite_session(self, tmp_path, monkeypatch):
        import autonomous_research as m

        seen_cycles = []

        def fake_run_cycle(cycle, journal, out_root, forced_frontier=None):
            seen_cycles.append(cycle)
            return self._record(m, cycle, status="failed")

        monkeypatch.setattr(m, "run_cycle", fake_run_cycle)
        monkeypatch.setattr(m.time, "sleep", lambda _: None)

        out_dir = tmp_path / "runs"
        exit_code = m.main([
            "--cycles", "0",
            "--out", str(out_dir),
            "--failure-backoff-seconds", "0",
            "--max-consecutive-failures", "2",
        ])

        assert exit_code == 0
        assert seen_cycles == [0, 1]

        status_data = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
        assert status_data["state"] == "failed_limit"
        assert status_data["consecutive_failures"] == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Frontier 1: Certified MCTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestCertifiedMCTS:
    """Smoke tests for the Certified MCTS frontier."""

    def test_certified_world_model_forward(self):
        from frontiers.certified_mcts import CertifiedWorldModel

        model = CertifiedWorldModel(hidden_dim=64, obs_dim=4)
        state = torch.randn(1, 64)
        obs = torch.randn(1, 4)
        dt = torch.ones(1, 1)

        result = model(state, obs, dt)
        # Returns 4 tensors: next_state, reward, done_logit, value_std
        assert len(result) == 4
        assert result[0].shape == (1, 64)
        assert result[1].shape == (1, 1)

    def test_lipschitz_estimation(self):
        from frontiers.certified_mcts import CertifiedWorldModel

        model = CertifiedWorldModel(hidden_dim=64, obs_dim=4)
        state = torch.randn(1, 64)
        dt = torch.ones(1, 1)

        lip = model.estimate_lipschitz(state, dt)
        assert isinstance(lip, float)
        assert lip > 0

    def test_spectral_norm_bound(self):
        from frontiers.certified_mcts import CertifiedWorldModel

        model = CertifiedWorldModel(hidden_dim=64, obs_dim=4)
        bound = model._spectral_norm_bound()
        assert isinstance(bound, float)
        assert bound > 0

    def test_certified_mcts_search(self):
        from frontiers.certified_mcts import CertifiedWorldModel, CertifiedMCTSEngine

        model = CertifiedWorldModel(hidden_dim=64, obs_dim=4)
        engine = CertifiedMCTSEngine(
            world_model=model,
            c_puct=1.5,
            lambda_return=0.8,
            gamma=0.99,
            lipschitz_threshold=5.0,
        )

        state = torch.randn(1, 64)
        obs = torch.randn(1, 4)
        result = engine.search(state, obs, num_simulations=8)

        assert "best_tau" in result
        assert "certified_fraction" in result
        assert 0 <= result["certified_fraction"] <= 1

    def test_pruning_with_low_threshold(self):
        """Very low threshold should prune most branches."""
        from frontiers.certified_mcts import CertifiedWorldModel, CertifiedMCTSEngine

        model = CertifiedWorldModel(hidden_dim=64, obs_dim=4)
        engine = CertifiedMCTSEngine(
            world_model=model,
            c_puct=1.5,
            lambda_return=0.8,
            gamma=0.99,
            lipschitz_threshold=0.001,  # very tight
        )

        state = torch.randn(1, 64)
        obs = torch.randn(1, 4)
        result = engine.search(state, obs, num_simulations=8)
        # Should have high pruned fraction
        assert result["pruned_fraction"] >= 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Frontier 2: WM-Guided Deliberation
# ═══════════════════════════════════════════════════════════════════════════════


class TestWMGuidedDeliberation:
    """Smoke tests for the WM-Guided Deliberation frontier."""

    def test_mini_rssm_observe(self):
        from frontiers.world_model_guided_deliberation import MiniRSSM, RSSMConfig

        cfg = RSSMConfig(obs_dim=4, action_dim=2, hidden_dim=64, stoch_dim=16, num_classes=16)
        rssm = MiniRSSM(cfg)

        state = rssm.initial_state(batch_size=1)
        obs = torch.randn(1, 4)
        action = torch.zeros(1, 2)

        new_state, prior_logits, post_logits = rssm.observe(state, action, obs)
        assert new_state["h"].shape == (1, 64)
        assert new_state["z"].shape == (1, 16, 16)

    def test_uncertainty_computation(self):
        from frontiers.world_model_guided_deliberation import MiniRSSM, RSSMConfig

        cfg = RSSMConfig(obs_dim=4, action_dim=2, hidden_dim=64, stoch_dim=16, num_classes=16)
        rssm = MiniRSSM(cfg)

        state = rssm.initial_state(batch_size=1)
        uncertainty = rssm.compute_uncertainty(state)
        assert isinstance(uncertainty, torch.Tensor)
        assert uncertainty.shape == (1,)
        assert (uncertainty >= 0).all()

    def test_uncertainty_guided_act_forward(self):
        from frontiers.world_model_guided_deliberation import UncertaintyGuidedACT, ACTConfig

        cfg = ACTConfig(obs_dim=4, action_dim=2, hidden_dim=64)
        act = UncertaintyGuidedACT(cfg)
        obs = torch.randn(1, 4)
        uncertainty = torch.tensor([0.5])

        result = act(obs, uncertainty)
        assert "action_logits" in result
        assert result["action_logits"].shape == (1, 2)
        assert "n_steps" in result
        assert torch.allclose(
            result["halt_probs"].sum(dim=-1),
            torch.ones(1),
            atol=1e-4,
        )
        assert (result["n_steps"] >= 1).all()
        assert (result["n_steps"] <= cfg.max_thinking_steps).all()

    def test_uncertainty_guided_act_diagnostics_are_normalized(self):
        from frontiers.world_model_guided_deliberation import (
            ACTConfig,
            UncertaintyGuidedACT,
            compute_pondering_diagnostics,
        )

        cfg = ACTConfig(
            obs_dim=4,
            action_dim=2,
            hidden_dim=32,
            base_thinking_steps=2,
            max_thinking_steps=5,
        )
        act = UncertaintyGuidedACT(cfg)
        obs = torch.randn(2, 4)
        uncertainty = torch.tensor([0.0, 2.0])

        result = act(obs, uncertainty)
        diag = compute_pondering_diagnostics(
            halt_probs=result["halt_probs"],
            uncertainties=uncertainty,
            n_steps=result["n_steps"],
            max_steps=cfg.max_thinking_steps,
        )

        assert diag["weight_sum_error"] < 1e-4
        assert diag["mean_ponder_depth"] >= 1.0
        assert diag["mean_active_steps"] >= diag["mean_ponder_depth"]


# ═══════════════════════════════════════════════════════════════════════════════
# Frontier 3: Multi-Scale Temporal World Model
# ═══════════════════════════════════════════════════════════════════════════════


class TestMultiScaleWM:
    """Smoke tests for the Multi-Scale Temporal World Model frontier."""

    def test_multiscale_model_forward(self):
        from frontiers.multiscale_temporal_world_model import MultiScaleTemporalWorldModel

        model = MultiScaleTemporalWorldModel(
            obs_dim=4, action_dim=2,
            fast_hidden_dim=32, slow_hidden_dim=64,
            fast_stoch_dim=8, slow_stoch_dim=16,
            num_classes=8, cross_scale_heads=2,
            slow_tick_every=4,
        )

        obs_seq = torch.randn(2, 8, 4)
        act_seq = torch.randn(2, 8, 2)

        outputs = model(obs_seq, act_seq)
        assert "loss" in outputs
        assert "fast_recon" in outputs
        assert "cross_consistency" in outputs

    def test_multiscale_loss_is_finite(self):
        from frontiers.multiscale_temporal_world_model import MultiScaleTemporalWorldModel

        model = MultiScaleTemporalWorldModel(
            obs_dim=4, action_dim=2,
            fast_hidden_dim=32, slow_hidden_dim=64,
            fast_stoch_dim=8, slow_stoch_dim=16,
            num_classes=8, cross_scale_heads=2,
            slow_tick_every=4,
        )

        obs_seq = torch.randn(2, 12, 4)
        act_seq = torch.randn(2, 12, 2)

        outputs = model(obs_seq, act_seq)
        assert torch.isfinite(outputs["loss"])


# ═══════════════════════════════════════════════════════════════════════════════
# Frontier 4: Temporal Consistency Distillation
# ═══════════════════════════════════════════════════════════════════════════════


class TestConsistencyDistillation:
    """Smoke tests for the Temporal Consistency Distillation frontier."""

    def test_student_network_forward(self):
        from frontiers.temporal_consistency_distillation import StudentNetwork

        student = StudentNetwork(obs_dim=4, action_dim=2, hidden_dim=32)
        obs = torch.randn(4, 4)

        logits, values, dts = student(obs)
        assert logits.shape == (4, 2)
        assert values.shape == (4, 1)
        assert dts.shape == (4, 1)
        assert (dts > 0).all()

    def test_lipschitz_consistency_loss(self):
        from frontiers.temporal_consistency_distillation import StudentNetwork, LipschitzConsistencyLoss

        student = StudentNetwork(obs_dim=4, action_dim=2, hidden_dim=32)
        loss_fn = LipschitzConsistencyLoss(margin=1.2)

        teacher_lip = torch.tensor([3.0])
        student_dt = torch.randn(4, 1)
        target_dt = torch.randn(4, 1)

        loss, info = loss_fn(student, teacher_lip, student_dt, target_dt)
        assert isinstance(loss.item(), float)
        assert loss.item() >= 0
        assert "student_lip" in info
        assert "lip_violation" in info

    def test_generate_targets_estimates_root_value_from_children(self, monkeypatch):
        from frontiers.temporal_consistency_distillation import TeacherMCTS, MCTSNode

        teacher = TeacherMCTS(obs_dim=4, action_dim=2, hidden_dim=16)
        hidden = teacher.world_model.initial_hidden(1, teacher.device)

        def fake_search(_obs, _n_sims, delta_tau):
            root = MCTSNode(hidden=hidden)
            root.children[0] = MCTSNode(hidden=hidden, value_sum=4.0, visit_count=2, prior=0.5)
            second_value = 3.0 if delta_tau <= teacher.delta_tau_nominal else 4.0
            root.children[1] = MCTSNode(hidden=hidden, value_sum=second_value, visit_count=1, prior=0.5)
            return root

        monkeypatch.setattr(teacher, "_search", fake_search)

        probs, value_estimate, lip = teacher.generate_targets(np.zeros(4, dtype=np.float32), n_sims=4)
        assert np.allclose(probs, np.array([2.0 / 3.0, 1.0 / 3.0], dtype=np.float32))
        assert math.isclose(value_estimate, 7.0 / 3.0, rel_tol=1e-6)
        assert math.isclose(lip, (1.0 / 3.0) / teacher.delta_tau_perturb, rel_tol=1e-6)

    def test_distillation_trainer_uses_deterministic_reset_seeds(self, monkeypatch):
        from frontiers import temporal_consistency_distillation as tcd

        envs = []

        class FakeEnv:
            def __init__(self):
                self.reset_seeds = []

            def reset(self, *, seed=None, options=None):
                self.reset_seeds.append(seed)
                return np.zeros(4, dtype=np.float32), {}

            def step(self, action):
                return np.zeros(4, dtype=np.float32), 1.0, True, False, {}

            def close(self):
                return None

        class DummyTeacher:
            def generate_targets(self, obs, n_sims=64):
                return np.array([1.0, 0.0], dtype=np.float32), 0.5, 0.2

            def act(self, obs, n_sims=64):
                return 0

        def fake_make(_env_id):
            env = FakeEnv()
            envs.append(env)
            return env

        monkeypatch.setattr(tcd.gym, "make", fake_make)

        student = tcd.StudentNetwork(obs_dim=4, action_dim=2, hidden_dim=16)
        trainer = tcd.DistillationTrainer(
            DummyTeacher(),
            student,
            env_id="CartPole-v1",
            seed=123,
            device=torch.device("cpu"),
        )

        trainer.generate_teacher_data(n_samples=2, n_sims=1, max_episode_steps=1)
        trainer.evaluate(n_episodes=2, max_steps=1, n_sims=1)

        assert envs[0].reset_seeds == [123, 124]
        assert envs[1].reset_seeds == [10123, 10123, 10123, 10124, 10124, 10124]

    def test_consistency_distillation_runner_preserves_seed(self, monkeypatch, tmp_path):
        import autonomous_research as ar
        from frontiers import temporal_consistency_distillation as tcd

        captured = {}

        class FakeExperiment:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run(self, out_dir):
                return {"composite_score": 0.5, "out_dir": str(out_dir)}

        monkeypatch.setattr(tcd, "ConsistencyDistillationExperiment", FakeExperiment)

        result = ar._run_consistency_distillation(
            {
                "obs_dim": 4,
                "action_dim": 2,
                "teacher_hidden_dim": 64,
                "student_hidden_dim": 32,
                "num_simulations": 8,
                "distill_steps": 10,
                "batch_size": 4,
                "n_eval_episodes": 2,
                "max_steps": 5,
                "seed": 123456,
                "lip_margin": 1.4,
                "delta_tau_nominal": 0.07,
                "use_spectral_norm": True,
            },
            tmp_path,
        )

        assert result["composite_score"] == 0.5
        assert captured["seed"] == 123456
        assert captured["lip_margin"] == 1.4
        assert captured["delta_tau_nominal"] == 0.07
        assert captured["use_spectral_norm"] is True
