"""Tests for stress failure analysis and ablation planning helpers."""

from __future__ import annotations

import json
from pathlib import Path

from deltatau_audit.stress_lab import (
    analyze_stress_summary,
    build_ablation_manifest,
    train_sb3_ablation_models,
    write_ablation_plan_artifacts,
    write_stress_analysis_artifacts,
    write_training_summary,
)


def _sample_summary_payload() -> dict:
    return {
        "summary": {
            "deployment_score": 0.86,
            "deployment_rating": "MILD",
            "stress_score": 0.45,
            "stress_rating": "FAIL",
            "stress_threshold": 0.50,
            "quadrant": "deployment_fragile",
        },
        "robustness": {
            "stress": {"worst_case": {"scenario": "speed_8x"}},
            "per_scenario_scores": {
                "speed_1x": {"return_ratio": 0.95, "ci_lower": 0.90, "rmse_ratio": 1.00},
                "speed_2x": {"return_ratio": 0.92, "ci_lower": 0.86, "rmse_ratio": 1.05},
                "speed_3x": {"return_ratio": 0.89, "ci_lower": 0.83, "rmse_ratio": 1.10},
                "speed_5x": {"return_ratio": 0.52, "ci_lower": 0.46, "rmse_ratio": 1.22},
                "speed_8x": {"return_ratio": 0.45, "ci_lower": 0.39, "rmse_ratio": 1.28},
            },
        },
        "diagnosis": {
            "summary_line": "1 FAIL scenario: speed_8x",
            "primary_pattern": "Extreme Frequency Fragility",
        },
    }


def test_analyze_stress_summary_detects_threshold_collapse(tmp_path: Path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(_sample_summary_payload()), encoding="utf-8")

    analysis = analyze_stress_summary(
        summary_path,
        stress_threshold=0.50,
        include_intervention3=False,
    )

    assert analysis["worst_scenario"]["scenario"] == "speed_8x"
    assert analysis["stress_gate_pass"] is False
    assert analysis["pattern"]["id"] == "threshold_collapse"
    assert analysis["mechanism"]["code"] == "A"
    assert analysis["ablation_variants"] == [
        "baseline",
        "intervention1_curriculum",
        "intervention2_time_feature",
        "intervention1_plus_2",
    ]


def test_build_ablation_manifest_includes_variant_and_seed_matrix():
    manifest = build_ablation_manifest(
        env="CartPole-v1",
        algo="ppo",
        model_template="checkpoints/{variant}/seed_{seed}/model.zip",
        seeds=[0, 1, 2],
        include_intervention3=True,
        output_dir="ablation_runs",
    )
    jobs = manifest["jobs"]
    assert len(jobs) >= 5
    assert all(job["command"] == "audit-sb3" for job in jobs)
    assert all("seed" in job["matrix"] for job in jobs)
    assert all("variant" in job["matrix"] for job in jobs)
    names = {job["name"] for job in jobs}
    assert "stress_ablation_intervention3_memory" in names
    tf_job = next(j for j in jobs if j["name"] == "stress_ablation_intervention2_time_feature")
    mem_job = next(j for j in jobs if j["name"] == "stress_ablation_intervention3_memory")
    assert tf_job["args"]["env_wrap_time_feature"] is True
    assert mem_job["args"]["env_wrap_frame_stack"] == 4
    assert tf_job["args"]["protocol"] == "research"
    assert tf_job["args"]["ci_gate_mode"] == "worst_ci_lower"


def test_write_stress_and_ablation_artifacts(tmp_path: Path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(_sample_summary_payload()), encoding="utf-8")

    analysis_artifacts = write_stress_analysis_artifacts(
        summary_path,
        out_dir=tmp_path / "stress",
        include_intervention3=True,
    )
    assert Path(analysis_artifacts["analysis_json"]).exists()
    assert Path(analysis_artifacts["analysis_md"]).exists()

    analysis = json.loads(Path(analysis_artifacts["analysis_json"]).read_text(encoding="utf-8"))
    manifest = build_ablation_manifest(
        env="CartPole-v1",
        algo="ppo",
        model_template="checkpoints/{variant}/seed_{seed}/model.zip",
        include_intervention3=True,
    )
    plan_artifacts = write_ablation_plan_artifacts(
        analysis=analysis,
        manifest=manifest,
        out_dir=tmp_path / "plan",
    )
    assert Path(plan_artifacts["ablation_manifest"]).exists()
    assert Path(plan_artifacts["ablation_plan_md"]).exists()


def test_train_sb3_ablation_models_with_fake_algo(tmp_path: Path, monkeypatch):
    class _FakeAlgo:
        def __init__(self, policy, env, seed, device, verbose):  # noqa: ARG002
            self._env = env
            self._seed = seed

        def learn(self, total_timesteps):  # noqa: ARG002
            return self

        def save(self, stem):
            Path(stem + ".zip").write_text(
                json.dumps({"seed": self._seed}), encoding="utf-8"
            )

    monkeypatch.setattr("deltatau_audit.stress_lab._algo_cls", lambda _a: _FakeAlgo)

    summary = train_sb3_ablation_models(
        env="CartPole-v1",
        algo="ppo",
        out_root=tmp_path / "ckpt",
        seeds=[0, 1],
        variants=["baseline", "intervention2_time_feature"],
        timesteps=10,
        verbose=0,
    )
    assert summary["status"] == "passed"
    assert summary["counts"]["trained"] == 4

    model_path = tmp_path / "ckpt" / "baseline" / "seed_0" / "model.zip"
    assert model_path.exists()

    artifacts = write_training_summary(summary, out_dir=tmp_path / "stress_out")
    assert Path(artifacts["training_json"]).exists()
    assert Path(artifacts["training_md"]).exists()
