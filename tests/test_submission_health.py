"""Tests for shared submission health helpers."""

from __future__ import annotations

import json
from pathlib import Path

import submission_health as sh


def test_expand_manifest_jobs_matrix():
    manifest = {
        "jobs": [
            {
                "name": "x",
                "matrix": {"seed": [0, 1]},
                "args": {"out": "bench_runs/foo/seed_{seed}", "seed": "{seed}"},
            }
        ]
    }

    expanded = sh.expand_manifest_jobs(manifest)

    assert len(expanded) == 2
    outs = sorted(row["args"]["out"] for row in expanded)
    assert outs == ["bench_runs/foo/seed_0", "bench_runs/foo/seed_1"]


def test_summary_targets_from_manifest(tmp_path: Path):
    manifest_path = tmp_path / "bench" / "mini.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "jobs:\n"
        "  - name: mini\n"
        "    matrix:\n"
        "      seed: [0, 1]\n"
        "    args:\n"
        "      out: bench_runs/mini/seed_{seed}\n",
        encoding="utf-8",
    )

    targets = sh.summary_targets_from_manifest(manifest_path, tmp_path / "bench_runs" / "mini")

    assert len(targets) == 2
    target_texts = sorted(str(p).replace("\\", "/") for p in targets)
    assert target_texts[0].endswith("/bench_runs/mini/seed_0/summary.json")
    assert target_texts[1].endswith("/bench_runs/mini/seed_1/summary.json")


def test_bench_counts_reads_status(tmp_path: Path):
    out_root = tmp_path / "bench_runs" / "mini"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "bench_summary.json").write_text(
        json.dumps({"status": "failed", "counts": {"passed": 1, "failed": 2, "skipped": 3}}),
        encoding="utf-8",
    )

    counts, updated, status = sh.bench_counts(out_root)

    assert counts == {"passed": 1, "failed": 2, "skipped": 3}
    assert updated is not None
    assert status == "failed"


def test_check_bench_execution_ready(tmp_path: Path):
    manifest = tmp_path / "bench" / "high_rigor_10seed_manifest.yaml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "jobs:\n"
        "  - name: mini\n"
        "    args:\n"
        "      out: bench_runs/mini/seed_0\n",
        encoding="utf-8",
    )
    out_root = tmp_path / "bench_runs" / "mini"
    (out_root / "seed_0").mkdir(parents=True, exist_ok=True)
    (out_root / "seed_0" / "summary.json").write_text("{}", encoding="utf-8")
    (out_root / "bench_summary.json").write_text(
        json.dumps({"status": "passed", "counts": {"passed": 1, "failed": 0, "skipped": 0}}),
        encoding="utf-8",
    )

    result = sh.check_bench_execution(manifest, out_root)

    assert result["ready"] is True
    assert result["expected_jobs"] == 1
    assert result["completed_jobs"] == 1
    assert result["bench_status"] == "passed"


def test_check_bench_execution_not_ready_when_failed(tmp_path: Path):
    manifest = tmp_path / "bench" / "mini.yaml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "jobs:\n"
        "  - name: mini\n"
        "    args:\n"
        "      out: bench_runs/mini/seed_0\n",
        encoding="utf-8",
    )
    out_root = tmp_path / "bench_runs" / "mini"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "bench_summary.json").write_text(
        json.dumps({"status": "failed", "counts": {"passed": 0, "failed": 1, "skipped": 0}}),
        encoding="utf-8",
    )

    result = sh.check_bench_execution(manifest, out_root)

    assert result["ready"] is False
    assert result["bench_status"] == "failed"
    assert result["counts"]["failed"] == 1
    assert any("failed jobs" in reason for reason in result["reasons"])


def test_bench_failure_breakdown_classifies_ci_gate_and_runtime(tmp_path: Path):
    out_root = tmp_path / "bench_runs" / "mini"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "bench_summary.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "counts": {"passed": 0, "failed": 2, "skipped": 0},
                "jobs": [
                    {
                        "id": "ci_gate_case",
                        "status": "failed",
                        "returncode": 2,
                        "summary_path": str(out_root / "ci_gate_case" / "summary.json"),
                    },
                    {
                        "id": "runtime_case",
                        "status": "failed",
                        "returncode": 1,
                        "summary_path": None,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    breakdown = sh.bench_failure_breakdown(out_root)

    assert breakdown["failed_total"] == 2
    assert breakdown["ci_gate_failures"] == 1
    assert breakdown["runtime_failures"] == 1
    assert len(breakdown["ci_gate_summary_paths"]) == 1
    ci_path = str(breakdown["ci_gate_summary_paths"][0]).replace("\\", "/")
    assert ci_path.endswith("ci_gate_case/summary.json")
    assert "ci_gate_case" in breakdown["failed_job_ids"]
    assert "runtime_case" in breakdown["failed_job_ids"]


def test_bench_failure_breakdown_counts_skipped_ci_gate_failures(tmp_path: Path):
    out_root = tmp_path / "bench_runs" / "mini"
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "seed_0" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("{}", encoding="utf-8")
    (out_root / "bench_summary.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "counts": {"passed": 0, "failed": 0, "skipped": 1},
                "jobs": [
                    {
                        "id": "ci_gate_case",
                        "status": "skipped",
                        "returncode": 0,
                        "summary_path": str(summary_path),
                        "result": {"stress_ci_gate_pass": False},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    breakdown = sh.bench_failure_breakdown(out_root)

    assert breakdown["failed_total"] == 1
    assert breakdown["ci_gate_failures"] == 1
    assert breakdown["runtime_failures"] == 0
    assert breakdown["failed_job_ids"] == ["ci_gate_case"]
    assert breakdown["ci_gate_summary_paths"] == [str(summary_path)]


def test_check_bench_execution_reports_breakdown_reasons(tmp_path: Path):
    manifest = tmp_path / "bench" / "mini.yaml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "jobs:\n"
        "  - name: mini\n"
        "    args:\n"
        "      out: bench_runs/mini/seed_0\n",
        encoding="utf-8",
    )
    out_root = tmp_path / "bench_runs" / "mini"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "bench_summary.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "counts": {"passed": 0, "failed": 2, "skipped": 0},
                "jobs": [
                    {
                        "id": "ci_gate_case",
                        "status": "failed",
                        "returncode": 2,
                        "summary_path": str(out_root / "ci_gate_case" / "summary.json"),
                    },
                    {
                        "id": "runtime_case",
                        "status": "failed",
                        "returncode": 1,
                        "summary_path": None,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = sh.check_bench_execution(manifest, out_root)

    assert result["failure_breakdown"]["ci_gate_failures"] == 1
    assert result["failure_breakdown"]["runtime_failures"] == 1
    assert any("runtime failures" in reason for reason in result["reasons"])
    assert any("quality-gate failures" in reason for reason in result["reasons"])


def test_check_bench_execution_not_ready_for_skipped_ci_gate_failure(tmp_path: Path):
    manifest = tmp_path / "bench" / "mini.yaml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "jobs:\n"
        "  - name: mini\n"
        "    args:\n"
        "      out: bench_runs/mini/seed_0\n",
        encoding="utf-8",
    )
    out_root = tmp_path / "bench_runs" / "mini"
    summary_path = out_root / "seed_0" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("{}", encoding="utf-8")
    (out_root / "bench_summary.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "counts": {"passed": 0, "failed": 0, "skipped": 1},
                "jobs": [
                    {
                        "id": "ci_gate_case",
                        "status": "skipped",
                        "returncode": 0,
                        "summary_path": str(summary_path),
                        "result": {"stress_ci_gate_pass": False},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = sh.check_bench_execution(manifest, out_root)

    assert result["ready"] is False
    assert result["bench_status"] == "passed"
    assert result["failure_breakdown"]["ci_gate_failures"] == 1
    assert any("quality-gate failures" in reason for reason in result["reasons"])


def test_check_bench_execution_exposes_quality_repair_plan(tmp_path: Path):
    manifest = tmp_path / "bench" / "high_rigor_10seed_manifest.yaml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "jobs:\n"
        "  - name: mini\n"
        "    args:\n"
        "      out: bench_runs/mini/seed_0\n",
        encoding="utf-8",
    )
    out_root = tmp_path / "bench_runs" / "mini"
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "seed_0" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("{}", encoding="utf-8")
    (out_root / "bench_summary.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "counts": {"passed": 0, "failed": 1, "skipped": 0},
                "jobs": [
                    {
                        "id": "cartpole_intervention1_plus_2_seed-6",
                        "status": "failed",
                        "returncode": 2,
                        "summary_path": str(summary_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = sh.check_bench_execution(manifest, out_root, protocol="paper", job_name="cartpole_high_rigor_bench")

    plan = result["quality_repair_plan"]
    assert isinstance(plan, dict)
    assert plan["job_name"] == "cartpole_high_rigor_bench"
    assert plan["protocol"] == "paper"
    assert plan["rerun_scope"] == "focused"
    assert len(plan["retrain_commands"]) == 1
    assert "--variants intervention1_plus_2 --seeds 6 --timesteps 45000 --force" in plan["retrain_commands"][0]
    assert plan["cleanup_summary_paths"] == [str(summary_path)]
    assert "python scripts/build_failed_job_manifest.py" in result["repair_command_chain"]
    assert "--out-manifest _status_demo/repair_manifests/cartpole_high_rigor_bench.yaml" in result["repair_command_chain"]
    assert "--output-dir _status_demo/repair_bench_runs/cartpole_high_rigor_bench" in result["repair_command_chain"]
    assert "python scripts/merge_bench_summaries.py" in result["repair_command_chain"]


def test_build_failed_job_subset_manifest_preserves_original_job_ids(tmp_path: Path):
    manifest = tmp_path / "bench" / "mini.yaml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "name: mini\n"
        "output_dir: bench_runs/mini\n"
        "jobs:\n"
        "  - name: cartpole_intervention1_plus_2\n"
        "    command: audit-sb3\n"
        "    matrix:\n"
        "      seed: [0, 1, 6]\n"
        "    args:\n"
        "      model: checkpoints_cartpole_ppo/intervention1_plus_2/seed_{seed}/model.zip\n"
        "      seed: \"{seed}\"\n"
        "      out: bench_runs/mini/intervention1_plus_2/seed_{seed}\n",
        encoding="utf-8",
    )
    out_root = tmp_path / "bench_runs" / "mini"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "bench_summary.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "counts": {"passed": 0, "failed": 2, "skipped": 0},
                "jobs": [
                    {"id": "cartpole_intervention1_plus_2_seed-1", "status": "failed", "summary_path": "a.json"},
                    {"id": "cartpole_intervention1_plus_2_seed-6", "status": "failed", "summary_path": "b.json"},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = sh.build_failed_job_subset_manifest(manifest, out_root)

    assert result["selected_count"] == 2
    subset = result["manifest"]
    assert subset["name"] == "mini_failed_subset"
    jobs = subset["jobs"]
    assert len(jobs) == 2
    assert jobs[0]["id"] == "cartpole_intervention1_plus_2_seed-1"
    assert jobs[1]["id"] == "cartpole_intervention1_plus_2_seed-6"
    assert "matrix" not in jobs[0]
    assert jobs[0]["args"]["seed"] == "1"
    assert jobs[1]["args"]["seed"] == "6"


def test_ci_gate_failed_summary_paths_deduplicates_and_filters():
    paths = sh.ci_gate_failed_summary_paths(
        {
            "ci_gate_summary_paths": [
                "C:/runs/a/summary.json",
                "C:/runs/a/summary.json",
                "  ",
                None,
                "C:/runs/b/summary.json",
            ]
        }
    )

    assert paths == ["C:/runs/a/summary.json", "C:/runs/b/summary.json"]


def test_cartpole_failed_variant_seeds_parses_breakdown():
    parsed = sh.cartpole_failed_variant_seeds(
        {
            "failed_job_ids": [
                "cartpole_intervention1_plus_2_seed-1",
                "cartpole_intervention1_plus_2_seed-7",
                "cartpole_intervention2_time_feature_seed-3",
                "not_cartpole_format",
            ]
        }
    )

    assert parsed == {
        "intervention1_plus_2": [1, 7],
        "intervention2_time_feature": [3],
    }


def test_summary_cleanup_commands_build_python_unlinks():
    commands = sh.summary_cleanup_commands(
        [
            "C:/runs/a/summary.json",
            "C:/runs/a/summary.json",
            "C:/runs/b/summary.json",
        ]
    )

    assert len(commands) == 2
    assert "Path(r'C:/runs/a/summary.json').unlink" in commands[0]
    assert "Path(r'C:/runs/b/summary.json').unlink" in commands[1]


def test_cartpole_retrain_commands_support_custom_speed_knobs():
    commands = sh.cartpole_retrain_commands(
        {"intervention1_plus_2": [1, 6, 7]},
        timesteps=90000,
        base_speed=4,
        jitter=3,
        phase_period=250,
    )

    assert len(commands) == 1
    assert "--timesteps 90000" in commands[0]
    assert "--base-speed 4" in commands[0]
    assert "--jitter 3" in commands[0]
    assert "--phase-period 250" in commands[0]


def test_build_quality_repair_plan_for_cartpole_quality_failures():
    plan = sh.build_quality_repair_plan(
        job_name="cartpole_high_rigor_bench",
        manifest="bench/high_rigor_10seed_manifest.yaml",
        output_root="bench_runs/cartpole_high_rigor_10seed",
        protocol="paper",
        failure_breakdown={
            "ci_gate_failures": 3,
            "failed_job_ids": [
                "cartpole_intervention1_plus_2_seed-1",
                "cartpole_intervention1_plus_2_seed-6",
                "cartpole_intervention2_time_feature_seed-4",
            ],
            "ci_gate_summary_paths": [
                "C:/runs/cartpole/intervention1_plus_2/seed_1/summary.json",
                "C:/runs/cartpole/intervention2_time_feature/seed_4/summary.json",
            ],
        },
    )

    assert plan is not None
    assert plan.use_no_resume is False
    assert plan.rerun_scope == "focused"
    assert len(plan.retrain_commands) == 2
    assert "--variants intervention1_plus_2 --seeds 1 6" in plan.retrain_commands[0] or "--variants intervention1_plus_2 --seeds 1 6" in plan.retrain_commands[1]
    assert len(plan.cleanup_summary_paths) == 2
    assert "python scripts/build_failed_job_manifest.py" in plan.rerun_command
    assert "--output-root bench_runs/cartpole_high_rigor_10seed" in plan.rerun_command
    assert "--output-dir _status_demo/repair_bench_runs/cartpole_high_rigor_bench" in plan.rerun_command
    assert "--job-id cartpole_intervention1_plus_2_seed-1" in plan.rerun_command
    assert "--no-resume" not in plan.rerun_command
    assert "python scripts/merge_bench_summaries.py" in plan.refresh_summary_command
    normalized_refresh = plan.refresh_summary_command.replace("\\", "/")
    assert "_status_demo/repair_bench_runs/cartpole_high_rigor_bench/bench_summary.json" in normalized_refresh
    assert any("merge focused rerun into full bench summary" in reason for reason in plan.reasons)
    assert any("failed cells" in reason for reason in plan.reasons)


def test_build_quality_repair_plan_without_summary_paths_forces_no_resume():
    plan = sh.build_quality_repair_plan(
        job_name="mini",
        manifest="bench/mini.yaml",
        protocol="research",
        failure_breakdown={
            "ci_gate_failures": 2,
            "failed_job_ids": ["mini_seed-1", "mini_seed-4"],
        },
    )

    assert plan is not None
    assert plan.use_no_resume is True
    assert plan.rerun_scope == "full"
    assert plan.cleanup_summary_paths == ()
    assert plan.retrain_commands == ()
    assert plan.rerun_command.endswith("--protocol research --no-resume")
    assert plan.refresh_summary_command == ""
    assert any("fallback to --no-resume" in reason for reason in plan.reasons)


def test_build_quality_repair_plan_for_widespread_cartpole_failure_prefers_diagnosis():
    plan = sh.build_quality_repair_plan(
        job_name="cartpole_high_rigor_bench",
        manifest="bench/high_rigor_10seed_manifest.yaml",
        output_root="bench_runs/cartpole_high_rigor_10seed_fresh_20260317",
        protocol="paper",
        expected_jobs=50,
        failure_breakdown={
            "ci_gate_failures": 50,
            "failed_job_ids": [
                "cartpole_baseline_seed-0",
                "cartpole_baseline_seed-1",
                "cartpole_intervention1_curriculum_seed-0",
                "cartpole_intervention1_plus_2_seed-0",
                "cartpole_intervention2_time_feature_seed-0",
                "cartpole_intervention3_memory_seed-0",
            ],
        },
    )

    assert plan is not None
    assert plan.strategy == "diagnose_protocol"
    assert plan.rerun_scope == "diagnose"
    assert plan.retrain_commands == ()
    assert plan.cleanup_summary_paths == ()
    assert plan.rerun_command == ""
    assert len(plan.diagnostic_commands) == 1
    assert "python scripts/analyze_bench_failures.py" in plan.diagnostic_commands[0]
    assert any("diagnose protocol/claim mismatch" in reason for reason in plan.reasons)


def test_repair_plan_commands_include_cleanup_and_post_commands():
    plan = sh.build_quality_repair_plan(
        job_name="mini",
        manifest="bench/mini.yaml",
        protocol="research",
        failure_breakdown={
            "ci_gate_failures": 1,
            "ci_gate_summary_paths": ["C:/runs/mini/seed_1/summary.json"],
        },
        include_retrain=False,
    )

    assert plan is not None
    commands = sh.repair_plan_commands(
        plan,
        post_commands=["python scripts/prepare_submission.py --check-only --strict-check"],
    )

    assert len(commands) == 3
    assert commands[0].startswith('python -c "from pathlib import Path;')
    assert commands[1] == "python -m deltatau_audit bench run --manifest bench/mini.yaml --protocol research"
    assert commands[2].endswith("--strict-check")


def test_bench_quality_analysis_summarizes_variants_and_signals(tmp_path: Path):
    out_root = tmp_path / "bench_runs" / "mini"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "bench_summary.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "counts": {"passed": 0, "failed": 4, "skipped": 0},
                "protocol": {"forced": "paper", "allow_override": False},
                "jobs": [
                    {
                        "id": "job_a",
                        "name": "baseline",
                        "status": "failed",
                        "result": {
                            "deployment_score": 0.20,
                            "deployment_rating": "FAIL",
                            "stress_score": 0.10,
                            "stress_rating": "FAIL",
                            "stress_worst_ci_lower": 0.09,
                            "stress_ci_gate_pass": False,
                            "diagnosis_pattern": "Adversarial Jitter Sensitivity",
                            "stress_worst_scenario": "adversarial_jitter",
                        },
                    },
                    {
                        "id": "job_b",
                        "name": "baseline",
                        "status": "failed",
                        "result": {
                            "deployment_score": 0.24,
                            "deployment_rating": "FAIL",
                            "stress_score": 0.11,
                            "stress_rating": "FAIL",
                            "stress_worst_ci_lower": 0.10,
                            "stress_ci_gate_pass": False,
                            "diagnosis_pattern": "Adversarial Jitter Sensitivity",
                            "stress_worst_scenario": "adversarial_jitter",
                        },
                    },
                    {
                        "id": "job_c",
                        "name": "memory",
                        "status": "failed",
                        "result": {
                            "deployment_score": 0.31,
                            "deployment_rating": "FAIL",
                            "stress_score": 0.14,
                            "stress_rating": "FAIL",
                            "stress_worst_ci_lower": 0.12,
                            "stress_ci_gate_pass": False,
                            "diagnosis_pattern": "Adversarial Jitter Sensitivity",
                            "stress_worst_scenario": "adversarial_jitter",
                        },
                    },
                    {
                        "id": "job_d",
                        "name": "memory",
                        "status": "failed",
                        "result": {
                            "deployment_score": 0.28,
                            "deployment_rating": "FAIL",
                            "stress_score": 0.15,
                            "stress_rating": "FAIL",
                            "stress_worst_ci_lower": 0.13,
                            "stress_ci_gate_pass": False,
                            "diagnosis_pattern": "Adversarial Jitter Sensitivity",
                            "stress_worst_scenario": "adversarial_jitter",
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    analysis = sh.bench_quality_analysis(out_root)

    assert analysis["exists"] is True
    assert analysis["protocol"] == "paper"
    assert analysis["ci_gate_failures"] == 4
    assert len(analysis["variants"]) == 2
    assert any("all 4 jobs failed the CI quality gate" in signal for signal in analysis["signals"])
    assert analysis["top_diagnosis_patterns"][0]["label"] == "Adversarial Jitter Sensitivity"
    baseline = next(item for item in analysis["variants"] if item["variant"] == "baseline")
    assert baseline["deployment_score"]["count"] == 2
    assert baseline["ci_gate_failures"] == 2


def test_compare_bench_quality_summarizes_metric_deltas(tmp_path: Path):
    base_root = tmp_path / "bench_runs" / "base"
    other_root = tmp_path / "bench_runs" / "other"
    base_root.mkdir(parents=True, exist_ok=True)
    other_root.mkdir(parents=True, exist_ok=True)
    base_payload = {
        "status": "failed",
        "counts": {"passed": 0, "failed": 2, "skipped": 0},
        "protocol": {"forced": "research"},
        "jobs": [
            {
                "id": "job_a",
                "name": "baseline",
                "status": "failed",
                "result": {
                    "deployment_score": 0.20,
                    "stress_score": 0.10,
                    "stress_worst_ci_lower": 0.09,
                    "stress_ci_gate_pass": False,
                    "diagnosis_pattern": "Pattern A",
                    "stress_worst_scenario": "speed_5x",
                },
            },
            {
                "id": "job_b",
                "name": "baseline",
                "status": "failed",
                "result": {
                    "deployment_score": 0.25,
                    "stress_score": 0.12,
                    "stress_worst_ci_lower": 0.11,
                    "stress_ci_gate_pass": False,
                    "diagnosis_pattern": "Pattern A",
                    "stress_worst_scenario": "speed_5x",
                },
            },
        ],
    }
    other_payload = {
        "status": "failed",
        "counts": {"passed": 0, "failed": 2, "skipped": 0},
        "protocol": {"forced": "paper"},
        "jobs": [
            {
                "id": "job_a",
                "name": "baseline",
                "status": "failed",
                "result": {
                    "deployment_score": 0.18,
                    "stress_score": 0.07,
                    "stress_worst_ci_lower": 0.06,
                    "stress_ci_gate_pass": False,
                    "diagnosis_pattern": "Pattern B",
                    "stress_worst_scenario": "adversarial_jitter",
                },
            },
            {
                "id": "job_b",
                "name": "baseline",
                "status": "failed",
                "result": {
                    "deployment_score": 0.22,
                    "stress_score": 0.08,
                    "stress_worst_ci_lower": 0.07,
                    "stress_ci_gate_pass": False,
                    "diagnosis_pattern": "Pattern B",
                    "stress_worst_scenario": "adversarial_jitter",
                },
            },
        ],
    }
    (base_root / "bench_summary.json").write_text(json.dumps(base_payload), encoding="utf-8")
    (other_root / "bench_summary.json").write_text(json.dumps(other_payload), encoding="utf-8")

    comparison = sh.compare_bench_quality(base_root, other_root)

    assert comparison["common_jobs"] == 2
    assert comparison["base_protocol"] == "research"
    assert comparison["other_protocol"] == "paper"
    assert comparison["deployment_score_delta"]["mean"] < 0
    assert comparison["stress_score_delta"]["mean"] < 0
    assert comparison["ci_gate_flips"]["unchanged_failed"] == 2
    assert comparison["diagnosis_pattern_changes"] == 2
    assert comparison["worst_scenario_changes"] == 2
