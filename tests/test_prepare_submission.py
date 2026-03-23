"""Tests for scripts/prepare_submission.py strict readiness checks."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "prepare_submission.py"
    spec = importlib.util.spec_from_file_location("prepare_submission", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_check_bench_execution_ready(tmp_path: Path):
    m = _load_module()

    manifest = tmp_path / "bench" / "mini.yaml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "jobs:\n"
        "  - name: mini\n"
        "    matrix:\n"
        "      seed: [0, 1]\n"
        "    args:\n"
        "      out: bench_runs/mini/seed_{seed}\n",
        encoding="utf-8",
    )

    output_root = tmp_path / "bench_runs" / "mini"
    for seed in (0, 1):
        out = output_root / f"seed_{seed}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text("{}", encoding="utf-8")

    (output_root / "bench_summary.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "counts": {"passed": 2, "failed": 0, "skipped": 0},
            }
        ),
        encoding="utf-8",
    )

    result = m.check_bench_execution(manifest, output_root)

    assert result["ready"] is True
    assert result["expected_jobs"] == 2
    assert result["completed_jobs"] == 2
    assert result["bench_status"] == "passed"
    assert result["counts"]["failed"] == 0


def test_check_bench_execution_detects_failures(tmp_path: Path):
    m = _load_module()

    manifest = tmp_path / "bench" / "mini.yaml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "jobs:\n"
        "  - name: mini\n"
        "    args:\n"
        "      out: bench_runs/mini/seed_0\n",
        encoding="utf-8",
    )

    output_root = tmp_path / "bench_runs" / "mini"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "bench_summary.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "counts": {"passed": 0, "failed": 1, "skipped": 0},
            }
        ),
        encoding="utf-8",
    )

    result = m.check_bench_execution(manifest, output_root)

    assert result["ready"] is False
    assert result["bench_status"] == "failed"
    assert result["counts"]["failed"] == 1
    assert any("failed jobs" in reason for reason in result["reasons"])


def test_main_check_only_strict_returns_nonzero(monkeypatch):
    m = _load_module()
    monkeypatch.setattr(m, "print_status_report", lambda **_kwargs: {"ready": False})
    monkeypatch.setattr(sys, "argv", ["prepare_submission.py", "--check-only", "--strict-check"])
    assert m.main() == 1


def test_main_check_only_strict_returns_zero_when_ready(monkeypatch):
    m = _load_module()
    monkeypatch.setattr(m, "print_status_report", lambda **_kwargs: {"ready": True})
    monkeypatch.setattr(sys, "argv", ["prepare_submission.py", "--check-only", "--strict-check"])
    assert m.main() == 0


def test_main_check_only_json_out_writes_report(monkeypatch, tmp_path: Path):
    m = _load_module()
    out_path = tmp_path / "status.json"
    monkeypatch.setattr(m, "print_status_report", lambda **_kwargs: {"ready": True, "n_ready": 7})
    monkeypatch.setattr(sys, "argv", ["prepare_submission.py", "--check-only", "--json-out", str(out_path)])

    assert m.main() == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload == {"ready": True, "n_ready": 7}


def test_prioritized_reasons_prefers_runtime_and_quality():
    m = _load_module()
    reasons = [
        "manifest expands to 0 jobs",
        "bench_summary status is 'failed'",
        "2 failed jobs in bench_summary",
        "2 runtime failures (missing summary or crashed jobs)",
        "5 quality-gate failures (summary exists but CI gate failed)",
    ]

    ranked = m._prioritized_reasons(reasons, max_items=3)

    assert "2 runtime failures (missing summary or crashed jobs)" in ranked
    assert "5 quality-gate failures (summary exists but CI gate failed)" in ranked
    assert len(ranked) == 3


def test_bench_repair_hint_runtime_prefers_resume():
    m = _load_module()
    hint = m._bench_repair_hint(
        label="dm_control",
        manifest="bench/dm_control_research_manifest.yaml",
        bench={
            "expected_jobs": 8,
            "completed_jobs": 6,
            "failure_breakdown": {"runtime_failures": 2, "ci_gate_failures": 0},
        },
    )

    assert hint.startswith("Resume missing/crashed dm_control jobs:")
    assert "--manifest bench/dm_control_research_manifest.yaml" in hint


def test_bench_repair_hint_ci_gate_prefers_retrain():
    m = _load_module()
    hint = m._bench_repair_hint(
        label="CartPole high-rigor",
        manifest="bench/high_rigor_10seed_manifest.yaml",
        bench={
            "expected_jobs": 50,
            "completed_jobs": 50,
            "failure_breakdown": {"runtime_failures": 0, "ci_gate_failures": 5},
        },
    )

    assert hint.startswith("CartPole high-rigor quality gate failed; retrain failing variants then rerun:")


def test_bench_repair_hint_all_failed_prefers_diagnosis():
    m = _load_module()
    hint = m._bench_repair_hint(
        label="CartPole high-rigor",
        manifest="bench/high_rigor_10seed_manifest.yaml",
        bench={
            "output_root": "bench_runs/cartpole_high_rigor_10seed_fresh_20260317",
            "expected_jobs": 50,
            "completed_jobs": 50,
            "failure_breakdown": {"runtime_failures": 0, "ci_gate_failures": 50},
        },
    )

    assert hint.startswith("CartPole high-rigor all jobs failed the quality gate; diagnose protocol/claim mismatch first:")
    assert "python scripts/analyze_bench_failures.py --bench bench_runs/cartpole_high_rigor_10seed_fresh_20260317" in hint


def test_cartpole_failed_variant_seeds_parses_ids():
    m = _load_module()
    parsed = m._cartpole_failed_variant_seeds(
        {
            "failure_breakdown": {
                "failed_job_ids": [
                    "cartpole_intervention2_time_feature_seed-3",
                    "cartpole_intervention2_time_feature_seed-4",
                    "cartpole_intervention1_plus_2_seed-1",
                    "not_cartpole_format",
                ]
            }
        }
    )

    assert parsed == {
        "intervention1_plus_2": [1],
        "intervention2_time_feature": [3, 4],
    }


def test_cartpole_retrain_commands_builds_targeted_invocations():
    m = _load_module()
    cmds = m._cartpole_retrain_commands(
        {
            "intervention1_plus_2": [1, 6, 7],
            "intervention2_time_feature": [3, 4],
        },
        timesteps=45000,
    )

    assert len(cmds) == 2
    assert "--variants intervention1_plus_2 --seeds 1 6 7 --timesteps 45000 --force" in cmds[0] or "--variants intervention1_plus_2 --seeds 1 6 7 --timesteps 45000 --force" in cmds[1]
    assert "--variants intervention2_time_feature --seeds 3 4 --timesteps 45000 --force" in cmds[0] or "--variants intervention2_time_feature --seeds 3 4 --timesteps 45000 --force" in cmds[1]


def test_main_check_only_passes_custom_bench_roots(monkeypatch, tmp_path: Path):
    m = _load_module()
    captured: dict[str, Path] = {}

    def _fake_print_status_report(**kwargs):
        captured.update(kwargs)
        return {"ready": True}

    monkeypatch.setattr(m, "print_status_report", _fake_print_status_report)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_submission.py",
            "--check-only",
            "--cartpole-bench-out",
            str(tmp_path / "cartpole_fresh"),
            "--dm-control-bench-out",
            str(tmp_path / "dm_fresh"),
        ],
    )

    assert m.main() == 0
    assert captured["cartpole_bench_out"] == (tmp_path / "cartpole_fresh").resolve()
    assert captured["dm_control_bench_out"] == (tmp_path / "dm_fresh").resolve()
