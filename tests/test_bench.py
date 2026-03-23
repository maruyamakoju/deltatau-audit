"""Tests for benchmark manifest runner."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from deltatau_audit.bench import run_manifest


def test_run_manifest_expands_matrix_and_runs_jobs(tmp_path, monkeypatch):
    manifest = {
        "output_dir": str(tmp_path / "out"),
        "jobs": [
            {
                "name": "sb3",
                "command": "audit-sb3",
                "matrix": {
                    "env": ["CartPole-v1", "Acrobot-v1"],
                    "seed": [0, 1],
                },
                "args": {
                    "algo": "ppo",
                    "model": "models/{env}.zip",
                    "env": "{env}",
                    "seed": "{seed}",
                },
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    calls: list[list[str]] = []

    def _fake_run(cmd, **kwargs):  # noqa: ARG001
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("subprocess.run", _fake_run)

    summary = run_manifest(str(manifest_path), resume=False)
    assert summary["status"] == "passed"
    assert summary["counts"]["passed"] == 4
    assert len(calls) == 4
    assert all("audit-sb3" in c for c in calls)


def test_run_manifest_resume_skips_existing_summary(tmp_path, monkeypatch):
    existing_out = tmp_path / "existing_run"
    existing_out.mkdir(parents=True, exist_ok=True)
    (existing_out / "summary.json").write_text(json.dumps({"summary": {}}), encoding="utf-8")

    manifest = {
        "jobs": [
            {
                "name": "resume_case",
                "command": "audit-sb3",
                "args": {
                    "algo": "ppo",
                    "model": "m.zip",
                    "env": "CartPole-v1",
                    "out": str(existing_out),
                },
            }
        ]
    }
    manifest_path = tmp_path / "manifest_resume.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def _should_not_run(*_args, **_kwargs):
        raise AssertionError("subprocess.run should not be called in resume skip case")

    monkeypatch.setattr("subprocess.run", _should_not_run)

    summary = run_manifest(str(manifest_path), resume=True)
    assert summary["counts"]["skipped"] == 1
    assert summary["jobs"][0]["status"] == "skipped"


def test_run_manifest_honors_explicit_job_id(tmp_path, monkeypatch):
    manifest = {
        "output_dir": str(tmp_path / "out"),
        "jobs": [
            {
                "name": "subset_case",
                "id": "cartpole_intervention1_plus_2_seed-6",
                "command": "audit-sb3",
                "args": {
                    "algo": "ppo",
                    "model": "m.zip",
                    "env": "CartPole-v1",
                },
            }
        ],
    }
    manifest_path = tmp_path / "manifest_id.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    calls: list[list[str]] = []

    def _fake_run(cmd, **kwargs):  # noqa: ARG001
        calls.append(cmd)
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps({"summary": {}}), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("subprocess.run", _fake_run)

    summary = run_manifest(str(manifest_path), resume=False)

    assert summary["status"] == "passed"
    assert summary["jobs"][0]["id"] == "cartpole_intervention1_plus_2_seed-6"
    assert calls
    assert "cartpole_intervention1_plus_2_seed-6" in calls[0][calls[0].index("--out") + 1]


def test_run_manifest_forces_protocol_and_writes_submission_tables(
    tmp_path, monkeypatch
):
    manifest = {
        "output_dir": str(tmp_path / "out"),
        "jobs": [
            {
                "name": "sb3_submit",
                "command": "audit-sb3",
                "matrix": {
                    "env": ["CartPole-v1"],
                    "seed": [0, 1],
                },
                "args": {
                    "algo": "ppo",
                    "model": "models/{env}_{seed}.zip",
                    "env": "{env}",
                    "seed": "{seed}",
                },
            }
        ],
    }
    manifest_path = tmp_path / "manifest_submit.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    calls: list[list[str]] = []

    def _arg(cmd: list[str], name: str) -> str | None:
        if name not in cmd:
            return None
        idx = cmd.index(name)
        if idx + 1 >= len(cmd):
            return None
        return cmd[idx + 1]

    def _fake_run(cmd, **kwargs):  # noqa: ARG001
        calls.append(cmd)
        out = _arg(cmd, "--out")
        if out:
            out_dir = Path(out)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "deployment_score": 0.88,
                            "deployment_rating": "MILD",
                            "stress_score": 0.56,
                            "stress_rating": "DEGRADED",
                            "stress_threshold": 0.50,
                            "quadrant": "deployment_ready",
                        },
                        "robustness": {
                            "stress": {"worst_case": {"scenario": "speed_5x"}},
                            "per_scenario_scores": {
                                "speed_5x": {
                                    "return_ratio": 0.56,
                                    "ci_lower": 0.51,
                                    "rmse_ratio": 1.2,
                                }
                            },
                        },
                        "diagnosis": {"primary_pattern": "Extreme Frequency Fragility"},
                        "manifest": {"protocol": {"name": "research"}},
                    }
                ),
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("subprocess.run", _fake_run)

    summary = run_manifest(
        str(manifest_path),
        resume=False,
        protocol_name="research",
        allow_protocol_override=False,
    )
    assert summary["status"] == "passed"
    assert summary["counts"]["passed"] == 2
    assert len(calls) == 2
    for call in calls:
        assert "--protocol" in call
        idx = call.index("--protocol")
        assert call[idx + 1] == "research"

    output_root = Path(summary["output_root"])
    csv_path = output_root / "submission_table.csv"
    md_path = output_root / "submission_table.md"
    assert csv_path.exists()
    assert md_path.exists()
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "deployment_score" in csv_text
    assert "speed_5x" in csv_text
