"""Tests for scripts/repair_submission_quality.py."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

# Tests that monkeypatch os.name="nt" must be skipped on POSIX. The patch
# leaks into pathlib.Path which then tries to instantiate WindowsPath at
# pytest_sessionfinish and crashes the runner.
windows_only = pytest.mark.skipif(
    os.name != "nt",
    reason="Patches os.name='nt' which leaks into pathlib on POSIX runners",
)


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "repair_submission_quality.py"
    spec = importlib.util.spec_from_file_location("repair_submission_quality", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_main_dry_run_reports_plan(monkeypatch, tmp_path: Path, capsys):
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

    out_root = tmp_path / "bench_runs" / "mini"
    out_root.mkdir(parents=True, exist_ok=True)
    failed_summary = out_root / "seed_0" / "summary.json"
    failed_summary.parent.mkdir(parents=True, exist_ok=True)
    failed_summary.write_text("{}", encoding="utf-8")
    (out_root / "bench_summary.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "counts": {"passed": 0, "failed": 1, "skipped": 0},
                "jobs": [
                    {
                        "id": "mini_seed-0",
                        "status": "failed",
                        "returncode": 2,
                        "summary_path": str(failed_summary),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rc = m.main(
        [
            "--manifest",
            str(manifest),
            "--output-root",
            str(out_root),
            "--protocol",
            "research",
            "--dry-run",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "ci_gate_failures: 1" in out
    assert "cleanup_paths: 1" in out
    assert "--protocol research" in out


def test_main_executes_plan_and_prefers_strict_check_rc(monkeypatch, tmp_path: Path):
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

    out_root = tmp_path / "bench_runs" / "mini"
    out_root.mkdir(parents=True, exist_ok=True)
    failed_summary = out_root / "seed_0" / "summary.json"
    failed_summary.parent.mkdir(parents=True, exist_ok=True)
    failed_summary.write_text("{}", encoding="utf-8")
    (out_root / "bench_summary.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "counts": {"passed": 0, "failed": 1, "skipped": 0},
                "jobs": [
                    {
                        "id": "mini_seed-0",
                        "status": "failed",
                        "returncode": 2,
                        "summary_path": str(failed_summary),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    calls: list[str] = []

    class Result:
        def __init__(self, returncode: int):
            self.returncode = returncode

    def fake_run(command, cwd=None, shell=None, check=None):
        calls.append(str(command))
        if "prepare_submission.py" in str(command):
            return Result(1)
        return Result(0)

    monkeypatch.setattr(m.subprocess, "run", fake_run)

    rc = m.main(
        [
            "--manifest",
            str(manifest),
            "--output-root",
            str(out_root),
            "--protocol",
            "research",
        ]
    )

    assert rc == 1
    assert not failed_summary.exists()
    assert any("deltatau_audit" in call and "--manifest" in call for call in calls)
    assert any("prepare_submission.py --check-only --strict-check" in call for call in calls)
    assert any("run_submission_pipeline.py --mode report" in call for call in calls)


def test_main_aborts_on_failed_retrain(monkeypatch, tmp_path: Path):
    m = _load_module()

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
    failed_summary = out_root / "seed_0" / "summary.json"
    failed_summary.parent.mkdir(parents=True, exist_ok=True)
    failed_summary.write_text("{}", encoding="utf-8")
    (out_root / "bench_summary.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "counts": {"passed": 0, "failed": 1, "skipped": 0},
                "jobs": [
                    {
                        "id": "cartpole_intervention1_plus_2_seed-1",
                        "status": "failed",
                        "returncode": 2,
                        "summary_path": str(failed_summary),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    calls: list[str] = []

    class Result:
        def __init__(self, returncode: int):
            self.returncode = returncode

    def fake_run(command, cwd=None, shell=None, check=None):
        calls.append(str(command))
        if "stress train-sb3" in str(command):
            return Result(7)
        return Result(0)

    monkeypatch.setattr(m.subprocess, "run", fake_run)

    rc = m.main(
        [
            "--manifest",
            str(manifest),
            "--output-root",
            str(out_root),
            "--protocol",
            "paper",
        ]
    )

    assert rc == 7
    assert failed_summary.exists()
    assert len(calls) == 1
    assert "stress train-sb3" in calls[0]


def test_main_launch_background_writes_state(monkeypatch, tmp_path: Path):
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

    out_root = tmp_path / "bench_runs" / "mini"
    out_root.mkdir(parents=True, exist_ok=True)
    failed_summary = out_root / "seed_0" / "summary.json"
    failed_summary.parent.mkdir(parents=True, exist_ok=True)
    failed_summary.write_text("{}", encoding="utf-8")
    (out_root / "bench_summary.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "counts": {"passed": 0, "failed": 1, "skipped": 0},
                "jobs": [
                    {
                        "id": "mini_seed-0",
                        "status": "failed",
                        "returncode": 2,
                        "summary_path": str(failed_summary),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    state_path = tmp_path / "repair_state.json"
    monkeypatch.setattr(m, "_launch_background_process", lambda argv, out_log, err_log: 4242)
    monkeypatch.setattr(m, "_run_argv", lambda argv, allow_failure: 0)

    rc = m.main(
        [
            "--manifest",
            str(manifest),
            "--output-root",
            str(out_root),
            "--protocol",
            "research",
            "--launch-background",
            "--launch-mode",
            "session",
            "--state-path",
            str(state_path),
            "--skip-strict-check",
            "--skip-pipeline-report",
        ]
    )

    assert rc == 0
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["pid"] == 4242
    assert payload["job_name"] == "mini"
    assert payload["launch_mode"] == "session"
    assert payload["summary_paths"] == [str(failed_summary)]
    assert "bench run --manifest" in payload["command"]
    assert "build_failed_job_manifest.py" in payload["prepare_command"]
    assert "--output-dir _status_demo/repair_bench_runs/mini" in payload["prepare_command"]
    assert "merge_bench_summaries.py" in payload["refresh_command"]
    assert not failed_summary.exists()


def test_main_status_reads_state_and_returns_success_when_alive(monkeypatch, tmp_path: Path, capsys):
    m = _load_module()

    state_path = tmp_path / "repair_state.json"
    state_path.write_text(
        json.dumps(
            {
                "job_name": "mini",
                "pid": 4242,
                "launched_at_utc": "2026-03-16T00:00:00Z",
                "command": "python -m deltatau_audit bench run --manifest mini.yaml",
                "out_log": "mini.out.log",
                "err_log": "mini.err.log",
                "summary_paths": [],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(m, "_pid_alive", lambda pid: pid == 4242)

    rc = m.main(
        [
            "--manifest",
            str(tmp_path / "bench" / "mini.yaml"),
            "--output-root",
            str(tmp_path / "bench_runs" / "mini"),
            "--status",
            "--state-path",
            str(state_path),
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "alive: True" in out
    assert "pid: 4242" in out


@windows_only
def test_schedule_background_task_writes_wrapper_and_uses_schtasks(monkeypatch, tmp_path: Path):
    m = _load_module()
    monkeypatch.setattr(m.os, "name", "nt", raising=False)
    monkeypatch.setattr(m, "_powershell_executable", lambda: r"C:\Program Files\PowerShell\7\pwsh.exe")

    calls: list[list[str]] = []

    class Result:
        def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(command, cwd=None, check=None, capture_output=None, text=None):
        calls.append(list(command))
        return Result()

    monkeypatch.setattr(m.subprocess, "run", fake_run)

    state_path = tmp_path / "repair_state.json"
    out_log = tmp_path / "repair.out.log"
    err_log = tmp_path / "repair.err.log"
    info = m._schedule_background_task(
        job_name="mini job",
        argv=[sys.executable, "-m", "deltatau_audit", "bench", "run", "--manifest", "mini.yaml", "--protocol", "paper"],
        refresh_argv=[sys.executable, "scripts/merge_bench_summaries.py", "--base-summary", "bench_runs/mini/bench_summary.json", "--patch-summary", "_status_demo/repair_bench_runs/mini/bench_summary.json", "--output-root", "bench_runs/mini"],
        state_path=state_path,
        out_log=out_log,
        err_log=err_log,
    )

    assert info["launch_mode"] == "durable"
    assert info["task_name"] == "CodexRepair_mini_job"
    wrapper_path = Path(str(info["wrapper_path"]))
    assert wrapper_path.exists()
    wrapper = wrapper_path.read_text(encoding="utf-8")
    assert "durable launch:" in wrapper
    assert "mini.yaml" in wrapper
    assert "refreshing full summary:" in wrapper
    assert len(calls) == 2
    assert calls[0][0] == "schtasks"
    assert "/Create" in calls[0]
    assert "/Run" in calls[1]


def test_parse_schtasks_query_extracts_state():
    m = _load_module()

    parsed = m._parse_schtasks_query(
        "\n".join(
            [
                "TaskName:  \\CodexRepair_mini",
                "Next Run Time:  N/A",
                "Status:  Running",
                "Last Run Time:  2026/03/16 23:13:00",
                "Last Result:  0",
            ]
        )
    )

    assert parsed is not None
    assert parsed["state"] == "Running"
    assert parsed["last_task_result"] == 0
    assert parsed["last_run_time"] == "2026/03/16 23:13:00"


@windows_only
def test_scheduled_task_status_falls_back_to_schtasks(monkeypatch):
    m = _load_module()
    monkeypatch.setattr(m.os, "name", "nt", raising=False)
    monkeypatch.setattr(m, "_powershell_executable", lambda: "pwsh.exe")

    class Result:
        def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(command, cwd=None, check=None, capture_output=None, text=None):
        if command[0] == "pwsh.exe":
            return Result(stdout="")
        return Result(
            stdout="\n".join(
                [
                    "TaskName:  \\CodexRepair_mini",
                    "Next Run Time:  N/A",
                    "Status:  Running",
                    "Last Run Time:  2026/03/16 23:13:00",
                    "Last Result:  267009",
                ]
            )
        )

    monkeypatch.setattr(m.subprocess, "run", fake_run)

    status = m._scheduled_task_status("CodexRepair_mini")

    assert status is not None
    assert status["state"] == "Running"
    assert status["last_task_result"] == 267009


def test_main_status_reads_task_exit_state(monkeypatch, tmp_path: Path, capsys):
    m = _load_module()

    exit_status_path = tmp_path / "repair_state.exit.json"
    exit_status_path.write_text(
        json.dumps({"exit_code": 0, "finished_at_utc": "2026-03-16T00:05:00Z"}),
        encoding="utf-8",
    )
    state_path = tmp_path / "repair_state.json"
    state_path.write_text(
        json.dumps(
            {
                "job_name": "mini",
                "pid": None,
                "launch_mode": "durable",
                "task_name": "CodexRepair_mini",
                "launched_at_utc": "2026-03-16T00:00:00Z",
                "command": "python -m deltatau_audit bench run --manifest mini.yaml",
                "prepare_command": "python scripts/build_failed_job_manifest.py --manifest mini.yaml",
                "refresh_command": "python scripts/merge_bench_summaries.py --base-summary bench_runs/mini/bench_summary.json --patch-summary _status_demo/repair_bench_runs/mini/bench_summary.json --output-root bench_runs/mini",
                "out_log": "mini.out.log",
                "err_log": "mini.err.log",
                "exit_status_path": str(exit_status_path),
                "summary_paths": [],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(m, "_scheduled_task_status", lambda task_name: {"state": "Ready", "last_task_result": 0})

    rc = m.main(
        [
            "--manifest",
            str(tmp_path / "bench" / "mini.yaml"),
            "--output-root",
            str(tmp_path / "bench_runs" / "mini"),
            "--status",
            "--state-path",
            str(state_path),
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "launch_mode: durable" in out
    assert "task_name: CodexRepair_mini" in out
    assert "task_state: Ready" in out
    assert "exit_code: 0" in out
    assert "refresh_command: python scripts/merge_bench_summaries.py --base-summary bench_runs/mini/bench_summary.json --patch-summary _status_demo/repair_bench_runs/mini/bench_summary.json --output-root bench_runs/mini" in out
