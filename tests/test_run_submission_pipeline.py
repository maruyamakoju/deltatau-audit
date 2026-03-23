"""Tests for scripts/run_submission_pipeline.py helpers."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_submission_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_submission_pipeline", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_expand_manifest_jobs_matrix():
    m = _load_module()
    manifest = {
        "jobs": [
            {
                "name": "x",
                "matrix": {"seed": [0, 1]},
                "args": {"out": "bench_runs/foo/seed_{seed}", "seed": "{seed}"},
            }
        ]
    }

    expanded = m._expand_manifest_jobs(manifest)

    assert len(expanded) == 2
    outs = sorted(row["args"]["out"] for row in expanded)
    assert outs == ["bench_runs/foo/seed_0", "bench_runs/foo/seed_1"]


def test_merge_jobs_keeps_active_pid():
    m = _load_module()
    m._pid_alive = lambda _pid: True
    defaults = [
        m.BenchJob(name="a", manifest="m1.yaml", output_root="o1"),
        m.BenchJob(name="b", manifest="m2.yaml", output_root="o2"),
    ]
    active = [
        m.BenchJob(name="a", manifest="old.yaml", output_root="old", pid=1234),
    ]

    merged = m._merge_jobs(defaults, active)

    assert len(merged) == 2
    assert merged[0].name == "a"
    assert merged[0].pid == 1234
    assert merged[0].manifest == "m1.yaml"  # defaults override stale metadata
    assert merged[1].name == "b"
    assert merged[1].pid is None


def test_merge_jobs_clears_dead_active_pid():
    m = _load_module()
    m._pid_alive = lambda _pid: False
    defaults = [m.BenchJob(name="a", manifest="m1.yaml", output_root="o1")]
    active = [m.BenchJob(name="a", manifest="old.yaml", output_root="old", pid=1234, started_at_utc="2026-03-01T00:00:00Z")]

    merged = m._merge_jobs(defaults, active)

    assert len(merged) == 1
    assert merged[0].pid is None
    assert merged[0].started_at_utc is None


def test_build_command_respects_no_resume():
    m = _load_module()
    job = m.BenchJob(
        name="x",
        manifest="bench/manifest.yaml",
        output_root="bench_runs/x",
        protocol="paper",
        no_resume=True,
    )

    cmd = m._build_command(job)

    assert "--manifest" in cmd
    assert "bench/manifest.yaml" in cmd
    assert "--protocol" in cmd
    assert "paper" in cmd
    assert "--no-resume" in cmd


def test_build_command_override_disables_no_resume():
    m = _load_module()
    job = m.BenchJob(
        name="x",
        manifest="bench/manifest.yaml",
        output_root="bench_runs/x",
        protocol="paper",
        no_resume=True,
    )

    cmd = m._build_command(job, no_resume_override=False)

    assert "--no-resume" not in cmd


def test_launch_job_forces_resume_when_partial_progress(monkeypatch, tmp_path: Path):
    m = _load_module()
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "STATUS_DIR", tmp_path / "_status_demo" / "long_runs")
    monkeypatch.setattr(m, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(m, "_job_progress", lambda _job: {"total": 10, "done": 6})
    monkeypatch.setattr(m, "_launch_popen_kwargs", lambda: {})

    captured: list[list[str]] = []

    class _Proc:
        pid = 999

    def _fake_popen(cmd, **kwargs):  # noqa: ARG001
        captured.append(cmd)
        return _Proc()

    monkeypatch.setattr(m.subprocess, "Popen", _fake_popen)

    job = m.BenchJob(
        name="mini",
        manifest="bench/mini.yaml",
        output_root="bench_runs/mini",
        protocol="paper",
        no_resume=True,
        out_log="logs/out.log",
        err_log="logs/err.log",
    )

    launched = m._launch_job(job, force_restart=False)

    assert launched.pid == 999
    assert captured
    assert "--no-resume" not in captured[0]


def test_launch_job_force_restart_keeps_no_resume(monkeypatch, tmp_path: Path):
    m = _load_module()
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "STATUS_DIR", tmp_path / "_status_demo" / "long_runs")
    monkeypatch.setattr(m, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(m, "_job_progress", lambda _job: {"total": 10, "done": 6})
    monkeypatch.setattr(m, "_launch_popen_kwargs", lambda: {})

    captured: list[list[str]] = []

    class _Proc:
        pid = 1001

    def _fake_popen(cmd, **kwargs):  # noqa: ARG001
        captured.append(cmd)
        return _Proc()

    monkeypatch.setattr(m.subprocess, "Popen", _fake_popen)

    job = m.BenchJob(
        name="mini",
        manifest="bench/mini.yaml",
        output_root="bench_runs/mini",
        protocol="paper",
        no_resume=True,
        out_log="logs/out.log",
        err_log="logs/err.log",
    )

    launched = m._launch_job(job, force_restart=True)

    assert launched.pid == 1001
    assert captured
    assert "--no-resume" in captured[0]


def test_pid_alive_false_for_invalid_pid():
    m = _load_module()
    assert m._pid_alive(None) is False
    assert m._pid_alive(-1) is False


def test_pid_alive_windows_tasklist_true(monkeypatch):
    m = _load_module()
    monkeypatch.setattr(m, "psutil", None)
    monkeypatch.setattr(m.os, "name", "nt", raising=False)

    class _Proc:
        def __init__(self, stdout: str):
            self.stdout = stdout
            self.stderr = ""

    monkeypatch.setattr(
        m.subprocess,
        "run",
        lambda *args, **kwargs: _Proc('"python.exe","1234","Console","1","10,000 K"\n'),
    )

    assert m._pid_alive(1234) is True


def test_pid_alive_windows_tasklist_false(monkeypatch):
    m = _load_module()
    monkeypatch.setattr(m, "psutil", None)
    monkeypatch.setattr(m.os, "name", "nt", raising=False)

    class _Proc:
        def __init__(self, stdout: str):
            self.stdout = stdout
            self.stderr = ""

    monkeypatch.setattr(
        m.subprocess,
        "run",
        lambda *args, **kwargs: _Proc("INFO: No tasks are running which match the specified criteria.\n"),
    )

    assert m._pid_alive(1234) is False


def test_terminate_job_process_falls_back_to_taskkill(monkeypatch):
    m = _load_module()
    monkeypatch.setattr(m.os, "name", "nt", raising=False)
    monkeypatch.setattr(m, "psutil", None)
    monkeypatch.setattr(m, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(m.time, "sleep", lambda _s: None)

    def _raise_kill(_pid, _sig):
        raise OSError("kill failed")

    monkeypatch.setattr(m.os, "kill", _raise_kill)
    calls: list[int] = []

    def _fake_taskkill(_pid, *, timeout_s=20):
        calls.append(int(timeout_s))
        return len(calls) >= 2

    monkeypatch.setattr(m, "_taskkill_process_tree", _fake_taskkill)

    ok = m._terminate_job_process(123, timeout_s=1)

    assert ok is True
    assert calls == [0, 1]


def test_load_active_jobs_backward_compatible_command_only(tmp_path: Path):
    m = _load_module()
    payload = {
        "generated_at_utc": "2026-03-01T00:00:00Z",
        "jobs": [
            {
                "name": "cartpole_high_rigor_bench",
                "command": (
                    "python -m deltatau_audit bench run "
                    "--manifest bench/high_rigor_10seed_manifest.yaml "
                    "--protocol paper"
                ),
                "pid": 1234,
                "out_log": "out.log",
                "err_log": "err.log",
            }
        ],
    }
    path = tmp_path / "active_jobs.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    jobs = m._load_active_jobs(path)

    assert len(jobs) == 1
    job = jobs[0]
    assert job.name == "cartpole_high_rigor_bench"
    assert job.manifest == "bench/high_rigor_10seed_manifest.yaml"
    assert job.protocol == "paper"
    assert job.pid == 1234
    assert job.out_log == "out.log"
    assert job.err_log == "err.log"
    assert job.output_root == ""


def test_atomic_write_text_overwrites_target(tmp_path: Path):
    m = _load_module()
    path = tmp_path / "state.json"
    path.write_text("old", encoding="utf-8")

    m._atomic_write_text(path, '{"ok": true}', encoding="utf-8")

    assert path.read_text(encoding="utf-8") == '{"ok": true}'
    tmp_files = list(tmp_path.glob("*.tmp.*"))
    assert tmp_files == []


def test_launch_popen_kwargs_windows(monkeypatch):
    m = _load_module()
    monkeypatch.setattr(m.os, "name", "nt", raising=False)

    kwargs = m._launch_popen_kwargs()

    flags = kwargs.get("creationflags", 0)
    assert isinstance(flags, int)
    assert flags & int(getattr(m.subprocess, "CREATE_NEW_PROCESS_GROUP", 0))


def test_launch_popen_kwargs_posix(monkeypatch):
    m = _load_module()
    monkeypatch.setattr(m.os, "name", "posix", raising=False)

    kwargs = m._launch_popen_kwargs()

    assert kwargs == {"start_new_session": True}


def test_launch_jobs_skips_completed_when_not_force_restart(monkeypatch):
    m = _load_module()
    jobs = [m.BenchJob(name="done", manifest="bench/done.yaml", output_root="bench_runs/done")]

    monkeypatch.setattr(m, "_is_job_completed", lambda _job: True)
    launched_calls: list[str] = []
    monkeypatch.setattr(m, "_launch_job", lambda job, force_restart=False: launched_calls.append(job.name) or job)
    monkeypatch.setattr(m, "_save_active_jobs", lambda _path, _jobs: None)

    launched = m._launch_jobs(jobs, force_restart=False)

    assert len(launched) == 1
    assert launched_calls == []


def test_job_progress_marks_stale_counts(tmp_path: Path, monkeypatch):
    m = _load_module()
    monkeypatch.setattr(m, "ROOT", tmp_path)

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
    seed_dir = out_root / "seed_0"
    seed_dir.mkdir(parents=True, exist_ok=True)
    (out_root / "bench_summary.json").write_text(
        json.dumps({"counts": {"passed": 1, "failed": 0, "skipped": 0}}),
        encoding="utf-8",
    )
    (seed_dir / "summary.json").write_text("{}", encoding="utf-8")

    stale_ts = time.time() - 3600
    summary_path = out_root / "bench_summary.json"
    os.utime(summary_path, (stale_ts, stale_ts))

    started = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    job = m.BenchJob(
        name="mini",
        manifest="bench/mini.yaml",
        output_root="bench_runs/mini",
        started_at_utc=started,
    )

    progress = m._job_progress(job)

    assert progress["counts"] == {"passed": 1, "failed": 0, "skipped": 0}
    assert progress["counts_stale"] is True


def test_finalize_uses_strict_prepare_submission(tmp_path: Path, monkeypatch):
    m = _load_module()
    monkeypatch.setattr(m, "ROOT", tmp_path)

    out_root = tmp_path / "bench_runs" / "mini"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "bench_summary.json").write_text("{}", encoding="utf-8")

    calls: list[list[str]] = []

    def _fake_run(cmd: list[str]) -> int:
        calls.append(cmd)
        return 0

    monkeypatch.setattr(m, "_run_cmd", _fake_run)

    rc = m._finalize(
        [
            m.BenchJob(
                name="mini",
                manifest="bench/mini.yaml",
                output_root="bench_runs/mini",
            )
        ]
    )

    assert rc == 0
    assert any(
        cmd[:4] == [sys.executable, "scripts/prepare_submission.py", "--check-only", "--strict-check"]
        for cmd in calls
    )


def test_build_preflight_manifest_reduces_matrix_and_overrides_args(tmp_path: Path, monkeypatch):
    m = _load_module()
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "STATUS_DIR", tmp_path / "_status_demo" / "long_runs")

    manifest = tmp_path / "bench" / "mini.yaml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "name: mini\n"
        "jobs:\n"
        "  - name: audit_case\n"
        "    matrix:\n"
        "      seed: [0, 1]\n"
        "      env: [CartPole-v1, Acrobot-v1]\n"
        "    args:\n"
        "      model: checkpoints/{env}/seed_{seed}.zip\n"
        "      seed: \"{seed}\"\n"
        "      env: \"{env}\"\n"
        "      out: bench_runs/mini/{env}/seed_{seed}\n",
        encoding="utf-8",
    )
    job = m.BenchJob(
        name="mini_job",
        manifest="bench/mini.yaml",
        output_root="bench_runs/mini",
    )
    target = tmp_path / "_status_demo" / "long_runs" / "preflight_manifests" / "mini_job.yaml"

    out_path = m._build_preflight_manifest(job, target)
    payload = m.yaml.safe_load(out_path.read_text(encoding="utf-8"))

    assert out_path == target
    assert isinstance(payload, dict)
    jobs = payload.get("jobs")
    assert isinstance(jobs, list) and len(jobs) == 1
    row = jobs[0]
    matrix = row["matrix"]
    assert matrix["seed"] == [0]
    assert matrix["env"] == ["CartPole-v1"]
    args = row["args"]
    assert args["episodes"] == 1
    assert args["speeds"] == [1]
    assert args["seeds"] == [0]
    assert args["workers"] == 1
    assert args["protocol"] == "custom"
    assert args["ci"] is False
    assert "ci_gate_mode" not in args
    assert args["out"] == "_status_demo/preflight/mini_job/audit_case"


def test_run_preflight_invokes_custom_bench_run(tmp_path: Path, monkeypatch):
    m = _load_module()
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "STATUS_DIR", tmp_path / "_status_demo" / "long_runs")

    # Minimal source manifest required by preflight builder.
    manifest = tmp_path / "bench" / "mini.yaml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "jobs:\n"
        "  - name: smoke\n"
        "    args:\n"
        "      out: bench_runs/mini/smoke\n",
        encoding="utf-8",
    )

    calls: list[list[str]] = []

    def _fake_run(cmd: list[str]) -> int:
        calls.append(cmd)
        return 0

    monkeypatch.setattr(m, "_run_cmd", _fake_run)

    rc = m._run_preflight(
        [m.BenchJob(name="mini", manifest="bench/mini.yaml", output_root="bench_runs/mini")]
    )

    assert rc == 0
    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[0] == sys.executable
    assert cmd[1:5] == ["-m", "deltatau_audit", "bench", "run"]
    assert "--protocol" in cmd and cmd[cmd.index("--protocol") + 1] == "custom"
    assert "--allow-protocol-override" in cmd
    assert "--no-resume" in cmd
    assert "--fail-fast" in cmd


def test_classify_error_signature_patterns():
    m = _load_module()
    assert m._classify_error_signature(["forrtl: error (200): program aborting"]) == "fortran_window_close"
    assert m._classify_error_signature(["Fatal Python error: init_import_site"]) == "python_site_init_failure"
    assert m._classify_error_signature(["KeyboardInterrupt"]) == "keyboard_interrupt"
    assert m._classify_error_signature(["Traceback (most recent call last):"]) == "python_exception"
    assert m._classify_error_signature([]) is None


def test_snapshot_metrics_computes_rate_and_eta():
    m = _load_module()
    metrics = m._snapshot_metrics(
        {
            "mini": {
                "timestamp_s": 0.0,
                "done": 2,
                "child_cpu_s_total": 10.0,
            }
        },
        job_name="mini",
        now_ts=3600.0,
        done=5,
        total=10,
        child_cpu_s_total=16.0,
    )

    assert metrics["window_s"] == 3600
    assert metrics["delta_done"] == 3
    assert metrics["jobs_per_hour"] == 3.0
    assert metrics["eta_s"] == 6000
    assert metrics["child_cpu_delta_s"] == 6.0


def test_collect_log_health_detects_latest_signature(tmp_path: Path):
    m = _load_module()
    root = tmp_path / "bench_runs" / "mini"
    old_dir = root / "job_a"
    new_dir = root / "job_b"
    old_dir.mkdir(parents=True, exist_ok=True)
    new_dir.mkdir(parents=True, exist_ok=True)

    old_err = old_dir / "bench_run.err.log"
    old_err.write_text("KeyboardInterrupt\n", encoding="utf-8")
    old_out = old_dir / "bench_run.log"
    old_out.write_text("old\n", encoding="utf-8")

    new_err = new_dir / "bench_run.err.log"
    new_err.write_text("forrtl: error (200): program aborting\n", encoding="utf-8")
    new_out = new_dir / "bench_run.log"
    new_out.write_text("new\n", encoding="utf-8")

    # Ensure deterministic "latest" ordering.
    now = time.time()
    os.utime(old_err, (now - 20, now - 20))
    os.utime(old_out, (now - 20, now - 20))
    os.utime(new_err, (now - 5, now - 5))
    os.utime(new_out, (now - 5, now - 5))

    health = m._collect_log_health(root)

    assert health["latest_error_signature"] == "fortran_window_close"
    assert isinstance(health["latest_stdout_log"], str)
    assert isinstance(health["latest_stderr_log"], str)
    assert health["latest_stdout_log"].endswith("job_b\\bench_run.log")
    assert health["latest_stderr_log"].endswith("job_b\\bench_run.err.log")


def test_print_diagnose_reports_stall(monkeypatch, capsys):
    m = _load_module()
    job = m.BenchJob(name="mini", manifest="bench/mini.yaml", output_root="bench_runs/mini", pid=123)

    monkeypatch.setattr(m, "_load_monitor_snapshot", lambda _path=None: {"jobs": {}})
    monkeypatch.setattr(m, "_save_monitor_snapshot", lambda _path, _payload: None)
    monkeypatch.setattr(m, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(
        m,
        "_job_progress",
        lambda _job: {
            "total": 10,
            "done": 2,
            "pct": 20.0,
            "log_health": {
                "latest_stdout_age_s": 1200,
                "latest_stderr_age_s": 1300,
                "latest_error_signature": None,
            },
        },
    )

    rc = m._print_diagnose([job], stall_seconds=900)
    out = capsys.readouterr().out

    assert rc == 1
    assert "possible_stall" in out


def test_print_diagnose_reports_running_compute_bound(monkeypatch, capsys):
    m = _load_module()
    job = m.BenchJob(name="mini", manifest="bench/mini.yaml", output_root="bench_runs/mini", pid=123)

    monkeypatch.setattr(
        m,
        "_load_monitor_snapshot",
        lambda _path=None: {
            "jobs": {
                "mini": {
                    "timestamp_s": 0.0,
                    "done": 2,
                    "total": 10,
                    "child_cpu_s_total": 100.0,
                }
            }
        },
    )
    monkeypatch.setattr(m, "_save_monitor_snapshot", lambda _path, _payload: None)
    monkeypatch.setattr(m.time, "time", lambda: 120.0)
    monkeypatch.setattr(m, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(
        m,
        "_job_progress",
        lambda _job: {
            "total": 10,
            "done": 2,
            "pct": 20.0,
            "log_health": {
                "latest_stdout_age_s": 1200,
                "latest_stderr_age_s": 1300,
                "latest_error_signature": None,
            },
            "process_health": {
                "child_count": 1,
                "child_cpu_s_total": 104.5,
                "newest_child_pid": 999,
            },
        },
    )

    rc = m._print_diagnose([job], stall_seconds=900)
    out = capsys.readouterr().out

    assert rc == 0
    assert "running_compute_bound" in out


def test_diagnose_job_classification():
    m = _load_module()

    blocked = m._diagnose_job(
        alive=False,
        total=10,
        done=3,
        out_age_s=10,
        err_age_s=10,
        child_count=0,
        child_cpu_delta_s=None,
        stall_seconds=900,
    )
    assert blocked.code == "blocked_dead"
    assert blocked.recoverable is True

    compute_bound = m._diagnose_job(
        alive=True,
        total=10,
        done=3,
        out_age_s=1800,
        err_age_s=1900,
        child_count=1,
        child_cpu_delta_s=12.5,
        stall_seconds=900,
    )
    assert compute_bound.code == "running_compute_bound"
    assert compute_bound.recoverable is False


def test_register_diagnosis_same_code_counter():
    m = _load_module()
    state_jobs = {
        "mini": {
            "restart_count": 0,
            "consecutive_recoverable": 2,
            "last_restart_ts": -1.0,
            "last_reason": "",
            "last_diagnosis": "blocked_dead",
        }
    }

    meta = m._register_diagnosis(
        state_jobs=state_jobs,
        job_name="mini",
        diagnosis=m.JobDiagnosis(
            code="possible_stall",
            summary="possible_stall",
            recoverable=True,
        ),
    )
    assert meta["consecutive_recoverable"] == 1

    meta = m._register_diagnosis(
        state_jobs=state_jobs,
        job_name="mini",
        diagnosis=m.JobDiagnosis(
            code="possible_stall",
            summary="possible_stall",
            recoverable=True,
        ),
    )
    assert meta["consecutive_recoverable"] == 2


def test_register_progress_counts_stagnation():
    m = _load_module()
    state_jobs = {"mini": {"last_done": 2, "no_progress_cycles": 1}}

    meta = m._register_progress(state_jobs=state_jobs, job_name="mini", done=2, total=10, now_ts=10.0)
    assert meta["no_progress_cycles"] == 2
    assert meta["last_progress_ts"] == 10.0

    meta = m._register_progress(state_jobs=state_jobs, job_name="mini", done=3, total=10, now_ts=20.0)
    assert meta["no_progress_cycles"] == 0
    assert meta["last_progress_ts"] == 20.0


def test_project_progress_meta_updates_stale_state():
    m = _load_module()
    meta = {
        "restart_count": 0,
        "consecutive_recoverable": 0,
        "no_progress_cycles": 5,
        "last_done": 2,
        "last_progress_ts": 0.0,
        "last_restart_ts": -1.0,
        "last_reason": "",
        "last_diagnosis": "running",
        "restarts_by_reason": {},
    }

    projected = m._project_progress_meta(meta, done=3, total=10, now_ts=100.0)

    assert projected["no_progress_cycles"] == 0
    assert projected["last_done"] == 3
    assert projected["last_progress_ts"] == 100.0


def test_supervisor_row_normalizes_restart_history():
    m = _load_module()
    state_jobs = {
        "mini": {
            "restart_times_s": [-1, "x", 30, "20", 10.5],
        }
    }

    meta = m._supervisor_row(state_jobs, "mini")

    assert meta["restart_times_s"] == [10.5, 20.0, 30.0]


def test_register_signature_observation_tracks_streak():
    m = _load_module()
    state_jobs = {"mini": {}}

    meta = m._register_signature_observation(state_jobs=state_jobs, job_name="mini", signature="fortran_window_close")
    assert meta["last_signature"] == "fortran_window_close"
    assert meta["consecutive_signature_hits"] == 1

    meta = m._register_signature_observation(state_jobs=state_jobs, job_name="mini", signature="fortran_window_close")
    assert meta["consecutive_signature_hits"] == 2

    meta = m._register_signature_observation(state_jobs=state_jobs, job_name="mini", signature=None)
    assert meta["last_signature"] == ""
    assert meta["consecutive_signature_hits"] == 0


def test_forced_recovery_reason_only_for_running_codes():
    m = _load_module()

    assert (
        m._forced_recovery_reason(
            diagnosis_code="possible_stall",
            incomplete=True,
            no_progress_cycles=100,
            no_progress_seconds=10000,
            max_no_progress_cycles=10,
            max_no_progress_seconds=3600,
        )
        is None
    )
    assert (
        m._forced_recovery_reason(
            diagnosis_code="running",
            incomplete=False,
            no_progress_cycles=100,
            no_progress_seconds=10000,
            max_no_progress_cycles=10,
            max_no_progress_seconds=3600,
        )
        is None
    )
    assert (
        m._forced_recovery_reason(
            diagnosis_code="running_compute_bound",
            incomplete=True,
            no_progress_cycles=10,
            no_progress_seconds=500,
            max_no_progress_cycles=10,
            max_no_progress_seconds=3600,
        )
        == "no_progress_timeout"
    )
    assert (
        m._forced_recovery_reason(
            diagnosis_code="running_compute_bound",
            incomplete=True,
            no_progress_cycles=10,
            no_progress_seconds=5000,
            max_no_progress_cycles=10,
            max_no_progress_seconds=3600,
            child_cpu_delta_s=12.5,
        )
        is None
    )
    assert (
        m._forced_recovery_reason(
            diagnosis_code="running_silent",
            incomplete=True,
            no_progress_cycles=1,
            no_progress_seconds=3600,
            max_no_progress_cycles=10,
            max_no_progress_seconds=3600,
        )
        == "no_progress_timeout_seconds"
    )


def test_decide_recovery_action_skips_reason_budget():
    m = _load_module()
    decision = m._decide_recovery_action(
        meta={
            "restart_count": 0,
            "consecutive_recoverable": 3,
            "last_restart_ts": -1.0,
            "restarts_by_reason": {"possible_stall": 2},
            "restarts_by_signature": {},
            "restart_times_s": [],
        },
        effective_reason="possible_stall",
        forced_reason=None,
        current_signature="fortran_window_close",
        now_ts=100.0,
        recover_after_consecutive=2,
        max_restarts_per_job=3,
        max_restarts_per_reason=2,
        max_restarts_per_signature=0,
        max_restarts_per_window=0,
        restart_window_seconds=0,
        restart_cooldown_seconds=0,
        restart_backoff_factor=1.0,
        max_restart_cooldown_seconds=0,
    )

    assert decision.allow is False
    assert decision.skip_event == "recovery_skipped_reason_budget"


def test_decide_recovery_action_forced_reason_bypasses_consecutive_gate():
    m = _load_module()
    decision = m._decide_recovery_action(
        meta={
            "restart_count": 0,
            "consecutive_recoverable": 0,
            "last_restart_ts": -1.0,
            "restarts_by_reason": {},
            "restarts_by_signature": {},
            "restart_times_s": [],
        },
        effective_reason="no_progress_timeout",
        forced_reason="no_progress_timeout",
        current_signature=None,
        now_ts=100.0,
        recover_after_consecutive=2,
        max_restarts_per_job=3,
        max_restarts_per_reason=2,
        max_restarts_per_signature=0,
        max_restarts_per_window=0,
        restart_window_seconds=0,
        restart_cooldown_seconds=0,
        restart_backoff_factor=1.0,
        max_restart_cooldown_seconds=0,
    )

    assert decision.allow is True


def test_decide_recovery_action_skips_cooldown():
    m = _load_module()
    decision = m._decide_recovery_action(
        meta={
            "restart_count": 1,
            "consecutive_recoverable": 3,
            "last_restart_ts": 90.0,
            "restarts_by_reason": {},
            "restarts_by_signature": {},
            "restart_times_s": [90.0],
        },
        effective_reason="possible_stall",
        forced_reason=None,
        current_signature="fortran_window_close",
        now_ts=100.0,
        recover_after_consecutive=2,
        max_restarts_per_job=3,
        max_restarts_per_reason=2,
        max_restarts_per_signature=0,
        max_restarts_per_window=10,
        restart_window_seconds=3600,
        restart_cooldown_seconds=60,
        restart_backoff_factor=1.0,
        max_restart_cooldown_seconds=0,
    )

    assert decision.allow is False
    assert decision.skip_event == "recovery_skipped_cooldown"


def test_decide_recovery_action_skips_signature_budget():
    m = _load_module()
    decision = m._decide_recovery_action(
        meta={
            "restart_count": 1,
            "consecutive_recoverable": 3,
            "last_restart_ts": -1.0,
            "restarts_by_reason": {},
            "restarts_by_signature": {"fortran_window_close": 2},
            "restart_times_s": [],
        },
        effective_reason="possible_stall",
        forced_reason=None,
        current_signature="fortran_window_close",
        now_ts=100.0,
        recover_after_consecutive=2,
        max_restarts_per_job=3,
        max_restarts_per_reason=2,
        max_restarts_per_signature=2,
        max_restarts_per_window=10,
        restart_window_seconds=3600,
        restart_cooldown_seconds=0,
        restart_backoff_factor=1.0,
        max_restart_cooldown_seconds=0,
    )

    assert decision.allow is False
    assert decision.skip_event == "recovery_skipped_signature_budget"


def test_emit_supervisor_event_tolerates_append_failure(monkeypatch):
    m = _load_module()

    def _boom(_path, _payload):
        raise OSError("disk full")

    monkeypatch.setattr(m, "_append_jsonl", _boom)

    m._emit_supervisor_event(
        Path("dummy.jsonl"),
        cycle=1,
        job_name="mini",
        event="diagnosis",
        payload={"ok": True},
    )


def test_effective_restart_cooldown_seconds_backoff():
    m = _load_module()

    assert (
        m._effective_restart_cooldown_seconds(
            restart_count=0,
            base_cooldown_seconds=60,
            backoff_factor=2.0,
            max_cooldown_seconds=0,
        )
        == 60
    )
    assert (
        m._effective_restart_cooldown_seconds(
            restart_count=2,
            base_cooldown_seconds=60,
            backoff_factor=2.0,
            max_cooldown_seconds=0,
        )
        == 240
    )
    assert (
        m._effective_restart_cooldown_seconds(
            restart_count=4,
            base_cooldown_seconds=60,
            backoff_factor=2.0,
            max_cooldown_seconds=300,
        )
        == 300
    )


def test_supervise_auto_recover_restarts_job(monkeypatch):
    m = _load_module()
    job = m.BenchJob(name="mini", manifest="bench/mini.yaml", output_root="bench_runs/mini", pid=111)

    monkeypatch.setattr(m, "_load_supervisor_state", lambda _path=None: {"jobs": {}})
    monkeypatch.setattr(m.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        m,
        "_collect_diagnose_rows",
        lambda _jobs, stall_seconds=0: (
            [
                {
                    "job": job,
                    "done": 2,
                    "total": 10,
                    "pct": 20.0,
                    "diagnosis": m.JobDiagnosis(
                        code="blocked_dead",
                        summary="blocked (process dead before completion)",
                        recoverable=True,
                    ),
                }
            ],
            {"updated_at_utc": "2026-03-02T00:00:00+00:00", "jobs": {}},
        ),
    )
    monkeypatch.setattr(m, "_save_monitor_snapshot", lambda _path, _payload: None)
    saved_state: list[dict] = []
    monkeypatch.setattr(m, "_save_supervisor_state", lambda _path, payload: saved_state.append(payload))
    saved_active: list[list[m.BenchJob]] = []
    monkeypatch.setattr(m, "_save_active_jobs", lambda _path, jobs: saved_active.append(jobs))

    restart_calls: list[str] = []

    def _fake_recover(_job, *, reason: str):
        restart_calls.append(reason)
        _job.pid = 222
        _job.started_at_utc = "2026-03-02T00:00:00+00:00"
        return _job, True

    monkeypatch.setattr(m, "_try_recover_job", _fake_recover)

    rc = m._supervise(
        [job],
        interval_seconds=1,
        stall_seconds=900,
        auto_recover=True,
        recover_after_consecutive=1,
        max_restarts_per_job=1,
        restart_cooldown_seconds=0,
        max_cycles=1,
    )

    assert rc == 1
    assert restart_calls == ["blocked_dead"]
    assert saved_active  # active job state is persisted after recovery
    assert saved_state
    jobs_payload = saved_state[-1]["jobs"]
    assert jobs_payload["mini"]["restart_count"] == 1


def test_supervise_respects_restart_budget(monkeypatch):
    m = _load_module()
    job = m.BenchJob(name="mini", manifest="bench/mini.yaml", output_root="bench_runs/mini", pid=111)

    monkeypatch.setattr(
        m,
        "_load_supervisor_state",
        lambda _path=None: {"jobs": {"mini": {"restart_count": 1, "last_restart_ts": 0.0}}},
    )
    monkeypatch.setattr(m.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        m,
        "_collect_diagnose_rows",
        lambda _jobs, stall_seconds=0: (
            [
                {
                    "job": job,
                    "done": 2,
                    "total": 10,
                    "pct": 20.0,
                    "diagnosis": m.JobDiagnosis(
                        code="possible_stall",
                        summary="possible_stall (no recent log activity >= 900s)",
                        recoverable=True,
                    ),
                }
            ],
            {"updated_at_utc": "2026-03-02T00:00:00+00:00", "jobs": {}},
        ),
    )
    monkeypatch.setattr(m, "_save_monitor_snapshot", lambda _path, _payload: None)
    monkeypatch.setattr(m, "_save_supervisor_state", lambda _path, _payload: None)
    monkeypatch.setattr(m, "_save_active_jobs", lambda _path, _jobs: None)

    called = {"recover": 0}

    def _fake_recover(_job, *, reason: str):
        called["recover"] += 1
        return _job, True

    monkeypatch.setattr(m, "_try_recover_job", _fake_recover)

    rc = m._supervise(
        [job],
        interval_seconds=1,
        stall_seconds=900,
        auto_recover=True,
        recover_after_consecutive=1,
        max_restarts_per_job=1,
        restart_cooldown_seconds=0,
        max_cycles=1,
    )

    assert rc == 1
    assert called["recover"] == 0


def test_supervise_waits_for_consecutive_recoverable(monkeypatch):
    m = _load_module()
    job = m.BenchJob(name="mini", manifest="bench/mini.yaml", output_root="bench_runs/mini", pid=111)

    monkeypatch.setattr(m, "_load_supervisor_state", lambda _path=None: {"jobs": {}})
    monkeypatch.setattr(m.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        m,
        "_collect_diagnose_rows",
        lambda _jobs, stall_seconds=0: (
            [
                {
                    "job": job,
                    "done": 2,
                    "total": 10,
                    "pct": 20.0,
                    "diagnosis": m.JobDiagnosis(
                        code="possible_stall",
                        summary="possible_stall (no recent log activity >= 900s)",
                        recoverable=True,
                    ),
                }
            ],
            {"updated_at_utc": "2026-03-02T00:00:00+00:00", "jobs": {}},
        ),
    )
    monkeypatch.setattr(m, "_save_monitor_snapshot", lambda _path, _payload: None)
    monkeypatch.setattr(m, "_save_supervisor_state", lambda _path, _payload: None)
    monkeypatch.setattr(m, "_save_active_jobs", lambda _path, _jobs: None)

    called = {"recover": 0}

    def _fake_recover(_job, *, reason: str):
        called["recover"] += 1
        return _job, True

    monkeypatch.setattr(m, "_try_recover_job", _fake_recover)

    rc = m._supervise(
        [job],
        interval_seconds=1,
        stall_seconds=900,
        auto_recover=True,
        recover_after_consecutive=2,
        max_restarts_per_job=2,
        restart_cooldown_seconds=0,
        max_cycles=1,
    )

    assert rc == 1
    assert called["recover"] == 0


def test_supervise_emits_events(monkeypatch, tmp_path: Path):
    m = _load_module()
    job = m.BenchJob(name="mini", manifest="bench/mini.yaml", output_root="bench_runs/mini", pid=111)

    monkeypatch.setattr(m, "_load_supervisor_state", lambda _path=None: {"jobs": {}})
    monkeypatch.setattr(m.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        m,
        "_collect_diagnose_rows",
        lambda _jobs, stall_seconds=0: (
            [
                {
                    "job": job,
                    "done": 2,
                    "total": 10,
                    "pct": 20.0,
                    "diagnosis": m.JobDiagnosis(
                        code="running",
                        summary="running (in progress)",
                        recoverable=False,
                    ),
                }
            ],
            {"updated_at_utc": "2026-03-02T00:00:00+00:00", "jobs": {}},
        ),
    )
    monkeypatch.setattr(m, "_save_monitor_snapshot", lambda _path, _payload: None)
    monkeypatch.setattr(m, "_save_supervisor_state", lambda _path, _payload: None)
    monkeypatch.setattr(m, "_save_active_jobs", lambda _path, _jobs: None)

    events: list[dict] = []
    monkeypatch.setattr(m, "_append_jsonl", lambda _path, payload: events.append(payload))

    rc = m._supervise(
        [job],
        interval_seconds=1,
        stall_seconds=900,
        auto_recover=False,
        recover_after_consecutive=2,
        max_restarts_per_job=2,
        restart_cooldown_seconds=0,
        max_cycles=1,
        events_path=tmp_path / "events.jsonl",
    )

    assert rc == 0
    assert events
    assert events[0]["event"] == "diagnosis"


def test_supervise_no_progress_timeout_triggers_recovery(monkeypatch):
    m = _load_module()
    job = m.BenchJob(name="mini", manifest="bench/mini.yaml", output_root="bench_runs/mini", pid=111)

    monkeypatch.setattr(
        m,
        "_load_supervisor_state",
        lambda _path=None: {"jobs": {"mini": {"last_done": 2, "no_progress_cycles": 0}}},
    )
    monkeypatch.setattr(m.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        m,
        "_collect_diagnose_rows",
        lambda _jobs, stall_seconds=0: (
            [
                {
                    "job": job,
                    "done": 2,
                    "total": 10,
                    "pct": 20.0,
                    "diagnosis": m.JobDiagnosis(
                        code="running_compute_bound",
                        summary="running_compute_bound",
                        recoverable=False,
                    ),
                }
            ],
            {"updated_at_utc": "2026-03-02T00:00:00+00:00", "jobs": {}},
        ),
    )
    monkeypatch.setattr(m, "_save_monitor_snapshot", lambda _path, _payload: None)
    monkeypatch.setattr(m, "_save_supervisor_state", lambda _path, _payload: None)
    monkeypatch.setattr(m, "_save_active_jobs", lambda _path, _jobs: None)
    monkeypatch.setattr(m, "_append_jsonl", lambda _path, _payload: None)

    reasons: list[str] = []

    def _fake_recover(_job, *, reason: str):
        reasons.append(reason)
        return _job, True

    monkeypatch.setattr(m, "_try_recover_job", _fake_recover)

    rc = m._supervise(
        [job],
        interval_seconds=1,
        stall_seconds=900,
        auto_recover=True,
        recover_after_consecutive=2,
        max_restarts_per_job=2,
        restart_cooldown_seconds=0,
        max_no_progress_cycles=1,
        max_cycles=1,
    )

    assert rc == 1
    assert reasons == ["no_progress_timeout"]


def test_supervise_no_progress_timeout_skips_compute_bound_with_cpu_growth(monkeypatch):
    m = _load_module()
    job = m.BenchJob(name="mini", manifest="bench/mini.yaml", output_root="bench_runs/mini", pid=111)

    monkeypatch.setattr(
        m,
        "_load_supervisor_state",
        lambda _path=None: {"jobs": {"mini": {"last_done": 2, "no_progress_cycles": 0, "last_progress_ts": 0.0}}},
    )
    monkeypatch.setattr(m.time, "time", lambda: 120.0)
    monkeypatch.setattr(m.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        m,
        "_collect_diagnose_rows",
        lambda _jobs, stall_seconds=0: (
            [
                {
                    "job": job,
                    "done": 2,
                    "total": 10,
                    "pct": 20.0,
                    "out_age": 3600,
                    "err_age": 3600,
                    "child_count": 1,
                    "metrics": {"child_cpu_delta_s": 7.0},
                    "diagnosis": m.JobDiagnosis(
                        code="running_compute_bound",
                        summary="running_compute_bound",
                        recoverable=False,
                    ),
                }
            ],
            {"updated_at_utc": "2026-03-02T00:00:00+00:00", "jobs": {}},
        ),
    )
    monkeypatch.setattr(m, "_save_monitor_snapshot", lambda _path, _payload: None)
    monkeypatch.setattr(m, "_save_supervisor_state", lambda _path, _payload: None)
    monkeypatch.setattr(m, "_save_active_jobs", lambda _path, _jobs: None)
    monkeypatch.setattr(m, "_append_jsonl", lambda _path, _payload: None)

    called = {"recover": 0}

    def _fake_recover(_job, *, reason: str):
        called["recover"] += 1
        return _job, True

    monkeypatch.setattr(m, "_try_recover_job", _fake_recover)

    rc = m._supervise(
        [job],
        interval_seconds=1,
        stall_seconds=900,
        auto_recover=True,
        recover_after_consecutive=2,
        max_restarts_per_job=2,
        restart_cooldown_seconds=0,
        max_no_progress_cycles=1,
        max_no_progress_seconds=60,
        max_cycles=1,
    )

    assert rc == 0
    assert called["recover"] == 0


def test_supervise_no_progress_seconds_timeout_triggers_recovery(monkeypatch):
    m = _load_module()
    job = m.BenchJob(name="mini", manifest="bench/mini.yaml", output_root="bench_runs/mini", pid=111)

    monkeypatch.setattr(
        m,
        "_load_supervisor_state",
        lambda _path=None: {"jobs": {"mini": {"last_done": 2, "no_progress_cycles": 0, "last_progress_ts": 0.0}}},
    )
    monkeypatch.setattr(m.time, "time", lambda: 120.0)
    monkeypatch.setattr(m.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        m,
        "_collect_diagnose_rows",
        lambda _jobs, stall_seconds=0: (
            [
                {
                    "job": job,
                    "done": 2,
                    "total": 10,
                    "pct": 20.0,
                    "out_age": 3600,
                    "err_age": 3600,
                    "child_count": 0,
                    "diagnosis": m.JobDiagnosis(
                        code="running_silent",
                        summary="running_silent",
                        recoverable=False,
                    ),
                }
            ],
            {"updated_at_utc": "2026-03-02T00:00:00+00:00", "jobs": {}},
        ),
    )
    monkeypatch.setattr(m, "_save_monitor_snapshot", lambda _path, _payload: None)
    monkeypatch.setattr(m, "_save_supervisor_state", lambda _path, _payload: None)
    monkeypatch.setattr(m, "_save_active_jobs", lambda _path, _jobs: None)
    monkeypatch.setattr(m, "_append_jsonl", lambda _path, _payload: None)

    reasons: list[str] = []

    def _fake_recover(_job, *, reason: str):
        reasons.append(reason)
        return _job, True

    monkeypatch.setattr(m, "_try_recover_job", _fake_recover)

    rc = m._supervise(
        [job],
        interval_seconds=1,
        stall_seconds=900,
        auto_recover=True,
        recover_after_consecutive=2,
        max_restarts_per_job=2,
        restart_cooldown_seconds=0,
        max_no_progress_cycles=0,
        max_no_progress_seconds=60,
        max_cycles=1,
    )

    assert rc == 1
    assert reasons == ["no_progress_timeout_seconds"]


def test_supervise_respects_reason_restart_budget(monkeypatch):
    m = _load_module()
    job = m.BenchJob(name="mini", manifest="bench/mini.yaml", output_root="bench_runs/mini", pid=111)

    monkeypatch.setattr(
        m,
        "_load_supervisor_state",
        lambda _path=None: {"jobs": {"mini": {"restarts_by_reason": {"possible_stall": 2}}}},
    )
    monkeypatch.setattr(m.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        m,
        "_collect_diagnose_rows",
        lambda _jobs, stall_seconds=0: (
            [
                {
                    "job": job,
                    "done": 2,
                    "total": 10,
                    "pct": 20.0,
                    "diagnosis": m.JobDiagnosis(
                        code="possible_stall",
                        summary="possible_stall",
                        recoverable=True,
                    ),
                }
            ],
            {"updated_at_utc": "2026-03-02T00:00:00+00:00", "jobs": {}},
        ),
    )
    monkeypatch.setattr(m, "_save_monitor_snapshot", lambda _path, _payload: None)
    monkeypatch.setattr(m, "_save_supervisor_state", lambda _path, _payload: None)
    monkeypatch.setattr(m, "_save_active_jobs", lambda _path, _jobs: None)
    monkeypatch.setattr(m, "_append_jsonl", lambda _path, _payload: None)

    called = {"recover": 0}

    def _fake_recover(_job, *, reason: str):
        called["recover"] += 1
        return _job, True

    monkeypatch.setattr(m, "_try_recover_job", _fake_recover)

    rc = m._supervise(
        [job],
        interval_seconds=1,
        stall_seconds=900,
        auto_recover=True,
        recover_after_consecutive=1,
        max_restarts_per_job=5,
        max_restarts_per_reason=2,
        restart_cooldown_seconds=0,
        max_no_progress_cycles=0,
        max_no_progress_seconds=0,
        max_cycles=1,
    )

    assert rc == 1
    assert called["recover"] == 0


def test_supervise_respects_signature_restart_budget(monkeypatch):
    m = _load_module()
    job = m.BenchJob(name="mini", manifest="bench/mini.yaml", output_root="bench_runs/mini", pid=111)

    monkeypatch.setattr(
        m,
        "_load_supervisor_state",
        lambda _path=None: {"jobs": {"mini": {"restarts_by_signature": {"fortran_window_close": 2}}}},
    )
    monkeypatch.setattr(m.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        m,
        "_collect_diagnose_rows",
        lambda _jobs, stall_seconds=0: (
            [
                {
                    "job": job,
                    "done": 2,
                    "total": 10,
                    "pct": 20.0,
                    "signature": "fortran_window_close",
                    "diagnosis": m.JobDiagnosis(
                        code="possible_stall",
                        summary="possible_stall",
                        recoverable=True,
                    ),
                }
            ],
            {"updated_at_utc": "2026-03-02T00:00:00+00:00", "jobs": {}},
        ),
    )
    monkeypatch.setattr(m, "_save_monitor_snapshot", lambda _path, _payload: None)
    monkeypatch.setattr(m, "_save_supervisor_state", lambda _path, _payload: None)
    monkeypatch.setattr(m, "_save_active_jobs", lambda _path, _jobs: None)
    monkeypatch.setattr(m, "_append_jsonl", lambda _path, _payload: None)

    called = {"recover": 0}

    def _fake_recover(_job, *, reason: str):
        called["recover"] += 1
        return _job, True

    monkeypatch.setattr(m, "_try_recover_job", _fake_recover)

    rc = m._supervise(
        [job],
        interval_seconds=1,
        stall_seconds=900,
        auto_recover=True,
        recover_after_consecutive=1,
        max_restarts_per_job=5,
        max_restarts_per_reason=5,
        max_restarts_per_signature=2,
        restart_cooldown_seconds=0,
        max_no_progress_cycles=0,
        max_no_progress_seconds=0,
        max_cycles=1,
    )

    assert rc == 1
    assert called["recover"] == 0


def test_supervise_respects_window_restart_budget(monkeypatch):
    m = _load_module()
    job = m.BenchJob(name="mini", manifest="bench/mini.yaml", output_root="bench_runs/mini", pid=111)

    monkeypatch.setattr(
        m,
        "_load_supervisor_state",
        lambda _path=None: {"jobs": {"mini": {"restart_times_s": [50.0, 90.0]}}},
    )
    monkeypatch.setattr(m.time, "time", lambda: 100.0)
    monkeypatch.setattr(m.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        m,
        "_collect_diagnose_rows",
        lambda _jobs, stall_seconds=0: (
            [
                {
                    "job": job,
                    "done": 2,
                    "total": 10,
                    "pct": 20.0,
                    "diagnosis": m.JobDiagnosis(
                        code="possible_stall",
                        summary="possible_stall",
                        recoverable=True,
                    ),
                }
            ],
            {"updated_at_utc": "2026-03-02T00:00:00+00:00", "jobs": {}},
        ),
    )
    monkeypatch.setattr(m, "_save_monitor_snapshot", lambda _path, _payload: None)
    monkeypatch.setattr(m, "_save_supervisor_state", lambda _path, _payload: None)
    monkeypatch.setattr(m, "_save_active_jobs", lambda _path, _jobs: None)
    monkeypatch.setattr(m, "_append_jsonl", lambda _path, _payload: None)

    called = {"recover": 0}

    def _fake_recover(_job, *, reason: str):
        called["recover"] += 1
        return _job, True

    monkeypatch.setattr(m, "_try_recover_job", _fake_recover)

    rc = m._supervise(
        [job],
        interval_seconds=1,
        stall_seconds=900,
        auto_recover=True,
        recover_after_consecutive=1,
        max_restarts_per_job=10,
        max_restarts_per_reason=10,
        max_restarts_per_window=2,
        restart_window_seconds=60,
        restart_cooldown_seconds=0,
        max_no_progress_cycles=0,
        max_no_progress_seconds=0,
        max_cycles=1,
    )

    assert rc == 1
    assert called["recover"] == 0


def test_supervise_suppresses_recovery_during_grace_window(monkeypatch):
    m = _load_module()
    job = m.BenchJob(name="mini", manifest="bench/mini.yaml", output_root="bench_runs/mini", pid=111)

    monkeypatch.setattr(
        m,
        "_load_supervisor_state",
        lambda _path=None: {"jobs": {"mini": {"last_restart_ts": 90.0, "restart_count": 1}}},
    )
    monkeypatch.setattr(m.time, "time", lambda: 100.0)
    monkeypatch.setattr(m.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        m,
        "_collect_diagnose_rows",
        lambda _jobs, stall_seconds=0: (
            [
                {
                    "job": job,
                    "done": 2,
                    "total": 10,
                    "pct": 20.0,
                    "diagnosis": m.JobDiagnosis(
                        code="possible_stall_low_cpu",
                        summary="possible_stall_low_cpu",
                        recoverable=True,
                    ),
                }
            ],
            {"updated_at_utc": "2026-03-02T00:00:00+00:00", "jobs": {}},
        ),
    )
    monkeypatch.setattr(m, "_save_monitor_snapshot", lambda _path, _payload: None)
    monkeypatch.setattr(m, "_save_supervisor_state", lambda _path, _payload: None)
    monkeypatch.setattr(m, "_save_active_jobs", lambda _path, _jobs: None)
    monkeypatch.setattr(m, "_append_jsonl", lambda _path, _payload: None)

    called = {"recover": 0}

    def _fake_recover(_job, *, reason: str):
        called["recover"] += 1
        return _job, True

    monkeypatch.setattr(m, "_try_recover_job", _fake_recover)

    rc = m._supervise(
        [job],
        interval_seconds=1,
        stall_seconds=900,
        auto_recover=True,
        recover_after_consecutive=1,
        max_restarts_per_job=3,
        max_restarts_per_reason=2,
        max_restarts_per_window=3,
        restart_window_seconds=3600,
        restart_cooldown_seconds=0,
        restart_backoff_factor=1.0,
        max_restart_cooldown_seconds=0,
        recovery_grace_seconds=300,
        max_no_progress_cycles=0,
        max_no_progress_seconds=0,
        max_cycles=1,
    )

    assert rc == 0
    assert called["recover"] == 0


def test_supervise_respects_total_restart_budget(monkeypatch):
    m = _load_module()
    job = m.BenchJob(name="mini", manifest="bench/mini.yaml", output_root="bench_runs/mini", pid=111)

    monkeypatch.setattr(
        m,
        "_load_supervisor_state",
        lambda _path=None: {"jobs": {"mini": {"restart_count": 3}}},
    )
    monkeypatch.setattr(m.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        m,
        "_collect_diagnose_rows",
        lambda _jobs, stall_seconds=0: (
            [
                {
                    "job": job,
                    "done": 2,
                    "total": 10,
                    "pct": 20.0,
                    "diagnosis": m.JobDiagnosis(
                        code="possible_stall",
                        summary="possible_stall",
                        recoverable=True,
                    ),
                }
            ],
            {"updated_at_utc": "2026-03-02T00:00:00+00:00", "jobs": {}},
        ),
    )
    monkeypatch.setattr(m, "_save_monitor_snapshot", lambda _path, _payload: None)
    monkeypatch.setattr(m, "_save_supervisor_state", lambda _path, _payload: None)
    monkeypatch.setattr(m, "_save_active_jobs", lambda _path, _jobs: None)
    monkeypatch.setattr(m, "_append_jsonl", lambda _path, _payload: None)

    called = {"recover": 0}

    def _fake_recover(_job, *, reason: str):
        called["recover"] += 1
        return _job, True

    monkeypatch.setattr(m, "_try_recover_job", _fake_recover)

    rc = m._supervise(
        [job],
        interval_seconds=1,
        stall_seconds=900,
        auto_recover=True,
        recover_after_consecutive=1,
        max_restarts_per_job=10,
        max_restarts_per_reason=10,
        max_total_restarts=3,
        restart_cooldown_seconds=0,
        max_no_progress_cycles=0,
        max_no_progress_seconds=0,
        max_cycles=1,
    )

    assert rc == 1
    assert called["recover"] == 0


def test_autopilot_runs_readiness_when_jobs_complete(monkeypatch):
    m = _load_module()
    jobs = [m.BenchJob(name="mini", manifest="bench/mini.yaml", output_root="bench_runs/mini")]

    calls: list[str] = []
    monkeypatch.setattr(m, "_run_preflight", lambda _jobs: calls.append("preflight") or 0)
    monkeypatch.setattr(m, "_launch_jobs", lambda _jobs, force_restart=False: calls.append("launch") or _jobs)
    monkeypatch.setattr(m, "_print_status", lambda _jobs, **_kwargs: calls.append("status"))
    monkeypatch.setattr(m, "_supervise", lambda *_args, **_kwargs: calls.append("supervise") or 0)
    monkeypatch.setattr(m, "_all_jobs_completed", lambda _jobs: True)
    monkeypatch.setattr(m, "_strict_readiness_check", lambda: calls.append("strict") or 0)
    monkeypatch.setattr(m, "_finalize", lambda _jobs: calls.append("finalize") or 0)
    monkeypatch.setattr(m, "_save_active_jobs", lambda _path, _jobs: None)

    rc = m._autopilot(
        jobs,
        do_preflight=True,
        force_restart=False,
        interval_seconds=1,
        stall_seconds=900,
        auto_recover=True,
        recover_after_consecutive=2,
        max_restarts_per_job=2,
        restart_cooldown_seconds=0,
        max_cycles=1,
        auto_finalize=False,
    )

    assert rc == 0
    assert calls == ["preflight", "launch", "status", "supervise", "strict"]


def test_autopilot_runs_finalize_when_enabled(monkeypatch):
    m = _load_module()
    jobs = [m.BenchJob(name="mini", manifest="bench/mini.yaml", output_root="bench_runs/mini")]

    calls: list[str] = []
    monkeypatch.setattr(m, "_run_preflight", lambda _jobs: 0)
    monkeypatch.setattr(m, "_launch_jobs", lambda _jobs, force_restart=False: _jobs)
    monkeypatch.setattr(m, "_print_status", lambda _jobs, **_kwargs: None)
    monkeypatch.setattr(m, "_supervise", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(m, "_all_jobs_completed", lambda _jobs: True)
    monkeypatch.setattr(m, "_strict_readiness_check", lambda: calls.append("strict") or 0)
    monkeypatch.setattr(m, "_finalize", lambda _jobs: calls.append("finalize") or 0)
    monkeypatch.setattr(m, "_save_active_jobs", lambda _path, _jobs: None)

    rc = m._autopilot(
        jobs,
        do_preflight=False,
        force_restart=False,
        interval_seconds=1,
        stall_seconds=900,
        auto_recover=True,
        recover_after_consecutive=2,
        max_restarts_per_job=2,
        restart_cooldown_seconds=0,
        max_cycles=1,
        auto_finalize=True,
    )

    assert rc == 0
    assert calls == ["finalize"]


def test_load_recent_events_parses_jsonl(tmp_path: Path):
    m = _load_module()
    path = tmp_path / "events.jsonl"
    path.write_text(
        '{"event":"diagnosis","job":"a"}\n'
        "not-json\n"
        '{"event":"recovered","job":"a"}\n',
        encoding="utf-8",
    )

    events = m._load_recent_events(path, max_lines=10)

    assert len(events) == 2
    assert events[0]["event"] == "diagnosis"
    assert events[1]["event"] == "recovered"


def test_summarize_supervisor_events_counts():
    m = _load_module()
    events = [
        {"event": "diagnosis", "job": "a"},
        {"event": "recovery_failed", "job": "a"},
        {"event": "recovery_failed", "job": "b"},
        {"event": "diagnosis"},
    ]

    counts, by_job = m._summarize_supervisor_events(events)

    assert counts["diagnosis"] == 2
    assert counts["recovery_failed"] == 2
    assert by_job["a"]["diagnosis"] == 1
    assert by_job["a"]["recovery_failed"] == 1
    assert by_job["b"]["recovery_failed"] == 1


def test_build_recommendation_finalize_when_all_complete():
    m = _load_module()
    job = m.BenchJob(name="mini", manifest="bench/mini.yaml", output_root="bench_runs/mini")
    rows = [
        {
            "job": job,
            "done": 10,
            "total": 10,
            "diagnosis": m.JobDiagnosis(code="ok", summary="no blocker", recoverable=False),
        }
    ]

    rec = m._build_recommendation(rows, state_jobs={}, now_ts=100.0)

    assert rec.action == "finalize"
    assert "mode finalize" in rec.command


def test_build_recommendation_completed_with_runtime_failures_prefers_rerun():
    m = _load_module()
    job = m.BenchJob(
        name="mini",
        manifest="bench/mini.yaml",
        output_root="bench_runs/mini",
        protocol="paper",
    )
    rows = [
        {
            "job": job,
            "done": 10,
            "total": 10,
            "progress": {
                "counts": {"passed": 8, "failed": 2, "skipped": 0},
                "bench_status": "failed",
                "failure_breakdown": {
                    "failed_total": 2,
                    "ci_gate_failures": 1,
                    "runtime_failures": 1,
                    "other_failures": 0,
                },
            },
            "diagnosis": m.JobDiagnosis(code="ok", summary="complete", recoverable=False),
        }
    ]

    rec = m._build_recommendation(rows, state_jobs={}, now_ts=100.0)

    assert rec.action == "rerun_runtime_failures"
    assert "python -m deltatau_audit bench run" in rec.command
    assert "--manifest bench/mini.yaml" in rec.command
    assert "--protocol paper" in rec.command


def test_build_recommendation_completed_with_ci_gate_failures_prefers_quality_action():
    m = _load_module()
    job = m.BenchJob(name="mini", manifest="bench/mini.yaml", output_root="bench_runs/mini")
    rows = [
        {
            "job": job,
            "done": 10,
            "total": 10,
            "progress": {
                "counts": {"passed": 8, "failed": 2, "skipped": 0},
                "bench_status": "failed",
                "failure_breakdown": {
                    "failed_total": 2,
                    "ci_gate_failures": 2,
                    "runtime_failures": 0,
                    "other_failures": 0,
                },
            },
            "diagnosis": m.JobDiagnosis(code="ok", summary="complete", recoverable=False),
        }
    ]

    rec = m._build_recommendation(rows, state_jobs={}, now_ts=100.0)

    assert rec.action == "improve_quality_gate_failures"
    assert "--strict-check" in rec.command
    assert "--mode report" in rec.command


def test_build_recommendation_cartpole_quality_failures_emits_targeted_repair_plan():
    m = _load_module()
    job = m.BenchJob(
        name="cartpole_high_rigor_bench",
        manifest="bench/high_rigor_10seed_manifest.yaml",
        output_root="bench_runs/cartpole_high_rigor_10seed",
        protocol="paper",
    )
    rows = [
        {
            "job": job,
            "done": 50,
            "total": 50,
            "progress": {
                "counts": {"passed": 45, "failed": 5, "skipped": 0},
                "bench_status": "failed",
                "failure_breakdown": {
                    "failed_total": 5,
                    "ci_gate_failures": 5,
                    "runtime_failures": 0,
                    "other_failures": 0,
                    "failed_job_ids": [
                        "cartpole_intervention1_plus_2_seed-1",
                        "cartpole_intervention1_plus_2_seed-6",
                        "cartpole_intervention1_plus_2_seed-7",
                        "cartpole_intervention2_time_feature_seed-3",
                        "cartpole_intervention2_time_feature_seed-4",
                    ],
                    "ci_gate_summary_paths": [
                        "C:/tmp/cartpole/intervention1_plus_2/seed_1/summary.json",
                        "C:/tmp/cartpole/intervention2_time_feature/seed_3/summary.json",
                    ],
                },
            },
            "diagnosis": m.JobDiagnosis(code="ok", summary="complete", recoverable=False),
        }
    ]

    rec = m._build_recommendation(rows, state_jobs={}, now_ts=100.0)

    assert rec.action == "improve_quality_gate_failures"
    assert "stress train-sb3" in rec.command
    assert "--variants intervention1_plus_2 --seeds 1 6 7 --timesteps 45000 --force" in rec.command
    assert "--variants intervention2_time_feature --seeds 3 4 --timesteps 45000 --force" in rec.command
    assert "python -c " in rec.command
    assert "python scripts/build_failed_job_manifest.py" in rec.command
    assert "--output-root bench_runs/cartpole_high_rigor_10seed" in rec.command
    assert "--out-manifest _status_demo/repair_manifests/cartpole_high_rigor_bench.yaml" in rec.command
    assert "--protocol paper" in rec.command
    assert "--no-resume" not in rec.command
    assert any("failed cells" in reason for reason in rec.reasons)


def test_build_recommendation_widespread_cartpole_quality_failures_prefers_diagnosis():
    m = _load_module()
    job = m.BenchJob(
        name="cartpole_high_rigor_bench",
        manifest="bench/high_rigor_10seed_manifest.yaml",
        output_root="bench_runs/cartpole_high_rigor_10seed_fresh_20260317",
        protocol="paper",
    )
    rows = [
        {
            "job": job,
            "done": 50,
            "total": 50,
            "progress": {
                "counts": {"passed": 0, "failed": 50, "skipped": 0},
                "bench_status": "failed",
                "failure_breakdown": {
                    "failed_total": 50,
                    "ci_gate_failures": 50,
                    "runtime_failures": 0,
                    "other_failures": 0,
                    "failed_job_ids": [
                        "cartpole_baseline_seed-0",
                        "cartpole_baseline_seed-1",
                        "cartpole_intervention1_curriculum_seed-0",
                        "cartpole_intervention1_plus_2_seed-0",
                        "cartpole_intervention2_time_feature_seed-0",
                        "cartpole_intervention3_memory_seed-0",
                    ],
                },
            },
            "diagnosis": m.JobDiagnosis(code="ok", summary="complete", recoverable=False),
        }
    ]

    rec = m._build_recommendation(rows, state_jobs={}, now_ts=100.0)

    assert rec.action == "improve_quality_gate_failures"
    assert "python scripts/analyze_bench_failures.py" in rec.command
    assert "stress train-sb3" not in rec.command
    assert any("diagnose protocol/claim mismatch" in reason for reason in rec.reasons)


def test_build_recommendation_quality_failures_without_summary_paths_falls_back_no_resume():
    m = _load_module()
    job = m.BenchJob(name="mini", manifest="bench/mini.yaml", output_root="bench_runs/mini")
    rows = [
        {
            "job": job,
            "done": 10,
            "total": 10,
            "progress": {
                "counts": {"passed": 8, "failed": 2, "skipped": 0},
                "bench_status": "failed",
                "failure_breakdown": {
                    "failed_total": 2,
                    "ci_gate_failures": 2,
                    "runtime_failures": 0,
                    "other_failures": 0,
                    "failed_job_ids": ["mini_seed-1", "mini_seed-4"],
                },
            },
            "diagnosis": m.JobDiagnosis(code="ok", summary="complete", recoverable=False),
        }
    ]

    rec = m._build_recommendation(rows, state_jobs={}, now_ts=100.0)

    assert rec.action == "improve_quality_gate_failures"
    assert "python scripts/build_failed_job_manifest.py" in rec.command
    assert "--output-root bench_runs/mini" in rec.command
    assert "--out-manifest _status_demo/repair_manifests/mini.yaml" in rec.command


def test_build_recommendation_incomplete_jobs_with_completed_quality_failures_prefers_parallel_repair():
    m = _load_module()
    cartpole_job = m.BenchJob(
        name="cartpole_high_rigor_bench",
        manifest="bench/high_rigor_10seed_manifest.yaml",
        output_root="bench_runs/cartpole_high_rigor_10seed",
        protocol="paper",
    )
    dm_job = m.BenchJob(
        name="dm_control_bench",
        manifest="bench/dm_control_research_manifest.yaml",
        output_root="bench_runs/dm_control",
        protocol="paper",
    )
    rows = [
        {
            "job": cartpole_job,
            "done": 50,
            "total": 50,
            "progress": {
                "counts": {"passed": 45, "failed": 5, "skipped": 0},
                "bench_status": "failed",
                "failure_breakdown": {
                    "failed_total": 5,
                    "ci_gate_failures": 5,
                    "runtime_failures": 0,
                    "other_failures": 0,
                    "failed_job_ids": [
                        "cartpole_intervention1_plus_2_seed-1",
                        "cartpole_intervention1_plus_2_seed-6",
                        "cartpole_intervention2_time_feature_seed-4",
                    ],
                    "ci_gate_summary_paths": [
                        "C:/tmp/cartpole/intervention1_plus_2/seed_1/summary.json",
                        "C:/tmp/cartpole/intervention2_time_feature/seed_4/summary.json",
                    ],
                },
            },
            "diagnosis": m.JobDiagnosis(code="done_quality", summary="quality failures", recoverable=False),
        },
        {
            "job": dm_job,
            "done": 6,
            "total": 8,
            "diagnosis": m.JobDiagnosis(code="running_compute_bound", summary="running", recoverable=False),
        },
    ]

    rec = m._build_recommendation(rows, state_jobs={}, now_ts=100.0)

    assert rec.action == "parallelize_quality_repairs"
    assert "stress train-sb3" in rec.command
    assert "--force" in rec.command
    assert "python scripts/build_failed_job_manifest.py" in rec.command
    assert "--output-root bench_runs/cartpole_high_rigor_10seed" in rec.command


def test_build_recommendation_recoverable_prefers_auto_recover():
    m = _load_module()
    job = m.BenchJob(name="mini", manifest="bench/mini.yaml", output_root="bench_runs/mini")
    rows = [
        {
            "job": job,
            "done": 2,
            "total": 10,
            "diagnosis": m.JobDiagnosis(code="possible_stall", summary="stall", recoverable=True),
        }
    ]

    rec = m._build_recommendation(rows, state_jobs={}, now_ts=100.0)

    assert rec.action == "supervise_auto_recover"
    assert "--auto-recover" in rec.command
    assert "--max-restarts-per-reason 2" in rec.command
    assert "--max-total-restarts 6" in rec.command
    assert "--max-restarts-per-signature 2" in rec.command
    assert "--max-restarts-per-window 3" in rec.command
    assert "--restart-window-seconds 10800" in rec.command
    assert "--recovery-grace-seconds 900" in rec.command


def test_build_recommendation_recoverable_with_failed_restarts_prefers_investigation():
    m = _load_module()
    job = m.BenchJob(name="mini", manifest="bench/mini.yaml", output_root="bench_runs/mini")
    rows = [
        {
            "job": job,
            "done": 2,
            "total": 10,
            "diagnosis": m.JobDiagnosis(code="possible_stall", summary="stall", recoverable=True),
        }
    ]
    recent_events = [
        {"event": "recovery_failed", "job": "mini"},
        {"event": "recovery_failed", "job": "mini"},
        {"event": "recovery_failed", "job": "mini"},
    ]

    rec = m._build_recommendation(
        rows,
        state_jobs={},
        recent_events=recent_events,
        now_ts=100.0,
    )

    assert rec.action == "investigate_recovery_failures"
    assert "--mode report" in rec.command
    assert "--mode diagnose" in rec.command


def test_build_recommendation_recoverable_with_signature_loop_prefers_investigation():
    m = _load_module()
    job = m.BenchJob(name="mini", manifest="bench/mini.yaml", output_root="bench_runs/mini")
    rows = [
        {
            "job": job,
            "done": 2,
            "total": 10,
            "diagnosis": m.JobDiagnosis(code="possible_stall", summary="stall", recoverable=True),
        }
    ]
    state_jobs = {"mini": {"last_signature": "fortran_window_close", "consecutive_signature_hits": 3}}

    rec = m._build_recommendation(
        rows,
        state_jobs=state_jobs,
        now_ts=100.0,
    )

    assert rec.action == "investigate_signature_loop"
    assert "--mode report" in rec.command
    assert "--mode diagnose" in rec.command


def test_build_recommendation_no_progress_seconds_prefers_timeout():
    m = _load_module()
    job = m.BenchJob(name="mini", manifest="bench/mini.yaml", output_root="bench_runs/mini")
    rows = [
        {
            "job": job,
            "done": 2,
            "total": 10,
            "diagnosis": m.JobDiagnosis(code="running_compute_bound", summary="compute bound", recoverable=False),
        }
    ]
    state_jobs = {"mini": {"last_done": 2, "last_progress_ts": 0.0}}

    rec = m._build_recommendation(rows, state_jobs=state_jobs, now_ts=8000.0)

    assert rec.action == "supervise_with_timeout"
    assert "--max-no-progress-seconds 7200" in rec.command


def test_build_recommendation_uses_projected_progress():
    m = _load_module()
    job = m.BenchJob(name="mini", manifest="bench/mini.yaml", output_root="bench_runs/mini")
    rows = [
        {
            "job": job,
            "done": 3,
            "total": 10,
            "diagnosis": m.JobDiagnosis(code="running", summary="running", recoverable=False),
        }
    ]
    state_jobs = {"mini": {"last_done": 2, "last_progress_ts": 0.0, "no_progress_cycles": 10}}

    rec = m._build_recommendation(rows, state_jobs=state_jobs, now_ts=8000.0)

    assert rec.action == "continue_supervision"


def test_try_recover_job_forces_resume_when_partial_progress(monkeypatch):
    m = _load_module()
    job = m.BenchJob(
        name="mini",
        manifest="bench/mini.yaml",
        output_root="bench_runs/mini",
        protocol="paper",
        no_resume=True,
        pid=123,
    )

    monkeypatch.setattr(m, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(
        m,
        "_job_progress",
        lambda _job: {
            "total": 10,
            "done": 6,
        },
    )
    launch_args: list[bool | None] = []

    def _fake_launch(_job, *, force_restart: bool, no_resume_override=None):
        assert force_restart is True
        launch_args.append(no_resume_override)
        _job.pid = 456
        return _job

    monkeypatch.setattr(m, "_launch_job", _fake_launch)

    relaunched, restarted = m._try_recover_job(job, reason="blocked_dead")

    assert restarted is True
    assert relaunched.pid == 456
    assert launch_args == [False]


def test_print_recommendation_execute_runs_shell_command(monkeypatch):
    m = _load_module()
    job = m.BenchJob(name="mini", manifest="bench/mini.yaml", output_root="bench_runs/mini")

    monkeypatch.setattr(
        m,
        "_collect_diagnose_rows",
        lambda _jobs, stall_seconds=0: (
            [{"job": job, "done": 1, "total": 1, "diagnosis": m.JobDiagnosis(code="ok", summary="ok", recoverable=False)}],
            {},
        ),
    )
    monkeypatch.setattr(m, "_load_supervisor_state", lambda _path=None: {"jobs": {}})
    monkeypatch.setattr(m, "_load_recent_events", lambda _path, max_lines=500: [])
    monkeypatch.setattr(m, "_summarize_supervisor_events", lambda _events: ({}, {}))
    monkeypatch.setattr(
        m,
        "_build_recommendation",
        lambda *_args, **_kwargs: m.Recommendation(action="x", command="echo hello", reasons=["r1"]),
    )

    called: list[str] = []
    monkeypatch.setattr(m, "_run_shell_cmd", lambda cmd: called.append(cmd) or 7)

    rc = m._print_recommendation([job], stall_seconds=900, execute=True)

    assert rc == 7
    assert called == ["echo hello"]


def test_print_recommendation_without_execute_is_non_destructive(monkeypatch):
    m = _load_module()
    job = m.BenchJob(name="mini", manifest="bench/mini.yaml", output_root="bench_runs/mini")

    monkeypatch.setattr(
        m,
        "_collect_diagnose_rows",
        lambda _jobs, stall_seconds=0: (
            [{"job": job, "done": 1, "total": 1, "diagnosis": m.JobDiagnosis(code="ok", summary="ok", recoverable=False)}],
            {},
        ),
    )
    monkeypatch.setattr(m, "_load_supervisor_state", lambda _path=None: {"jobs": {}})
    monkeypatch.setattr(m, "_load_recent_events", lambda _path, max_lines=500: [])
    monkeypatch.setattr(m, "_summarize_supervisor_events", lambda _events: ({}, {}))
    monkeypatch.setattr(
        m,
        "_build_recommendation",
        lambda *_args, **_kwargs: m.Recommendation(action="x", command="echo hello", reasons=[]),
    )

    called = {"run": 0}
    monkeypatch.setattr(m, "_run_shell_cmd", lambda _cmd: called.__setitem__("run", called["run"] + 1) or 0)

    rc = m._print_recommendation([job], stall_seconds=900, execute=False)

    assert rc == 0
    assert called["run"] == 0
