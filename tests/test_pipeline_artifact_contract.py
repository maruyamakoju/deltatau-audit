from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_submission_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_submission_pipeline_contract", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def _load_golden_text(name: str) -> str:
    return (_GOLDEN_DIR / name).read_text(encoding="utf-8")


def _load_golden_json(name: str):
    return json.loads(_load_golden_text(name))


def test_pipeline_state_and_event_artifacts_match_contract(tmp_path: Path, monkeypatch):
    m = _load_module()
    tmp_path.mkdir(parents=True, exist_ok=True)

    active_path = tmp_path / "active_jobs.json"
    snapshot_path = tmp_path / "monitor_snapshot.json"
    supervisor_state_path = tmp_path / "supervisor_state.json"
    supervisor_events_path = tmp_path / "supervisor_events.jsonl"

    monkeypatch.setattr(m, "ACTIVE_PATH", active_path)
    monkeypatch.setattr(m, "SNAPSHOT_PATH", snapshot_path)
    monkeypatch.setattr(m, "SUPERVISOR_STATE_PATH", supervisor_state_path)
    monkeypatch.setattr(m, "SUPERVISOR_EVENTS_PATH", supervisor_events_path)
    monkeypatch.setattr(m, "_utc_now", lambda: "2026-03-02T00:00:00+00:00")
    monkeypatch.setattr(m.time, "sleep", lambda _s: None)
    monkeypatch.setattr(m.time, "time", lambda: 100.0)

    job = m.BenchJob(
        name="mini",
        manifest="bench/mini.yaml",
        output_root="bench_runs/mini",
        protocol="paper",
        no_resume=False,
        out_log="_status_demo/long_runs/mini.out.log",
        err_log="_status_demo/long_runs/mini.err.log",
        pid=111,
        started_at_utc="2026-03-01T23:59:00+00:00",
    )

    m._save_active_jobs(active_path, [job])

    monkeypatch.setattr(m, "_load_supervisor_state", lambda _path=None: {"jobs": {}})
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
                    "metrics": {},
                    "signature": None,
                }
            ],
            {
                "updated_at_utc": "2026-03-02T00:00:00+00:00",
                "jobs": {
                    "mini": {
                        "timestamp_utc": "2026-03-02T00:00:00+00:00",
                        "timestamp_s": 100.0,
                        "done": 2,
                        "total": 10,
                        "pct": 20.0,
                        "child_cpu_s_total": None,
                    }
                },
            },
        ),
    )

    rc = m._supervise(
        [job],
        interval_seconds=1,
        stall_seconds=900,
        auto_recover=False,
        recover_after_consecutive=2,
        max_restarts_per_job=2,
        max_cycles=1,
        events_path=supervisor_events_path,
    )

    assert rc == 0
    assert json.loads(active_path.read_text(encoding="utf-8")) == _load_golden_json(
        "pipeline_active_jobs_contract.json"
    )
    assert json.loads(snapshot_path.read_text(encoding="utf-8")) == _load_golden_json(
        "pipeline_monitor_snapshot_contract.json"
    )
    assert json.loads(supervisor_state_path.read_text(encoding="utf-8")) == _load_golden_json(
        "pipeline_supervisor_state_contract.json"
    )
    assert supervisor_events_path.read_text(encoding="utf-8") == _load_golden_text(
        "pipeline_supervisor_events_contract.jsonl"
    )
