#!/usr/bin/env python3
"""Preflight, launch, monitor, and finalize long-running submission benchmark jobs."""

from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
from submission_health import (
    bench_failure_breakdown as _shared_bench_failure_breakdown,
    bench_counts as _shared_bench_counts,
    build_quality_repair_plan as _shared_build_quality_repair_plan,
    cartpole_retrain_commands as _shared_cartpole_retrain_commands,
    expand_manifest_jobs as _shared_expand_manifest_jobs,
    load_manifest as _shared_load_manifest,
    repair_plan_commands as _shared_repair_plan_commands,
    summary_cleanup_commands as _shared_summary_cleanup_commands,
    summary_targets_from_manifest as _shared_summary_targets_from_manifest,
)

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None

STATUS_DIR = ROOT / "_status_demo" / "long_runs"
ACTIVE_PATH = STATUS_DIR / "active_jobs.json"
SNAPSHOT_PATH = STATUS_DIR / "monitor_snapshot.json"
SUPERVISOR_STATE_PATH = STATUS_DIR / "supervisor_state.json"
SUPERVISOR_EVENTS_PATH = STATUS_DIR / "supervisor_events.jsonl"
MAX_RESTART_HISTORY_ITEMS = 512
RECOMMENDED_MAX_RESTARTS_PER_WINDOW = 3
RECOMMENDED_RESTART_WINDOW_SECONDS = 10800
RECOMMENDED_RECOVERY_GRACE_SECONDS = 900
RECOMMENDED_MAX_RESTARTS_PER_SIGNATURE = 2
RUNNING_DIAGNOSIS_CODES = frozenset({"running", "running_silent", "running_compute_bound"})


@dataclass
class BenchJob:
    name: str
    manifest: str
    output_root: str
    protocol: str = "paper"
    no_resume: bool = False
    out_log: str = ""
    err_log: str = ""
    pid: int | None = None
    started_at_utc: str | None = None


@dataclass
class JobDiagnosis:
    code: str
    summary: str
    recoverable: bool


@dataclass
class Recommendation:
    action: str
    command: str
    reasons: list[str]


@dataclass
class RecoveryDecision:
    allow: bool
    skip_event: str = ""
    skip_message: str = ""
    skip_payload: dict[str, Any] | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_utc(value: str | None) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _default_jobs() -> list[BenchJob]:
    return [
        BenchJob(
            name="cartpole_high_rigor_bench",
            manifest="bench/high_rigor_10seed_manifest.yaml",
            output_root="bench_runs/cartpole_high_rigor_10seed",
            protocol="paper",
            no_resume=False,
            out_log="_status_demo/long_runs/cartpole_bench_resume.out.log",
            err_log="_status_demo/long_runs/cartpole_bench_resume.err.log",
        ),
        BenchJob(
            name="dm_control_bench",
            manifest="bench/dm_control_research_manifest.yaml",
            output_root="bench_runs/dm_control",
            protocol="paper",
            no_resume=True,
            out_log="_status_demo/long_runs/dm_control_bench_resume.out.log",
            err_log="_status_demo/long_runs/dm_control_bench_resume.err.log",
        ),
    ]


def _pid_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    if psutil is not None:
        try:
            return bool(psutil.pid_exists(int(pid)))
        except Exception:
            pass
    if os.name == "nt":
        try:
            proc = subprocess.run(
                ["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"],
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            text = (proc.stdout or "").strip()
            if not text:
                return False
            if text.lower().startswith("info:"):
                return False
            return f'"{int(pid)}"' in text
        except Exception:
            pass
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _load_active_jobs(path: Path) -> list[BenchJob]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        return []
    out: list[BenchJob] = []
    for raw in jobs:
        job = _job_from_raw(raw)
        if job is None:
            continue
        out.append(job)
    return out


def _to_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text and text.lstrip("+-").isdigit():
            return int(text)
    return None


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _normalize_signature(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip().lower()
    if not text:
        return ""
    if len(text) > 120:
        text = text[:120]
    return text


def _parse_legacy_command(raw_command: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    try:
        tokens = shlex.split(raw_command)
    except Exception:
        return fields

    for idx, token in enumerate(tokens):
        if token == "--manifest" and idx + 1 < len(tokens):
            fields["manifest"] = tokens[idx + 1]
        elif token == "--protocol" and idx + 1 < len(tokens):
            fields["protocol"] = tokens[idx + 1]
        elif token == "--no-resume":
            fields["no_resume"] = True
    return fields


def _job_from_raw(raw: Any) -> BenchJob | None:
    if not isinstance(raw, dict):
        return None

    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        return None

    command = raw.get("command")
    legacy_fields: dict[str, Any] = {}
    if isinstance(command, str) and command.strip():
        legacy_fields = _parse_legacy_command(command)

    manifest = raw.get("manifest", legacy_fields.get("manifest", ""))
    output_root = raw.get("output_root", "")
    protocol = raw.get("protocol", legacy_fields.get("protocol", "paper"))
    no_resume = bool(raw.get("no_resume", legacy_fields.get("no_resume", False)))

    if not isinstance(manifest, str):
        manifest = ""
    if not isinstance(output_root, str):
        output_root = ""
    if not isinstance(protocol, str) or not protocol:
        protocol = "paper"

    out_log = raw.get("out_log", "")
    err_log = raw.get("err_log", "")
    if not isinstance(out_log, str):
        out_log = ""
    if not isinstance(err_log, str):
        err_log = ""

    pid = _to_int(raw.get("pid"))
    started_at_utc = raw.get("started_at_utc")
    if not isinstance(started_at_utc, str):
        started_at_utc = None

    return BenchJob(
        name=name.strip(),
        manifest=manifest,
        output_root=output_root,
        protocol=protocol,
        no_resume=no_resume,
        out_log=out_log,
        err_log=err_log,
        pid=pid,
        started_at_utc=started_at_utc,
    )


def _save_active_jobs(path: Path, jobs: list[BenchJob]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": _utc_now(),
        "jobs": [asdict(job) for job in jobs],
    }
    _atomic_write_text(path, json.dumps(payload, indent=2), encoding="utf-8")


def _load_manifest(path: Path) -> dict[str, Any]:
    return _shared_load_manifest(path)


def _expand_manifest_jobs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return _shared_expand_manifest_jobs(manifest)


def _job_summary_targets(manifest_path: Path, output_root: Path) -> list[Path]:
    return _shared_summary_targets_from_manifest(manifest_path, output_root)


def _bench_counts(output_root: Path) -> tuple[dict[str, int] | None, datetime | None, str | None]:
    return _shared_bench_counts(output_root)


def _bench_failure_breakdown(output_root: Path) -> dict[str, Any]:
    return _shared_bench_failure_breakdown(output_root)


def _job_progress(job: BenchJob) -> dict[str, Any]:
    manifest_path = (ROOT / job.manifest).resolve()
    output_root = (ROOT / job.output_root).resolve()

    total = 0
    done = 0
    if manifest_path.exists():
        targets = _job_summary_targets(manifest_path, output_root)
        total = len(targets)
        done = sum(1 for path in targets if path.exists())

    counts, counts_updated_at, bench_status = _bench_counts(output_root)
    failure_breakdown = _bench_failure_breakdown(output_root)
    started_at = _parse_utc(job.started_at_utc)
    counts_stale = bool(
        counts is not None
        and counts_updated_at is not None
        and started_at is not None
        and counts_updated_at < started_at
    )
    pct = (100.0 * done / total) if total > 0 else 0.0
    log_health = _collect_log_health(output_root)
    process_health = _collect_process_health(job.pid)
    return {
        "total": total,
        "done": done,
        "pct": pct,
        "counts": counts,
        "bench_status": bench_status,
        "failure_breakdown": failure_breakdown,
        "counts_stale": counts_stale,
        "counts_updated_at_utc": counts_updated_at.isoformat() if counts_updated_at else None,
        "log_health": log_health,
        "process_health": process_health,
    }


def _read_tail_lines(path: Path, n_lines: int = 120) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fp:
            return list(deque(fp, maxlen=max(1, int(n_lines))))
    except Exception:
        return []


def _latest_log_file(output_root: Path, filename: str) -> Path | None:
    if not output_root.exists():
        return None
    latest: Path | None = None
    latest_mtime = -1.0
    for path in output_root.rglob(filename):
        try:
            mtime = path.stat().st_mtime
        except Exception:
            continue
        if mtime > latest_mtime:
            latest = path
            latest_mtime = mtime
    return latest


def _classify_error_signature(lines: list[str]) -> str | None:
    if not lines:
        return None
    text = "\n".join(lines).lower()
    if "forrtl: error (200)" in text:
        return "fortran_window_close"
    if "fatal python error: init_import_site" in text:
        return "python_site_init_failure"
    if "keyboardinterrupt" in text:
        return "keyboard_interrupt"
    if "modulenotfounderror" in text:
        return "module_not_found"
    if "filenotfounderror" in text:
        return "file_not_found"
    if "traceback (most recent call last)" in text:
        return "python_exception"
    return None


def _collect_log_health(output_root: Path) -> dict[str, Any]:
    now = time.time()
    latest_out = _latest_log_file(output_root, "bench_run.log")
    latest_err = _latest_log_file(output_root, "bench_run.err.log")

    out_age_s: int | None = None
    err_age_s: int | None = None
    if latest_out is not None:
        try:
            out_age_s = max(0, int(now - latest_out.stat().st_mtime))
        except Exception:
            out_age_s = None
    if latest_err is not None:
        try:
            err_age_s = max(0, int(now - latest_err.stat().st_mtime))
        except Exception:
            err_age_s = None

    err_tail = _read_tail_lines(latest_err, n_lines=160) if latest_err is not None else []
    signature = _classify_error_signature(err_tail)

    return {
        "latest_stdout_log": str(latest_out) if latest_out is not None else None,
        "latest_stderr_log": str(latest_err) if latest_err is not None else None,
        "latest_stdout_age_s": out_age_s,
        "latest_stderr_age_s": err_age_s,
        "latest_error_signature": signature,
    }


def _collect_process_health(pid: int | None) -> dict[str, Any]:
    if not isinstance(pid, int) or pid <= 0:
        return {
            "psutil_available": psutil is not None,
            "child_count": 0,
            "newest_child_age_s": None,
            "child_cpu_s_total": None,
        }
    if psutil is None:
        return {
            "psutil_available": False,
            "child_count": 0,
            "newest_child_age_s": None,
            "child_cpu_s_total": None,
        }
    try:
        proc = psutil.Process(pid)
    except Exception:
        return {
            "psutil_available": True,
            "child_count": 0,
            "newest_child_age_s": None,
            "child_cpu_s_total": None,
        }

    now = time.time()
    newest_age: int | None = None
    child_count = 0
    child_cpu_s_total = 0.0
    newest_child_pid: int | None = None
    newest_child_cmd: str | None = None
    try:
        children = proc.children(recursive=False)
    except Exception:
        children = []
    for child in children:
        child_count += 1
        try:
            age = max(0, int(now - child.create_time()))
        except Exception:
            continue
        if newest_age is None or age < newest_age:
            newest_age = age
            newest_child_pid = int(child.pid)
            try:
                cmdline = child.cmdline()
                if isinstance(cmdline, list) and cmdline:
                    newest_child_cmd = " ".join(str(part) for part in cmdline)
            except Exception:
                newest_child_cmd = None
        try:
            cpu_times = child.cpu_times()
            child_cpu_s_total += float(getattr(cpu_times, "user", 0.0)) + float(getattr(cpu_times, "system", 0.0))
        except Exception:
            pass
    return {
        "psutil_available": True,
        "child_count": child_count,
        "newest_child_age_s": newest_age,
        "child_cpu_s_total": round(child_cpu_s_total, 3),
        "newest_child_pid": newest_child_pid,
        "newest_child_cmdline": newest_child_cmd,
    }


def _format_duration(seconds: int | None) -> str:
    if not isinstance(seconds, int) or seconds < 0:
        return "unknown"
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _load_monitor_snapshot(path: Path = SNAPSHOT_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"jobs": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"jobs": {}}
    if not isinstance(payload, dict):
        return {"jobs": {}}
    jobs = payload.get("jobs")
    if not isinstance(jobs, dict):
        payload["jobs"] = {}
    return payload


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        temp_path.write_text(text, encoding=encoding)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


def _save_monitor_snapshot(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(path, json.dumps(payload, indent=2), encoding="utf-8")


def _load_supervisor_state(path: Path = SUPERVISOR_STATE_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"jobs": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"jobs": {}}
    if not isinstance(payload, dict):
        return {"jobs": {}}
    jobs = payload.get("jobs")
    if not isinstance(jobs, dict):
        payload["jobs"] = {}
    return payload


def _save_supervisor_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(path, json.dumps(payload, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _emit_supervisor_event(
    events_path: Path | None,
    *,
    cycle: int,
    job_name: str,
    event: str,
    payload: dict[str, Any] | None = None,
) -> None:
    if events_path is None:
        return
    event_payload: dict[str, Any] = {
        "time_utc": _utc_now(),
        "cycle": int(cycle),
        "job": job_name,
        "event": event,
    }
    if isinstance(payload, dict):
        event_payload.update(payload)
    try:
        _append_jsonl(events_path, event_payload)
    except Exception as exc:
        print(f"warning: failed to append supervisor event '{event}': {exc}", file=sys.stderr)


def _snapshot_metrics(
    snapshot_jobs: dict[str, Any],
    *,
    job_name: str,
    now_ts: float,
    done: int,
    total: int,
    child_cpu_s_total: float | None,
) -> dict[str, Any]:
    row = snapshot_jobs.get(job_name)
    if not isinstance(row, dict):
        return {
            "window_s": None,
            "delta_done": None,
            "jobs_per_hour": None,
            "eta_s": None,
            "child_cpu_delta_s": None,
        }
    prev_ts = _to_float(row.get("timestamp_s"))
    prev_done = _to_int(row.get("done"))
    prev_child_cpu = _to_float(row.get("child_cpu_s_total"))
    if prev_ts is None or prev_done is None:
        return {
            "window_s": None,
            "delta_done": None,
            "jobs_per_hour": None,
            "eta_s": None,
            "child_cpu_delta_s": None,
        }
    window_s = max(0, int(now_ts - prev_ts))
    delta_done = done - prev_done
    jobs_per_hour: float | None = None
    eta_s: int | None = None
    if window_s > 0 and delta_done >= 0:
        jobs_per_hour = (float(delta_done) * 3600.0) / float(window_s)
        remaining = max(0, int(total - done))
        if jobs_per_hour > 0.0 and remaining > 0:
            eta_s = int((float(remaining) / jobs_per_hour) * 3600.0)
    child_cpu_delta: float | None = None
    if isinstance(child_cpu_s_total, float) and isinstance(prev_child_cpu, float):
        child_cpu_delta = round(child_cpu_s_total - prev_child_cpu, 3)
    return {
        "window_s": window_s,
        "delta_done": delta_done,
        "jobs_per_hour": jobs_per_hour,
        "eta_s": eta_s,
        "child_cpu_delta_s": child_cpu_delta,
    }


def _effective_restart_cooldown_seconds(
    *,
    restart_count: int,
    base_cooldown_seconds: int,
    backoff_factor: float,
    max_cooldown_seconds: int,
) -> int:
    base = max(0, int(base_cooldown_seconds))
    if base <= 0:
        return 0
    factor = float(backoff_factor)
    if factor <= 1.0:
        cooldown = base
    else:
        safe_restart_count = max(0, int(restart_count))
        cooldown = int(round(base * (factor ** safe_restart_count)))
    max_cap = max(0, int(max_cooldown_seconds))
    if max_cap > 0:
        cooldown = min(cooldown, max_cap)
    return max(0, int(cooldown))


def _diagnose_job(
    *,
    alive: bool,
    total: int,
    done: int,
    out_age_s: int | None,
    err_age_s: int | None,
    child_count: int | None,
    child_cpu_delta_s: float | None,
    stall_seconds: int,
) -> JobDiagnosis:
    incomplete = total > 0 and done < total
    if not alive and incomplete:
        return JobDiagnosis(
            code="blocked_dead",
            summary="blocked (process dead before completion)",
            recoverable=True,
        )
    if not alive or not incomplete:
        return JobDiagnosis(
            code="ok",
            summary="no obvious blocker",
            recoverable=False,
        )

    stalled = False
    if isinstance(out_age_s, int) and out_age_s >= stall_seconds:
        stalled = True
    if isinstance(err_age_s, int) and err_age_s >= stall_seconds:
        stalled = True
    if not stalled:
        return JobDiagnosis(
            code="running",
            summary="running (in progress)",
            recoverable=False,
        )

    if isinstance(child_count, int) and child_count > 0:
        if isinstance(child_cpu_delta_s, float):
            if child_cpu_delta_s > 0.0:
                return JobDiagnosis(
                    code="running_compute_bound",
                    summary="running_compute_bound (logs stale, but child CPU is increasing)",
                    recoverable=False,
                )
            return JobDiagnosis(
                code="possible_stall_low_cpu",
                summary="possible_stall_low_cpu (logs stale and child CPU did not increase)",
                recoverable=True,
            )
        return JobDiagnosis(
            code="running_silent",
            summary=f"running_silent (logs stale >= {stall_seconds}s, but child process is active)",
            recoverable=False,
        )

    return JobDiagnosis(
        code="possible_stall",
        summary=f"possible_stall (no recent log activity >= {stall_seconds}s)",
        recoverable=True,
    )


def _collect_diagnose_rows(
    jobs: list[BenchJob],
    *,
    stall_seconds: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    now_utc = _utc_now()
    now_ts = time.time()
    snapshot = _load_monitor_snapshot(SNAPSHOT_PATH)
    snapshot_jobs = snapshot.get("jobs")
    if not isinstance(snapshot_jobs, dict):
        snapshot_jobs = {}

    rows: list[dict[str, Any]] = []
    updated_snapshot_jobs: dict[str, Any] = {}
    for job in jobs:
        progress = _job_progress(job)
        alive = _pid_alive(job.pid)
        total = _to_int(progress.get("total")) or 0
        done = _to_int(progress.get("done")) or 0
        pct = _to_float(progress.get("pct")) or 0.0
        log_health = progress.get("log_health", {})
        out_age = _to_int(log_health.get("latest_stdout_age_s")) if isinstance(log_health, dict) else None
        err_age = _to_int(log_health.get("latest_stderr_age_s")) if isinstance(log_health, dict) else None
        signature = log_health.get("latest_error_signature") if isinstance(log_health, dict) else None
        process_health = progress.get("process_health", {})
        child_count = _to_int(process_health.get("child_count")) if isinstance(process_health, dict) else None
        child_cpu_total = _to_float(process_health.get("child_cpu_s_total")) if isinstance(process_health, dict) else None
        newest_child_pid = process_health.get("newest_child_pid") if isinstance(process_health, dict) else None
        metrics = _snapshot_metrics(
            snapshot_jobs,
            job_name=job.name,
            now_ts=now_ts,
            done=done,
            total=total,
            child_cpu_s_total=child_cpu_total,
        )
        child_cpu_delta = _to_float(metrics.get("child_cpu_delta_s"))
        diagnosis = _diagnose_job(
            alive=bool(alive),
            total=total,
            done=done,
            out_age_s=out_age,
            err_age_s=err_age,
            child_count=child_count,
            child_cpu_delta_s=child_cpu_delta,
            stall_seconds=max(1, int(stall_seconds)),
        )
        breakdown = progress.get("failure_breakdown")
        if isinstance(breakdown, dict) and total > 0 and done >= total:
            failed_total = _to_int(breakdown.get("failed_total")) or 0
            if failed_total > 0:
                runtime_failed = _to_int(breakdown.get("runtime_failures")) or 0
                ci_gate_failed = _to_int(breakdown.get("ci_gate_failures")) or 0
                if runtime_failed > 0:
                    diagnosis = JobDiagnosis(
                        code="completed_with_runtime_failures",
                        summary=(
                            "completed_with_runtime_failures "
                            f"(failed={failed_total}, runtime={runtime_failed}, ci_gate={ci_gate_failed})"
                        ),
                        recoverable=False,
                    )
                else:
                    diagnosis = JobDiagnosis(
                        code="completed_with_quality_failures",
                        summary=(
                            "completed_with_quality_failures "
                            f"(failed={failed_total}, ci_gate={ci_gate_failed})"
                        ),
                        recoverable=False,
                    )
        rows.append(
            {
                "job": job,
                "progress": progress,
                "alive": alive,
                "total": total,
                "done": done,
                "pct": pct,
                "out_age": out_age,
                "err_age": err_age,
                "signature": signature,
                "child_count": child_count,
                "child_cpu_total": child_cpu_total,
                "newest_child_pid": _to_int(newest_child_pid),
                "metrics": metrics,
                "diagnosis": diagnosis,
            }
        )
        updated_snapshot_jobs[job.name] = {
            "timestamp_utc": now_utc,
            "timestamp_s": now_ts,
            "done": done,
            "total": total,
            "pct": round(pct, 4),
            "child_cpu_s_total": child_cpu_total,
        }

    snapshot_payload = {
        "updated_at_utc": now_utc,
        "jobs": updated_snapshot_jobs,
    }
    return rows, snapshot_payload


def _terminate_job_process(pid: int | None, *, timeout_s: int = 20) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return True
    if not _pid_alive(pid):
        return True
    if _taskkill_process_tree(pid, timeout_s=0):
        return True
    if psutil is not None:
        try:
            proc = psutil.Process(pid)
            children = proc.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except Exception:
                    pass
            try:
                proc.terminate()
            except Exception:
                pass
            _, alive = psutil.wait_procs([*children, proc], timeout=max(1, int(timeout_s)))
            for rem in alive:
                try:
                    rem.kill()
                except Exception:
                    pass
            if not _pid_alive(pid):
                return True
        except Exception:
            pass
    try:
        os.kill(pid, signal.SIGTERM)
    except Exception:
        return _taskkill_process_tree(pid, timeout_s=timeout_s)
    waited = 0.0
    while waited < float(timeout_s):
        if not _pid_alive(pid):
            return True
        time.sleep(0.5)
        waited += 0.5
    if not _pid_alive(pid):
        return True
    return _taskkill_process_tree(pid, timeout_s=timeout_s)


def _taskkill_process_tree(pid: int | None, *, timeout_s: int = 20) -> bool:
    if os.name != "nt":
        return False
    if not isinstance(pid, int) or pid <= 0:
        return False
    if not _pid_alive(pid):
        return True
    try:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        return False
    waited = 0.0
    while waited < float(max(0, int(timeout_s))):
        if not _pid_alive(pid):
            return True
        time.sleep(0.25)
        waited += 0.25
    return not _pid_alive(pid)


def _build_command(job: BenchJob, *, no_resume_override: bool | None = None) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "deltatau_audit",
        "bench",
        "run",
        "--manifest",
        job.manifest,
        "--protocol",
        job.protocol,
    ]
    use_no_resume = job.no_resume if no_resume_override is None else bool(no_resume_override)
    if use_no_resume:
        cmd.append("--no-resume")
    return cmd


def _launch_popen_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        new_group = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        detached = int(getattr(subprocess, "DETACHED_PROCESS", 0))
        return {"creationflags": new_group | detached}
    return {"start_new_session": True}


def _launch_job(
    job: BenchJob,
    *,
    force_restart: bool,
    no_resume_override: bool | None = None,
) -> BenchJob:
    if _pid_alive(job.pid):
        if not force_restart:
            return job
        stopped = _terminate_job_process(job.pid)
        if not stopped and _pid_alive(job.pid):
            return job

    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (ROOT / job.out_log).resolve()
    err_path = (ROOT / job.err_log).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    effective_no_resume_override = no_resume_override
    if effective_no_resume_override is None and job.no_resume and not force_restart:
        progress = _job_progress(job)
        total = _to_int(progress.get("total")) or 0
        done = _to_int(progress.get("done")) or 0
        if total > 0 and 0 < done < total:
            effective_no_resume_override = False
            print(
                f"[{job.name}] launch: forcing resume mode "
                f"(partial progress detected: {done}/{total}; --no-resume temporarily disabled)"
            )

    out_fp = open(out_path, "ab")
    err_fp = open(err_path, "ab")
    proc = subprocess.Popen(
        _build_command(job, no_resume_override=effective_no_resume_override),
        cwd=str(ROOT),
        stdout=out_fp,
        stderr=err_fp,
        **_launch_popen_kwargs(),
    )
    out_fp.close()
    err_fp.close()

    job.pid = int(proc.pid)
    job.started_at_utc = _utc_now()
    return job


def _print_status(jobs: list[BenchJob], *, stall_seconds: int = 900) -> None:
    rows, snapshot_payload = _collect_diagnose_rows(jobs, stall_seconds=max(1, int(stall_seconds)))
    print(f"status_time_utc: {snapshot_payload.get('updated_at_utc')}")
    for row in rows:
        job = row["job"]
        progress = row["progress"]
        diagnosis: JobDiagnosis = row["diagnosis"]
        print()
        print(f"[{job.name}]")
        print(f"  pid:      {job.pid}")
        print(f"  alive:    {row['alive']}")
        print(f"  manifest: {job.manifest}")
        print(f"  out_root: {job.output_root}")
        print(f"  progress: {row['done']}/{row['total']} ({row['pct']:.1f}%)")
        counts = progress.get("counts")
        if isinstance(counts, dict):
            stale_suffix = " (stale; pre-launch snapshot)" if progress.get("counts_stale") else ""
            bench_status = progress.get("bench_status")
            status_suffix = f", status={bench_status}" if isinstance(bench_status, str) else ""
            print(
                "  bench_summary counts: "
                f"passed={counts.get('passed', 0)} "
                f"failed={counts.get('failed', 0)} "
                f"skipped={counts.get('skipped', 0)}"
                f"{status_suffix}"
                f"{stale_suffix}"
            )
            breakdown = progress.get("failure_breakdown")
            if isinstance(breakdown, dict):
                failed_total = _to_int(breakdown.get("failed_total")) or 0
                if failed_total > 0:
                    ci_gate = _to_int(breakdown.get("ci_gate_failures")) or 0
                    runtime = _to_int(breakdown.get("runtime_failures")) or 0
                    other = _to_int(breakdown.get("other_failures")) or 0
                    print(
                        "  bench_failures: "
                        f"total={failed_total} ci_gate={ci_gate} runtime={runtime} other={other}"
                    )
            updated_at = progress.get("counts_updated_at_utc")
            if isinstance(updated_at, str) and updated_at:
                print(f"  bench_summary updated_at_utc: {updated_at}")
        log_health = progress.get("log_health")
        if isinstance(log_health, dict):
            out_log = log_health.get("latest_stdout_log")
            err_log = log_health.get("latest_stderr_log")
            out_age = log_health.get("latest_stdout_age_s")
            err_age = log_health.get("latest_stderr_age_s")
            sig = log_health.get("latest_error_signature")
            if isinstance(out_log, str):
                age_text = f"{out_age}s ago" if isinstance(out_age, int) else "unknown age"
                print(f"  latest bench_run.log: {out_log} ({age_text})")
            if isinstance(err_log, str):
                age_text = f"{err_age}s ago" if isinstance(err_age, int) else "unknown age"
                print(f"  latest bench_run.err.log: {err_log} ({age_text})")
            if isinstance(sig, str):
                print(f"  latest error signature: {sig}")
        process_health = progress.get("process_health")
        if isinstance(process_health, dict):
            child_count = process_health.get("child_count")
            newest_age = process_health.get("newest_child_age_s")
            child_cpu = process_health.get("child_cpu_s_total")
            newest_child_pid = process_health.get("newest_child_pid")
            newest_child_cmd = process_health.get("newest_child_cmdline")
            if isinstance(child_count, int):
                if isinstance(newest_age, int):
                    if isinstance(child_cpu, (int, float)):
                        print(
                            "  child_processes: "
                            f"{child_count} (newest age: {newest_age}s, cpu_s_total: {child_cpu})"
                        )
                    else:
                        print(f"  child_processes: {child_count} (newest age: {newest_age}s)")
                else:
                    print(f"  child_processes: {child_count}")
            if isinstance(newest_child_pid, int):
                print(f"  newest_child_pid: {newest_child_pid}")
            if isinstance(newest_child_cmd, str) and newest_child_cmd:
                cmd = newest_child_cmd
                if len(cmd) > 220:
                    cmd = cmd[:217] + "..."
                print(f"  newest_child_cmd: {cmd}")

        metrics = row["metrics"]
        window_s = _to_int(metrics.get("window_s"))
        delta_done = _to_int(metrics.get("delta_done"))
        jobs_per_hour = _to_float(metrics.get("jobs_per_hour"))
        eta_s = _to_int(metrics.get("eta_s"))
        if isinstance(window_s, int) and window_s > 0 and isinstance(delta_done, int):
            sign = "+" if delta_done >= 0 else ""
            print(f"  progress_delta: {sign}{delta_done} jobs over {window_s}s")
            if isinstance(jobs_per_hour, float):
                print(f"  throughput_jobs_per_hour: {jobs_per_hour:.3f}")
            if isinstance(eta_s, int):
                print(f"  eta_to_complete: {_format_duration(eta_s)}")
        print(f"  diagnosis_hint: {diagnosis.code}")
    _save_monitor_snapshot(SNAPSHOT_PATH, snapshot_payload)


def _print_diagnose(jobs: list[BenchJob], *, stall_seconds: int) -> int:
    rc = 0
    rows, snapshot_payload = _collect_diagnose_rows(jobs, stall_seconds=max(1, int(stall_seconds)))
    print(f"diagnose_time_utc: {snapshot_payload.get('updated_at_utc')}")
    for row in rows:
        job = row["job"]
        metrics = row["metrics"]
        diagnosis = row["diagnosis"]
        print()
        print(f"[{job.name}]")
        print(f"  alive: {row['alive']}")
        print(f"  progress: {row['done']}/{row['total']} ({row['pct']:.1f}%)")
        if isinstance(row.get("out_age"), int):
            print(f"  latest_stdout_age_s: {row['out_age']}")
        if isinstance(row.get("err_age"), int):
            print(f"  latest_stderr_age_s: {row['err_age']}")
        child_count = row.get("child_count")
        child_cpu_total = row.get("child_cpu_total")
        if isinstance(child_count, int):
            if isinstance(child_cpu_total, float):
                print(f"  child_processes: {child_count} (cpu_s_total: {child_cpu_total})")
            else:
                print(f"  child_processes: {child_count}")
        newest_child_pid = row.get("newest_child_pid")
        if isinstance(newest_child_pid, int):
            print(f"  newest_child_pid: {newest_child_pid}")
        signature = row.get("signature")
        if isinstance(signature, str):
            print(f"  signature: {signature}")
        window_s = _to_int(metrics.get("window_s"))
        delta_done = _to_int(metrics.get("delta_done"))
        child_cpu_delta = _to_float(metrics.get("child_cpu_delta_s"))
        if isinstance(window_s, int) and window_s > 0 and isinstance(delta_done, int):
            sign = "+" if delta_done >= 0 else ""
            print(f"  progress_delta: {sign}{delta_done} jobs over {window_s}s")
        if isinstance(window_s, int) and window_s > 0 and isinstance(child_cpu_delta, float):
            print(f"  child_cpu_delta_s: {child_cpu_delta:.3f} over {window_s}s")
        print(f"  diagnosis: {diagnosis.summary}")
        if diagnosis.recoverable:
            rc |= 1
    _save_monitor_snapshot(SNAPSHOT_PATH, snapshot_payload)
    return rc


def _normalize_restart_history(raw: Any) -> list[float]:
    if not isinstance(raw, list):
        return []
    history: list[float] = []
    for value in raw:
        ts = _to_float(value)
        if ts is None or ts < 0:
            continue
        history.append(float(ts))
    history.sort()
    if len(history) > MAX_RESTART_HISTORY_ITEMS:
        history = history[-MAX_RESTART_HISTORY_ITEMS:]
    return history


def _count_restarts_in_window(
    restart_times_s: list[float],
    *,
    now_ts: float,
    window_seconds: int,
) -> int:
    if window_seconds <= 0:
        return 0
    cutoff = float(now_ts) - float(window_seconds)
    return sum(1 for ts in restart_times_s if ts >= cutoff)


def _forced_recovery_reason(
    *,
    diagnosis_code: str,
    incomplete: bool,
    no_progress_cycles: int,
    no_progress_seconds: int,
    max_no_progress_cycles: int,
    max_no_progress_seconds: int,
    child_cpu_delta_s: float | None = None,
    out_age_s: int | None = None,
    err_age_s: int | None = None,
    child_count: int | None = None,
    stall_seconds: int = 0,
) -> str | None:
    if not incomplete:
        return None
    if diagnosis_code not in RUNNING_DIAGNOSIS_CODES:
        return None
    # Guard against false-positive restarts for jobs that still show active work.
    if diagnosis_code == "running":
        return None
    if diagnosis_code == "running_compute_bound" and isinstance(child_cpu_delta_s, float) and child_cpu_delta_s > 0.0:
        return None
    if (
        diagnosis_code == "running_silent"
        and isinstance(child_count, int)
        and child_count > 0
        and isinstance(out_age_s, int)
        and out_age_s < max(1, int(stall_seconds))
        and isinstance(err_age_s, int)
        and err_age_s < max(1, int(stall_seconds))
    ):
        return None
    if max_no_progress_cycles > 0 and no_progress_cycles >= max_no_progress_cycles:
        return "no_progress_timeout"
    if max_no_progress_seconds > 0 and no_progress_seconds >= max_no_progress_seconds:
        return "no_progress_timeout_seconds"
    return None


def _decide_recovery_action(
    *,
    meta: dict[str, Any],
    effective_reason: str,
    forced_reason: str | None,
    current_signature: str | None,
    now_ts: float,
    recover_after_consecutive: int,
    max_restarts_per_job: int,
    max_restarts_per_reason: int,
    max_restarts_per_signature: int,
    max_restarts_per_window: int,
    restart_window_seconds: int,
    restart_cooldown_seconds: int,
    restart_backoff_factor: float,
    max_restart_cooldown_seconds: int,
) -> RecoveryDecision:
    restart_count = _to_int(meta.get("restart_count")) or 0
    if restart_count >= max_restarts_per_job:
        return RecoveryDecision(
            allow=False,
            skip_event="recovery_skipped_budget",
            skip_message=f"(restart budget exhausted: {restart_count}/{max_restarts_per_job})",
            skip_payload={
                "restart_count": restart_count,
                "max_restarts_per_job": max_restarts_per_job,
            },
        )

    if max_restarts_per_reason > 0:
        reason_counts = meta.get("restarts_by_reason")
        reason_count = 0
        if isinstance(reason_counts, dict):
            reason_count = _to_int(reason_counts.get(effective_reason)) or 0
        if reason_count >= max_restarts_per_reason:
            return RecoveryDecision(
                allow=False,
                skip_event="recovery_skipped_reason_budget",
                skip_message=(
                    f"(reason budget exhausted for '{effective_reason}': "
                    f"{reason_count}/{max_restarts_per_reason})"
                ),
                skip_payload={
                    "reason": effective_reason,
                    "reason_restart_count": reason_count,
                    "max_restarts_per_reason": max_restarts_per_reason,
                },
            )

    signature_key = _normalize_signature(current_signature)
    if max_restarts_per_signature > 0 and signature_key:
        signature_counts = meta.get("restarts_by_signature")
        signature_restart_count = 0
        if isinstance(signature_counts, dict):
            signature_restart_count = _to_int(signature_counts.get(signature_key)) or 0
        if signature_restart_count >= max_restarts_per_signature:
            return RecoveryDecision(
                allow=False,
                skip_event="recovery_skipped_signature_budget",
                skip_message=(
                    f"(signature budget exhausted for '{signature_key}': "
                    f"{signature_restart_count}/{max_restarts_per_signature})"
                ),
                skip_payload={
                    "signature": signature_key,
                    "signature_restart_count": signature_restart_count,
                    "max_restarts_per_signature": max_restarts_per_signature,
                },
            )

    if max_restarts_per_window > 0 and restart_window_seconds > 0:
        restart_times_s = _normalize_restart_history(meta.get("restart_times_s"))
        restarts_in_window = _count_restarts_in_window(
            restart_times_s,
            now_ts=now_ts,
            window_seconds=restart_window_seconds,
        )
        if restarts_in_window >= max_restarts_per_window:
            return RecoveryDecision(
                allow=False,
                skip_event="recovery_skipped_window_budget",
                skip_message=(
                    f"(window budget exhausted in {restart_window_seconds}s: "
                    f"{restarts_in_window}/{max_restarts_per_window})"
                ),
                skip_payload={
                    "restarts_in_window": restarts_in_window,
                    "restart_window_seconds": restart_window_seconds,
                    "max_restarts_per_window": max_restarts_per_window,
                },
            )

    consecutive = _to_int(meta.get("consecutive_recoverable")) or 0
    require_consecutive_gate = forced_reason is None
    if require_consecutive_gate and consecutive < recover_after_consecutive:
        return RecoveryDecision(
            allow=False,
            skip_event="recovery_skipped_consecutive",
            skip_message=f"(consecutive recoverable {consecutive}/{recover_after_consecutive})",
            skip_payload={
                "consecutive_recoverable": consecutive,
                "recover_after_consecutive": recover_after_consecutive,
            },
        )

    last_restart_ts = _to_float(meta.get("last_restart_ts"))
    if last_restart_ts is None:
        last_restart_ts = -1.0
    effective_cooldown = _effective_restart_cooldown_seconds(
        restart_count=restart_count,
        base_cooldown_seconds=restart_cooldown_seconds,
        backoff_factor=restart_backoff_factor,
        max_cooldown_seconds=max_restart_cooldown_seconds,
    )
    cooldown_left = 0
    if last_restart_ts >= 0:
        cooldown_left = int(effective_cooldown - (now_ts - last_restart_ts))
    if cooldown_left > 0:
        return RecoveryDecision(
            allow=False,
            skip_event="recovery_skipped_cooldown",
            skip_message=f"(cooldown active: {cooldown_left}s left)",
            skip_payload={
                "cooldown_left_s": cooldown_left,
                "restart_cooldown_seconds": restart_cooldown_seconds,
                "effective_cooldown_s": effective_cooldown,
                "restart_backoff_factor": restart_backoff_factor,
            },
        )

    return RecoveryDecision(allow=True)


def _supervisor_row(state_jobs: dict[str, Any], job_name: str) -> dict[str, Any]:
    current = state_jobs.get(job_name)
    if not isinstance(current, dict):
        current = {}
    restart_count = _to_int(current.get("restart_count")) or 0
    consecutive_recoverable = _to_int(current.get("consecutive_recoverable")) or 0
    consecutive_signature_hits = _to_int(current.get("consecutive_signature_hits")) or 0
    no_progress_cycles = _to_int(current.get("no_progress_cycles")) or 0
    last_done = _to_int(current.get("last_done"))
    if last_done is None:
        last_done = -1
    last_progress_ts = _to_float(current.get("last_progress_ts"))
    if last_progress_ts is None:
        last_progress_ts = -1.0
    last_restart_ts = _to_float(current.get("last_restart_ts"))
    if last_restart_ts is None:
        last_restart_ts = -1.0
    restarts_by_reason_raw = current.get("restarts_by_reason")
    restarts_by_reason: dict[str, int] = {}
    if isinstance(restarts_by_reason_raw, dict):
        for key, value in restarts_by_reason_raw.items():
            if not isinstance(key, str) or not key:
                continue
            count = _to_int(value) or 0
            if count > 0:
                restarts_by_reason[key] = count
    restarts_by_signature_raw = current.get("restarts_by_signature")
    restarts_by_signature: dict[str, int] = {}
    if isinstance(restarts_by_signature_raw, dict):
        for key, value in restarts_by_signature_raw.items():
            signature_key = _normalize_signature(key)
            if not signature_key:
                continue
            count = _to_int(value) or 0
            if count > 0:
                restarts_by_signature[signature_key] = count
    restart_times_s = _normalize_restart_history(current.get("restart_times_s"))
    return {
        "restart_count": restart_count,
        "consecutive_recoverable": consecutive_recoverable,
        "consecutive_signature_hits": consecutive_signature_hits,
        "no_progress_cycles": no_progress_cycles,
        "last_done": last_done,
        "last_progress_ts": last_progress_ts,
        "last_restart_ts": last_restart_ts,
        "last_reason": str(current.get("last_reason", "")),
        "last_diagnosis": str(current.get("last_diagnosis", "")),
        "last_signature": _normalize_signature(current.get("last_signature")),
        "restarts_by_reason": restarts_by_reason,
        "restarts_by_signature": restarts_by_signature,
        "restart_times_s": restart_times_s,
    }


def _register_restart(
    *,
    state_jobs: dict[str, Any],
    job_name: str,
    reason: str,
    signature: str | None = None,
    now_ts: float,
) -> None:
    row = _supervisor_row(state_jobs, job_name)
    restarts_by_reason = dict(row.get("restarts_by_reason") or {})
    restarts_by_reason[reason] = (restarts_by_reason.get(reason, 0) or 0) + 1
    restarts_by_signature = dict(row.get("restarts_by_signature") or {})
    signature_key = _normalize_signature(signature)
    if signature_key:
        restarts_by_signature[signature_key] = (restarts_by_signature.get(signature_key, 0) or 0) + 1
    restart_times_s = _normalize_restart_history(row.get("restart_times_s"))
    restart_times_s.append(float(now_ts))
    restart_times_s = _normalize_restart_history(restart_times_s)
    state_jobs[job_name] = {
        "restart_count": row["restart_count"] + 1,
        "consecutive_recoverable": 0,
        "consecutive_signature_hits": 0,
        "no_progress_cycles": 0,
        "last_done": row["last_done"],
        "last_progress_ts": now_ts,
        "last_restart_ts": now_ts,
        "last_restart_utc": _utc_now(),
        "last_reason": reason,
        "last_diagnosis": reason,
        "last_signature": "",
        "restarts_by_reason": restarts_by_reason,
        "restarts_by_signature": restarts_by_signature,
        "restart_times_s": restart_times_s,
    }


def _register_diagnosis(
    *,
    state_jobs: dict[str, Any],
    job_name: str,
    diagnosis: JobDiagnosis,
) -> dict[str, Any]:
    row = _supervisor_row(state_jobs, job_name)
    if diagnosis.recoverable:
        if row["last_diagnosis"] == diagnosis.code:
            consecutive = row["consecutive_recoverable"] + 1
        else:
            consecutive = 1
    else:
        consecutive = 0
    state_jobs[job_name] = {
        "restart_count": row["restart_count"],
        "consecutive_recoverable": consecutive,
        "consecutive_signature_hits": row["consecutive_signature_hits"],
        "no_progress_cycles": row["no_progress_cycles"],
        "last_done": row["last_done"],
        "last_progress_ts": row["last_progress_ts"],
        "last_restart_ts": row["last_restart_ts"],
        "last_reason": row["last_reason"],
        "last_diagnosis": diagnosis.code,
        "last_signature": row["last_signature"],
        "restarts_by_reason": row.get("restarts_by_reason") or {},
        "restarts_by_signature": row.get("restarts_by_signature") or {},
        "restart_times_s": row.get("restart_times_s") or [],
    }
    return _supervisor_row(state_jobs, job_name)


def _register_signature_observation(
    *,
    state_jobs: dict[str, Any],
    job_name: str,
    signature: str | None,
) -> dict[str, Any]:
    row = _supervisor_row(state_jobs, job_name)
    signature_key = _normalize_signature(signature)
    if signature_key:
        if signature_key == row["last_signature"]:
            consecutive_signature_hits = row["consecutive_signature_hits"] + 1
        else:
            consecutive_signature_hits = 1
    else:
        consecutive_signature_hits = 0
    state_jobs[job_name] = {
        "restart_count": row["restart_count"],
        "consecutive_recoverable": row["consecutive_recoverable"],
        "consecutive_signature_hits": consecutive_signature_hits,
        "no_progress_cycles": row["no_progress_cycles"],
        "last_done": row["last_done"],
        "last_progress_ts": row["last_progress_ts"],
        "last_restart_ts": row["last_restart_ts"],
        "last_reason": row["last_reason"],
        "last_diagnosis": row["last_diagnosis"],
        "last_signature": signature_key,
        "restarts_by_reason": row.get("restarts_by_reason") or {},
        "restarts_by_signature": row.get("restarts_by_signature") or {},
        "restart_times_s": row.get("restart_times_s") or [],
    }
    return _supervisor_row(state_jobs, job_name)


def _register_progress(
    *,
    state_jobs: dict[str, Any],
    job_name: str,
    done: int,
    total: int,
    now_ts: float | None = None,
) -> dict[str, Any]:
    if now_ts is None:
        now_ts = time.time()
    row = _supervisor_row(state_jobs, job_name)
    incomplete = total > 0 and done < total
    progressed = False
    if not incomplete:
        no_progress_cycles = 0
        progressed = True
    elif row["last_done"] < 0:
        no_progress_cycles = 0
        progressed = True
    elif done <= row["last_done"]:
        no_progress_cycles = row["no_progress_cycles"] + 1
    else:
        no_progress_cycles = 0
        progressed = True
    last_progress_ts = float(row["last_progress_ts"])
    if progressed or last_progress_ts < 0:
        last_progress_ts = float(now_ts)
    state_jobs[job_name] = {
        "restart_count": row["restart_count"],
        "consecutive_recoverable": row["consecutive_recoverable"],
        "consecutive_signature_hits": row["consecutive_signature_hits"],
        "no_progress_cycles": no_progress_cycles,
        "last_done": done,
        "last_progress_ts": last_progress_ts,
        "last_restart_ts": row["last_restart_ts"],
        "last_reason": row["last_reason"],
        "last_diagnosis": row["last_diagnosis"],
        "last_signature": row["last_signature"],
        "restarts_by_reason": row.get("restarts_by_reason") or {},
        "restarts_by_signature": row.get("restarts_by_signature") or {},
        "restart_times_s": row.get("restart_times_s") or [],
    }
    return _supervisor_row(state_jobs, job_name)


def _project_progress_meta(
    meta: dict[str, Any],
    *,
    done: int,
    total: int,
    now_ts: float,
) -> dict[str, Any]:
    row = {
        "restart_count": _to_int(meta.get("restart_count")) or 0,
        "consecutive_recoverable": _to_int(meta.get("consecutive_recoverable")) or 0,
        "consecutive_signature_hits": _to_int(meta.get("consecutive_signature_hits")) or 0,
        "no_progress_cycles": _to_int(meta.get("no_progress_cycles")) or 0,
        "last_done": _to_int(meta.get("last_done")) if _to_int(meta.get("last_done")) is not None else -1,
        "last_progress_ts": _to_float(meta.get("last_progress_ts")) if _to_float(meta.get("last_progress_ts")) is not None else -1.0,
        "last_restart_ts": _to_float(meta.get("last_restart_ts")) if _to_float(meta.get("last_restart_ts")) is not None else -1.0,
        "last_reason": str(meta.get("last_reason", "")),
        "last_diagnosis": str(meta.get("last_diagnosis", "")),
        "last_signature": _normalize_signature(meta.get("last_signature")),
        "restarts_by_reason": dict(meta.get("restarts_by_reason") or {}),
        "restarts_by_signature": dict(meta.get("restarts_by_signature") or {}),
        "restart_times_s": _normalize_restart_history(meta.get("restart_times_s")),
    }
    incomplete = total > 0 and done < total
    progressed = False
    if not incomplete:
        no_progress_cycles = 0
        progressed = True
    elif row["last_done"] < 0:
        no_progress_cycles = 0
        progressed = True
    elif done <= row["last_done"]:
        no_progress_cycles = row["no_progress_cycles"] + 1
    else:
        no_progress_cycles = 0
        progressed = True
    last_progress_ts = float(row["last_progress_ts"])
    if progressed or last_progress_ts < 0:
        last_progress_ts = float(now_ts)
    return {
        "restart_count": row["restart_count"],
        "consecutive_recoverable": row["consecutive_recoverable"],
        "consecutive_signature_hits": row["consecutive_signature_hits"],
        "no_progress_cycles": no_progress_cycles,
        "last_done": done,
        "last_progress_ts": last_progress_ts,
        "last_restart_ts": row["last_restart_ts"],
        "last_reason": row["last_reason"],
        "last_diagnosis": row["last_diagnosis"],
        "last_signature": row["last_signature"],
        "restarts_by_reason": row["restarts_by_reason"],
        "restarts_by_signature": row["restarts_by_signature"],
        "restart_times_s": row["restart_times_s"],
    }


def _try_recover_job(
    job: BenchJob,
    *,
    reason: str,
) -> tuple[BenchJob, bool]:
    print(f"  recovery: reason={reason}")
    if _pid_alive(job.pid):
        stopped = _terminate_job_process(job.pid)
        print(f"  recovery: stop_pid={job.pid} stopped={stopped}")
        if not stopped and _pid_alive(job.pid):
            print("  recovery: aborted (process still alive after stop attempt)")
            return job, False
    no_resume_override: bool | None = None
    if job.no_resume and reason in {
        "blocked_dead",
        "possible_stall",
        "possible_stall_low_cpu",
        "no_progress_timeout",
        "no_progress_timeout_seconds",
    }:
        progress = _job_progress(job)
        total = _to_int(progress.get("total")) or 0
        done = _to_int(progress.get("done")) or 0
        if total > 0 and 0 < done < total:
            no_resume_override = False
            print(
                "  recovery: forcing resume mode "
                f"(partial progress detected: {done}/{total}; --no-resume temporarily disabled)"
            )

    relaunched = _launch_job(job, force_restart=True, no_resume_override=no_resume_override)
    print(f"  recovery: relaunched_pid={relaunched.pid}")
    return relaunched, True


def _supervise(
    jobs: list[BenchJob],
    *,
    interval_seconds: int,
    stall_seconds: int,
    auto_recover: bool,
    recover_after_consecutive: int,
    max_restarts_per_job: int,
    max_restarts_per_reason: int = 0,
    max_restarts_per_signature: int = 0,
    max_total_restarts: int = 0,
    max_restarts_per_window: int = 0,
    restart_window_seconds: int = 0,
    restart_cooldown_seconds: int = 0,
    restart_backoff_factor: float = 1.0,
    max_restart_cooldown_seconds: int = 0,
    recovery_grace_seconds: int = 0,
    max_no_progress_cycles: int = 0,
    max_no_progress_seconds: int = 0,
    max_cycles: int = 0,
    events_path: Path | None = None,
) -> int:
    interval_seconds = max(1, int(interval_seconds))
    stall_seconds = max(1, int(stall_seconds))
    recover_after_consecutive = max(1, int(recover_after_consecutive))
    max_restarts_per_job = max(0, int(max_restarts_per_job))
    max_restarts_per_reason = max(0, int(max_restarts_per_reason))
    max_restarts_per_signature = max(0, int(max_restarts_per_signature))
    max_total_restarts = max(0, int(max_total_restarts))
    max_restarts_per_window = max(0, int(max_restarts_per_window))
    restart_window_seconds = max(0, int(restart_window_seconds))
    restart_cooldown_seconds = max(0, int(restart_cooldown_seconds))
    restart_backoff_factor = max(1.0, float(restart_backoff_factor))
    max_restart_cooldown_seconds = max(0, int(max_restart_cooldown_seconds))
    recovery_grace_seconds = max(0, int(recovery_grace_seconds))
    max_no_progress_cycles = max(0, int(max_no_progress_cycles))
    max_no_progress_seconds = max(0, int(max_no_progress_seconds))
    max_cycles = max(0, int(max_cycles))

    rc = 0
    cycle = 0
    state = _load_supervisor_state(SUPERVISOR_STATE_PATH)
    state_jobs = state.get("jobs")
    if not isinstance(state_jobs, dict):
        state_jobs = {}

    print(
        "supervise_config: "
        f"interval_s={interval_seconds} "
        f"stall_s={stall_seconds} "
        f"auto_recover={auto_recover} "
        f"recover_after_consecutive={recover_after_consecutive} "
        f"max_restarts_per_job={max_restarts_per_job} "
        f"max_restarts_per_reason={max_restarts_per_reason} "
        f"max_restarts_per_signature={max_restarts_per_signature} "
        f"max_total_restarts={max_total_restarts} "
        f"max_restarts_per_window={max_restarts_per_window} "
        f"restart_window_s={restart_window_seconds} "
        f"restart_cooldown_s={restart_cooldown_seconds} "
        f"restart_backoff_factor={restart_backoff_factor:.3f} "
        f"max_restart_cooldown_s={max_restart_cooldown_seconds} "
        f"recovery_grace_s={recovery_grace_seconds} "
        f"max_no_progress_cycles={max_no_progress_cycles} "
        f"max_no_progress_seconds={max_no_progress_seconds} "
        f"max_cycles={max_cycles if max_cycles > 0 else 'infinite'}"
    )
    try:
        while True:
            cycle += 1
            now_ts = time.time()
            print()
            print(f"supervise_cycle: {cycle} time_utc={_utc_now()}")
            rows, snapshot_payload = _collect_diagnose_rows(jobs, stall_seconds=stall_seconds)
            recoveries: list[str] = []
            for row in rows:
                job = row["job"]
                diagnosis: JobDiagnosis = row["diagnosis"]
                print(f"[{job.name}] {row['done']}/{row['total']} ({row['pct']:.1f}%) -> {diagnosis.code}")
                _register_progress(
                    state_jobs=state_jobs,
                    job_name=job.name,
                    done=int(row["done"]),
                    total=int(row["total"]),
                    now_ts=now_ts,
                )
                meta = _register_diagnosis(
                    state_jobs=state_jobs,
                    job_name=job.name,
                    diagnosis=diagnosis,
                )
                meta = _register_signature_observation(
                    state_jobs=state_jobs,
                    job_name=job.name,
                    signature=row.get("signature"),
                )
                no_progress_cycles = int(meta["no_progress_cycles"])
                no_progress_seconds = 0
                last_progress_ts = _to_float(meta.get("last_progress_ts"))
                if isinstance(last_progress_ts, float) and last_progress_ts >= 0:
                    no_progress_seconds = max(0, int(now_ts - last_progress_ts))
                if no_progress_cycles > 0:
                    print(f"  no_progress_cycles: {no_progress_cycles}")
                if no_progress_seconds > 0:
                    print(f"  no_progress_seconds: {no_progress_seconds}")
                incomplete = int(row["total"]) > 0 and int(row["done"]) < int(row["total"])
                metrics = row.get("metrics")
                metrics_dict = metrics if isinstance(metrics, dict) else {}
                forced_reason = _forced_recovery_reason(
                    diagnosis_code=diagnosis.code,
                    incomplete=incomplete,
                    no_progress_cycles=no_progress_cycles,
                    no_progress_seconds=no_progress_seconds,
                    max_no_progress_cycles=max_no_progress_cycles,
                    max_no_progress_seconds=max_no_progress_seconds,
                    child_cpu_delta_s=_to_float(metrics_dict.get("child_cpu_delta_s")),
                    out_age_s=_to_int(row.get("out_age")),
                    err_age_s=_to_int(row.get("err_age")),
                    child_count=_to_int(row.get("child_count")),
                    stall_seconds=stall_seconds,
                )
                if forced_reason == "no_progress_timeout":
                    print(
                        "  escalation: "
                        f"{forced_reason} ({no_progress_cycles} cycles without progress)"
                    )
                elif forced_reason == "no_progress_timeout_seconds":
                    print(
                        "  escalation: "
                        f"{forced_reason} ({no_progress_seconds}s without progress)"
                    )
                _emit_supervisor_event(
                    events_path,
                    cycle=cycle,
                    job_name=job.name,
                    event="diagnosis",
                    payload={
                        "diagnosis_code": diagnosis.code,
                        "diagnosis_recoverable": diagnosis.recoverable,
                        "done": row["done"],
                        "total": row["total"],
                        "pid": job.pid,
                        "no_progress_cycles": no_progress_cycles,
                        "no_progress_seconds": no_progress_seconds,
                        "forced_recovery_reason": forced_reason,
                    },
                )
                suppressed_by_grace = False
                if (
                    recovery_grace_seconds > 0
                    and forced_reason is None
                    and diagnosis.recoverable
                    and diagnosis.code != "blocked_dead"
                ):
                    last_restart_ts = _to_float(meta.get("last_restart_ts"))
                    since_restart_s = -1
                    if isinstance(last_restart_ts, float) and last_restart_ts >= 0:
                        since_restart_s = max(0, int(now_ts - last_restart_ts))
                    if 0 <= since_restart_s < recovery_grace_seconds:
                        suppressed_by_grace = True
                        print(
                            "  recovery: suppressed "
                            f"(grace window active: {since_restart_s}/{recovery_grace_seconds}s, "
                            f"diagnosis={diagnosis.code})"
                        )
                        _emit_supervisor_event(
                            events_path,
                            cycle=cycle,
                            job_name=job.name,
                            event="recovery_suppressed_grace",
                            payload={
                                "diagnosis_code": diagnosis.code,
                                "since_last_restart_s": since_restart_s,
                                "recovery_grace_seconds": recovery_grace_seconds,
                            },
                        )
                effective_recoverable = (diagnosis.recoverable and not suppressed_by_grace) or forced_reason is not None
                effective_reason = forced_reason or diagnosis.code
                if not effective_recoverable:
                    continue
                rc |= 1
                if not auto_recover:
                    continue
                if max_total_restarts > 0:
                    total_restarts = sum(
                        (_to_int(_supervisor_row(state_jobs, name).get("restart_count")) or 0)
                        for name in state_jobs
                    )
                    if total_restarts >= max_total_restarts:
                        print(
                            "  recovery: skipped "
                            f"(global restart budget exhausted: {total_restarts}/{max_total_restarts})"
                        )
                        _emit_supervisor_event(
                            events_path,
                            cycle=cycle,
                            job_name=job.name,
                            event="recovery_skipped_total_budget",
                            payload={
                                "total_restarts": total_restarts,
                                "max_total_restarts": max_total_restarts,
                            },
                        )
                        continue
                decision = _decide_recovery_action(
                    meta=meta,
                    effective_reason=effective_reason,
                    forced_reason=forced_reason,
                    current_signature=row.get("signature"),
                    now_ts=now_ts,
                    recover_after_consecutive=recover_after_consecutive,
                    max_restarts_per_job=max_restarts_per_job,
                    max_restarts_per_reason=max_restarts_per_reason,
                    max_restarts_per_signature=max_restarts_per_signature,
                    max_restarts_per_window=max_restarts_per_window,
                    restart_window_seconds=restart_window_seconds,
                    restart_cooldown_seconds=restart_cooldown_seconds,
                    restart_backoff_factor=restart_backoff_factor,
                    max_restart_cooldown_seconds=max_restart_cooldown_seconds,
                )
                if not decision.allow:
                    if decision.skip_message:
                        print(f"  recovery: skipped {decision.skip_message}")
                    _emit_supervisor_event(
                        events_path,
                        cycle=cycle,
                        job_name=job.name,
                        event=decision.skip_event,
                        payload=decision.skip_payload or {},
                    )
                    continue

                relaunched, restarted = _try_recover_job(job, reason=effective_reason)
                if not restarted:
                    _emit_supervisor_event(
                        events_path,
                        cycle=cycle,
                        job_name=job.name,
                        event="recovery_failed",
                        payload={"reason": effective_reason},
                    )
                    continue
                if relaunched.pid != job.pid:
                    # Defensive fallback; current implementation mutates in place.
                    job.pid = relaunched.pid
                    job.started_at_utc = relaunched.started_at_utc
                _register_restart(
                    state_jobs=state_jobs,
                    job_name=job.name,
                    reason=effective_reason,
                    signature=_normalize_signature(row.get("signature")),
                    now_ts=now_ts,
                )
                recoveries.append(job.name)
                _emit_supervisor_event(
                    events_path,
                    cycle=cycle,
                    job_name=job.name,
                    event="recovered",
                    payload={
                        "reason": effective_reason,
                        "new_pid": job.pid,
                    },
                )

            _save_monitor_snapshot(SNAPSHOT_PATH, snapshot_payload)
            _save_supervisor_state(
                SUPERVISOR_STATE_PATH,
                {
                    "updated_at_utc": _utc_now(),
                    "jobs": state_jobs,
                },
            )
            if recoveries:
                _save_active_jobs(ACTIVE_PATH, jobs)

            all_done = all((row["total"] > 0 and row["done"] >= row["total"]) for row in rows)
            if all_done:
                print("supervise: all jobs completed.")
                return rc
            if max_cycles > 0 and cycle >= max_cycles:
                print("supervise: reached max cycles.")
                return rc
            print(f"supervise: sleeping {interval_seconds}s")
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("\nsupervise: interrupted.")
        return rc


def _launch_jobs(jobs: list[BenchJob], *, force_restart: bool) -> list[BenchJob]:
    launched: list[BenchJob] = []
    for job in jobs:
        if not force_restart and _is_job_completed(job):
            print(f"[{job.name}] launch: skipped (already complete)")
            launched.append(job)
            continue
        launched.append(_launch_job(job, force_restart=force_restart))
    _save_active_jobs(ACTIVE_PATH, launched)
    return launched


def _all_jobs_completed(jobs: list[BenchJob]) -> bool:
    if not jobs:
        return False
    for job in jobs:
        progress = _job_progress(job)
        total = _to_int(progress.get("total")) or 0
        done = _to_int(progress.get("done")) or 0
        if total <= 0 or done < total:
            return False
    return True


def _is_job_completed(job: BenchJob) -> bool:
    progress = _job_progress(job)
    total = _to_int(progress.get("total")) or 0
    done = _to_int(progress.get("done")) or 0
    return total > 0 and done >= total


def _strict_readiness_check() -> int:
    return _run_cmd([sys.executable, "scripts/prepare_submission.py", "--check-only", "--strict-check"])


def _autopilot(
    jobs: list[BenchJob],
    *,
    do_preflight: bool,
    force_restart: bool,
    interval_seconds: int,
    stall_seconds: int,
    auto_recover: bool,
    recover_after_consecutive: int,
    max_restarts_per_job: int,
    max_restarts_per_reason: int = 0,
    max_restarts_per_signature: int = 0,
    max_total_restarts: int = 0,
    max_restarts_per_window: int = 0,
    restart_window_seconds: int = 0,
    restart_cooldown_seconds: int = 0,
    restart_backoff_factor: float = 1.0,
    max_restart_cooldown_seconds: int = 0,
    recovery_grace_seconds: int = 0,
    max_no_progress_cycles: int = 0,
    max_no_progress_seconds: int = 0,
    max_cycles: int = 0,
    auto_finalize: bool = False,
) -> int:
    print(
        "autopilot_config: "
        f"preflight={do_preflight} "
        f"force_restart={force_restart} "
        f"interval_s={interval_seconds} "
        f"stall_s={stall_seconds} "
        f"auto_recover={auto_recover} "
        f"recover_after_consecutive={recover_after_consecutive} "
        f"max_restarts_per_job={max_restarts_per_job} "
        f"max_restarts_per_reason={max_restarts_per_reason} "
        f"max_restarts_per_signature={max_restarts_per_signature} "
        f"max_total_restarts={max_total_restarts} "
        f"max_restarts_per_window={max_restarts_per_window} "
        f"restart_window_s={restart_window_seconds} "
        f"restart_cooldown_s={restart_cooldown_seconds} "
        f"restart_backoff_factor={restart_backoff_factor:.3f} "
        f"max_restart_cooldown_s={max_restart_cooldown_seconds} "
        f"recovery_grace_s={recovery_grace_seconds} "
        f"max_no_progress_cycles={max_no_progress_cycles} "
        f"max_no_progress_seconds={max_no_progress_seconds} "
        f"max_cycles={max_cycles if max_cycles > 0 else 'infinite'} "
        f"auto_finalize={auto_finalize}"
    )

    rc = 0
    if do_preflight:
        preflight_rc = _run_preflight(jobs)
        rc |= preflight_rc
        if preflight_rc != 0:
            print("autopilot: preflight failed.")
            return rc

    launched = _launch_jobs(jobs, force_restart=force_restart)
    _print_status(launched, stall_seconds=stall_seconds)

    supervise_rc = _supervise(
        launched,
        interval_seconds=interval_seconds,
        stall_seconds=stall_seconds,
        auto_recover=auto_recover,
        recover_after_consecutive=recover_after_consecutive,
        max_restarts_per_job=max_restarts_per_job,
        max_restarts_per_reason=max_restarts_per_reason,
        max_restarts_per_signature=max_restarts_per_signature,
        max_total_restarts=max_total_restarts,
        max_restarts_per_window=max_restarts_per_window,
        restart_window_seconds=restart_window_seconds,
        restart_cooldown_seconds=restart_cooldown_seconds,
        restart_backoff_factor=restart_backoff_factor,
        max_restart_cooldown_seconds=max_restart_cooldown_seconds,
        recovery_grace_seconds=recovery_grace_seconds,
        max_no_progress_cycles=max_no_progress_cycles,
        max_no_progress_seconds=max_no_progress_seconds,
        max_cycles=max_cycles,
        events_path=SUPERVISOR_EVENTS_PATH,
    )
    rc |= supervise_rc
    _save_active_jobs(ACTIVE_PATH, launched)

    if not _all_jobs_completed(launched):
        print("autopilot: jobs not yet complete; skipping readiness/finalize.")
        return rc | 1

    if auto_finalize:
        finalize_rc = _finalize(launched)
        rc |= finalize_rc
        return rc

    readiness_rc = _strict_readiness_check()
    rc |= readiness_rc
    return rc


def _recommended_autopilot_command(
    *,
    max_no_progress_cycles: int,
    max_no_progress_seconds: int,
) -> str:
    return (
        "python scripts/run_submission_pipeline.py --mode autopilot "
        "--auto-recover --recover-after-consecutive 2 --max-restarts-per-job 2 "
        "--max-restarts-per-reason 2 --max-total-restarts 6 "
        f"--max-restarts-per-signature {RECOMMENDED_MAX_RESTARTS_PER_SIGNATURE} "
        f"--max-restarts-per-window {RECOMMENDED_MAX_RESTARTS_PER_WINDOW} "
        f"--restart-window-seconds {RECOMMENDED_RESTART_WINDOW_SECONDS} "
        "--restart-cooldown-seconds 1800 "
        "--restart-backoff-factor 1.5 --max-restart-cooldown-seconds 21600 "
        f"--recovery-grace-seconds {RECOMMENDED_RECOVERY_GRACE_SECONDS} "
        f"--max-no-progress-cycles {max(0, int(max_no_progress_cycles))} "
        f"--max-no-progress-seconds {max(0, int(max_no_progress_seconds))} "
        "--interval 120 --stall-seconds 1800 --max-cycles 0"
    )


def _completed_failure_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in rows:
        total = _to_int(row.get("total")) or 0
        done = _to_int(row.get("done")) or 0
        if total <= 0 or done < total:
            continue
        progress = row.get("progress")
        if not isinstance(progress, dict):
            continue
        counts = progress.get("counts")
        failed_jobs = _to_int(counts.get("failed")) if isinstance(counts, dict) else None
        if failed_jobs is None:
            failed_jobs = 0
        breakdown = progress.get("failure_breakdown")
        breakdown_dict = breakdown if isinstance(breakdown, dict) else {}
        failed_total = _to_int(breakdown_dict.get("failed_total")) or 0
        if failed_jobs <= 0 and failed_total <= 0:
            continue
        ci_gate = _to_int(breakdown_dict.get("ci_gate_failures")) or 0
        runtime = _to_int(breakdown_dict.get("runtime_failures")) or 0
        other = _to_int(breakdown_dict.get("other_failures")) or 0
        failures.append(
            {
                "row": row,
                "failed_jobs": max(failed_jobs, failed_total),
                "ci_gate_failures": ci_gate,
                "runtime_failures": runtime,
                "other_failures": other,
            }
        )
    return failures


def _runtime_failure_rerun_command(failures: list[dict[str, Any]]) -> str:
    commands: list[str] = []
    seen: set[tuple[str, str]] = set()
    for item in failures:
        row = item.get("row")
        if not isinstance(row, dict):
            continue
        job = row.get("job")
        if not isinstance(job, BenchJob):
            continue
        key = (job.manifest, job.protocol)
        if key in seen:
            continue
        seen.add(key)
        manifest = shlex.quote(job.manifest)
        protocol = shlex.quote(job.protocol)
        # Intentionally omit --no-resume to rerun only missing/crashed jobs.
        commands.append(f"python -m deltatau_audit bench run --manifest {manifest} --protocol {protocol}")
    return " && ".join(commands)


def _cartpole_retrain_commands(
    variant_seeds: dict[str, list[int]],
    *,
    timesteps: int = 45000,
    force: bool = True,
    base_speed: int = 3,
    jitter: int = 2,
    phase_period: int = 200,
) -> list[str]:
    return _shared_cartpole_retrain_commands(
        variant_seeds,
        timesteps=timesteps,
        force=force,
        base_speed=base_speed,
        jitter=jitter,
        phase_period=phase_period,
    )


def _ci_gate_summary_cleanup_command(paths: list[str]) -> str:
    return " && ".join(_shared_summary_cleanup_commands(paths))


def _quality_failure_repair_command(failures: list[dict[str, Any]]) -> tuple[str, list[str]]:
    commands: list[str] = []
    detail_reasons: list[str] = []
    seen_manifests: set[tuple[str, str]] = set()

    for item in failures:
        ci_gate_failures = _to_int(item.get("ci_gate_failures")) or 0
        if ci_gate_failures <= 0:
            continue
        row = item.get("row")
        if not isinstance(row, dict):
            continue
        job = row.get("job")
        if not isinstance(job, BenchJob):
            continue
        progress = row.get("progress")
        breakdown = progress.get("failure_breakdown") if isinstance(progress, dict) else {}
        breakdown_dict = breakdown if isinstance(breakdown, dict) else {}

        key = (job.manifest, job.protocol)
        if key in seen_manifests:
            continue
        seen_manifests.add(key)
        plan = _shared_build_quality_repair_plan(
            job_name=job.name,
            manifest=job.manifest,
            output_root=job.output_root,
            protocol=job.protocol,
            failure_breakdown=breakdown_dict,
            base_speed=3,
            jitter=2,
            phase_period=200,
            expected_jobs=_to_int(row.get("total")),
        )
        if plan is None:
            continue
        commands.extend(_shared_repair_plan_commands(plan))
        detail_reasons.extend(plan.reasons)

    if not commands:
        return "", []

    commands.append("python scripts/prepare_submission.py --check-only --strict-check")
    commands.append("python scripts/run_submission_pipeline.py --mode report --event-tail 500 --stall-seconds 1800")
    return " && ".join(commands), detail_reasons


def _build_recommendation(
    rows: list[dict[str, Any]],
    *,
    state_jobs: dict[str, Any],
    recent_events: list[dict[str, Any]] | None = None,
    now_ts: float | None = None,
) -> Recommendation:
    if now_ts is None:
        now_ts = time.time()
    incomplete_rows = [
        row for row in rows if int(row.get("total", 0)) <= 0 or int(row.get("done", 0)) < int(row.get("total", 0))
    ]
    if not incomplete_rows:
        completed_failures = _completed_failure_rows(rows)
        if completed_failures:
            total_failed = sum(item["failed_jobs"] for item in completed_failures)
            total_runtime = sum(item["runtime_failures"] for item in completed_failures)
            total_ci_gate = sum(item["ci_gate_failures"] for item in completed_failures)
            total_other = sum(item["other_failures"] for item in completed_failures)
            if total_runtime > 0:
                rerun_cmd = _runtime_failure_rerun_command(completed_failures)
                fallback_cmd = (
                    "python scripts/run_submission_pipeline.py --mode report --event-tail 500 --stall-seconds 1800 && "
                    "python scripts/run_submission_pipeline.py --mode diagnose --stall-seconds 1800"
                )
                return Recommendation(
                    action="rerun_runtime_failures",
                    command=rerun_cmd or fallback_cmd,
                    reasons=[
                        f"all jobs completed but bench_summary still reports failed jobs ({total_failed})",
                        f"runtime failures detected: {total_runtime} (ci_gate={total_ci_gate}, other={total_other})",
                    ],
                )
            quality_cmd, quality_details = _quality_failure_repair_command(completed_failures)
            if not quality_cmd:
                quality_cmd = (
                    "python scripts/prepare_submission.py --check-only --strict-check && "
                    "python scripts/run_submission_pipeline.py --mode report --event-tail 500 --stall-seconds 1800"
                )
            return Recommendation(
                action="improve_quality_gate_failures",
                command=quality_cmd,
                reasons=(
                    [
                        f"all jobs completed but CI quality gates failed ({total_ci_gate}/{total_failed} failures)",
                        "retraining or intervention tuning is required before finalize",
                    ]
                    + quality_details
                ),
            )
        return Recommendation(
            action="finalize",
            command=(
                "python scripts/prepare_submission.py --check-only --strict-check && "
                "python scripts/run_submission_pipeline.py --mode finalize"
            ),
            reasons=["all tracked jobs are complete"],
        )

    parallel_quality_cmd = ""
    parallel_quality_details: list[str] = []
    completed_failures = _completed_failure_rows(rows)
    completed_quality_failures = [
        item
        for item in completed_failures
        if (_to_int(item.get("ci_gate_failures")) or 0) > 0 and (_to_int(item.get("runtime_failures")) or 0) <= 0
    ]
    if completed_quality_failures:
        parallel_quality_cmd, parallel_quality_details = _quality_failure_repair_command(completed_quality_failures)

    has_recoverable_diagnosis = any(
        isinstance(row.get("diagnosis"), JobDiagnosis) and row["diagnosis"].recoverable for row in incomplete_rows
    )
    event_counts: dict[str, int] = {}
    if isinstance(recent_events, list) and recent_events:
        event_counts, _ = _summarize_supervisor_events(recent_events)
    recovery_failed_count = _to_int(event_counts.get("recovery_failed")) or 0
    recovered_count = _to_int(event_counts.get("recovered")) or 0
    max_no_progress_seconds = 0
    max_signature_hits = 0
    signature_loop_name = ""
    signature_loop_value = ""
    for row in incomplete_rows:
        job = row.get("job")
        if not isinstance(job, BenchJob):
            continue
        meta = _project_progress_meta(
            _supervisor_row(state_jobs, job.name),
            done=_to_int(row.get("done")) or 0,
            total=_to_int(row.get("total")) or 0,
            now_ts=float(now_ts),
        )
        last_progress_ts = _to_float(meta.get("last_progress_ts"))
        if isinstance(last_progress_ts, float) and last_progress_ts >= 0:
            max_no_progress_seconds = max(max_no_progress_seconds, int(now_ts - last_progress_ts))
        signature_hits = _to_int(meta.get("consecutive_signature_hits")) or 0
        last_signature = _normalize_signature(meta.get("last_signature"))
        if signature_hits > max_signature_hits and last_signature:
            max_signature_hits = signature_hits
            signature_loop_name = job.name
            signature_loop_value = last_signature

    if has_recoverable_diagnosis and recovery_failed_count >= 3 and recovered_count == 0:
        return Recommendation(
            action="investigate_recovery_failures",
            command=(
                "python scripts/run_submission_pipeline.py --mode report --event-tail 500 --stall-seconds 1800 && "
                "python scripts/run_submission_pipeline.py --mode diagnose --stall-seconds 1800"
            ),
            reasons=[
                f"recent supervisor events include {recovery_failed_count} recovery_failed and no successful recovery",
                "inspect logs/diagnosis before additional restart attempts",
            ],
        )

    if has_recoverable_diagnosis and max_signature_hits >= 3 and signature_loop_name and signature_loop_value:
        return Recommendation(
            action="investigate_signature_loop",
            command=(
                "python scripts/run_submission_pipeline.py --mode report --event-tail 500 --stall-seconds 1800 && "
                "python scripts/run_submission_pipeline.py --mode diagnose --stall-seconds 1800"
            ),
            reasons=[
                f"job '{signature_loop_name}' reports repeated signature '{signature_loop_value}' "
                f"({max_signature_hits} consecutive cycles)",
                "investigate root cause before spending additional restart budget",
            ],
        )

    if has_recoverable_diagnosis:
        return Recommendation(
            action="supervise_auto_recover",
            command=_recommended_autopilot_command(
                max_no_progress_cycles=60,
                max_no_progress_seconds=7200,
            ),
            reasons=["at least one job has recoverable diagnosis"],
        )

    if max_no_progress_seconds >= 7200:
        return Recommendation(
            action="supervise_with_timeout",
            command=_recommended_autopilot_command(
                max_no_progress_cycles=0,
                max_no_progress_seconds=7200,
            ),
            reasons=[f"max no-progress duration is high ({max_no_progress_seconds}s)"],
        )

    if parallel_quality_cmd:
        return Recommendation(
            action="parallelize_quality_repairs",
            command=parallel_quality_cmd,
            reasons=(
                [
                    "at least one bench is complete with CI quality-gate failures while others are still running",
                    "run quality repairs in parallel so strict-check can pass when remaining jobs finish",
                ]
                + parallel_quality_details
            ),
        )

    return Recommendation(
        action="continue_supervision",
        command=_recommended_autopilot_command(
            max_no_progress_cycles=0,
            max_no_progress_seconds=0,
        ),
        reasons=["jobs are still in progress and no immediate recovery signal was detected"],
    )


def _print_recommendation(jobs: list[BenchJob], *, stall_seconds: int, execute: bool = False) -> int:
    rows, _ = _collect_diagnose_rows(jobs, stall_seconds=max(1, int(stall_seconds)))
    state = _load_supervisor_state(SUPERVISOR_STATE_PATH)
    state_jobs = state.get("jobs")
    if not isinstance(state_jobs, dict):
        state_jobs = {}
    recent_events = _load_recent_events(SUPERVISOR_EVENTS_PATH, max_lines=500)
    event_counts, _ = _summarize_supervisor_events(recent_events)
    recommendation = _build_recommendation(
        rows,
        state_jobs=state_jobs,
        recent_events=recent_events,
        now_ts=time.time(),
    )

    print(f"recommend_time_utc: {_utc_now()}")
    if event_counts:
        parts = [f"{k}={event_counts[k]}" for k in sorted(event_counts)]
        print(f"recent_event_counts: {' '.join(parts)}")
    print(f"recommended_action: {recommendation.action}")
    print(f"recommended_command: {recommendation.command}")
    if recommendation.reasons:
        print("reasons:")
        for reason in recommendation.reasons:
            print(f"  - {reason}")
    if execute:
        print("executing_recommended_command: true")
        return _run_shell_cmd(recommendation.command)
    return 0


def _load_recent_events(path: Path, *, max_lines: int = 500) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw_lines = _read_tail_lines(path, n_lines=max(1, int(max_lines)))
    events: list[dict[str, Any]] = []
    for line in raw_lines:
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except Exception:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _summarize_supervisor_events(events: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    event_counts: dict[str, int] = {}
    event_counts_by_job: dict[str, dict[str, int]] = {}
    for event in events:
        name = event.get("event")
        if not isinstance(name, str) or not name:
            continue
        event_counts[name] = event_counts.get(name, 0) + 1
        job_name = event.get("job")
        if not isinstance(job_name, str) or not job_name:
            continue
        bucket = event_counts_by_job.setdefault(job_name, {})
        bucket[name] = bucket.get(name, 0) + 1
    return event_counts, event_counts_by_job


def _print_supervisor_report(
    jobs: list[BenchJob],
    *,
    stall_seconds: int,
    event_tail: int,
) -> int:
    rows, _ = _collect_diagnose_rows(jobs, stall_seconds=max(1, int(stall_seconds)))
    state = _load_supervisor_state(SUPERVISOR_STATE_PATH)
    state_jobs = state.get("jobs")
    if not isinstance(state_jobs, dict):
        state_jobs = {}
    now_ts = time.time()
    events = _load_recent_events(SUPERVISOR_EVENTS_PATH, max_lines=max(1, int(event_tail)))
    event_counts, event_counts_by_job = _summarize_supervisor_events(events)

    print(f"report_time_utc: {_utc_now()}")
    print(f"state_updated_at_utc: {state.get('updated_at_utc')}")
    print(f"events_tail_loaded: {len(events)}")
    if event_counts:
        parts = [f"{k}={event_counts[k]}" for k in sorted(event_counts)]
        print(f"event_counts: {' '.join(parts)}")
    for row in rows:
        job = row["job"]
        diagnosis: JobDiagnosis = row["diagnosis"]
        meta = _project_progress_meta(
            _supervisor_row(state_jobs, job.name),
            done=_to_int(row.get("done")) or 0,
            total=_to_int(row.get("total")) or 0,
            now_ts=now_ts,
        )
        print()
        print(f"[{job.name}]")
        print(f"  progress: {row['done']}/{row['total']} ({row['pct']:.1f}%)")
        print(f"  diagnosis: {diagnosis.code}")
        progress = row.get("progress")
        if isinstance(progress, dict):
            breakdown = progress.get("failure_breakdown")
            if isinstance(breakdown, dict):
                failed_total = _to_int(breakdown.get("failed_total")) or 0
                if failed_total > 0:
                    ci_gate = _to_int(breakdown.get("ci_gate_failures")) or 0
                    runtime = _to_int(breakdown.get("runtime_failures")) or 0
                    other = _to_int(breakdown.get("other_failures")) or 0
                    print(
                        "  bench_failures: "
                        f"total={failed_total} ci_gate={ci_gate} runtime={runtime} other={other}"
                    )
        print(f"  restart_count: {meta['restart_count']}")
        job_event_counts = event_counts_by_job.get(job.name)
        if isinstance(job_event_counts, dict) and job_event_counts:
            parts = [f"{k}:{job_event_counts[k]}" for k in sorted(job_event_counts)]
            print(f"  recent_events: {' '.join(parts)}")
        last_signature = _normalize_signature(meta.get("last_signature"))
        if last_signature:
            print(f"  last_error_signature: {last_signature}")
            print(f"  consecutive_signature_hits: {meta.get('consecutive_signature_hits', 0)}")
        restart_times_s = _normalize_restart_history(meta.get("restart_times_s"))
        if restart_times_s:
            recent_window_s = RECOMMENDED_RESTART_WINDOW_SECONDS
            restarts_in_recent_window = _count_restarts_in_window(
                restart_times_s,
                now_ts=now_ts,
                window_seconds=recent_window_s,
            )
            print(f"  restart_count_last_{recent_window_s}s: {restarts_in_recent_window}")
        reason_counts = meta.get("restarts_by_reason")
        if isinstance(reason_counts, dict) and reason_counts:
            parts = [f"{k}:{reason_counts[k]}" for k in sorted(reason_counts)]
            print(f"  restarts_by_reason: {' '.join(parts)}")
        signature_counts = meta.get("restarts_by_signature")
        if isinstance(signature_counts, dict) and signature_counts:
            parts = [f"{k}:{signature_counts[k]}" for k in sorted(signature_counts)]
            print(f"  restarts_by_signature: {' '.join(parts)}")
        print(f"  consecutive_recoverable: {meta['consecutive_recoverable']}")
        print(f"  no_progress_cycles: {meta['no_progress_cycles']}")
        last_progress_ts = _to_float(meta.get("last_progress_ts"))
        if isinstance(last_progress_ts, float) and last_progress_ts >= 0:
            no_progress_s = max(0, int(now_ts - last_progress_ts))
            print(f"  no_progress_seconds: {no_progress_s}")
        last_restart_ts = _to_float(meta.get("last_restart_ts"))
        if isinstance(last_restart_ts, float) and last_restart_ts >= 0:
            since_restart = max(0, int(now_ts - last_restart_ts))
            print(f"  since_last_restart_s: {since_restart}")
    return 0


def _run_cmd(cmd: list[str]) -> int:
    print(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(ROOT))
    return int(proc.returncode)


def _run_shell_cmd(command: str) -> int:
    print(f"$ {command}")
    proc = subprocess.run(command, cwd=str(ROOT), shell=True)
    return int(proc.returncode)


def _build_preflight_manifest(job: BenchJob, target_path: Path) -> Path:
    source_manifest_path = (ROOT / job.manifest).resolve()
    source_manifest = _load_manifest(source_manifest_path)
    source_jobs = source_manifest.get("jobs")
    if not isinstance(source_jobs, list) or not source_jobs:
        raise ValueError(f"Manifest has no jobs for preflight: {source_manifest_path}")

    reduced_jobs: list[dict[str, Any]] = []
    for idx, raw_job in enumerate(source_jobs):
        if not isinstance(raw_job, dict):
            continue
        reduced_job: dict[str, Any] = copy.deepcopy(raw_job)

        matrix = reduced_job.get("matrix")
        if isinstance(matrix, dict) and matrix:
            reduced_matrix: dict[str, list[Any]] = {}
            for key, values in matrix.items():
                if isinstance(values, list) and values:
                    reduced_matrix[key] = [values[0]]
                else:
                    reduced_matrix[key] = [0]
            reduced_job["matrix"] = reduced_matrix

        args = reduced_job.get("args", {})
        if not isinstance(args, dict):
            args = {}
        args = dict(args)
        reduced_name = reduced_job.get("name", f"job_{idx}")
        if not isinstance(reduced_name, str) or not reduced_name:
            reduced_name = f"job_{idx}"
        args["episodes"] = 1
        args["speeds"] = [1]
        args["seeds"] = [0]
        args["workers"] = 1
        args["protocol"] = "custom"
        # Preflight validates runtime wiring, not performance thresholds.
        args["ci"] = False
        for noisy_key in (
            "ci_gate_mode",
            "ci_min_deployment_pass_rate",
            "ci_min_stress_pass_rate",
            "target_ci_width",
            "bootstrap_samples",
        ):
            args.pop(noisy_key, None)
        args["out"] = f"_status_demo/preflight/{job.name}/{reduced_name}"
        reduced_job["args"] = args
        reduced_jobs.append(reduced_job)

    if not reduced_jobs:
        raise ValueError(f"Manifest had no valid job entries for preflight: {source_manifest_path}")

    preflight_manifest: dict[str, Any] = {
        "name": f"preflight_{job.name}",
        "description": f"Auto-generated preflight manifest for {job.name}",
        "output_dir": f"_status_demo/preflight/{job.name}",
        "jobs": reduced_jobs,
    }
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(yaml.safe_dump(preflight_manifest, sort_keys=False), encoding="utf-8")
    return target_path


def _run_preflight(jobs: list[BenchJob]) -> int:
    rc = 0
    preflight_dir = STATUS_DIR / "preflight_manifests"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    print(f"preflight_time_utc: {_utc_now()}")
    for job in jobs:
        target_manifest = preflight_dir / f"{job.name}.yaml"
        try:
            manifest_path = _build_preflight_manifest(job, target_manifest)
        except Exception as exc:
            print()
            print(f"[{job.name}]")
            print(f"  preflight: FAILED to build manifest ({exc})")
            rc |= 1
            continue

        print()
        print(f"[{job.name}]")
        print(f"  preflight_manifest: {manifest_path}")
        cmd = [
            sys.executable,
            "-m",
            "deltatau_audit",
            "bench",
            "run",
            "--manifest",
            str(manifest_path),
            "--protocol",
            "custom",
            "--allow-protocol-override",
            "--no-resume",
            "--fail-fast",
        ]
        cmd_rc = _run_cmd(cmd)
        print(f"  preflight_rc: {cmd_rc}")
        rc |= cmd_rc
    return rc


def _finalize(jobs: list[BenchJob]) -> int:
    rc = 0
    rc |= _run_cmd([sys.executable, "scripts/check_release_consistency.py"])
    rc |= _strict_readiness_check()
    for job in jobs:
        summary_path = (ROOT / job.output_root).resolve()
        if (summary_path / "bench_summary.json").exists():
            rc |= _run_cmd(
                [
                    sys.executable,
                    "-m",
                    "deltatau_audit",
                    "bench",
                    "table",
                    "--summary",
                    str(summary_path),
                ]
            )
    return rc


def _merge_jobs(defaults: list[BenchJob], active: list[BenchJob]) -> list[BenchJob]:
    by_name = {j.name: j for j in active}
    merged: list[BenchJob] = []
    for d in defaults:
        cur = by_name.get(d.name)
        if cur is None:
            merged.append(d)
            continue
        active_pid = cur.pid if _pid_alive(cur.pid) else None
        active_started_at = cur.started_at_utc if active_pid is not None else None
        merged.append(
            BenchJob(
                name=d.name,
                manifest=d.manifest,
                output_root=d.output_root,
                protocol=d.protocol,
                no_resume=d.no_resume,
                out_log=d.out_log,
                err_log=d.err_log,
                pid=active_pid,
                started_at_utc=active_started_at,
            )
        )
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Submission pipeline launcher/monitor/finalizer")
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "launch",
            "status",
            "finalize",
            "preflight",
            "diagnose",
            "supervise",
            "autopilot",
            "report",
            "recommend",
        ],
        required=True,
        help="Operation mode.",
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        default=False,
        help="Restart jobs even if recorded PID is still alive.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        default=False,
        help="In status mode, keep polling until interrupted.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Polling interval in seconds for --watch mode.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        default=False,
        help="Run lightweight manifest preflight before launch mode.",
    )
    parser.add_argument(
        "--stall-seconds",
        type=int,
        default=900,
        help="Diagnose mode threshold for stale log activity.",
    )
    parser.add_argument(
        "--auto-recover",
        action="store_true",
        default=False,
        help="In supervise mode, restart recoverable stalled/blocked jobs automatically.",
    )
    parser.add_argument(
        "--recover-after-consecutive",
        type=int,
        default=2,
        help="Require this many consecutive recoverable diagnoses before auto-recovery.",
    )
    parser.add_argument(
        "--max-restarts-per-job",
        type=int,
        default=2,
        help="Maximum automatic restarts per job in supervise mode.",
    )
    parser.add_argument(
        "--max-restarts-per-reason",
        type=int,
        default=2,
        help="Maximum automatic restarts per job for the same recovery reason (0 disables reason budget).",
    )
    parser.add_argument(
        "--max-restarts-per-signature",
        type=int,
        default=0,
        help="Maximum automatic restarts per job for the same error signature (0 disables signature budget).",
    )
    parser.add_argument(
        "--max-total-restarts",
        type=int,
        default=0,
        help="Maximum automatic restarts across all jobs in the current supervisor state (0 disables).",
    )
    parser.add_argument(
        "--max-restarts-per-window",
        type=int,
        default=0,
        help="Maximum automatic restarts per job within restart-window-seconds (0 disables).",
    )
    parser.add_argument(
        "--restart-window-seconds",
        type=int,
        default=0,
        help="Window size in seconds for max-restarts-per-window (0 disables).",
    )
    parser.add_argument(
        "--restart-cooldown-seconds",
        type=int,
        default=1800,
        help="Minimum seconds between automatic restarts of the same job in supervise mode.",
    )
    parser.add_argument(
        "--restart-backoff-factor",
        type=float,
        default=1.0,
        help="Exponential cooldown multiplier per restart (1.0 disables backoff).",
    )
    parser.add_argument(
        "--max-restart-cooldown-seconds",
        type=int,
        default=0,
        help="Cap for exponential restart cooldown (0 means no cap).",
    )
    parser.add_argument(
        "--recovery-grace-seconds",
        type=int,
        default=0,
        help="Suppress non-dead recoverable restarts for this long after a restart (0 disables).",
    )
    parser.add_argument(
        "--max-no-progress-cycles",
        type=int,
        default=60,
        help="Escalate to recovery when progress stays unchanged for this many supervise cycles (0 disables).",
    )
    parser.add_argument(
        "--max-no-progress-seconds",
        type=int,
        default=0,
        help="Escalate to recovery when progress stays unchanged for this many seconds (0 disables).",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=0,
        help="Maximum supervise cycles (0 means infinite loop).",
    )
    parser.add_argument(
        "--auto-finalize",
        action="store_true",
        default=False,
        help="In autopilot mode, run finalize after jobs complete.",
    )
    parser.add_argument(
        "--event-tail",
        type=int,
        default=500,
        help="In report mode, number of recent supervisor events to summarize.",
    )
    parser.add_argument(
        "--run-recommendation",
        action="store_true",
        default=False,
        help="In recommend mode, execute the recommended command in the repo root.",
    )
    args = parser.parse_args()

    defaults = _default_jobs()
    active = _load_active_jobs(ACTIVE_PATH)
    jobs = _merge_jobs(defaults, active)

    if args.mode == "launch":
        if args.preflight:
            preflight_rc = _run_preflight(jobs)
            if preflight_rc != 0:
                print("preflight failed; aborting launch.")
                return preflight_rc
        launched = _launch_jobs(jobs, force_restart=bool(args.force_restart))
        _print_status(launched, stall_seconds=max(1, int(args.stall_seconds)))
        return 0

    if args.mode == "status":
        if args.watch:
            try:
                while True:
                    _print_status(jobs, stall_seconds=max(1, int(args.stall_seconds)))
                    print()
                    print(f"sleeping {args.interval}s ...")
                    time.sleep(max(1, int(args.interval)))
            except KeyboardInterrupt:
                print("\ninterrupted.")
                return 0
        _print_status(jobs, stall_seconds=max(1, int(args.stall_seconds)))
        return 0

    if args.mode == "preflight":
        return _run_preflight(jobs)

    if args.mode == "diagnose":
        return _print_diagnose(jobs, stall_seconds=max(1, int(args.stall_seconds)))

    if args.mode == "supervise":
        return _supervise(
            jobs,
            interval_seconds=max(1, int(args.interval)),
            stall_seconds=max(1, int(args.stall_seconds)),
            auto_recover=bool(args.auto_recover),
            recover_after_consecutive=max(1, int(args.recover_after_consecutive)),
            max_restarts_per_job=max(0, int(args.max_restarts_per_job)),
            max_restarts_per_reason=max(0, int(args.max_restarts_per_reason)),
            max_restarts_per_signature=max(0, int(args.max_restarts_per_signature)),
            max_total_restarts=max(0, int(args.max_total_restarts)),
            max_restarts_per_window=max(0, int(args.max_restarts_per_window)),
            restart_window_seconds=max(0, int(args.restart_window_seconds)),
            restart_cooldown_seconds=max(0, int(args.restart_cooldown_seconds)),
            restart_backoff_factor=max(1.0, float(args.restart_backoff_factor)),
            max_restart_cooldown_seconds=max(0, int(args.max_restart_cooldown_seconds)),
            recovery_grace_seconds=max(0, int(args.recovery_grace_seconds)),
            max_no_progress_cycles=max(0, int(args.max_no_progress_cycles)),
            max_no_progress_seconds=max(0, int(args.max_no_progress_seconds)),
            max_cycles=max(0, int(args.max_cycles)),
            events_path=SUPERVISOR_EVENTS_PATH,
        )

    if args.mode == "autopilot":
        return _autopilot(
            jobs,
            do_preflight=bool(args.preflight),
            force_restart=bool(args.force_restart),
            interval_seconds=max(1, int(args.interval)),
            stall_seconds=max(1, int(args.stall_seconds)),
            auto_recover=bool(args.auto_recover),
            recover_after_consecutive=max(1, int(args.recover_after_consecutive)),
            max_restarts_per_job=max(0, int(args.max_restarts_per_job)),
            max_restarts_per_reason=max(0, int(args.max_restarts_per_reason)),
            max_restarts_per_signature=max(0, int(args.max_restarts_per_signature)),
            max_total_restarts=max(0, int(args.max_total_restarts)),
            max_restarts_per_window=max(0, int(args.max_restarts_per_window)),
            restart_window_seconds=max(0, int(args.restart_window_seconds)),
            restart_cooldown_seconds=max(0, int(args.restart_cooldown_seconds)),
            restart_backoff_factor=max(1.0, float(args.restart_backoff_factor)),
            max_restart_cooldown_seconds=max(0, int(args.max_restart_cooldown_seconds)),
            recovery_grace_seconds=max(0, int(args.recovery_grace_seconds)),
            max_no_progress_cycles=max(0, int(args.max_no_progress_cycles)),
            max_no_progress_seconds=max(0, int(args.max_no_progress_seconds)),
            max_cycles=max(0, int(args.max_cycles)),
            auto_finalize=bool(args.auto_finalize),
        )

    if args.mode == "report":
        return _print_supervisor_report(
            jobs,
            stall_seconds=max(1, int(args.stall_seconds)),
            event_tail=max(1, int(args.event_tail)),
        )

    if args.mode == "recommend":
        return _print_recommendation(
            jobs,
            stall_seconds=max(1, int(args.stall_seconds)),
            execute=bool(args.run_recommendation),
        )

    if args.mode == "finalize":
        return _finalize(jobs)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
