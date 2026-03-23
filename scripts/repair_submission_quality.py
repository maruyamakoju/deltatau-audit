"""Repair bench quality-gate failures with a reproducible retrain/rerun flow."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submission_health import build_quality_repair_plan, check_bench_execution

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency path
    psutil = None


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))
    return safe.strip("._-") or "repair_job"


def _default_state_paths(job_name: str) -> tuple[Path, Path, Path]:
    slug = _slug(job_name)
    base = ROOT / "_status_demo" / "repair_runs"
    return (
        base / f"{slug}.json",
        base / f"{slug}.out.log",
        base / f"{slug}.err.log",
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _pid_alive(pid: int | None) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    if psutil is not None:
        try:
            return bool(psutil.pid_exists(pid))
        except Exception:
            pass
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _shell_join(argv: list[str]) -> str:
    items = [str(part) for part in argv if str(part)]
    if not items:
        return ""
    if os.name == "nt":
        return subprocess.list2cmdline(items)
    return shlex.join(items)


def _launch_popen_kwargs() -> dict[str, object]:
    if os.name == "nt":
        create_new = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        detached = int(getattr(subprocess, "DETACHED_PROCESS", 0x00000008))
        return {"creationflags": create_new | detached}
    return {"start_new_session": True}


def _launch_background_process(argv: list[str], *, out_log: Path, err_log: Path) -> int:
    out_log.parent.mkdir(parents=True, exist_ok=True)
    err_log.parent.mkdir(parents=True, exist_ok=True)
    out_fp = out_log.open("ab")
    err_fp = err_log.open("ab")
    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(ROOT),
            stdout=out_fp,
            stderr=err_fp,
            **_launch_popen_kwargs(),
        )
    finally:
        out_fp.close()
        err_fp.close()
    return int(proc.pid)


def _powershell_executable() -> str:
    for candidate in ("pwsh.exe", "pwsh", "powershell.exe", "powershell"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return "powershell.exe"


def _powershell_quote(text: str) -> str:
    return "'" + str(text).replace("'", "''") + "'"


def _powershell_invoke(argv: list[str]) -> str:
    parts = [_powershell_quote(part) for part in argv if str(part)]
    return "& " + " ".join(parts)


def _scheduled_task_name(job_name: str) -> str:
    return f"CodexRepair_{_slug(job_name)}"


def _write_task_wrapper(
    *,
    wrapper_path: Path,
    argv: list[str],
    refresh_argv: list[str] | None,
    out_log: Path,
    err_log: Path,
    exit_status_path: Path,
) -> None:
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    command_display = _shell_join(argv)
    ps_command = _powershell_invoke(argv)
    refresh_display = _shell_join(refresh_argv or [])
    refresh_ps_command = _powershell_invoke(refresh_argv or []) if refresh_argv else ""
    content = "\n".join(
        [
            "$ErrorActionPreference = 'Stop'",
            f"Set-Location {_powershell_quote(str(ROOT))}",
            "$env:PYTHONUNBUFFERED = '1'",
            f"$stdoutPath = {_powershell_quote(str(out_log))}",
            f"$stderrPath = {_powershell_quote(str(err_log))}",
            f"$exitPath = {_powershell_quote(str(exit_status_path))}",
            f"$commandDisplay = {_powershell_quote(command_display)}",
            f"$refreshDisplay = {_powershell_quote(refresh_display)}",
            "\"[\" + [DateTime]::UtcNow.ToString('o') + \"] durable launch: \" + $commandDisplay | Out-File -FilePath $stdoutPath -Append -Encoding utf8",
            "try {",
            f"  {ps_command} 1>> $stdoutPath 2>> $stderrPath",
            "  $exitCode = $LASTEXITCODE",
            "  if ($refreshDisplay) {",
            "    \"[\" + [DateTime]::UtcNow.ToString('o') + \"] refreshing full summary: \" + $refreshDisplay | Out-File -FilePath $stdoutPath -Append -Encoding utf8",
            f"    {refresh_ps_command} 1>> $stdoutPath 2>> $stderrPath" if refresh_ps_command else "    $null = $null",
            "    $refreshExit = $LASTEXITCODE",
            "    if ($exitCode -eq 0) { $exitCode = $refreshExit }",
            "  }",
            "} catch {",
            "  $_ | Out-String | Out-File -FilePath $stderrPath -Append -Encoding utf8",
            "  $exitCode = 1",
            "}",
            "$payload = @{",
            "  exit_code = $exitCode",
            "  finished_at_utc = [DateTime]::UtcNow.ToString('o')",
            "  command = $commandDisplay",
            "  refresh_command = $refreshDisplay",
            "}",
            "$payload | ConvertTo-Json | Set-Content -Path $exitPath -Encoding utf8",
            "exit $exitCode",
            "",
        ]
    )
    wrapper_path.write_text(content, encoding="utf-8")


def _run_checked(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )


def _parse_schtasks_query(text: str) -> dict[str, object] | None:
    rows: dict[str, str] = {}
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        rows[key.strip().lower()] = value.strip()
    if not rows:
        return None
    last_result_raw = rows.get("last result", "")
    last_task_result: int | str | None = None
    if last_result_raw:
        try:
            last_task_result = int(last_result_raw)
        except Exception:
            last_task_result = last_result_raw
    return {
        "exists": True,
        "state": rows.get("status", ""),
        "last_task_result": last_task_result,
        "last_run_time": rows.get("last run time", ""),
        "next_run_time": rows.get("next run time", ""),
    }


def _schedule_background_task(
    *,
    job_name: str,
    argv: list[str],
    refresh_argv: list[str] | None,
    state_path: Path,
    out_log: Path,
    err_log: Path,
) -> dict[str, object]:
    task_name = _scheduled_task_name(job_name)
    wrapper_path = state_path.with_suffix(".task.ps1")
    exit_status_path = state_path.with_suffix(".exit.json")
    if exit_status_path.exists():
        exit_status_path.unlink()
    _write_task_wrapper(
        wrapper_path=wrapper_path,
        argv=argv,
        refresh_argv=refresh_argv,
        out_log=out_log,
        err_log=err_log,
        exit_status_path=exit_status_path,
    )

    start_time = (datetime.now().astimezone() + timedelta(minutes=1)).strftime("%H:%M")
    powershell = _powershell_executable()
    task_command = subprocess.list2cmdline(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(wrapper_path),
        ]
    )
    create_cmd = [
        "schtasks",
        "/Create",
        "/TN",
        task_name,
        "/SC",
        "ONCE",
        "/ST",
        start_time,
        "/TR",
        task_command,
        "/F",
    ]
    run_cmd = ["schtasks", "/Run", "/TN", task_name]

    created = _run_checked(create_cmd)
    if created.returncode != 0:
        raise RuntimeError(
            "scheduled task create failed: "
            + (created.stderr.strip() or created.stdout.strip() or f"rc={created.returncode}")
        )
    started = _run_checked(run_cmd)
    if started.returncode != 0:
        raise RuntimeError(
            "scheduled task run failed: "
            + (started.stderr.strip() or started.stdout.strip() or f"rc={started.returncode}")
        )
    return {
        "launch_mode": "durable",
        "pid": None,
        "task_name": task_name,
        "wrapper_path": str(wrapper_path),
        "exit_status_path": str(exit_status_path),
    }


def _scheduled_task_status(task_name: str) -> dict[str, object] | None:
    if os.name != "nt":
        return None
    powershell = _powershell_executable()
    query = "\n".join(
        [
            "$ErrorActionPreference = 'Stop'",
            f"$taskName = {_powershell_quote(task_name)}",
            "try {",
            "  $task = Get-ScheduledTask -TaskName $taskName -ErrorAction Stop",
            "  $info = Get-ScheduledTaskInfo -TaskName $taskName -ErrorAction Stop",
            "  [pscustomobject]@{",
            "    exists = $true",
            "    state = [string]$task.State",
            "    last_task_result = [int]$info.LastTaskResult",
            "    last_run_time = if ($info.LastRunTime) { $info.LastRunTime.ToString('o') } else { '' }",
            "    next_run_time = if ($info.NextRunTime) { $info.NextRunTime.ToString('o') } else { '' }",
            "  } | ConvertTo-Json -Compress",
            "} catch {",
            "  [pscustomobject]@{ exists = $false; error = $_.Exception.Message } | ConvertTo-Json -Compress",
            "}",
        ]
    )
    proc = subprocess.run(
        [powershell, "-NoProfile", "-Command", query],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    text = (proc.stdout or "").strip()
    if text:
        try:
            payload = json.loads(text)
        except Exception:
            payload = None
        if isinstance(payload, dict) and payload.get("exists", True) is not False:
            return payload
    fallback = subprocess.run(
        ["schtasks", "/Query", "/TN", task_name, "/V", "/FO", "LIST"],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    fallback_text = (fallback.stdout or fallback.stderr or "").strip()
    if not fallback_text or fallback.returncode != 0:
        return None
    return _parse_schtasks_query(fallback_text)


def _summary_progress(payload: dict[str, object]) -> tuple[int, int]:
    raw = payload.get("summary_paths")
    if not isinstance(raw, list):
        return 0, 0
    targets = [Path(str(item)) for item in raw]
    done = sum(1 for path in targets if path.exists())
    return done, len(targets)


def _build_manifest_argv(plan) -> list[str] | None:
    if not getattr(plan, "repair_manifest_path", ""):
        return None
    if getattr(plan, "rerun_scope", "") != "focused":
        return None
    focused_job_ids = [str(job_id) for job_id in getattr(plan, "focused_job_ids", ()) if str(job_id)]
    if not focused_job_ids:
        return None
    argv = [
        sys.executable,
        "scripts/build_failed_job_manifest.py",
        "--manifest",
        str(plan.manifest),
        "--output-root",
        str(plan.output_root),
        "--out-manifest",
        str(plan.repair_manifest_path),
    ]
    repair_output_dir = str(getattr(plan, "repair_output_dir", "")).strip()
    if repair_output_dir:
        argv.extend(["--output-dir", repair_output_dir])
    for job_id in focused_job_ids:
        argv.extend(["--job-id", job_id])
    return argv


def _build_rerun_argv(plan) -> list[str]:
    manifest = str(plan.manifest)
    if getattr(plan, "rerun_scope", "") == "focused" and getattr(plan, "repair_manifest_path", ""):
        manifest = str(plan.repair_manifest_path)
    argv = [
        sys.executable,
        "-m",
        "deltatau_audit",
        "bench",
        "run",
        "--manifest",
        manifest,
        "--protocol",
        str(plan.protocol),
    ]
    if bool(getattr(plan, "use_no_resume", False)):
        argv.append("--no-resume")
    return argv


def _build_refresh_summary_argv(plan) -> list[str] | None:
    if getattr(plan, "rerun_scope", "") != "focused":
        return None
    output_root = str(getattr(plan, "output_root", "")).strip()
    repair_output_dir = str(getattr(plan, "repair_output_dir", "")).strip()
    if not output_root or not repair_output_dir:
        return None
    return [
        sys.executable,
        "scripts/merge_bench_summaries.py",
        "--base-summary",
        str(Path(output_root) / "bench_summary.json"),
        "--patch-summary",
        str(Path(repair_output_dir) / "bench_summary.json"),
        "--output-root",
        output_root,
    ]


def _prepare_background_launch(plan) -> tuple[list[str] | None, list[str], list[str] | None]:
    return _build_manifest_argv(plan), _build_rerun_argv(plan), _build_refresh_summary_argv(plan)


def _read_exit_status(path: Path) -> dict[str, object] | None:
    return _load_json(path)


def _background_alive(payload: dict[str, object]) -> bool:
    pid_raw = payload.get("pid")
    pid = int(pid_raw) if isinstance(pid_raw, int) else None
    if _pid_alive(pid):
        return True
    task_name = payload.get("task_name")
    if isinstance(task_name, str) and task_name.strip():
        task = _scheduled_task_status(task_name.strip())
        state = str(task.get("state", "")) if isinstance(task, dict) else ""
        if state.lower() in {"running", "queued"}:
            return True
    return False


def _print_status_from_state(state_path: Path) -> int:
    payload = _load_json(state_path)
    if payload is None:
        print(f"state missing: {state_path}")
        return 1
    pid_raw = payload.get("pid")
    pid = int(pid_raw) if isinstance(pid_raw, int) else None
    alive = _background_alive(payload)
    done, total = _summary_progress(payload)
    task_name = payload.get("task_name")
    task_status = None
    if isinstance(task_name, str) and task_name.strip():
        task_status = _scheduled_task_status(task_name.strip())
    exit_status = None
    exit_status_path = payload.get("exit_status_path")
    if isinstance(exit_status_path, str) and exit_status_path.strip():
        exit_status = _read_exit_status(Path(exit_status_path))
    print(f"state_path: {state_path}")
    print(f"job_name: {payload.get('job_name')}")
    print(f"pid: {pid}")
    print(f"alive: {alive}")
    print(f"launch_mode: {payload.get('launch_mode')}")
    print(f"launched_at_utc: {payload.get('launched_at_utc')}")
    print(f"progress: {done}/{total}")
    print(f"rerun_command: {payload.get('command')}")
    prepare_command = payload.get("prepare_command")
    if isinstance(prepare_command, str) and prepare_command:
        print(f"prepare_command: {prepare_command}")
    refresh_command = payload.get("refresh_command")
    if isinstance(refresh_command, str) and refresh_command:
        print(f"refresh_command: {refresh_command}")
    print(f"out_log: {payload.get('out_log')}")
    print(f"err_log: {payload.get('err_log')}")
    if isinstance(task_name, str) and task_name:
        print(f"task_name: {task_name}")
    if isinstance(task_status, dict):
        print(f"task_state: {task_status.get('state')}")
        print(f"task_last_result: {task_status.get('last_task_result')}")
    if isinstance(exit_status_path, str) and exit_status_path:
        print(f"exit_status_path: {exit_status_path}")
    if isinstance(exit_status, dict):
        print(f"exit_code: {exit_status.get('exit_code')}")
        print(f"finished_at_utc: {exit_status.get('finished_at_utc')}")
    if alive:
        return 0
    if total > 0 and done >= total:
        return 0
    exit_code = exit_status.get("exit_code") if isinstance(exit_status, dict) else None
    if isinstance(exit_code, int):
        return 0 if exit_code == 0 else 1
    return 1


def _run_shell(command: str, *, allow_failure: bool) -> int:
    print(f"$ {command}", flush=True)
    started = time.time()
    proc = subprocess.run(command, cwd=str(ROOT), shell=True, check=False)
    elapsed = time.time() - started
    print(f"  rc={proc.returncode} elapsed_s={elapsed:.1f}", flush=True)
    if proc.returncode != 0 and not allow_failure:
        print("  aborting on non-zero exit code", flush=True)
    return int(proc.returncode)


def _run_argv(argv: list[str], *, allow_failure: bool) -> int:
    display = _shell_join(argv)
    print(f"$ {display}", flush=True)
    started = time.time()
    proc = subprocess.run(argv, cwd=str(ROOT), check=False)
    elapsed = time.time() - started
    print(f"  rc={proc.returncode} elapsed_s={elapsed:.1f}", flush=True)
    if proc.returncode != 0 and not allow_failure:
        print("  aborting on non-zero exit code", flush=True)
    return int(proc.returncode)


def _cleanup_failed_summaries(paths: tuple[str, ...]) -> tuple[int, int]:
    removed = 0
    missing = 0
    for raw in paths:
        path = Path(raw)
        if path.exists():
            path.unlink()
            removed += 1
        else:
            missing += 1
    print(
        f"cleanup_failed_summaries: removed={removed} missing={missing}",
        flush=True,
    )
    return removed, missing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repair bench quality-gate failures and rerun strict checks.",
    )
    parser.add_argument("--manifest", required=True, help="Bench manifest path.")
    parser.add_argument("--output-root", required=True, help="Bench output root.")
    parser.add_argument(
        "--state-path",
        default="",
        help="Optional state JSON path for background repair runs.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        default=False,
        help="Read the repair state file and print background job status.",
    )
    parser.add_argument(
        "--launch-background",
        action="store_true",
        default=False,
        help="Launch the rerun command in the background after retrain/cleanup.",
    )
    parser.add_argument(
        "--launch-mode",
        choices=["auto", "session", "durable"],
        default="auto",
        help="Background launch mode. auto uses durable tasks on Windows and session detachment elsewhere.",
    )
    parser.add_argument(
        "--protocol",
        default="paper",
        help="Protocol used when rerunning the bench (default: paper).",
    )
    parser.add_argument(
        "--job-name",
        default=None,
        help="Optional label shown in the repair summary.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=45000,
        help="Timesteps for targeted CartPole retraining (default: 45000).",
    )
    parser.add_argument(
        "--base-speed",
        type=int,
        default=3,
        help="Base speed for targeted CartPole jitter retraining (default: 3).",
    )
    parser.add_argument(
        "--jitter",
        type=int,
        default=2,
        help="Jitter range for targeted CartPole retraining (default: 2).",
    )
    parser.add_argument(
        "--phase-period",
        type=int,
        default=200,
        help="Phase period for time-feature retraining variants (default: 200).",
    )
    parser.add_argument(
        "--skip-retrain",
        action="store_true",
        default=False,
        help="Skip retraining even when the repair plan includes retrain steps.",
    )
    parser.add_argument(
        "--no-force-retrain",
        action="store_true",
        default=False,
        help="Do not pass --force to targeted retrain commands.",
    )
    parser.add_argument(
        "--skip-strict-check",
        action="store_true",
        default=False,
        help="Skip prepare_submission --strict-check after rerun.",
    )
    parser.add_argument(
        "--skip-pipeline-report",
        action="store_true",
        default=False,
        help="Skip pipeline report after rerun.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print the repair plan without executing it.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    manifest_path = _resolve_path(args.manifest)
    output_root = _resolve_path(args.output_root)
    state_path, out_log, err_log = _default_state_paths(args.job_name or output_root.name)
    if args.state_path:
        state_path = _resolve_path(args.state_path)
        out_log = state_path.with_suffix(".out.log")
        err_log = state_path.with_suffix(".err.log")

    if args.status:
        return _print_status_from_state(state_path)

    bench = check_bench_execution(manifest_path, output_root)
    failure_breakdown = bench.get("failure_breakdown")
    plan = build_quality_repair_plan(
        job_name=args.job_name or output_root.name,
        manifest=args.manifest,
        output_root=args.output_root,
        protocol=args.protocol,
        failure_breakdown=failure_breakdown,
        timesteps=int(args.timesteps),
        force_retrain=not bool(args.no_force_retrain),
        include_retrain=not bool(args.skip_retrain),
        base_speed=int(args.base_speed),
        jitter=int(args.jitter),
        phase_period=int(args.phase_period),
    )

    if plan is None:
        print("No CI quality-gate failures detected; nothing to repair.")
        return 0

    print(f"repair_job: {plan.job_name}")
    print(f"manifest: {plan.manifest}")
    print(f"output_root: {output_root}")
    print(f"ci_gate_failures: {plan.ci_gate_failures}")
    for reason in plan.reasons:
        print(f"reason: {reason}")
    for command in plan.retrain_commands:
        print(f"retrain: {command}")
    if plan.cleanup_summary_paths:
        print(f"cleanup_paths: {len(plan.cleanup_summary_paths)}")
    print(f"rerun: {plan.rerun_command}")
    if args.launch_background:
        print(f"background_state: {state_path}")
        print(f"background_out_log: {out_log}")
        print(f"background_err_log: {err_log}")
    if not args.skip_strict_check:
        print("post_check: python scripts/prepare_submission.py --check-only --strict-check")
    if not args.skip_pipeline_report:
        print("post_check: python scripts/run_submission_pipeline.py --mode report --event-tail 500 --stall-seconds 1800")

    if args.dry_run:
        return 0

    for command in plan.retrain_commands:
        rc = _run_shell(command, allow_failure=False)
        if rc != 0:
            return rc

    _cleanup_failed_summaries(plan.cleanup_summary_paths)
    prepare_argv, rerun_argv, refresh_argv = _prepare_background_launch(plan)
    prepare_command = _shell_join(prepare_argv) if prepare_argv else ""
    rerun_command = _shell_join(rerun_argv)
    refresh_command = _shell_join(refresh_argv) if refresh_argv else ""

    if args.launch_background:
        if prepare_argv:
            prepare_rc = _run_argv(prepare_argv, allow_failure=False)
            if prepare_rc != 0:
                return prepare_rc
        try:
            launch_mode = args.launch_mode
            if launch_mode == "auto":
                launch_mode = "durable" if os.name == "nt" else "session"
            if launch_mode == "durable":
                launch_info = _schedule_background_task(
                    job_name=plan.job_name,
                    argv=rerun_argv,
                    refresh_argv=refresh_argv,
                    state_path=state_path,
                    out_log=out_log,
                    err_log=err_log,
                )
            else:
                launch_info = {
                    "launch_mode": "session",
                    "pid": _launch_background_process(rerun_argv, out_log=out_log, err_log=err_log),
                }
        except Exception as exc:
            print(f"background launch failed: {exc}")
            return 1
        state_payload = {
            "job_name": plan.job_name,
            "manifest": str(manifest_path),
            "output_root": str(output_root),
            "protocol": args.protocol,
            "command": rerun_command,
            "prepare_command": prepare_command,
            "refresh_command": refresh_command,
            "pid": launch_info.get("pid"),
            "launch_mode": launch_info.get("launch_mode"),
            "launched_at_utc": _utc_now(),
            "out_log": str(out_log),
            "err_log": str(err_log),
            "rerun_scope": plan.rerun_scope,
            "repair_manifest_path": plan.repair_manifest_path,
            "focused_job_ids": list(plan.focused_job_ids),
            "summary_paths": list(plan.cleanup_summary_paths),
            "task_name": launch_info.get("task_name"),
            "wrapper_path": launch_info.get("wrapper_path"),
            "exit_status_path": launch_info.get("exit_status_path"),
        }
        _write_json(state_path, state_payload)
        if isinstance(launch_info.get("pid"), int):
            print(f"background_pid: {launch_info.get('pid')}")
        if isinstance(launch_info.get("task_name"), str) and launch_info.get("task_name"):
            print(f"background_task: {launch_info.get('task_name')}")
        return 0

    if prepare_argv:
        prepare_rc = _run_argv(prepare_argv, allow_failure=False)
        if prepare_rc != 0:
            return prepare_rc
    rerun_rc = _run_argv(rerun_argv, allow_failure=True)
    refresh_rc = 0
    if refresh_argv:
        refresh_rc = _run_argv(refresh_argv, allow_failure=True)
    strict_rc = 0
    report_rc = 0

    if not args.skip_strict_check:
        strict_rc = _run_shell(
            "python scripts/prepare_submission.py --check-only --strict-check",
            allow_failure=True,
        )
    if not args.skip_pipeline_report:
        report_rc = _run_shell(
            "python scripts/run_submission_pipeline.py --mode report --event-tail 500 --stall-seconds 1800",
            allow_failure=True,
        )

    if strict_rc != 0:
        return strict_rc
    if rerun_rc != 0:
        return rerun_rc
    if refresh_rc != 0:
        return refresh_rc
    return report_rc


if __name__ == "__main__":
    raise SystemExit(main())
