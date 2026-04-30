#!/usr/bin/env python3
"""Codex-driven autonomous research loop.

This module puts the LLM itself in the loop:
  1. Codex inspects the workspace and proposes the next experiment.
  2. The chosen frontier is executed with the proposed hyperparameters.
  3. Codex critiques the result and recommends the next move.
  4. Token usage, prompts, and raw Codex JSONL traces are persisted.

Usage:
    python experiments/codex_autonomous_lab.py --cycles 0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from jsonschema import ValidationError
from jsonschema import validate as jsonschema_validate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import autonomous_research as base
from _orch_shared import (
    CodexCallRecord,  # noqa: F401  (backwards-compat re-export)
    CodexLabJournal,  # noqa: F401  (backwards-compat re-export)
    CodexUsage,  # noqa: F401  (backwards-compat re-export)
    LLMRunnerBase,
    _coerce_subprocess_output,
    _safe_int,
    build_experiment_command,
    experiment_record_from_dict,
    should_stop,
    usage_summary,
    write_json,
)

WINDOWS_CONTROL_EVENT_EXIT = 0xC000013A


def parse_codex_exec_output(
    *,
    label: str,
    stdout_text: str,
    stderr_text: str,
    returncode: int,
    duration_sec: float,
    prompt_path: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> CodexCallRecord:
    """Parse Codex JSONL stdout into a structured call record."""
    usage = CodexUsage()
    final_message = ""
    session_id = None
    parsed_json = None
    parse_error = None
    event_error = None

    try:
        if stdout_text.strip():
            for raw_line in stdout_text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue

                event_type = str(event.get("type", ""))
                if event_type == "thread.started":
                    session_id = event.get("thread_id") or session_id
                elif event_type == "item.completed":
                    item = event.get("item", {})
                    if isinstance(item, dict) and item.get("type") == "agent_message":
                        final_message = str(item.get("text", "") or "")
                elif event_type == "turn.completed":
                    turn_usage = event.get("usage", {})
                    if isinstance(turn_usage, dict):
                        usage.input_tokens = _safe_int(turn_usage.get("input_tokens", usage.input_tokens))
                        usage.cached_input_tokens = _safe_int(
                            turn_usage.get("cached_input_tokens", usage.cached_input_tokens)
                        )
                        usage.output_tokens = _safe_int(turn_usage.get("output_tokens", usage.output_tokens))
                elif event_type == "error":
                    event_error = str(event.get("message", "") or "").strip() or event_error

            # Backward-compatible fallback for older single-object payloads.
            if not final_message:
                event = json.loads(stdout_text.strip())
                if isinstance(event, dict):
                    session_id = event.get("session_id") or session_id
                    final_message = str(event.get("response", "") or "")
                    stats = event.get("stats", {})
                    if isinstance(stats, dict):
                        tokens = stats.get("tokens", {})
                        if isinstance(tokens, dict):
                            usage.input_tokens = _safe_int(tokens.get("input", usage.input_tokens))
                            usage.cached_input_tokens = _safe_int(tokens.get("cached", usage.cached_input_tokens))
                            usage.output_tokens = _safe_int(tokens.get("candidates", usage.output_tokens))
    except Exception:
        pass

    if final_message:
        try:
            # Strip potential markdown code blocks before JSON parsing
            clean_message = final_message.strip()
            if clean_message.startswith("```json"):
                clean_message = clean_message[7:]
            elif clean_message.startswith("```"):
                clean_message = clean_message[3:]
            if clean_message.endswith("```"):
                clean_message = clean_message[:-3]
            clean_message = clean_message.strip()

            parsed_json = json.loads(clean_message)
        except json.JSONDecodeError as exc:
            parse_error = f"Final message was not valid JSON: {exc}"

    error = None
    if returncode != 0:
        detail = stderr_text.strip() or stdout_text.strip()
        if event_error:
            detail = event_error
        error = f"codex exec failed with exit code {returncode}: {detail[:1000]}"
    elif not final_message:
        error = "codex exec did not return a final agent message"
    elif parse_error:
        error = parse_error

    return CodexCallRecord(
        label=label,
        timestamp=datetime.now(timezone.utc).isoformat(),
        duration_sec=duration_sec,
        returncode=returncode,
        session_id=session_id,
        final_message=final_message,
        parsed_json=parsed_json if isinstance(parsed_json, dict) else None,
        usage=usage,
        prompt_path=str(prompt_path),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        error=error,
    )


class CodexExecRunner(LLMRunnerBase):
    """Thin wrapper around `codex exec --json`."""

    _CLI_CANDIDATES = ("codex.cmd", "codex", "codex.ps1", "gemini.cmd", "gemini", "gemini.ps1")

    def __init__(self, model: Optional[str], timeout_seconds: int, max_retries: int = 2) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, int(max_retries))
        self.codex_command = self._resolve_cli_command(self._CLI_CANDIDATES, name="codex")

    @staticmethod
    def _is_retryable_windows_exit(returncode: int) -> bool:
        """Treat console-close / Ctrl+C style exits as transient on Windows."""
        return os.name == "nt" and int(returncode) == WINDOWS_CONTROL_EVENT_EXIT

    def run_json_prompt(
        self,
        *,
        label: str,
        prompt: str,
        schema_path: Path,
        out_dir: Path,
    ) -> CodexCallRecord:
        prompt_path = out_dir / f"{label}_prompt.txt"
        stdout_path = out_dir / f"{label}_stdout.jsonl"
        stderr_path = out_dir / f"{label}_stderr.log"
        prompt_path.write_text(prompt, encoding="utf-8")

        command = [
            self.codex_command,
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
        ]
        if self.model:
            command.extend(["--model", self.model])
        command.append("-")

        attempts: List[CodexCallRecord] = []
        attempt_count = self.max_retries + 1
        for attempt_index in range(attempt_count):
            start = time.perf_counter()
            try:
                completed = subprocess.run(
                    command,
                    input=prompt,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    capture_output=True,
                    cwd=PROJECT_ROOT,
                    timeout=self.timeout_seconds,
                    check=False,
                    creationflags=self._creationflags(),
                )
                duration = time.perf_counter() - start
                stdout_path.write_text(completed.stdout, encoding="utf-8")
                stderr_path.write_text(completed.stderr, encoding="utf-8")
                record = parse_codex_exec_output(
                    label=label,
                    stdout_text=completed.stdout,
                    stderr_text=completed.stderr,
                    returncode=completed.returncode,
                    duration_sec=duration,
                    prompt_path=prompt_path,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )
                if record.parsed_json is not None:
                    try:
                        schema = json.loads(schema_path.read_text(encoding="utf-8"))
                        jsonschema_validate(record.parsed_json, schema)
                    except ValidationError as exc:
                        record.error = f"Codex JSON failed schema validation: {exc.message}"
                    except Exception as exc:
                        record.error = f"Failed to validate Codex JSON: {exc}"

                if not self._is_retryable_windows_exit(record.returncode) or attempt_index >= self.max_retries:
                    return record

                retry_note = (
                    f"Transient Windows console-control exit {record.returncode}; "
                    f"retrying ({attempt_index + 1}/{self.max_retries})"
                )
                stderr_path.write_text(
                    (completed.stderr or "") + f"\n{retry_note}\n",
                    encoding="utf-8",
                )
                attempts.append(record)
                time.sleep(min(5.0, 1.5 * (attempt_index + 1)))
            except subprocess.TimeoutExpired as exc:
                duration = time.perf_counter() - start
                stdout_text = _coerce_subprocess_output(exc.stdout)
                stderr_text = _coerce_subprocess_output(exc.stderr)
                stdout_path.write_text(stdout_text, encoding="utf-8")
                stderr_path.write_text(stderr_text, encoding="utf-8")

                record = parse_codex_exec_output(
                    label=label,
                    stdout_text=stdout_text,
                    stderr_text=stderr_text,
                    returncode=0,
                    duration_sec=duration,
                    prompt_path=prompt_path,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )
                if record.final_message:
                    record.returncode = 0
                    record.error = None
                    return record

                record.returncode = 124
                record.error = f"codex exec timed out after {self.timeout_seconds} seconds"
                return record

        if attempts:
            return attempts[-1]
        raise RuntimeError("Unreachable codex exec retry state")


def _recent_record_summary(journal: base.ResearchJournal, limit: int = 5) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for record in journal.records[-limit:]:
        summary.append({
            "cycle": record.cycle,
            "frontier": record.frontier,
            "status": record.status,
            "composite_score": record.metrics.get("composite_score", 0.0),
            "duration_sec": round(record.duration_sec, 2),
            "finding": record.finding,
        })
    return summary


def _frontier_summary(journal: base.ResearchJournal) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for name in base.FRONTIER_REGISTRY:
        scores = journal.frontier_scores.get(name, [])
        best = journal.best_per_frontier.get(name, {})
        durations = [r.duration_sec for r in journal.records if r.frontier == name and r.status == "success"]
        summary[name] = {
            "description": base.FRONTIER_REGISTRY[name].description,
            "runs": len(scores),
            "best_score": best.get("score", 0.0),
            "best_cycle": best.get("cycle"),
            "mean_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "mean_success_duration_sec": round(sum(durations) / len(durations), 2) if durations else None,
        }
    return summary


def build_strategy_prompt(
    cycle: int,
    journal: base.ResearchJournal,
    out_root: Path,
) -> str:
    frontier_summary = json.dumps(_frontier_summary(journal), indent=2, ensure_ascii=False)
    recent_records = json.dumps(_recent_record_summary(journal), indent=2, ensure_ascii=False)

    return f"""
You are the strategy lead for a long-horizon autonomous research lab running inside this repository.

This is not a cheap reply. Spend real effort and tokens inspecting the workspace before answering.
Read the current journal, inspect relevant frontier code, and choose one concrete next experiment.
You are in the strategy phase only: do not edit files in this phase.

Required workspace context to inspect before deciding:
- {out_root / 'journal.json'}
- {PROJECT_ROOT / 'experiments' / 'autonomous_research.py'}
- one or more relevant files under {PROJECT_ROOT / 'experiments' / 'frontiers'}

Mission:
- push beyond the current best composite score with a nontrivial but executable experiment
- prefer experiments that are likely to generate useful signal without wasting wall-clock time
- if a frontier looks saturated or too slow, switch aggressively

Current cycle: {cycle}
Current frontier summary:
{frontier_summary}

Recent experiment records:
{recent_records}

Return JSON only with exactly these keys:
{{
  "objective": "short string",
  "selected_frontier": "one frontier name",
  "hyperparams": {{"param_name": 1}},
  "rationale": "why this is the best next shot",
  "predicted_upside": 0.0,
  "risk_factors": ["risk 1"],
  "read_set": ["path/you/read"],
  "experiment_brief": "one paragraph",
  "code_change_required": false,
  "confidence": 0.0
}}
""".strip()


def build_critique_prompt(
    cycle: int,
    journal: base.ResearchJournal,
    strategy_payload: Dict[str, Any],
    experiment_record: base.ExperimentRecord,
) -> str:
    recent_records = json.dumps(_recent_record_summary(journal), indent=2, ensure_ascii=False)
    experiment_json = json.dumps(asdict(experiment_record), indent=2, ensure_ascii=False)
    strategy_json = json.dumps(strategy_payload, indent=2, ensure_ascii=False)

    return f"""
You are the scientific critic for the same autonomous research lab.

This is not a cheap summary. Spend real effort reading the latest result in the context of prior runs.
Explain whether the signal is real, what mechanism may explain it, and what the next move should be.
You may inspect code and prior artifacts, but do not edit files in this critique phase.

Cycle: {cycle}
Strategy that led to the run:
{strategy_json}

Latest experiment record:
{experiment_json}

Recent experiment records:
{recent_records}

Return JSON only with exactly these keys:
{{
  "summary": "short string",
  "mechanistic_take": "mechanistic interpretation",
  "signal_quality": "weak",
  "next_action": "explore_more",
  "followup_frontier": "frontier name",
  "followup_hyperparams": {{"param_name": 1}},
  "red_flags": ["flag 1"],
  "novelty_assessment": "short string",
  "proposed_new_frontier": null
}}

Special option — proposing a NEW frontier axis:
- Set next_action to "propose_new_frontier" and populate proposed_new_frontier when the existing frontier set looks saturated or mis-targeted. Leave it null otherwise.
- The proposal must include: name (snake_case, 3-40 chars, unique), description (one line), rationale (why this matters now), hypothesis (testable), and optional skeleton_python (a Python file stub with a run(params) -> dict entry point).
- This is the only way the lab can grow its search space beyond the current 10 axes. Use it when you have a concrete, testable new idea — not as filler.
""".strip()


def run_experiment_isolated(
    *,
    cycle: int,
    frontier_name: str,
    params: Dict[str, Any],
    journal_path: Path,
    out_root: Path,
    codex_cycle_dir: Path,
    timeout_seconds: int,
) -> base.ExperimentRecord:
    """Run one experiment in a child Python process to isolate CUDA state."""
    params_path = codex_cycle_dir / "experiment_params.json"
    result_path = codex_cycle_dir / "experiment_record.json"
    stdout_path = codex_cycle_dir / "experiment_stdout.log"
    stderr_path = codex_cycle_dir / "experiment_stderr.log"
    params_path.write_text(json.dumps(params, indent=2, default=str), encoding="utf-8")

    command = build_experiment_command(
        cycle=cycle,
        frontier_name=frontier_name,
        out_root=out_root,
        journal_path=journal_path,
        params_path=params_path,
        result_path=result_path,
    )
    start = time.perf_counter()

    try:
        completed = subprocess.run(
            command,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            cwd=PROJECT_ROOT,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - start
        stdout_path.write_text(exc.stdout or "", encoding="utf-8")
        stderr_path.write_text(exc.stderr or "", encoding="utf-8")
        return base.ExperimentRecord(
            frontier=frontier_name,
            cycle=cycle,
            timestamp=datetime.now(timezone.utc).isoformat(),
            hyperparams=params,
            metrics={},
            duration_sec=duration,
            status="failed",
            finding=f"FAILED: TimeoutExpired after {timeout_seconds}s",
            error=f"Child experiment timed out after {timeout_seconds} seconds",
        )

    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")

    if completed.returncode != 0:
        duration = time.perf_counter() - start
        detail = (completed.stderr or completed.stdout).strip()[:4000]
        return base.ExperimentRecord(
            frontier=frontier_name,
            cycle=cycle,
            timestamp=datetime.now(timezone.utc).isoformat(),
            hyperparams=params,
            metrics={},
            duration_sec=duration,
            status="failed",
            finding=f"FAILED: ChildProcessError exit={completed.returncode}",
            error=detail or f"Child experiment exited with code {completed.returncode}",
        )

    if not result_path.exists():
        duration = time.perf_counter() - start
        return base.ExperimentRecord(
            frontier=frontier_name,
            cycle=cycle,
            timestamp=datetime.now(timezone.utc).isoformat(),
            hyperparams=params,
            metrics={},
            duration_sec=duration,
            status="failed",
            finding="FAILED: Child process produced no experiment record",
            error="Missing experiment_record.json from child process",
        )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    return experiment_record_from_dict(payload)


def build_status_payload(
    *,
    state: str,
    phase: str,
    started_at: str,
    out_root: Path,
    journal_path: Path,
    llm_journal_path: Path,
    dashboard_path: Path,
    status_path: Path,
    stop_path: Path,
    journal: base.ResearchJournal,
    llm_journal: CodexLabJournal,
    next_cycle: int,
    target_cycles: int,
    last_record: Optional[base.ExperimentRecord],
    last_strategy: Optional[CodexCallRecord],
    last_critique: Optional[CodexCallRecord],
) -> Dict[str, Any]:
    total_usage = llm_journal.total_usage
    return {
        "state": state,
        "phase": phase,
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "output_root": str(out_root),
        "journal_path": str(journal_path),
        "llm_journal_path": str(llm_journal_path),
        "dashboard_path": str(dashboard_path),
        "status_path": str(status_path),
        "stop_file": str(stop_path),
        "next_cycle": next_cycle,
        "session_target_cycles": None if target_cycles == 0 else target_cycles,
        "total_experiment_cycles": journal.total_cycles,
        "total_codex_calls": llm_journal.total_codex_calls,
        "total_input_tokens": total_usage.input_tokens,
        "total_cached_input_tokens": total_usage.cached_input_tokens,
        "total_output_tokens": total_usage.output_tokens,
        "total_observed_tokens": total_usage.observed_tokens,
        "total_billable_proxy_tokens": total_usage.billable_proxy_tokens,
        "best_frontier": max(
            (
                {
                    "name": name,
                    "score": payload.get("score", 0.0),
                    "cycle": payload.get("cycle"),
                }
                for name, payload in journal.best_per_frontier.items()
            ),
            key=lambda item: float(item["score"]),
            default=None,
        ),
        "last_record": asdict(last_record) if last_record else None,
        "last_strategy": asdict(last_strategy) if last_strategy else None,
        "last_critique": asdict(last_critique) if last_critique else None,
    }


def persist_state(
    *,
    state: str,
    phase: str,
    started_at: str,
    out_root: Path,
    journal_path: Path,
    llm_journal_path: Path,
    dashboard_path: Path,
    status_path: Path,
    stop_path: Path,
    journal: base.ResearchJournal,
    llm_journal: CodexLabJournal,
    next_cycle: int,
    target_cycles: int,
    last_record: Optional[base.ExperimentRecord],
    last_strategy: Optional[CodexCallRecord],
    last_critique: Optional[CodexCallRecord],
) -> None:
    journal.save(journal_path)
    llm_journal.save(llm_journal_path)
    base.generate_dashboard_safely(journal_path, dashboard_path)
    write_json(
        status_path,
        build_status_payload(
            state=state,
            phase=phase,
            started_at=started_at,
            out_root=out_root,
            journal_path=journal_path,
            llm_journal_path=llm_journal_path,
            dashboard_path=dashboard_path,
            status_path=status_path,
            stop_path=stop_path,
            journal=journal,
            llm_journal=llm_journal,
            next_cycle=next_cycle,
            target_cycles=target_cycles,
            last_record=last_record,
            last_strategy=last_strategy,
            last_critique=last_critique,
        ),
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Codex-driven autonomous research lab")
    parser.add_argument("--cycles", type=int, default=0, help="Number of cycles (0=infinite)")
    parser.add_argument("--out", type=str, default="research_runs", help="Output directory")
    parser.add_argument("--journal", type=str, default=None, help="Experiment journal path")
    parser.add_argument("--llm-journal", type=str, default=None, help="Codex token journal path")
    parser.add_argument("--dashboard", type=str, default=None, help="Dashboard HTML path")
    parser.add_argument("--status", type=str, default=None, help="Status JSON path")
    parser.add_argument("--stop-file", type=str, default=None, help="Cooperative stop file path")
    parser.add_argument("--model", type=str, default=None, help="Optional Codex model override")
    parser.add_argument("--codex-timeout-seconds", type=int, default=900, help="Timeout per Codex call")
    parser.add_argument(
        "--experiment-timeout-seconds",
        type=int,
        default=7200,
        help="Timeout per frontier experiment subprocess",
    )
    parser.add_argument("--cycle-delay-seconds", type=float, default=5.0, help="Delay between cycles")
    args = parser.parse_args(argv)

    if args.cycles < 0:
        parser.error("--cycles must be >= 0")
    if args.codex_timeout_seconds <= 0:
        parser.error("--codex-timeout-seconds must be > 0")
    if args.experiment_timeout_seconds <= 0:
        parser.error("--experiment-timeout-seconds must be > 0")
    if args.cycle_delay_seconds < 0:
        parser.error("--cycle-delay-seconds must be >= 0")

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    journal_path = Path(args.journal).resolve() if args.journal else out_root / "journal.json"
    llm_journal_path = (
        Path(args.llm_journal).resolve() if args.llm_journal else out_root / "codex_lab_journal.json"
    )
    dashboard_path = Path(args.dashboard).resolve() if args.dashboard else out_root / "dashboard.html"
    status_path = Path(args.status).resolve() if args.status else out_root / "codex_lab_status.json"
    stop_path = Path(args.stop_file).resolve() if args.stop_file else out_root / "CODEX_STOP"
    strategy_schema_path = PROJECT_ROOT / "configs" / "codex_lab_strategy.schema.json"
    critique_schema_path = PROJECT_ROOT / "configs" / "codex_lab_critique.schema.json"

    journal = base.ResearchJournal.load(journal_path)
    llm_journal = CodexLabJournal.load(llm_journal_path)
    runner = CodexExecRunner(model=args.model, timeout_seconds=args.codex_timeout_seconds)

    started_at = datetime.now(timezone.utc).isoformat()
    cycle = journal.total_cycles
    stop_after_cycle = None if args.cycles == 0 else cycle + args.cycles
    last_record: Optional[base.ExperimentRecord] = None
    last_strategy: Optional[CodexCallRecord] = None
    last_critique: Optional[CodexCallRecord] = None
    state = "starting"
    phase = "idle"

    print(f"\n{'#' * 72}")
    print("#  CODEX AUTONOMOUS LAB")
    print("#  Codex is inside the research loop")
    print(f"#  Output: {out_root}")
    print(f"#  Experiment journal: {journal_path}")
    print(f"#  Codex journal: {llm_journal_path}")
    print(f"#  Status: {status_path}")
    print(f"#  Stop file: {stop_path}")
    print(f"#  Cycles this session: {'infinite' if args.cycles == 0 else args.cycles}")
    print(f"{'#' * 72}")

    persist_state(
        state=state,
        phase=phase,
        started_at=started_at,
        out_root=out_root,
        journal_path=journal_path,
        llm_journal_path=llm_journal_path,
        dashboard_path=dashboard_path,
        status_path=status_path,
        stop_path=stop_path,
        journal=journal,
        llm_journal=llm_journal,
        next_cycle=cycle,
        target_cycles=args.cycles,
        last_record=last_record,
        last_strategy=last_strategy,
        last_critique=last_critique,
    )

    try:
        while stop_after_cycle is None or cycle < stop_after_cycle:
            if should_stop(stop_path):
                print(f"\nStop signal detected at {stop_path}. Exiting cleanly.")
                state = "stopped"
                break

            cycle_dir = out_root / f"cycle_{cycle:05d}_codex_lab"
            cycle_dir.mkdir(parents=True, exist_ok=True)

            state = "running"
            phase = "strategy"
            persist_state(
                state=state,
                phase=phase,
                started_at=started_at,
                out_root=out_root,
                journal_path=journal_path,
                llm_journal_path=llm_journal_path,
                dashboard_path=dashboard_path,
                status_path=status_path,
                stop_path=stop_path,
                journal=journal,
                llm_journal=llm_journal,
                next_cycle=cycle,
                target_cycles=args.cycles,
                last_record=last_record,
                last_strategy=last_strategy,
                last_critique=last_critique,
            )

            strategy_prompt = build_strategy_prompt(cycle, journal, out_root)
            strategy = runner.run_json_prompt(
                label="codex_strategy",
                prompt=strategy_prompt,
                schema_path=strategy_schema_path,
                out_dir=cycle_dir,
            )
            llm_journal.add_call(strategy)
            last_strategy = strategy
            write_json(cycle_dir / "codex_strategy.json", asdict(strategy))
            print(
                f"[cycle {cycle}][strategy] "
                f"rc={strategy.returncode} dur={strategy.duration_sec:.1f}s "
                f"{usage_summary(strategy.usage)}"
            )
            if strategy.error:
                print(f"[cycle {cycle}][strategy][error] {strategy.error}")

            if strategy.error or not strategy.parsed_json:
                raise RuntimeError(strategy.error or "Codex strategy phase returned no JSON payload")

            strategy_payload = strategy.parsed_json
            selected_frontier = str(strategy_payload.get("selected_frontier", "")).strip()
            if selected_frontier not in base.FRONTIER_REGISTRY:
                selected_frontier = base.select_frontier(journal)

            requested_params = strategy_payload.get("hyperparams", {})
            if not isinstance(requested_params, dict):
                requested_params = {}
            params = base.prepare_frontier_params(selected_frontier, requested_params)

            phase = "experiment"
            persist_state(
                state=state,
                phase=phase,
                started_at=started_at,
                out_root=out_root,
                journal_path=journal_path,
                llm_journal_path=llm_journal_path,
                dashboard_path=dashboard_path,
                status_path=status_path,
                stop_path=stop_path,
                journal=journal,
                llm_journal=llm_journal,
                next_cycle=cycle,
                target_cycles=args.cycles,
                last_record=last_record,
                last_strategy=last_strategy,
                last_critique=last_critique,
            )

            experiment = run_experiment_isolated(
                cycle=cycle,
                frontier_name=selected_frontier,
                params=params,
                journal_path=journal_path,
                out_root=out_root,
                codex_cycle_dir=cycle_dir,
                timeout_seconds=args.experiment_timeout_seconds,
            )
            if "BREAKTHROUGH" in experiment.finding:
                journal.breakthroughs.append(f"Cycle {cycle}: {experiment.finding}")
            journal.add(experiment)
            last_record = experiment
            print(
                f"[cycle {cycle}][experiment] frontier={selected_frontier} "
                f"status={experiment.status} dur={experiment.duration_sec:.1f}s "
                f"finding={experiment.finding}"
            )

            phase = "critique"
            persist_state(
                state=state,
                phase=phase,
                started_at=started_at,
                out_root=out_root,
                journal_path=journal_path,
                llm_journal_path=llm_journal_path,
                dashboard_path=dashboard_path,
                status_path=status_path,
                stop_path=stop_path,
                journal=journal,
                llm_journal=llm_journal,
                next_cycle=cycle + 1,
                target_cycles=args.cycles,
                last_record=last_record,
                last_strategy=last_strategy,
                last_critique=last_critique,
            )

            critique_prompt = build_critique_prompt(cycle, journal, strategy_payload, experiment)
            critique = runner.run_json_prompt(
                label="codex_critique",
                prompt=critique_prompt,
                schema_path=critique_schema_path,
                out_dir=cycle_dir,
            )
            llm_journal.add_call(critique)
            last_critique = critique
            write_json(cycle_dir / "codex_critique.json", asdict(critique))
            print(
                f"[cycle {cycle}][critique] "
                f"rc={critique.returncode} dur={critique.duration_sec:.1f}s "
                f"{usage_summary(critique.usage)}"
            )
            if critique.error:
                print(f"[cycle {cycle}][critique][error] {critique.error}")

            try:
                import frontier_proposals  # local import; same sys.path as this module
                proposal = (critique.parsed_json or {}).get("proposed_new_frontier")
                result = frontier_proposals.materialize_proposal(
                    proposal,
                    cycle=cycle,
                    critic_session_id=critique.session_id,
                    out_root=out_root,
                )
                if result.accepted:
                    print(f"[cycle {cycle}][frontier-proposal] accepted: {result.name} -> {result.path}")
                elif proposal:
                    print(f"[cycle {cycle}][frontier-proposal] rejected: {result.reason}")
            except Exception as exc:
                print(f"[cycle {cycle}][frontier-proposal][error] {exc}")

            llm_journal.add_cycle(
                cycle=cycle,
                frontier=selected_frontier,
                experiment=experiment,
                strategy=strategy,
                critique=critique,
            )
            print(
                f"[cycle {cycle}][totals] codex_calls={llm_journal.total_codex_calls} "
                f"{usage_summary(llm_journal.total_usage)}"
            )

            cycle += 1
            persist_state(
                state=state,
                phase="idle",
                started_at=started_at,
                out_root=out_root,
                journal_path=journal_path,
                llm_journal_path=llm_journal_path,
                dashboard_path=dashboard_path,
                status_path=status_path,
                stop_path=stop_path,
                journal=journal,
                llm_journal=llm_journal,
                next_cycle=cycle,
                target_cycles=args.cycles,
                last_record=last_record,
                last_strategy=last_strategy,
                last_critique=last_critique,
            )

            if args.cycle_delay_seconds > 0 and (stop_after_cycle is None or cycle < stop_after_cycle):
                time.sleep(args.cycle_delay_seconds)

    except KeyboardInterrupt:
        print("\n\nCodex autonomous lab paused by operator.")
        state = "paused"
    except Exception as exc:
        print(f"\n\nCodex autonomous lab failed: {exc}")
        traceback.print_exc()
        state = "failed"

    if state == "running":
        state = "completed"

    persist_state(
        state=state,
        phase="idle",
        started_at=started_at,
        out_root=out_root,
        journal_path=journal_path,
        llm_journal_path=llm_journal_path,
        dashboard_path=dashboard_path,
        status_path=status_path,
        stop_path=stop_path,
        journal=journal,
        llm_journal=llm_journal,
        next_cycle=cycle,
        target_cycles=args.cycles,
        last_record=last_record,
        last_strategy=last_strategy,
        last_critique=last_critique,
    )
    print(f"Final experiment journal: {journal_path}")
    print(f"Final Codex journal: {llm_journal_path}")
    print(f"Status: {status_path}")
    return 0 if state in {"completed", "stopped", "paused"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
