#!/usr/bin/env python3
"""Claude-driven variant of the autonomous research lab.

Forks codex_autonomous_lab.py only in the strategy phase: the critique
and experiment phases are reused unchanged so we can A/B the strategy
engine (Claude vs Codex) without diverging the rest of the pipeline.

Auth: `claude -p` uses OAuth / keychain by default, which routes
through the user's Claude Pro subscription. We deliberately do NOT set
ANTHROPIC_API_KEY or pass --max-budget-usd.

Usage:
    python experiments/claude_autonomous_lab.py --cycles 1
    python experiments/claude_autonomous_lab.py --cycles 1 --strategy-engine codex
    python experiments/claude_autonomous_lab.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import autonomous_research as base
import codex_autonomous_lab as cal


class ClaudeExecRunner:
    """Thin wrapper around `claude -p --output-format json`.

    Returns a ``cal.CodexCallRecord`` so downstream code (journal, persist,
    status) can treat Claude and Codex runners interchangeably.
    """

    def __init__(
        self,
        model: Optional[str],
        timeout_seconds: int,
        max_retries: int = 1,
        allowed_tools: str = "Read,Grep,Glob",
    ) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, int(max_retries))
        self.allowed_tools = allowed_tools
        self.claude_command = self._resolve_claude_command()

    @staticmethod
    def _resolve_claude_command() -> str:
        for candidate in ("claude.cmd", "claude", "claude.ps1"):
            path = shutil.which(candidate)
            if path:
                return path
        raise FileNotFoundError("Could not find claude CLI on PATH")

    @staticmethod
    def _creationflags() -> int:
        if os.name != "nt":
            return 0
        return int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) | int(
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )

    @staticmethod
    def _subscription_env() -> dict:
        """Strip ANTHROPIC_API_KEY so claude -p falls back to OAuth/keychain."""
        env = os.environ.copy()
        for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
            env.pop(key, None)
        return env

    def run_json_prompt(
        self,
        *,
        label: str,
        prompt: str,
        schema_path: Path,
        out_dir: Path,
    ) -> cal.CodexCallRecord:
        prompt_path = out_dir / f"{label}_prompt.txt"
        stdout_path = out_dir / f"{label}_stdout.jsonl"
        stderr_path = out_dir / f"{label}_stderr.log"
        prompt_path.write_text(prompt, encoding="utf-8")

        # Long prompts passed via argv (`-p <prompt>`) silently caused
        # Claude to emit markdown with no envelope in testing. Streaming
        # JSONL from stdin is reliable across prompt sizes and captures the
        # full tool-use trace for debugging.
        command = [
            self.claude_command,
            "-p",
            "--input-format", "text",
            "--output-format", "stream-json",
            "--verbose",
            "--permission-mode", "bypassPermissions",
            "--allowedTools", self.allowed_tools,
            "--add-dir", str(PROJECT_ROOT),
            "--no-session-persistence",
        ]
        if self.model:
            command.extend(["--model", self.model])

        env = self._subscription_env()

        attempts = self.max_retries + 1
        last_record: Optional[cal.CodexCallRecord] = None
        for attempt in range(attempts):
            start = time.perf_counter()
            try:
                completed = subprocess.run(
                    command,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=PROJECT_ROOT,
                    timeout=self.timeout_seconds,
                    check=False,
                    env=env,
                    creationflags=self._creationflags(),
                )
                duration = time.perf_counter() - start
                stdout_path.write_text(completed.stdout or "", encoding="utf-8")
                stderr_path.write_text(completed.stderr or "", encoding="utf-8")
                record = _parse_claude_output(
                    label=label,
                    stdout_text=completed.stdout or "",
                    stderr_text=completed.stderr or "",
                    returncode=completed.returncode,
                    duration_sec=duration,
                    prompt_path=prompt_path,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    schema_path=schema_path,
                )
                if record.error is None or attempt >= self.max_retries:
                    return record
                last_record = record
                time.sleep(min(5.0, 1.5 * (attempt + 1)))
            except subprocess.TimeoutExpired as exc:
                duration = time.perf_counter() - start
                stdout_text = cal._coerce_subprocess_output(exc.stdout)
                stderr_text = cal._coerce_subprocess_output(exc.stderr)
                stdout_path.write_text(stdout_text, encoding="utf-8")
                stderr_path.write_text(stderr_text, encoding="utf-8")
                last_record = cal.CodexCallRecord(
                    label=label,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    duration_sec=duration,
                    returncode=124,
                    session_id=None,
                    final_message=stdout_text,
                    parsed_json=None,
                    usage=cal.CodexUsage(),
                    prompt_path=str(prompt_path),
                    stdout_path=str(stdout_path),
                    stderr_path=str(stderr_path),
                    error=f"claude -p timeout after {self.timeout_seconds}s",
                )
                if attempt >= self.max_retries:
                    return last_record
                time.sleep(min(5.0, 1.5 * (attempt + 1)))
        assert last_record is not None
        return last_record


def _extract_json_block(text: str) -> str:
    """Extract a JSON object from a Claude response.

    Claude often precedes the JSON with prose and wraps it in ```json ... ```
    fences. Falls back to the widest top-level ``{ ... }`` pair if no fence
    is present.
    """
    fence_start = text.find("```json")
    if fence_start != -1:
        inside = text[fence_start + len("```json"):]
        fence_end = inside.find("```")
        if fence_end != -1:
            return inside[:fence_end].strip()
        return inside.strip()
    fence_start = text.find("```")
    if fence_start != -1:
        inside = text[fence_start + 3:]
        fence_end = inside.find("```")
        if fence_end != -1:
            return inside[:fence_end].strip()
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        return text[first_brace:last_brace + 1].strip()
    return text.strip()


def _parse_claude_output(
    *,
    label: str,
    stdout_text: str,
    stderr_text: str,
    returncode: int,
    duration_sec: float,
    prompt_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    schema_path: Path,
) -> cal.CodexCallRecord:
    """Parse `claude -p --output-format stream-json` JSONL.

    Streaming mode emits one JSON event per line. We scan for the terminal
    ``type=result`` event which mirrors the single-envelope shape:
      {"type": "result", "subtype": "success", "result": "<assistant text>",
       "session_id": "...", "is_error": false,
       "usage": {"input_tokens": n, "output_tokens": n,
                 "cache_read_input_tokens": n, ...}}
    """
    usage = cal.CodexUsage()
    final_message = ""
    session_id: Optional[str] = None
    parsed_json: Optional[dict] = None
    envelope_error: Optional[str] = None
    parse_error: Optional[str] = None
    result_envelope: Optional[dict] = None

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
        etype = str(event.get("type", ""))
        if etype == "system" and event.get("subtype") == "init":
            session_id = event.get("session_id") or session_id
        elif etype == "result":
            result_envelope = event
            break

    if result_envelope is None:
        envelope_error = "no terminal result event in claude stream"
    else:
        session_id = result_envelope.get("session_id") or session_id
        final_message = str(result_envelope.get("result", "") or "")
        u = result_envelope.get("usage")
        if isinstance(u, dict):
            usage.input_tokens = cal._safe_int(u.get("input_tokens"))
            usage.cached_input_tokens = cal._safe_int(
                u.get("cache_read_input_tokens", u.get("cached_input_tokens", 0))
            )
            usage.output_tokens = cal._safe_int(u.get("output_tokens"))
        if result_envelope.get("is_error"):
            envelope_error = (
                str(result_envelope.get("error") or result_envelope.get("subtype") or "").strip()
                or "claude reported is_error=true"
            )

    if final_message:
        candidate_text = _extract_json_block(final_message)
        try:
            candidate = json.loads(candidate_text)
            if isinstance(candidate, dict):
                parsed_json = candidate
        except json.JSONDecodeError as exc:
            parse_error = f"Final message was not valid JSON: {exc}"

    error: Optional[str] = None
    if returncode != 0:
        detail = stderr_text.strip() or stdout_text.strip()
        error = f"claude -p failed with exit code {returncode}: {detail[:1000]}"
    elif envelope_error:
        error = envelope_error
    elif not final_message:
        error = "claude -p returned no 'result' text"
    elif parse_error:
        error = parse_error

    record = cal.CodexCallRecord(
        label=label,
        timestamp=datetime.now(timezone.utc).isoformat(),
        duration_sec=duration_sec,
        returncode=returncode,
        session_id=session_id,
        final_message=final_message,
        parsed_json=parsed_json,
        usage=usage,
        prompt_path=str(prompt_path),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        error=error,
    )

    if record.parsed_json is not None and record.error is None:
        try:
            from jsonschema import ValidationError, validate as jsonschema_validate
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            jsonschema_validate(record.parsed_json, schema)
        except ValidationError as exc:
            record.error = f"Claude JSON failed schema validation: {exc.message}"
        except Exception as exc:
            record.error = f"Failed to validate Claude JSON: {exc}"

    return record


def run_loop(args: argparse.Namespace) -> int:
    """Main loop. Mirrors cal.main but with split strategy/critique engines."""
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    journal_path = Path(args.journal).resolve() if args.journal else out_root / "journal.json"
    llm_journal_path = (
        Path(args.llm_journal).resolve()
        if args.llm_journal
        else out_root / f"{args.strategy_engine}_lab_journal.json"
    )
    dashboard_path = (
        Path(args.dashboard).resolve() if args.dashboard else out_root / "dashboard.html"
    )
    status_path = (
        Path(args.status).resolve()
        if args.status
        else out_root / f"{args.strategy_engine}_lab_status.json"
    )
    stop_path = Path(args.stop_file).resolve() if args.stop_file else out_root / "CLAUDE_STOP"
    strategy_schema_path = PROJECT_ROOT / "configs" / "codex_lab_strategy.schema.json"
    critique_schema_path = PROJECT_ROOT / "configs" / "codex_lab_critique.schema.json"

    journal = base.ResearchJournal.load(journal_path)
    llm_journal = cal.CodexLabJournal.load(llm_journal_path)

    if args.strategy_engine == "claude":
        strategy_runner = ClaudeExecRunner(
            model=args.claude_model,
            timeout_seconds=args.strategy_timeout,
        )
    else:
        strategy_runner = cal.CodexExecRunner(
            model=args.codex_model, timeout_seconds=args.strategy_timeout
        )
    if args.critique_engine == "claude":
        critique_runner = ClaudeExecRunner(
            model=args.claude_model,
            timeout_seconds=args.critique_timeout,
        )
    else:
        critique_runner = cal.CodexExecRunner(
            model=args.codex_model, timeout_seconds=args.critique_timeout
        )

    started_at = datetime.now(timezone.utc).isoformat()
    cycle = journal.total_cycles
    stop_after_cycle = None if args.cycles == 0 else cycle + args.cycles
    last_record: Optional[base.ExperimentRecord] = None
    last_strategy: Optional[cal.CodexCallRecord] = None
    last_critique: Optional[cal.CodexCallRecord] = None
    state = "starting"
    phase = "idle"

    print(f"\n{'#' * 72}")
    print(
        f"#  AUTONOMOUS LAB (strategy={args.strategy_engine}, critique={args.critique_engine})"
    )
    print(f"#  Output: {out_root}")
    print(f"#  Cycles this session: {'infinite' if args.cycles == 0 else args.cycles}")
    print(f"{'#' * 72}")

    def snapshot(phase_name: str, next_cycle: int) -> None:
        cal.persist_state(
            state=state,
            phase=phase_name,
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
            target_cycles=args.cycles,
            last_record=last_record,
            last_strategy=last_strategy,
            last_critique=last_critique,
        )

    snapshot(phase, cycle)

    try:
        while stop_after_cycle is None or cycle < stop_after_cycle:
            if cal.should_stop(stop_path):
                print(f"\nStop signal detected at {stop_path}. Exiting cleanly.")
                state = "stopped"
                break

            cycle_dir = out_root / f"cycle_{cycle:05d}_{args.strategy_engine}_lab"
            cycle_dir.mkdir(parents=True, exist_ok=True)

            state = "running"
            phase = "strategy"
            snapshot(phase, cycle)

            strategy_prompt = cal.build_strategy_prompt(cycle, journal, out_root)
            strategy = strategy_runner.run_json_prompt(
                label=f"{args.strategy_engine}_strategy",
                prompt=strategy_prompt,
                schema_path=strategy_schema_path,
                out_dir=cycle_dir,
            )
            llm_journal.add_call(strategy)
            last_strategy = strategy
            cal.write_json(cycle_dir / f"{args.strategy_engine}_strategy.json", asdict(strategy))
            print(
                f"[cycle {cycle}][strategy] engine={args.strategy_engine} "
                f"rc={strategy.returncode} dur={strategy.duration_sec:.1f}s "
                f"{cal.usage_summary(strategy.usage)}"
            )
            if strategy.error:
                print(f"[cycle {cycle}][strategy][error] {strategy.error}")

            if strategy.error or not strategy.parsed_json:
                raise RuntimeError(
                    strategy.error or f"{args.strategy_engine} strategy returned no JSON payload"
                )

            strategy_payload = strategy.parsed_json
            selected_frontier = str(strategy_payload.get("selected_frontier", "")).strip()
            if selected_frontier not in base.FRONTIER_REGISTRY:
                selected_frontier = base.select_frontier(journal)

            requested_params = strategy_payload.get("hyperparams", {})
            if not isinstance(requested_params, dict):
                requested_params = {}
            params = base.prepare_frontier_params(selected_frontier, requested_params)

            phase = "experiment"
            snapshot(phase, cycle)

            experiment = cal.run_experiment_isolated(
                cycle=cycle,
                frontier_name=selected_frontier,
                params=params,
                journal_path=journal_path,
                out_root=out_root,
                codex_cycle_dir=cycle_dir,
                timeout_seconds=args.experiment_timeout,
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
            snapshot(phase, cycle + 1)

            critique_prompt = cal.build_critique_prompt(cycle, journal, strategy_payload, experiment)
            critique = critique_runner.run_json_prompt(
                label="codex_critique",
                prompt=critique_prompt,
                schema_path=critique_schema_path,
                out_dir=cycle_dir,
            )
            llm_journal.add_call(critique)
            last_critique = critique
            cal.write_json(cycle_dir / "codex_critique.json", asdict(critique))
            print(
                f"[cycle {cycle}][critique] rc={critique.returncode} "
                f"dur={critique.duration_sec:.1f}s {cal.usage_summary(critique.usage)}"
            )
            if critique.error:
                print(f"[cycle {cycle}][critique][error] {critique.error}")

            try:
                import frontier_proposals
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
                f"[cycle {cycle}][totals] llm_calls={llm_journal.total_codex_calls} "
                f"{cal.usage_summary(llm_journal.total_usage)}"
            )

            cycle += 1
            snapshot("idle", cycle)

            if args.cycle_delay_seconds > 0 and (
                stop_after_cycle is None or cycle < stop_after_cycle
            ):
                time.sleep(args.cycle_delay_seconds)

    except KeyboardInterrupt:
        print(f"\n\n{args.strategy_engine} autonomous lab paused by operator.")
        state = "paused"
    except Exception as exc:
        print(f"\n\n{args.strategy_engine} autonomous lab failed: {exc}")
        traceback.print_exc()
        state = "failed"

    if state == "running":
        state = "completed"

    snapshot("idle", cycle)
    print(f"Final experiment journal: {journal_path}")
    print(f"Final LLM journal: {llm_journal_path}")
    print(f"Status: {status_path}")
    return 0 if state in {"completed", "stopped", "paused"} else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Claude-driven autonomous research lab")
    parser.add_argument("--cycles", type=int, default=1, help="Number of cycles (0=infinite)")
    parser.add_argument(
        "--strategy-engine",
        choices=("claude", "codex"),
        default="claude",
        help="Which engine handles the strategy phase (default: claude)",
    )
    parser.add_argument(
        "--critique-engine",
        choices=("claude", "codex"),
        default="codex",
        help="Which engine handles the critique phase (default: codex). "
        "Set 'claude' for a full-Pro-subscription loop.",
    )
    parser.add_argument("--claude-model", default="opus", help="Model alias for claude -p")
    parser.add_argument("--codex-model", default=None, help="Optional codex model override")
    parser.add_argument("--strategy-timeout", type=int, default=900)
    parser.add_argument("--critique-timeout", type=int, default=900)
    parser.add_argument("--experiment-timeout", type=int, default=7200)
    parser.add_argument("--out", type=str, default="research_runs")
    parser.add_argument("--journal", type=str, default=None)
    parser.add_argument("--llm-journal", type=str, default=None)
    parser.add_argument("--dashboard", type=str, default=None)
    parser.add_argument("--status", type=str, default=None)
    parser.add_argument("--stop-file", type=str, default=None)
    parser.add_argument("--cycle-delay-seconds", type=float, default=5.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build strategy prompt + exit (no LLM call). Useful for A/B prompt checks.",
    )
    args = parser.parse_args(argv)

    if args.cycles < 0:
        parser.error("--cycles must be >= 0")
    for key in ("strategy_timeout", "critique_timeout", "experiment_timeout"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be > 0")
    if args.cycle_delay_seconds < 0:
        parser.error("--cycle-delay-seconds must be >= 0")

    if args.dry_run:
        out_root = Path(args.out).resolve()
        journal_path = Path(args.journal).resolve() if args.journal else out_root / "journal.json"
        journal = base.ResearchJournal.load(journal_path)
        prompt = cal.build_strategy_prompt(journal.total_cycles, journal, out_root)
        print("=" * 60)
        print(f"DRY RUN — strategy prompt for cycle {journal.total_cycles}")
        print(f"Engine that would be called: {args.strategy_engine}")
        print(f"Claude model: {args.claude_model}")
        print(f"Prompt length: {len(prompt)} chars")
        print("=" * 60)
        print(prompt[:2000])
        if len(prompt) > 2000:
            print(f"... [{len(prompt) - 2000} chars trimmed]")
        return 0

    return run_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
