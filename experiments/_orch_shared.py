"""Shared scaffolding for the LLM-driven autonomous research labs.

The Codex-driven (``codex_autonomous_lab.py``) and Claude-driven
(``claude_autonomous_lab.py``) loops both:

    1. Spawn an LLM CLI as a subprocess with stdin-fed prompts.
    2. Capture JSONL or single-envelope JSON output to disk.
    3. Track token usage (input / cached / output).
    4. Persist a long-running journal of calls + cycles.
    5. Build standardised status snapshots for an external dashboard.

Before this module the dataclasses, helpers, and prompt-template logic
all lived inside ``codex_autonomous_lab.py`` and were imported by the
Claude lab via ``import codex_autonomous_lab as cal``. The names were
``CodexUsage`` / ``CodexCallRecord`` / ``CodexLabJournal`` even when used
for Claude — the engines are swappable but the contract is the same.

This module:

    - Renames the data classes to engine-neutral ``LLM*`` names.
    - Keeps ``Codex*`` aliases so the old import paths still work
      (the existing test suite and downstream tooling use them).
    - Centralises the truly-shared helpers
      (``_safe_int``, ``_coerce_subprocess_output``, ``should_stop``,
      ``write_json``, ``usage_summary``, ``_extract_json_block``,
      ``experiment_record_from_dict``, ``build_experiment_command``).
    - Provides ``LLMRunnerBase`` with the cross-platform plumbing
      (``_creationflags``, ``_resolve_cli_command``) that both runners
      need but were independently re-implementing.

Engine-specific concerns deliberately stay out of here:
    - JSONL parsers (``parse_codex_exec_output`` vs ``_parse_claude_output``)
      because the event schemas differ.
    - Retry policies (Codex retries Windows control events;
      Claude retries timeouts).
    - Subscription/auth env handling (Claude strips ``ANTHROPIC_*`` keys
      to fall back on OAuth; Codex doesn't).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


__all__ = [
    # Data classes
    "LLMUsage",
    "LLMCallRecord",
    "LLMLabJournal",
    # Backwards-compatible aliases
    "CodexUsage",
    "CodexCallRecord",
    "CodexLabJournal",
    # Pure helpers
    "_safe_int",
    "_coerce_subprocess_output",
    "should_stop",
    "write_json",
    "usage_summary",
    "_extract_json_block",
    "experiment_record_from_dict",
    "build_experiment_command",
    # Subprocess base
    "LLMRunnerBase",
]


# ─────────────────────────────────────────────────────────────────────────────
# Token accounting
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class LLMUsage:
    """Token counts reported by an LLM CLI for a single call.

    ``cached_input_tokens`` are billed at a discount (or free) by most
    providers; ``billable_proxy_tokens`` excludes them as a rough
    cost-side-of-the-house estimate. ``observed_tokens`` is the raw
    sum useful for usage telemetry.
    """

    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0

    @property
    def observed_tokens(self) -> int:
        return self.input_tokens + self.cached_input_tokens + self.output_tokens

    @property
    def billable_proxy_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, other: "LLMUsage") -> None:
        self.input_tokens += other.input_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.output_tokens += other.output_tokens


# ─────────────────────────────────────────────────────────────────────────────
# Per-call record
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class LLMCallRecord:
    """The structured result of one LLM CLI invocation.

    Both Codex and Claude runners produce this shape so the journal
    persistence and dashboard logic can treat them interchangeably.
    """

    label: str
    timestamp: str
    duration_sec: float
    returncode: int
    session_id: Optional[str]
    final_message: str
    parsed_json: Optional[Dict[str, Any]]
    usage: LLMUsage
    prompt_path: str
    stdout_path: str
    stderr_path: str
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Long-running journal
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class LLMLabJournal:
    """Append-only journal of LLM calls + research cycles.

    The on-disk schema is engine-neutral and intentionally stable;
    upstream dashboards key off ``total_codex_calls`` for historical
    reasons (the field name pre-dates the engine-neutral split).
    """

    started_at: str
    total_codex_calls: int = 0
    total_usage: LLMUsage = field(default_factory=LLMUsage)
    recent_calls: List[Dict[str, Any]] = field(default_factory=list)
    recent_cycles: List[Dict[str, Any]] = field(default_factory=list)

    def add_call(self, record: LLMCallRecord) -> None:
        self.total_codex_calls += 1
        self.total_usage.add(record.usage)
        self.recent_calls.append(
            {
                "label": record.label,
                "timestamp": record.timestamp,
                "duration_sec": record.duration_sec,
                "returncode": record.returncode,
                "session_id": record.session_id,
                "usage": asdict(record.usage),
                "error": record.error,
            }
        )
        self.recent_calls = self.recent_calls[-100:]

    def add_cycle(
        self,
        cycle: int,
        frontier: str,
        experiment: Any,  # base.ExperimentRecord; not imported to avoid cycle
        strategy: LLMCallRecord,
        critique: LLMCallRecord,
    ) -> None:
        usage = LLMUsage()
        usage.add(strategy.usage)
        usage.add(critique.usage)
        self.recent_cycles.append(
            {
                "cycle": cycle,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "frontier": frontier,
                "experiment_status": experiment.status,
                "experiment_finding": experiment.finding,
                "strategy_session_id": strategy.session_id,
                "critique_session_id": critique.session_id,
                "codex_usage": asdict(usage),
            }
        )
        self.recent_cycles = self.recent_cycles[-100:]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "started_at": self.started_at,
            "total_codex_calls": self.total_codex_calls,
            "total_usage": asdict(self.total_usage),
            "recent_calls": self.recent_calls,
            "recent_cycles": self.recent_cycles,
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LLMLabJournal":
        if not path.exists():
            return cls(started_at=datetime.now(timezone.utc).isoformat())
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            total_usage = LLMUsage(**payload.get("total_usage", {}))
            return cls(
                started_at=payload.get("started_at", datetime.now(timezone.utc).isoformat()),
                total_codex_calls=int(payload.get("total_codex_calls", 0)),
                total_usage=total_usage,
                recent_calls=list(payload.get("recent_calls", [])),
                recent_cycles=list(payload.get("recent_cycles", [])),
            )
        except Exception:
            return cls(started_at=datetime.now(timezone.utc).isoformat())


# Backwards-compat aliases. Existing code does:
#     from codex_autonomous_lab import CodexCallRecord
#     cal.CodexUsage(...)
# These will keep working after the migration.
CodexUsage = LLMUsage
CodexCallRecord = LLMCallRecord
CodexLabJournal = LLMLabJournal


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers (no global state, no I/O beyond file reads/writes)
# ─────────────────────────────────────────────────────────────────────────────


def _safe_int(value: Any) -> int:
    """Best-effort conversion to int; returns 0 on any failure.

    Used pervasively when consuming CLI JSONL where token counts may be
    missing, ``None``, or accidentally string-typed.
    """
    try:
        return int(value)
    except Exception:
        return 0


def _coerce_subprocess_output(value: Any) -> str:
    """Normalise possibly-bytes / possibly-None subprocess output to str."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def should_stop(stop_path: Optional[Path]) -> bool:
    """Cooperative stop signal — checks for the existence of a stop file."""
    return stop_path is not None and stop_path.exists()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write ``payload`` as indented JSON to ``path``, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def usage_summary(usage: LLMUsage) -> str:
    """Compact, human-readable token breakdown used in console logs."""
    return (
        f"in={usage.input_tokens:,} "
        f"cached={usage.cached_input_tokens:,} "
        f"out={usage.output_tokens:,} "
        f"obs={usage.observed_tokens:,}"
    )


def _extract_json_block(text: str) -> str:
    """Best-effort extraction of a JSON object from prose-y LLM output.

    Tries in order: a ```json fenced block, a plain ``` fenced block,
    or the widest top-level ``{...}`` brace span. Returns the raw text
    stripped if nothing recognisable is found — caller handles parse
    errors.
    """
    fence_start = text.find("```json")
    if fence_start != -1:
        inside = text[fence_start + len("```json") :]
        fence_end = inside.find("```")
        if fence_end != -1:
            return inside[:fence_end].strip()
        return inside.strip()
    fence_start = text.find("```")
    if fence_start != -1:
        inside = text[fence_start + 3 :]
        fence_end = inside.find("```")
        if fence_end != -1:
            return inside[:fence_end].strip()
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        return text[first_brace : last_brace + 1].strip()
    return text.strip()


def experiment_record_from_dict(payload: Dict[str, Any]) -> Any:
    """Reconstruct a ``base.ExperimentRecord`` from a JSON-deserialised dict.

    The ``base`` import is local to avoid a circular import: this module
    is imported by ``codex_autonomous_lab`` which itself imports
    ``autonomous_research`` (= ``base``).
    """
    import autonomous_research as base

    return base.ExperimentRecord(
        frontier=str(payload.get("frontier", "unknown")),
        cycle=int(payload.get("cycle", 0)),
        timestamp=str(payload.get("timestamp", "")),
        hyperparams=dict(payload.get("hyperparams", {})),
        metrics=dict(payload.get("metrics", {})),
        duration_sec=float(payload.get("duration_sec", 0.0)),
        status=str(payload.get("status", "failed")),
        finding=str(payload.get("finding", "")),
        error=payload.get("error"),
    )


def build_experiment_command(
    *,
    cycle: int,
    frontier_name: str,
    out_root: Path,
    journal_path: Path,
    params_path: Path,
    result_path: Path,
) -> List[str]:
    """Build the child-process command that runs one frontier in isolation."""
    helper = PROJECT_ROOT / "experiments" / "run_frontier_once.py"
    return [
        sys.executable,
        str(helper),
        "--cycle",
        str(cycle),
        "--frontier",
        frontier_name,
        "--out",
        str(out_root),
        "--journal",
        str(journal_path),
        "--params-json",
        str(params_path),
        "--result-json",
        str(result_path),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess plumbing common to both LLM runners
# ─────────────────────────────────────────────────────────────────────────────


class LLMRunnerBase:
    """Common subprocess plumbing for Codex- and Claude-style CLI runners.

    Subclasses are responsible for:
        - Resolving the CLI binary (``_resolve_cli_command``)
        - Building the per-call command
        - Parsing the CLI's stdout into an :class:`LLMCallRecord`
        - Defining the retry policy

    This class only owns the cross-platform bits — Windows
    ``creationflags`` and the binary lookup helper.
    """

    @staticmethod
    def _creationflags() -> int:
        """Detach child processes from the parent console on Windows.

        Returns 0 on non-Windows platforms. The two flags combined keep
        the child process from popping a console window AND prevent a
        parent Ctrl+Break from cascading into the child mid-run.
        """
        if os.name != "nt":
            return 0
        return int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) | int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))

    @staticmethod
    def _resolve_cli_command(candidates: Iterable[str], *, name: str) -> str:
        """Locate an LLM CLI binary on PATH or raise ``FileNotFoundError``.

        ``candidates`` should be a list ordered from most-preferred to
        least; the first one ``shutil.which`` resolves wins. ``name`` is
        used only in the error message.
        """
        for candidate in candidates:
            path = shutil.which(candidate)
            if path:
                return path
        raise FileNotFoundError(f"Could not find {name} CLI on PATH")
