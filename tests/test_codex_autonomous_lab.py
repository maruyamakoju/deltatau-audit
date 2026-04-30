"""Tests for the Codex-driven autonomous lab."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Add experiments to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))


class TestCodexExecParsing:
    """Validate parsing of `codex exec --json` output."""

    def test_parse_codex_exec_output_extracts_usage_and_json(self, tmp_path):
        from codex_autonomous_lab import parse_codex_exec_output

        stdout_text = "\n".join([
            "Reading additional input from stdin...",
            json.dumps({"type": "thread.started", "thread_id": "thread-123"}),
            json.dumps({
                "type": "item.completed",
                "item": {
                    "type": "agent_message",
                    "text": json.dumps({
                        "objective": "Push the frontier",
                        "selected_frontier": "certified_mcts",
                        "hyperparams": {"num_simulations": 96},
                        "rationale": "Test deeper search",
                        "predicted_upside": 0.12,
                        "risk_factors": ["slow"],
                        "read_set": ["research_runs/journal.json"],
                        "experiment_brief": "Increase search depth",
                        "code_change_required": False,
                        "confidence": 0.61,
                    }),
                },
            }),
            json.dumps({
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 15000,
                    "cached_input_tokens": 4000,
                    "output_tokens": 180,
                },
            }),
        ])

        record = parse_codex_exec_output(
            label="codex_strategy",
            stdout_text=stdout_text,
            stderr_text="",
            returncode=0,
            duration_sec=2.5,
            prompt_path=tmp_path / "prompt.txt",
            stdout_path=tmp_path / "stdout.jsonl",
            stderr_path=tmp_path / "stderr.log",
        )

        assert record.error is None
        assert record.session_id == "thread-123"
        assert record.parsed_json["selected_frontier"] == "certified_mcts"
        assert record.usage.input_tokens == 15000
        assert record.usage.cached_input_tokens == 4000
        assert record.usage.output_tokens == 180

    def test_parse_codex_exec_output_reports_invalid_json(self, tmp_path):
        from codex_autonomous_lab import parse_codex_exec_output

        stdout_text = "\n".join([
            json.dumps({"type": "thread.started", "thread_id": "thread-123"}),
            json.dumps({
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "not-json"},
            }),
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 7, "output_tokens": 3}}),
        ])

        record = parse_codex_exec_output(
            label="codex_strategy",
            stdout_text=stdout_text,
            stderr_text="",
            returncode=0,
            duration_sec=1.0,
            prompt_path=tmp_path / "prompt.txt",
            stdout_path=tmp_path / "stdout.jsonl",
            stderr_path=tmp_path / "stderr.log",
        )

        assert record.error is not None
        assert "valid JSON" in record.error


class TestCodexLabJournal:
    """Test token accounting persistence."""

    def test_lab_journal_round_trip(self, tmp_path):
        from codex_autonomous_lab import CodexCallRecord, CodexLabJournal, CodexUsage

        journal = CodexLabJournal(started_at="2026-04-05T00:00:00Z")
        record = CodexCallRecord(
            label="codex_strategy",
            timestamp="2026-04-05T00:00:00Z",
            duration_sec=2.0,
            returncode=0,
            session_id="thread-123",
            final_message="{}",
            parsed_json={},
            usage=CodexUsage(input_tokens=1000, cached_input_tokens=200, output_tokens=50),
            prompt_path="prompt.txt",
            stdout_path="stdout.jsonl",
            stderr_path="stderr.log",
            error=None,
        )
        journal.add_call(record)

        path = tmp_path / "codex_lab_journal.json"
        journal.save(path)
        loaded = CodexLabJournal.load(path)

        assert loaded.total_codex_calls == 1
        assert loaded.total_usage.input_tokens == 1000
        assert loaded.total_usage.cached_input_tokens == 200
        assert loaded.total_usage.output_tokens == 50
        assert loaded.recent_calls[0]["session_id"] == "thread-123"


class TestCodexExecRunner:
    """Validate Windows-friendly Codex execution behavior."""

    @pytest.mark.skipif(
        os.name != "nt",
        reason=(
            "Test patches os.name='nt' to exercise the Windows control-event "
            "retry path; on POSIX this leaks into pathlib.Path which then "
            "tries to instantiate WindowsPath and crashes pytest_sessionfinish."
        ),
    )
    def test_runner_retries_transient_windows_control_event(self, tmp_path, monkeypatch):
        import codex_autonomous_lab as m

        schema_path = tmp_path / "schema.json"
        schema_path.write_text(json.dumps({
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        }), encoding="utf-8")

        calls = []
        responses = [
            subprocess.CompletedProcess(["codex"], m.WINDOWS_CONTROL_EVENT_EXIT, stdout="", stderr="Reading prompt from stdin..."),
            subprocess.CompletedProcess(
                ["codex"],
                0,
                stdout="\n".join([
                    json.dumps({"type": "thread.started", "thread_id": "thread-123"}),
                    json.dumps({
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": json.dumps({"ok": True})},
                    }),
                    json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 2}}),
                ]),
                stderr="Reading prompt from stdin...",
            ),
        ]

        def fake_run(command, **kwargs):
            calls.append(kwargs)
            return responses.pop(0)

        # CLI lookup now happens inside _orch_shared.LLMRunnerBase, so patch there.
        import _orch_shared
        monkeypatch.setattr(_orch_shared.shutil, "which", lambda *_: "codex.cmd")
        monkeypatch.setattr(m.subprocess, "run", fake_run)
        monkeypatch.setattr(m.time, "sleep", lambda *_: None)
        monkeypatch.setattr(m.os, "name", "nt")

        runner = m.CodexExecRunner(model=None, timeout_seconds=60, max_retries=1)
        record = runner.run_json_prompt(
            label="codex_strategy",
            prompt='{"ok":true}',
            schema_path=schema_path,
            out_dir=tmp_path,
        )

        assert record.returncode == 0
        assert record.error is None
        assert record.parsed_json == {"ok": True}
        assert len(calls) == 2
        assert calls[0]["creationflags"] != 0


class TestIsolatedExperimentRunner:
    """Validate subprocess-isolated experiment execution plumbing."""

    def test_experiment_record_from_dict_roundtrip(self):
        from codex_autonomous_lab import experiment_record_from_dict

        record = experiment_record_from_dict({
            "frontier": "certified_mcts",
            "cycle": 7,
            "timestamp": "2026-04-05T00:00:00Z",
            "hyperparams": {"num_simulations": 64},
            "metrics": {"composite_score": 0.6},
            "duration_sec": 12.5,
            "status": "success",
            "finding": "ok",
            "error": None,
        })

        assert record.frontier == "certified_mcts"
        assert record.cycle == 7
        assert record.metrics["composite_score"] == 0.6

    def test_run_experiment_isolated_reads_child_record(self, tmp_path, monkeypatch):
        import codex_autonomous_lab as m

        codex_cycle_dir = tmp_path / "cycle_00001_codex_lab"
        codex_cycle_dir.mkdir(parents=True)

        def fake_run(command, **kwargs):
            result_path = Path(command[-1])
            result_path.write_text(json.dumps({
                "frontier": "certified_mcts",
                "cycle": 1,
                "timestamp": "2026-04-05T00:00:00Z",
                "hyperparams": {"num_simulations": 64},
                "metrics": {"composite_score": 0.61},
                "duration_sec": 8.0,
                "status": "success",
                "finding": "ok",
                "error": None,
            }), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="child ok", stderr="")

        monkeypatch.setattr(m.subprocess, "run", fake_run)

        record = m.run_experiment_isolated(
            cycle=1,
            frontier_name="certified_mcts",
            params={"num_simulations": 64},
            journal_path=tmp_path / "journal.json",
            out_root=tmp_path / "runs",
            codex_cycle_dir=codex_cycle_dir,
            timeout_seconds=60,
        )

        assert record.status == "success"
        assert record.metrics["composite_score"] == 0.61
        assert (codex_cycle_dir / "experiment_params.json").exists()
        assert (codex_cycle_dir / "experiment_stdout.log").exists()
