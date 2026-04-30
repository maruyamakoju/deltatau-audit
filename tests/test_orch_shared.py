"""Bracketing tests for the orchestration shared module.

These tests pin down the behaviour of helpers that are about to migrate
from ``codex_autonomous_lab.py`` into a new ``_orch_shared.py`` module.
They will keep passing both before and during the migration; the entire
file targets ``_orch_shared`` once the move is complete.

Coverage:
    - ``LLMUsage`` (currently ``CodexUsage``) arithmetic + properties.
    - ``LLMCallRecord`` (currently ``CodexCallRecord``) round-trip.
    - ``LLMLabJournal`` (currently ``CodexLabJournal``) load/save.
    - Helpers: ``_safe_int``, ``_coerce_subprocess_output``, ``should_stop``,
      ``usage_summary``, ``_extract_json_block``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add experiments/ to path so we can import the orchestration modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))


# ─────────────────────────────────────────────────────────────────────────────
# LLMUsage / CodexUsage (alias)
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMUsage:
    def test_observed_tokens_includes_cached(self):
        from _orch_shared import LLMUsage

        usage = LLMUsage(input_tokens=100, cached_input_tokens=50, output_tokens=20)
        assert usage.observed_tokens == 170

    def test_billable_excludes_cached(self):
        from _orch_shared import LLMUsage

        usage = LLMUsage(input_tokens=100, cached_input_tokens=50, output_tokens=20)
        assert usage.billable_proxy_tokens == 120

    def test_add_accumulates_in_place(self):
        from _orch_shared import LLMUsage

        a = LLMUsage(input_tokens=10, cached_input_tokens=2, output_tokens=3)
        b = LLMUsage(input_tokens=5, cached_input_tokens=1, output_tokens=2)
        a.add(b)
        assert a.input_tokens == 15
        assert a.cached_input_tokens == 3
        assert a.output_tokens == 5

    def test_zero_usage_default(self):
        from _orch_shared import LLMUsage

        u = LLMUsage()
        assert u.observed_tokens == 0
        assert u.billable_proxy_tokens == 0


class TestCodexUsageAlias:
    """``CodexUsage`` must remain a drop-in alias for ``LLMUsage``."""

    def test_alias_is_same_type(self):
        from _orch_shared import CodexUsage, LLMUsage

        assert CodexUsage is LLMUsage

    def test_codex_usage_imports_from_codex_lab(self):
        # Backwards compat: existing call sites do ``from codex_autonomous_lab import CodexUsage``
        from codex_autonomous_lab import CodexUsage

        u = CodexUsage(input_tokens=1)
        assert u.input_tokens == 1


# ─────────────────────────────────────────────────────────────────────────────
# LLMCallRecord / CodexCallRecord (alias)
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMCallRecord:
    def test_alias_compatibility(self):
        from _orch_shared import CodexCallRecord, LLMCallRecord

        assert CodexCallRecord is LLMCallRecord

    def test_dataclass_fields_present(self):
        from _orch_shared import LLMCallRecord, LLMUsage

        rec = LLMCallRecord(
            label="strategy",
            timestamp="2026-04-30T00:00:00Z",
            duration_sec=1.0,
            returncode=0,
            session_id="s-1",
            final_message="ok",
            parsed_json={"k": 1},
            usage=LLMUsage(input_tokens=5),
            prompt_path="prompt.txt",
            stdout_path="stdout.jsonl",
            stderr_path="stderr.log",
            error=None,
        )
        assert rec.label == "strategy"
        assert rec.usage.input_tokens == 5


# ─────────────────────────────────────────────────────────────────────────────
# LLMLabJournal / CodexLabJournal (alias)
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMLabJournal:
    def test_alias_compatibility(self):
        from _orch_shared import CodexLabJournal, LLMLabJournal

        assert CodexLabJournal is LLMLabJournal

    def test_round_trip_preserves_call_count(self, tmp_path):
        from _orch_shared import LLMCallRecord, LLMLabJournal, LLMUsage

        journal = LLMLabJournal(started_at="2026-04-30T00:00:00Z")
        rec = LLMCallRecord(
            label="strategy",
            timestamp="2026-04-30T00:00:00Z",
            duration_sec=1.0,
            returncode=0,
            session_id="s-1",
            final_message="ok",
            parsed_json={},
            usage=LLMUsage(input_tokens=10, output_tokens=5),
            prompt_path="prompt.txt",
            stdout_path="stdout.jsonl",
            stderr_path="stderr.log",
            error=None,
        )
        journal.add_call(rec)
        path = tmp_path / "journal.json"
        journal.save(path)
        loaded = LLMLabJournal.load(path)
        assert loaded.total_codex_calls == 1
        assert loaded.total_usage.input_tokens == 10

    def test_load_returns_fresh_journal_when_missing(self, tmp_path):
        from _orch_shared import LLMLabJournal

        loaded = LLMLabJournal.load(tmp_path / "absent.json")
        assert loaded.total_codex_calls == 0

    def test_load_recovers_from_corrupt_file(self, tmp_path):
        from _orch_shared import LLMLabJournal

        path = tmp_path / "corrupt.json"
        path.write_text("not json{", encoding="utf-8")
        loaded = LLMLabJournal.load(path)
        assert loaded.total_codex_calls == 0


# ─────────────────────────────────────────────────────────────────────────────
# Plain helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestSafeInt:
    def test_int_passthrough(self):
        from _orch_shared import _safe_int

        assert _safe_int(5) == 5
        assert _safe_int(-3) == -3

    def test_float_truncates(self):
        from _orch_shared import _safe_int

        assert _safe_int(3.9) == 3

    def test_string_numeric(self):
        from _orch_shared import _safe_int

        assert _safe_int("42") == 42

    def test_string_garbage_returns_zero(self):
        from _orch_shared import _safe_int

        assert _safe_int("abc") == 0

    def test_none_returns_zero(self):
        from _orch_shared import _safe_int

        assert _safe_int(None) == 0


class TestCoerceSubprocessOutput:
    def test_none_returns_empty(self):
        from _orch_shared import _coerce_subprocess_output

        assert _coerce_subprocess_output(None) == ""

    def test_bytes_decoded(self):
        from _orch_shared import _coerce_subprocess_output

        assert _coerce_subprocess_output(b"hello") == "hello"

    def test_invalid_utf8_replaced_not_raised(self):
        from _orch_shared import _coerce_subprocess_output

        # Lone surrogate-half byte sequence in latin-1
        out = _coerce_subprocess_output(b"\xff\xfe\xfdok")
        assert isinstance(out, str)
        assert "ok" in out

    def test_string_passthrough(self):
        from _orch_shared import _coerce_subprocess_output

        assert _coerce_subprocess_output("already string") == "already string"


class TestShouldStop:
    def test_none_path_returns_false(self):
        from _orch_shared import should_stop

        assert should_stop(None) is False

    def test_missing_path_returns_false(self, tmp_path):
        from _orch_shared import should_stop

        assert should_stop(tmp_path / "nope") is False

    def test_existing_path_returns_true(self, tmp_path):
        from _orch_shared import should_stop

        path = tmp_path / "STOP"
        path.write_text("", encoding="utf-8")
        assert should_stop(path) is True


class TestUsageSummary:
    def test_includes_all_fields(self):
        from _orch_shared import LLMUsage, usage_summary

        out = usage_summary(LLMUsage(input_tokens=1234, cached_input_tokens=200, output_tokens=50))
        assert "in=1,234" in out
        assert "cached=200" in out
        assert "out=50" in out
        assert "obs=1,484" in out


class TestExtractJsonBlock:
    def test_fenced_json_block(self):
        from _orch_shared import _extract_json_block

        text = 'preamble\n```json\n{"a": 1}\n```\ntrailing'
        out = _extract_json_block(text)
        assert json.loads(out) == {"a": 1}

    def test_unfenced_json_extracted_by_brace_span(self):
        from _orch_shared import _extract_json_block

        text = 'leading words {"a": 1, "b": [2, 3]} trailing'
        out = _extract_json_block(text)
        assert json.loads(out) == {"a": 1, "b": [2, 3]}

    def test_plain_fence_without_json_lang_tag(self):
        from _orch_shared import _extract_json_block

        text = '```\n{"k": "v"}\n```'
        out = _extract_json_block(text)
        assert json.loads(out) == {"k": "v"}

    def test_no_json_returns_input_stripped(self):
        from _orch_shared import _extract_json_block

        out = _extract_json_block("  hello  ")
        assert out == "hello"
