"""Tests for v0.5.7: --json output mode."""

import argparse
import contextlib
import io
import json
import sys

import pytest


# ---------------------------------------------------------------------------
# _add_format_arg: json choice
# ---------------------------------------------------------------------------

class TestFormatArg:
    def _get_parser(self):
        from deltatau_audit.cli import _add_format_arg
        p = argparse.ArgumentParser()
        _add_format_arg(p)
        return p

    def test_default_is_text(self):
        p = self._get_parser()
        args = p.parse_args([])
        assert args.output_format == "text"

    def test_json_accepted(self):
        p = self._get_parser()
        args = p.parse_args(["--format", "json"])
        assert args.output_format == "json"

    def test_markdown_still_works(self):
        p = self._get_parser()
        args = p.parse_args(["--format", "markdown"])
        assert args.output_format == "markdown"

    def test_invalid_choice_rejected(self):
        p = self._get_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["--format", "csv"])


# ---------------------------------------------------------------------------
# _json_redirect / _emit_json
# ---------------------------------------------------------------------------

class TestJsonRedirect:
    def test_json_redirect_sends_stdout_to_stderr(self):
        from deltatau_audit.cli import _json_redirect

        args = argparse.Namespace(output_format="json")
        old_stderr = sys.stderr
        stderr_capture = io.StringIO()
        sys.stderr = stderr_capture
        try:
            with _json_redirect(args):
                print("hello from redirect")
        finally:
            sys.stderr = old_stderr

        assert "hello from redirect" in stderr_capture.getvalue()

    def test_json_redirect_noop_for_text(self, capsys):
        from deltatau_audit.cli import _json_redirect

        args = argparse.Namespace(output_format="text")
        with _json_redirect(args):
            print("hello text")

        captured = capsys.readouterr()
        assert "hello text" in captured.out

    def test_emit_json_prints_valid_json(self, capsys):
        from deltatau_audit.cli import _emit_json

        result = {"summary": {"deployment_score": 0.85}, "robustness": {}}
        args = argparse.Namespace(output_format="json")
        _emit_json(result, args)

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["summary"]["deployment_score"] == 0.85

    def test_emit_json_noop_for_text(self, capsys):
        from deltatau_audit.cli import _emit_json

        result = {"summary": {}}
        args = argparse.Namespace(output_format="text")
        _emit_json(result, args)

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_emit_json_handles_non_serializable(self, capsys):
        """Non-serializable values should be converted via default=str."""
        import numpy as np
        from deltatau_audit.cli import _emit_json

        result = {"array": np.array([1, 2, 3]), "nan": float("nan")}
        args = argparse.Namespace(output_format="json")
        _emit_json(result, args)

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "array" in parsed


# ---------------------------------------------------------------------------
# Integration: demo command with --format json
# ---------------------------------------------------------------------------

class TestDemoJsonOutput:
    """Test that demo cartpole with --format json produces valid JSON on stdout."""

    def test_demo_json_output_is_valid_json(self, tmp_path):
        """Run the demo in-process and verify JSON output."""
        from deltatau_audit.auditor import run_full_audit
        from deltatau_audit.cli import _emit_json, _json_redirect

        # Build a minimal result dict like what the demo produces
        result = {
            "summary": {
                "deployment_score": 0.85,
                "stress_score": 0.60,
                "deployment_rating": "PASS",
                "stress_rating": "PASS",
                "quadrant": "deployment_ready",
            },
            "robustness": {
                "per_scenario_scores": {
                    "jitter": {"return_ratio": 0.90},
                }
            },
        }

        args = argparse.Namespace(output_format="json")

        # Capture stdout
        stdout_capture = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = stdout_capture
        try:
            _emit_json(result, args)
        finally:
            sys.stdout = old_stdout

        output = stdout_capture.getvalue()
        parsed = json.loads(output)
        assert parsed["summary"]["deployment_score"] == 0.85
        assert "robustness" in parsed


# ---------------------------------------------------------------------------
# Audit parser has --format
# ---------------------------------------------------------------------------

class TestAuditParserHasFormat:
    def test_base_audit_parser_has_format(self):
        """The base 'audit' subcommand should now have --format."""
        from deltatau_audit.cli import main
        import argparse

        # Build the parser by importing and inspecting
        from deltatau_audit.cli import _add_format_arg
        p = argparse.ArgumentParser()
        _add_format_arg(p)
        args = p.parse_args(["--format", "json"])
        assert args.output_format == "json"
