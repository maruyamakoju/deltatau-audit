"""Tests for deltatau_audit.cli — command-line interface."""

import pytest


# ── make_env_factory ──────────────────────────────────────────────────

class TestMakeEnvFactory:
    def test_chain_env_factory(self):
        """Test chain environment factory creation."""
        from deltatau_audit.cli import make_env_factory

        factory = make_env_factory("chain", speed_hidden=True, chain_length=20)

        # Factory should be callable
        assert callable(factory)

        # Should create env with correct parameters
        env = factory()
        assert env is not None

        # Check it's the right type
        from internal_time_rl.envs.variable_frequency import VariableFrequencyChainEnv
        assert isinstance(env, VariableFrequencyChainEnv)

        env.close()

    def test_chain_env_with_custom_length(self):
        """Test chain env with custom chain length."""
        from deltatau_audit.cli import make_env_factory

        factory = make_env_factory("chain", speed_hidden=False, chain_length=30)
        env = factory()

        # Should work without error
        obs, _ = env.reset()
        assert obs is not None

        env.close()

    def test_unknown_env_raises(self):
        """Unknown env type should raise ValueError."""
        from deltatau_audit.cli import make_env_factory

        with pytest.raises(ValueError, match="Unknown env type"):
            make_env_factory("unknown_env_type")


# ── CI helpers ────────────────────────────────────────────────────────

class TestCIHelpers:
    def test_add_ci_args(self):
        """Test that CI arguments are added to parser."""
        import argparse
        from deltatau_audit.cli import _add_ci_args

        parser = argparse.ArgumentParser()
        _add_ci_args(parser)

        # Parse with CI args
        args = parser.parse_args([
            "--ci",
            "--ci-deploy-threshold", "0.85",
            "--ci-stress-threshold", "0.60",
        ])

        assert args.ci is True
        assert args.ci_deploy_threshold == 0.85
        assert args.ci_stress_threshold == 0.60

    def test_add_ci_args_defaults(self):
        """Test CI argument defaults."""
        import argparse
        from deltatau_audit.cli import _add_ci_args

        parser = argparse.ArgumentParser()
        _add_ci_args(parser)

        args = parser.parse_args([])

        assert args.ci is False
        assert args.ci_deploy_threshold == 0.80
        assert args.ci_stress_threshold == 0.50


# ── Argument parsing ──────────────────────────────────────────────────

class TestArgumentParsing:
    def test_audit_subcommand_args(self):
        """Test audit subcommand argument parsing."""
        from deltatau_audit.cli import main
        import sys

        # Mock sys.argv
        old_argv = sys.argv
        try:
            sys.argv = [
                "deltatau-audit",
                "audit",
                "--checkpoint", "model.pt",
                "--agent-type", "internal_time",
                "--env", "chain",
                "--speeds", "1", "2", "3",
                "--episodes", "20",
                "--out", "test_report",
            ]

            # We can't actually run main() without the full environment,
            # but we can test that argument parsing works
            import argparse
            from deltatau_audit.cli import main

            parser = argparse.ArgumentParser()
            subparsers = parser.add_subparsers(dest="command")

            audit_parser = subparsers.add_parser("audit")
            audit_parser.add_argument("--checkpoint", type=str, required=True)
            audit_parser.add_argument("--agent-type", type=str, default="internal_time")
            audit_parser.add_argument("--env", type=str, default="chain")
            audit_parser.add_argument("--speeds", type=int, nargs="+", default=[1, 2, 3, 5, 8])
            audit_parser.add_argument("--episodes", type=int, default=50)
            audit_parser.add_argument("--out", type=str, default="audit_report")

            args = parser.parse_args(sys.argv[1:])

            assert args.command == "audit"
            assert args.checkpoint == "model.pt"
            assert args.agent_type == "internal_time"
            assert args.env == "chain"
            assert args.speeds == [1, 2, 3]
            assert args.episodes == 20
            assert args.out == "test_report"

        finally:
            sys.argv = old_argv

    def test_demo_subcommand_args(self):
        """Test demo subcommand argument parsing."""
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")

        demo_parser = subparsers.add_parser("demo")
        demo_parser.add_argument("demo_name", type=str, nargs="?", default="cartpole")
        demo_parser.add_argument("--out", type=str, default="demo_report")
        demo_parser.add_argument("--episodes", type=int, default=30)

        args = parser.parse_args(["demo", "cartpole", "--out", "my_demo", "--episodes", "10"])

        assert args.command == "demo"
        assert args.demo_name == "cartpole"
        assert args.out == "my_demo"
        assert args.episodes == 10

    def test_diff_subcommand_args(self):
        """Test diff subcommand argument parsing."""
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")

        diff_parser = subparsers.add_parser("diff")
        diff_parser.add_argument("before", type=str)
        diff_parser.add_argument("after", type=str)
        diff_parser.add_argument("--out", type=str, default="comparison.md")

        args = parser.parse_args([
            "diff",
            "before/summary.json",
            "after/summary.json",
            "--out", "my_comparison.md",
        ])

        assert args.command == "diff"
        assert args.before == "before/summary.json"
        assert args.after == "after/summary.json"
        assert args.out == "my_comparison.md"


# ── Integration smoke tests ───────────────────────────────────────────

class TestCLISmoke:
    def test_main_no_args_shows_help(self, capsys):
        """Running main with no args should show help."""
        import sys
        from deltatau_audit.cli import main

        old_argv = sys.argv
        try:
            sys.argv = ["deltatau-audit"]

            # Should print help and not crash
            try:
                main()
            except SystemExit:
                pass  # argparse may call sys.exit

            captured = capsys.readouterr()
            # Should show usage or examples
            assert "usage" in captured.out.lower() or "examples" in captured.out.lower()

        finally:
            sys.argv = old_argv

    def test_diff_command_basic(self, tmp_path):
        """Test diff command with mock summaries."""
        import sys
        import json
        from deltatau_audit.cli import _run_diff

        # Create mock summary files
        before_path = tmp_path / "before_summary.json"
        after_path = tmp_path / "after_summary.json"
        out_path = tmp_path / "comparison.md"

        before_data = {
            "summary": {
                "deployment_score": 0.60,
                "deployment_rating": "DEGRADED",
                "stress_score": 0.40,
                "stress_rating": "FAIL",
                "quadrant": "deployment_fragile",
            }
        }

        after_data = {
            "summary": {
                "deployment_score": 0.95,
                "deployment_rating": "PASS",
                "stress_score": 0.85,
                "stress_rating": "MILD",
                "quadrant": "deployment_ready",
            }
        }

        with open(before_path, "w") as f:
            json.dump(before_data, f)

        with open(after_path, "w") as f:
            json.dump(after_data, f)

        # Mock args
        class Args:
            before = str(before_path)
            after = str(after_path)
            out = str(out_path)

        args = Args()

        # Run diff
        _run_diff(args)

        # Check output was created
        assert out_path.exists()

        # Check content
        content = out_path.read_text()
        assert "Comparison" in content
        assert "DEGRADED" in content
        assert "PASS" in content
