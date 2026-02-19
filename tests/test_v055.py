"""Tests for v0.5.5: WandB / MLflow experiment tracker integration."""

import argparse
import sys
import types
from unittest.mock import MagicMock, call, patch

import pytest

from deltatau_audit.tracker import (
    _build_metrics,
    _build_params,
    log_to_mlflow,
    log_to_wandb,
    maybe_log,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_result(
    deployment_score=0.85,
    stress_score=0.60,
    robustness_score=0.75,
    reliance_score=3.5,
    sensitivity_mean=0.12,
    deployment_rating="PASS",
    stress_rating="PASS",
    reliance_rating="time_aware",
    quadrant="time_aware_robust",
):
    result = {
        "summary": {
            "deployment_score": deployment_score,
            "stress_score": stress_score,
            "robustness_score": robustness_score,
            "reliance_score": reliance_score,
            "sensitivity_mean": sensitivity_mean,
            "deployment_rating": deployment_rating,
            "stress_rating": stress_rating,
            "reliance_rating": reliance_rating,
            "quadrant": quadrant,
        },
        "robustness": {
            "per_scenario_scores": {
                "jitter": {"return_ratio": 0.92},
                "delay": {"return_ratio": 0.88},
                "speed_5x": {"return_ratio": 0.61},
            }
        },
    }
    return result


def _make_args(**kwargs):
    defaults = dict(
        wandb=False,
        wandb_project="deltatau-audit",
        wandb_run=None,
        mlflow=False,
        mlflow_experiment="deltatau-audit",
        title=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# _build_metrics
# ---------------------------------------------------------------------------

class TestBuildMetrics:
    def test_core_scalars_present(self):
        result = _make_result()
        m = _build_metrics(result)
        assert "deployment_score" in m
        assert "stress_score" in m
        assert "robustness_score" in m

    def test_reliance_score_included_when_present(self):
        result = _make_result(reliance_score=3.5)
        m = _build_metrics(result)
        assert m["reliance_score"] == pytest.approx(3.5)

    def test_reliance_score_absent_when_not_in_summary(self):
        result = _make_result()
        del result["summary"]["reliance_score"]
        m = _build_metrics(result)
        assert "reliance_score" not in m

    def test_sensitivity_mean_included(self):
        result = _make_result(sensitivity_mean=0.12)
        m = _build_metrics(result)
        assert m["sensitivity_mean"] == pytest.approx(0.12)

    def test_per_scenario_return_ratios(self):
        result = _make_result()
        m = _build_metrics(result)
        assert m["scenario/jitter/return_ratio"] == pytest.approx(0.92)
        assert m["scenario/delay/return_ratio"] == pytest.approx(0.88)
        assert m["scenario/speed_5x/return_ratio"] == pytest.approx(0.61)

    def test_empty_robustness(self):
        result = _make_result()
        result["robustness"] = {}
        m = _build_metrics(result)
        # Should not raise; per-scenario keys simply absent
        assert not any(k.startswith("scenario/") for k in m)


# ---------------------------------------------------------------------------
# _build_params
# ---------------------------------------------------------------------------

class TestBuildParams:
    def test_categorical_params_present(self):
        result = _make_result()
        p = _build_params(result)
        assert p["deployment_rating"] == "PASS"
        assert p["stress_rating"] == "PASS"
        assert p["reliance_rating"] == "time_aware"
        assert p["quadrant"] == "time_aware_robust"

    def test_missing_keys_give_question_mark(self):
        result = {"summary": {}}
        p = _build_params(result)
        assert p["deployment_rating"] == "?"
        assert p["quadrant"] == "?"


# ---------------------------------------------------------------------------
# log_to_wandb
# ---------------------------------------------------------------------------

class TestLogToWandB:
    def test_raises_import_error_when_wandb_missing(self):
        with patch.dict(sys.modules, {"wandb": None}):
            with pytest.raises(ImportError, match="wandb"):
                log_to_wandb(_make_result())

    def test_calls_wandb_init_and_log(self):
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            log_to_wandb(_make_result(), run_name="my-run", project="my-proj")

        mock_wandb.init.assert_called_once()
        init_kwargs = mock_wandb.init.call_args.kwargs
        assert init_kwargs["project"] == "my-proj"
        assert init_kwargs["name"] == "my-run"

        mock_wandb.log.assert_called_once()
        logged_metrics = mock_wandb.log.call_args.args[0]
        assert "deployment_score" in logged_metrics

        mock_run.finish.assert_called_once()

    def test_run_finish_called_even_on_log_error(self):
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.log.side_effect = RuntimeError("log failed")

        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            with pytest.raises(RuntimeError):
                log_to_wandb(_make_result())

        mock_run.finish.assert_called_once()

    def test_tags_passed_to_init(self):
        mock_wandb = MagicMock()
        mock_wandb.init.return_value = MagicMock()

        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            log_to_wandb(_make_result(), tags=["ci", "v0.5.5"])

        init_kwargs = mock_wandb.init.call_args.kwargs
        assert "ci" in init_kwargs["tags"]
        assert "v0.5.5" in init_kwargs["tags"]

    def test_config_includes_version(self):
        mock_wandb = MagicMock()
        mock_wandb.init.return_value = MagicMock()

        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            log_to_wandb(_make_result())

        config = mock_wandb.init.call_args.kwargs["config"]
        assert "_deltatau_version" in config


# ---------------------------------------------------------------------------
# log_to_mlflow
# ---------------------------------------------------------------------------

class TestLogToMLflow:
    def test_raises_import_error_when_mlflow_missing(self):
        with patch.dict(sys.modules, {"mlflow": None}):
            with pytest.raises(ImportError, match="mlflow"):
                log_to_mlflow(_make_result())

    def test_calls_set_experiment_and_start_run(self):
        mock_mlflow = MagicMock()
        mock_run_ctx = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run_ctx)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            log_to_mlflow(_make_result(), run_name="test-run", experiment_name="my-exp")

        mock_mlflow.set_experiment.assert_called_once_with("my-exp")
        mock_mlflow.start_run.assert_called_once_with(run_name="test-run")

    def test_logs_metrics_with_slash_replaced(self):
        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            log_to_mlflow(_make_result())

        logged_keys = [c.args[0] for c in mock_mlflow.log_metric.call_args_list]
        # scenario/jitter/return_ratio → scenario_jitter_return_ratio
        assert "scenario_jitter_return_ratio" in logged_keys
        assert not any("/" in k for k in logged_keys)

    def test_logs_params_and_version(self):
        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            log_to_mlflow(_make_result())

        logged_param_keys = [c.args[0] for c in mock_mlflow.log_param.call_args_list]
        assert "deployment_rating" in logged_param_keys
        assert "_deltatau_version" in logged_param_keys

    def test_extra_tags_set(self):
        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            log_to_mlflow(_make_result(), extra_tags={"env": "HalfCheetah"})

        mock_mlflow.set_tags.assert_called_once_with({"env": "HalfCheetah"})


# ---------------------------------------------------------------------------
# maybe_log
# ---------------------------------------------------------------------------

class TestMaybeLog:
    def test_no_trackers_enabled_is_noop(self):
        result = _make_result()
        args = _make_args()
        # Should not raise or call anything
        maybe_log(result, args)

    def test_wandb_flag_triggers_wandb_log(self):
        result = _make_result()
        args = _make_args(wandb=True, wandb_project="proj", wandb_run="run1")

        with patch("deltatau_audit.tracker.log_to_wandb") as mock_log:
            maybe_log(result, args)

        mock_log.assert_called_once_with(result, run_name="run1", project="proj")

    def test_mlflow_flag_triggers_mlflow_log(self):
        result = _make_result()
        args = _make_args(mlflow=True, mlflow_experiment="exp1", title="my-agent")

        with patch("deltatau_audit.tracker.log_to_mlflow") as mock_log:
            maybe_log(result, args)

        mock_log.assert_called_once_with(result, run_name="my-agent", experiment_name="exp1")

    def test_both_flags_trigger_both(self):
        result = _make_result()
        args = _make_args(wandb=True, mlflow=True)

        with patch("deltatau_audit.tracker.log_to_wandb") as mock_wb, \
             patch("deltatau_audit.tracker.log_to_mlflow") as mock_mf:
            maybe_log(result, args)

        mock_wb.assert_called_once()
        mock_mf.assert_called_once()

    def test_import_error_prints_warning_not_raises(self, capsys):
        result = _make_result()
        args = _make_args(wandb=True)

        with patch("deltatau_audit.tracker.log_to_wandb",
                   side_effect=ImportError("wandb not installed")):
            maybe_log(result, args)  # should NOT raise

        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_title_used_as_wandb_run_when_wandb_run_is_none(self):
        result = _make_result()
        args = _make_args(wandb=True, wandb_run=None, title="experiment-42")

        with patch("deltatau_audit.tracker.log_to_wandb") as mock_log:
            maybe_log(result, args)

        mock_log.assert_called_once_with(result, run_name="experiment-42", project="deltatau-audit")


# ---------------------------------------------------------------------------
# CLI parser flags
# ---------------------------------------------------------------------------

class TestCliTrackerFlags:
    def _get_parser(self):
        """Import and build the CLI parser."""
        from deltatau_audit.cli import _add_tracker_args
        p = argparse.ArgumentParser()
        _add_tracker_args(p)
        return p

    def test_wandb_flag_default_false(self):
        p = self._get_parser()
        args = p.parse_args([])
        assert args.wandb is False

    def test_wandb_flag_set(self):
        p = self._get_parser()
        args = p.parse_args(["--wandb"])
        assert args.wandb is True

    def test_wandb_project_default(self):
        p = self._get_parser()
        args = p.parse_args([])
        assert args.wandb_project == "deltatau-audit"

    def test_wandb_project_custom(self):
        p = self._get_parser()
        args = p.parse_args(["--wandb-project", "my-project"])
        assert args.wandb_project == "my-project"

    def test_wandb_run_default_none(self):
        p = self._get_parser()
        args = p.parse_args([])
        assert args.wandb_run is None

    def test_mlflow_flag_default_false(self):
        p = self._get_parser()
        args = p.parse_args([])
        assert args.mlflow is False

    def test_mlflow_flag_set(self):
        p = self._get_parser()
        args = p.parse_args(["--mlflow"])
        assert args.mlflow is True

    def test_mlflow_experiment_default(self):
        p = self._get_parser()
        args = p.parse_args([])
        assert args.mlflow_experiment == "deltatau-audit"

    def test_mlflow_experiment_custom(self):
        p = self._get_parser()
        args = p.parse_args(["--mlflow-experiment", "my-exp"])
        assert args.mlflow_experiment == "my-exp"
