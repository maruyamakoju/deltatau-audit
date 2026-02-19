"""Experiment tracker integration: Weights & Biases and MLflow.

After running ``run_full_audit()``, call ``maybe_log(result, args)`` (from CLI)
or use ``log_to_wandb`` / ``log_to_mlflow`` directly from Python code to push
metrics to your tracking server.
"""

from typing import Any, Dict, List, Optional


def _build_metrics(result: Dict) -> Dict[str, Any]:
    """Extract flat metrics dict from an audit result.

    Returns scalar metrics suitable for logging to WandB or MLflow.
    """
    summary = result["summary"]
    rob = result.get("robustness", {})

    metrics: Dict[str, Any] = {
        "deployment_score": summary.get("deployment_score", 0.0),
        "stress_score": summary.get("stress_score", 0.0),
        "robustness_score": summary.get("robustness_score", 0.0),
    }

    rel_score = summary.get("reliance_score")
    if rel_score is not None:
        metrics["reliance_score"] = rel_score

    sens = summary.get("sensitivity_mean")
    if sens is not None:
        metrics["sensitivity_mean"] = sens

    # Per-scenario return ratios
    for sc, sc_data in rob.get("per_scenario_scores", {}).items():
        metrics[f"scenario/{sc}/return_ratio"] = sc_data.get("return_ratio", 0.0)

    return metrics


def _build_params(result: Dict) -> Dict[str, str]:
    """Extract string/categorical params for logging."""
    summary = result["summary"]
    return {
        "deployment_rating": str(summary.get("deployment_rating", "?")),
        "stress_rating": str(summary.get("stress_rating", "?")),
        "reliance_rating": str(summary.get("reliance_rating", "?")),
        "quadrant": str(summary.get("quadrant", "?")),
    }


def log_to_wandb(
    result: Dict,
    run_name: Optional[str] = None,
    project: str = "deltatau-audit",
    run_config: Optional[Dict] = None,
    tags: Optional[List[str]] = None,
) -> None:
    """Log audit metrics to a Weights & Biases run.

    Args:
        result: Output of ``run_full_audit()``.
        run_name: WandB run name (default: None → auto-generated).
        project: WandB project name (default: "deltatau-audit").
        run_config: Extra config dict to log to the run.
        tags: List of WandB tags.

    Raises:
        ImportError: If ``wandb`` is not installed. Install with
            ``pip install "deltatau-audit[wandb]"``.
    """
    try:
        import wandb  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "WandB is not installed. Install it with:\n"
            '  pip install "deltatau-audit[wandb]"'
        )

    from . import __version__
    metrics = _build_metrics(result)
    params = _build_params(result)
    config = {**(run_config or {}), **params, "_deltatau_version": __version__}

    run = wandb.init(
        project=project,
        name=run_name,
        config=config,
        tags=tags or [],
        reinit=True,
    )
    try:
        wandb.log(metrics)
    finally:
        run.finish()


def log_to_mlflow(
    result: Dict,
    run_name: Optional[str] = None,
    experiment_name: str = "deltatau-audit",
    extra_tags: Optional[Dict[str, str]] = None,
) -> None:
    """Log audit metrics to an MLflow run.

    Args:
        result: Output of ``run_full_audit()``.
        run_name: MLflow run name (default: None → auto-generated).
        experiment_name: MLflow experiment name (default: "deltatau-audit").
        extra_tags: Additional MLflow tags to set on the run.

    Raises:
        ImportError: If ``mlflow`` is not installed. Install with
            ``pip install "deltatau-audit[mlflow]"``.
    """
    try:
        import mlflow  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "MLflow is not installed. Install it with:\n"
            '  pip install "deltatau-audit[mlflow]"'
        )

    from . import __version__
    metrics = _build_metrics(result)
    params = _build_params(result)

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        for k, v in metrics.items():
            # MLflow metric keys cannot contain '/' — replace with '_'
            mlflow.log_metric(k.replace("/", "_"), v)
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("_deltatau_version", __version__)
        if extra_tags:
            mlflow.set_tags(extra_tags)


def maybe_log(result: Dict, args: Any) -> None:
    """Log to configured trackers after an audit.

    Called by the CLI after each audit command completes.
    Reads ``--wandb``, ``--wandb-project``, ``--wandb-run``,
    ``--mlflow``, ``--mlflow-experiment`` from ``args``.
    """
    title = getattr(args, "title", None)

    if getattr(args, "wandb", False):
        project = getattr(args, "wandb_project", "deltatau-audit")
        run_name = getattr(args, "wandb_run", None) or title
        try:
            log_to_wandb(result, run_name=run_name, project=project)
            print(f"  → WandB: metrics logged to project '{project}'")
        except ImportError as e:
            print(f"  WARNING: {e}")

    if getattr(args, "mlflow", False):
        exp = getattr(args, "mlflow_experiment", "deltatau-audit")
        run_name = title
        try:
            log_to_mlflow(result, run_name=run_name, experiment_name=exp)
            print(f"  → MLflow: metrics logged to experiment '{exp}'")
        except ImportError as e:
            print(f"  WARNING: {e}")
