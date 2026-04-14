"""CLI entry point for deltatau-audit.

Usage:
    # Audit an internal-time agent on chain env
    python -m deltatau_audit audit \
        --checkpoint runs/.../final.pt \
        --agent-type internal_time \
        --env chain --out audit_report/

    # Run the CartPole demo (bundled checkpoints)
    python -m deltatau_audit demo cartpole --out demo_report/

    # CI mode (exit codes for pipeline gates)
    python -m deltatau_audit demo cartpole --ci --out ci_report/
    python -m deltatau_audit audit ... --ci --out ci_report/
"""

import argparse
import contextlib
import json as json_mod
import os
import sys
import time


# _parse_kwargs: re-exported here for backward compatibility
# (tests import directly from deltatau_audit.cli)
def _parse_kwargs(kwargs_str):
    """Parse key=value,key=value string into a dict."""
    if not kwargs_str:
        return {}
    result = {}
    for pair in kwargs_str.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        try:
            result[k.strip()] = int(v.strip())
        except ValueError:
            try:
                result[k.strip()] = float(v.strip())
            except ValueError:
                result[k.strip()] = v.strip()
    return result


def make_env_factory(env_type: str, speed_hidden: bool = True, chain_length: int = 20):
    """Create an env factory based on env type string."""
    if env_type == "chain":
        from internal_time_rl.envs.variable_frequency import VariableFrequencyChainEnv

        def factory():
            return VariableFrequencyChainEnv(
                chain_length=chain_length,
                delay=10,
                max_agent_steps=100,
                train_speeds=(1, 2, 3),
                speed_in_obs=not speed_hidden,
            )

        return factory
    else:
        raise ValueError(
            f"Unknown env type: {env_type}. Currently supported: 'chain'. For custom envs, use the Python API directly."
        )


def _add_ci_args(parser):
    """Add CI-related arguments to a subparser."""
    parser.add_argument(
        "--ci",
        action="store_true",
        default=False,
        help="CI mode: write ci_summary.json/md, exit code based on thresholds",
    )
    parser.add_argument(
        "--ci-deploy-threshold", type=float, default=0.80, help="Deployment return ratio threshold (default: 0.80)"
    )
    parser.add_argument(
        "--ci-stress-threshold", type=float, default=0.50, help="Stress return ratio threshold (default: 0.50)"
    )
    parser.add_argument(
        "--ci-min-deployment-pass-rate",
        type=float,
        default=0.80,
        help="Multi-seed CI gate: minimum deployment pass rate (default: 0.80). Used when --seeds is set.",
    )
    parser.add_argument(
        "--ci-min-stress-pass-rate",
        type=float,
        default=0.50,
        help="Multi-seed CI gate: minimum stress pass rate (default: 0.50). Used when --seeds is set.",
    )
    parser.add_argument(
        "--ci-gate-mode",
        type=str,
        choices=["score", "pass_rate", "worst_ci_lower"],
        default="score",
        help="CI gate rule: score (default), pass_rate (multi-seed), or worst_ci_lower (strict research gate).",
    )


def _add_threshold_args(parser):
    """Add --deploy-threshold and --stress-threshold flags."""
    parser.add_argument(
        "--deploy-threshold",
        type=float,
        default=0.80,
        metavar="RATIO",
        help="Deployment return ratio threshold for quadrant classification "
        "(default: 0.80). Below this → deployment_fragile / time_*_fragile.",
    )
    parser.add_argument(
        "--stress-threshold",
        type=float,
        default=0.50,
        metavar="RATIO",
        help="Stress return ratio threshold for CI pass/warn (default: 0.50). Stored in summary for downstream use.",
    )


def _add_quiet_arg(parser):
    """Add --quiet flag to suppress episode-level progress output."""
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        default=False,
        help="Suppress episode-level progress bars and verbose output. Final summary is still shown.",
    )


def _add_tracker_args(parser):
    """Add --wandb / --mlflow experiment tracker flags."""
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help='Log audit metrics to Weights & Biases after the audit. Requires: pip install "deltatau-audit[wandb]".',
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="deltatau-audit",
        metavar="PROJECT",
        help="WandB project name (default: deltatau-audit). Used with --wandb.",
    )
    parser.add_argument(
        "--wandb-run",
        type=str,
        default=None,
        metavar="RUN",
        help="WandB run name (default: audit title). Used with --wandb.",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        default=False,
        help='Log audit metrics to MLflow after the audit. Requires: pip install "deltatau-audit[mlflow]".',
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="deltatau-audit",
        metavar="EXP",
        help="MLflow experiment name (default: deltatau-audit). Used with --mlflow.",
    )


def _add_adaptive_args(parser):
    """Add --adaptive, --target-ci-width, --max-episodes for adaptive sampling."""
    parser.add_argument(
        "--adaptive",
        action="store_true",
        default=False,
        help="Use adaptive episode sampling: keep adding episode batches until "
        "every scenario's 95%% bootstrap CI width on the return ratio is "
        "below --target-ci-width (or --max-episodes is reached).",
    )
    parser.add_argument(
        "--target-ci-width",
        type=float,
        default=0.10,
        metavar="WIDTH",
        help="Target 95%% CI width for adaptive sampling (default: 0.10). Ignored unless --adaptive is set.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=500,
        metavar="N",
        help="Hard cap on episodes per scenario in adaptive mode (default: 500). Ignored unless --adaptive is set.",
    )


def _add_stats_args(parser):
    """Add statistical-rigor arguments."""
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        metavar="N",
        help="Bootstrap resamples for return-ratio confidence intervals (default: 2000).",
    )


def _add_seed_arg(parser):
    """Add --seed for reproducible audits."""
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible results (default: None = non-deterministic)",
    )


def _add_seeds_arg(parser):
    """Add --seeds for multi-seed protocol audits."""
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        metavar="SEED",
        help="Run a multi-seed sweep (example: --seeds 0 1 2 3 4). When set, --seed is ignored.",
    )


def _add_protocol_args(parser):
    """Add protocol preset controls (`custom`/`ci`/`research`/`paper`)."""
    parser.add_argument(
        "--protocol",
        type=str,
        choices=["custom", "ci", "research", "paper"],
        default="custom",
        help="Audit protocol preset. `research` enforces strict reproducibility "
        "defaults; `paper` enforces publication-grade defaults.",
    )
    parser.add_argument(
        "--allow-protocol-override",
        action="store_true",
        default=False,
        help="Allow explicit CLI flags to override protocol preset values.",
    )


def _add_explain_fail_arg(parser):
    """Add --explain-fail to print concise root-cause guidance."""
    parser.add_argument(
        "--explain-fail",
        action="store_true",
        default=False,
        help="Print an explicit failure explanation block after the audit.",
    )


def _add_workers_arg(parser):
    """Add --workers for parallel episode execution."""
    parser.add_argument(
        "--workers",
        type=str,
        default="1",
        help="Parallel workers for episode collection. "
        "Use an integer (e.g. 4) or 'auto' to use all "
        "CPU cores. Default: 1 (serial).",
    )


def _add_eval_env_wrap_args(parser):
    """Add optional evaluation-time observation wrappers for external policies."""
    parser.add_argument(
        "--env-wrap-time-feature",
        action="store_true",
        default=False,
        help="Wrap eval env with TimeFeatureWrapper (adds dt/elapsed/phase features).",
    )
    parser.add_argument(
        "--env-wrap-phase-period",
        type=int,
        default=200,
        help="Phase period for --env-wrap-time-feature (default: 200).",
    )
    parser.add_argument(
        "--env-wrap-frame-stack",
        type=int,
        default=0,
        help="Stack N observations at eval time (0/1 disables).",
    )
    parser.add_argument(
        "--env-wrap-flatten-obs",
        action="store_true",
        default=False,
        help="Flatten stacked observations (recommended with MlpPolicy).",
    )


def _resolve_workers(args) -> int:
    """Parse --workers value, resolving 'auto' to os.cpu_count()."""
    import os

    raw = getattr(args, "workers", "1") or "1"
    if str(raw).strip().lower() in ("auto", "0", "-1"):
        return max(1, os.cpu_count() or 1)
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def _wrap_external_eval_env(base_env, args):
    """Apply optional external-eval wrappers configured by CLI flags."""
    env = base_env

    # Normalize dict observations (e.g. shimmy dm_control) to a flat Box
    # so SB3 MlpPolicy checkpoints can be audited without custom adapters.
    try:
        import gymnasium as gym

        if isinstance(env.observation_space, gym.spaces.Dict):
            from gymnasium.wrappers import FlattenObservation

            env = FlattenObservation(env)
    except Exception:
        pass

    if bool(getattr(args, "env_wrap_time_feature", False)):
        from deltatau_audit.wrappers.time_feature import TimeFeatureWrapper

        env = TimeFeatureWrapper(
            env,
            phase_period=int(getattr(args, "env_wrap_phase_period", 200)),
        )

    stack = int(getattr(args, "env_wrap_frame_stack", 0) or 0)
    if stack > 1:
        try:
            from gymnasium.wrappers import FrameStackObservation
        except Exception:
            from gymnasium.wrappers import FrameStack as FrameStackObservation
        env = FrameStackObservation(env, stack_size=stack)
        if bool(getattr(args, "env_wrap_flatten_obs", False)):
            from gymnasium.wrappers import FlattenObservation

            env = FlattenObservation(env)

    return env


def _add_compare_arg(parser):
    """Add --compare for generating a comparison.html against a previous audit."""
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        metavar="SUMMARY_JSON",
        help="Path to a previous summary.json to compare against. Generates comparison.html in the output directory.",
    )


def _maybe_compare(args, out_dir: str):
    """If --compare was given, generate comparison.html in the output dir."""
    compare = getattr(args, "compare", None)
    if not compare:
        return
    import pathlib

    compare_path = pathlib.Path(compare)
    if not compare_path.exists():
        print(f"  WARNING: --compare path not found: {compare}")
        return
    import os

    from deltatau_audit.diff import generate_comparison_html

    new_json = os.path.join(out_dir, "summary.json")
    html_path = os.path.join(out_dir, "comparison.html")
    try:
        generate_comparison_html(compare_path, new_json, output_path=html_path)
        print(f"  Comparison:  {html_path}")
    except Exception as e:
        print(f"  WARNING: Could not generate comparison: {e}")


def _json_redirect(args):
    """Return a context manager that redirects stdout to stderr in JSON mode."""
    fmt = getattr(args, "output_format", "text")
    if fmt == "json":
        return contextlib.redirect_stdout(sys.stderr)
    return contextlib.nullcontext()


def _emit_json(result: dict, args) -> None:
    """Print JSON result to stdout if --format json is active."""
    fmt = getattr(args, "output_format", "text")
    if fmt == "json":
        print(json_mod.dumps(result, indent=2, default=str))


def _add_format_arg(parser):
    """Add --format for output format selection."""
    parser.add_argument(
        "--format",
        type=str,
        default="text",
        choices=["text", "markdown", "json"],
        dest="output_format",
        help="Output format: text (default), markdown (PR-ready table, "
        "writes to $GITHUB_STEP_SUMMARY), or json (structured JSON "
        "to stdout for piping; progress goes to stderr).",
    )


def _print_markdown_summary(result: dict, label: str = "") -> str:
    """Print and return a markdown summary suitable for GitHub PR comments.

    When $GITHUB_STEP_SUMMARY is set (GitHub Actions), also appends to it.
    """
    summary = result["summary"]
    scores = result["robustness"].get("per_scenario_scores", {})

    dep_r = summary.get("deployment_rating", "?")
    dep_s = summary.get("deployment_score", 0)
    str_r = summary.get("stress_rating", "?")
    str_s = summary.get("stress_score", 0)
    quad = summary.get("quadrant", "?")

    icon = {"PASS": "✅", "MILD": "🟡", "DEGRADED": "⚠️", "FAIL": "❌"}.get(dep_r, "❓")

    header = f"## {icon} Time Robustness Audit: **{dep_r}**"
    if label:
        header += f" - {label}"

    lines = [
        header,
        "",
        "| Badge | Rating | Score |",
        "|-------|--------|-------|",
        f"| **Deployment** | **{dep_r}** | {dep_s:.2f} |",
        f"| **Stress** | **{str_r}** | {str_s:.2f} |",
        "",
        f"**Quadrant:** `{quad}`",
        "",
        "**Per-scenario results:**",
        "",
        "| Scenario | Category | Return | Significant |",
        "|----------|----------|--------|-------------|",
    ]

    _deploy = {"jitter", "delay", "spike", "obs_noise"}
    for sc, info in scores.items():
        ret = info.get("return_ratio", 0) * 100
        cat = "Deployment" if sc in _deploy else "Stress"
        sig = "⚠️" if info.get("significant") else "—"
        lines.append(f"| `{sc}` | {cat} | {ret:.0f}% | {sig} |")

    # Failure diagnosis block
    diagnosis = result.get("diagnosis")
    if diagnosis and diagnosis.get("issues"):
        lines.extend(["", "**Failure Analysis:**", ""])
        lines.append(f"> ⚠️ {diagnosis['summary_line']}")
        lines.append(">")
        lines.append(f"> **Pattern:** {diagnosis['primary_pattern']}")
        lines.append(">")
        lines.append(f"> **Cause:** {diagnosis['root_cause']}")
        lines.append(">")
        lines.append(f"> **Fix:** {diagnosis['fix_recommendation']}")

    from deltatau_audit import __version__

    lines.extend(
        [
            "",
            f"*Generated by [deltatau-audit](https://github.com/maruyamakoju/deltatau-audit) v{__version__}*",
        ]
    )

    md = "\n".join(lines)
    print(md)

    # Write to GitHub Actions step summary if available
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        try:
            with open(step_summary, "a", encoding="utf-8") as f:
                f.write(md + "\n\n")
            print("\n  → Written to $GITHUB_STEP_SUMMARY")
        except OSError:
            pass

    return md


def _handle_ci(result, out_dir, args):
    """Write CI summary and return exit code if --ci is set."""
    if not args.ci:
        return 0

    from deltatau_audit.ci import write_ci_summary, write_seed_sweep_ci_summary

    seed_sweep = result.get("seed_sweep")
    gate_mode = getattr(args, "ci_gate_mode", "score")
    if isinstance(seed_sweep, dict) and isinstance(seed_sweep.get("aggregate"), dict):
        if gate_mode == "score":
            gate_mode = "pass_rate"
        if gate_mode == "pass_rate":
            exit_code = write_seed_sweep_ci_summary(
                result["summary"],
                seed_sweep["aggregate"],
                out_dir,
                deploy_threshold=args.ci_deploy_threshold,
                stress_threshold=args.ci_stress_threshold,
                min_deploy_pass_rate=args.ci_min_deployment_pass_rate,
                min_stress_pass_rate=args.ci_min_stress_pass_rate,
                gate_mode="pass_rate",
            )
        else:
            # For strict modes, gate directly on aggregate scenario CI-lower.
            exit_code = write_ci_summary(
                result["summary"],
                result["robustness"],
                out_dir,
                deploy_threshold=args.ci_deploy_threshold,
                stress_threshold=args.ci_stress_threshold,
                gate_mode=gate_mode,
            )
    else:
        exit_code = write_ci_summary(
            result["summary"],
            result["robustness"],
            out_dir,
            deploy_threshold=args.ci_deploy_threshold,
            stress_threshold=args.ci_stress_threshold,
            gate_mode=gate_mode,
        )

    status = {0: "pass", 1: "warn", 2: "fail"}[exit_code]
    dep = result["summary"]["deployment_score"]
    stress = result["summary"]["stress_score"]

    print(f"\n  CI: {status.upper()} (deployment={dep:.2f}, stress={stress:.2f})")
    if gate_mode != "score":
        print(f"  CI gate mode: {gate_mode}")
    if isinstance(seed_sweep, dict) and isinstance(seed_sweep.get("aggregate"), dict):
        pass_rates = seed_sweep["aggregate"].get("pass_rates", {})
        dep_rate = float(pass_rates.get("deployment", 0.0))
        str_rate = float(pass_rates.get("stress", 0.0))
        n_seeds = int(seed_sweep.get("n_seeds", 0))
        print(f"  Multi-seed gate: n={n_seeds}, deployment_pass_rate={dep_rate:.1%}, stress_pass_rate={str_rate:.1%}")
    print(f"  ci_summary.json -> {out_dir}/ci_summary.json")
    print(f"  ci_summary.md   -> {out_dir}/ci_summary.md")

    return exit_code


def _resolve_seed_list(args):
    """Return ordered unique seed list from --seeds, or None if unset."""
    seeds = getattr(args, "seeds", None)
    if not seeds:
        return None
    ordered = []
    for s in seeds:
        if s not in ordered:
            ordered.append(int(s))
    return ordered


def _activate_protocol(args) -> dict:
    """Apply protocol preset to args and return protocol metadata.

    Reads --protocol flag and applies preset defaults to args if not overridden.
    """
    from deltatau_audit.protocols import PROTOCOL_PRESETS

    name = getattr(args, "protocol", "custom") or "custom"
    preset = PROTOCOL_PRESETS.get(name, {})

    if name == "custom" or not preset:
        return {"name": "custom", "applied": {}, "ignored": {}}

    allow_override = getattr(args, "allow_protocol_override", False)
    applied = {}
    ignored = {}

    # Map preset keys to argparse attribute names
    _key_map = {
        "episodes": "episodes",
        "speeds": "speeds",
        "adaptive": "adaptive",
        "target_ci_width": "target_ci_width",
        "bootstrap_samples": "bootstrap_samples",
        "seeds": "seeds",
    }
    for preset_key, arg_attr in _key_map.items():
        if preset_key not in preset:
            continue
        preset_val = preset[preset_key]
        current_val = getattr(args, arg_attr, None)
        if current_val is None or not allow_override:
            if current_val != preset_val:
                try:
                    setattr(args, arg_attr, preset_val)
                    applied[arg_attr] = preset_val
                except AttributeError:
                    pass
        else:
            ignored[arg_attr] = current_val

    if getattr(args, "output_format", "text") != "json":
        print(f"  Protocol: {name}")
        if applied:
            print(f"  Applied preset fields: {', '.join(sorted(applied.keys()))}")
        if ignored:
            print(f"  Note: --allow-protocol-override: {', '.join(sorted(ignored.keys()))}")
        if name == "research":
            print("  Research gate: CI uses worst-case 95% CI lower bound.")
        elif name == "paper":
            print("  Paper gate: 10 seeds + tighter CI + larger bootstrap budget.")

    return {"name": name, "applied": applied, "ignored": ignored}


def _result_experiment_manifest(
    args,
    *,
    title: str | None,
    seed_list: list[int] | None,
    n_workers: int,
    protocol_meta: dict | None,
) -> dict:
    """Build manifest metadata payload for reproducible reports."""
    from deltatau_audit.provenance import build_manifest
    from deltatau_audit.schema import SCHEMA_VERSION

    def _arg(name: str, default=None):
        return getattr(args, name, default)

    experiment = {
        "title": title,
        "out_dir": _arg("out"),
        "command": _arg("command"),
        "env": _arg("env"),
        "algo": _arg("algo"),
        "model": _arg("model"),
        "checkpoint": _arg("checkpoint"),
        "repo": _arg("repo"),
        "agent_module": _arg("agent_module"),
        "agent_class": _arg("agent_class"),
        "device": _arg("device"),
        "speeds": list(_arg("speeds", []) or []),
        "episodes": int(_arg("episodes", 0) or 0),
        "workers": int(n_workers),
        "seed": _arg("seed"),
        "seeds": seed_list or [],
        "adaptive": bool(_arg("adaptive", False)),
        "target_ci_width": _arg("target_ci_width"),
        "max_episodes": _arg("max_episodes"),
        "bootstrap_samples": _arg("bootstrap_samples"),
        "env_wrap_time_feature": bool(_arg("env_wrap_time_feature", False)),
        "env_wrap_phase_period": _arg("env_wrap_phase_period"),
        "env_wrap_frame_stack": _arg("env_wrap_frame_stack"),
        "env_wrap_flatten_obs": bool(_arg("env_wrap_flatten_obs", False)),
        "deploy_threshold": _arg("deploy_threshold"),
        "stress_threshold": _arg("stress_threshold"),
        "ci": bool(_arg("ci", False)),
        "ci_gate_mode": _arg("ci_gate_mode", "score"),
        "ci_deploy_threshold": _arg("ci_deploy_threshold"),
        "ci_stress_threshold": _arg("ci_stress_threshold"),
        "ci_min_deployment_pass_rate": _arg("ci_min_deployment_pass_rate"),
        "ci_min_stress_pass_rate": _arg("ci_min_stress_pass_rate"),
        "schema_version": SCHEMA_VERSION,
    }

    protocol_name = "custom"
    protocol_config = {}
    if isinstance(protocol_meta, dict):
        protocol_name = str(protocol_meta.get("name", "custom"))
        protocol_config = dict(protocol_meta.get("applied", {}))
        if protocol_meta.get("ignored"):
            protocol_config["_ignored_overrides"] = dict(protocol_meta["ignored"])

    return build_manifest(
        command=_arg("command"),
        argv=sys.argv[1:],
        protocol_name=protocol_name,
        protocol_config=protocol_config,
        experiment=experiment,
        cwd=os.getcwd(),
    )


def _print_failure_explanation(result: dict) -> None:
    """Print concise root-cause guidance for failing/degraded runs."""
    diagnosis = result.get("diagnosis", {})
    if not isinstance(diagnosis, dict):
        print("  explain-fail: diagnosis unavailable.")
        return
    issues = diagnosis.get("issues", [])
    if not isinstance(issues, list) or not issues:
        print("  explain-fail: no critical timing failures detected.")
        return

    print()
    print("  explain-fail")
    print("  ────────────")
    print(f"  Summary: {diagnosis.get('summary_line', 'N/A')}")
    print(f"  Pattern: {diagnosis.get('primary_pattern', 'N/A')}")
    print(f"  Cause:   {diagnosis.get('root_cause', 'N/A')}")
    print(f"  Fix:     {diagnosis.get('fix_recommendation', 'N/A')}")


def _build_seed_sweep_result(seed_payload: dict, args) -> dict:
    """Convert run_seed_sweep payload into a report/CI-compatible audit result."""
    from deltatau_audit.seed_sweep import seed_sweep_payload_to_result

    return seed_sweep_payload_to_result(
        seed_payload,
        speeds=list(getattr(args, "speeds", [])),
        n_episodes=int(getattr(args, "episodes", 0)),
        deploy_threshold=float(getattr(args, "deploy_threshold", 0.80)),
        stress_threshold=float(getattr(args, "stress_threshold", 0.50)),
    )


_DM_CONTROL_HINT_TOKENS = ("dm_control", "dm-control", "dmcontrol")


def _is_dm_control_env_id(env_id: str) -> bool:
    env_lower = env_id.lower()
    return any(token in env_lower for token in _DM_CONTROL_HINT_TOKENS)


def _require_module(module_name: str, *, error: str, hint: str | None = None) -> None:
    """Import a required module or print a friendly install hint and exit."""
    try:
        __import__(module_name)
    except ImportError:
        print(error)
        if hint:
            print(f"  {hint}")
        sys.exit(1)


def _print_env_install_hint(
    env_id: str,
    err_lower: str,
    *,
    extras: str | None = None,
    include_atari: bool = False,
) -> None:
    """Print dependency hints for common Gymnasium environment families."""
    env_lower = env_id.lower()
    if _is_dm_control_env_id(env_id) or "namespace dm_control not found" in err_lower:
        print("\n  pip install shimmy[dm-control] dm-control")
        return
    if any(token in env_lower or token in err_lower for token in _MUJOCO_HINT_TOKENS):
        if extras:
            print(f'\n  pip install "deltatau-audit[{extras},mujoco]"')
        else:
            print("\n  pip install gymnasium[mujoco]")
        return
    if "box2d" in err_lower or any(token in env_lower for token in _BOX2D_HINT_TOKENS):
        print("\n  pip install gymnasium[box2d]")
        return
    if include_atari and ("ale" in err_lower or "atari" in env_lower):
        print("\n  pip install gymnasium[atari] autorom[accept-rom-license]")
        return
    print(f"\n  Check the environment ID is correct: {env_id}")


def _validate_gym_env_or_exit(env_id: str, *, extras: str | None = None, include_atari: bool = False):
    """Validate a Gymnasium env ID and return imported gym module."""
    if _is_dm_control_env_id(env_id):
        _require_module(
            "shimmy",
            error="ERROR: dm_control environments require shimmy registration.",
            hint="Install with: pip install shimmy[dm-control] dm-control",
        )
        import shimmy  # noqa: F401

    import gymnasium as gym

    try:
        test_env = gym.make(env_id)
        test_env.close()
    except Exception as e:
        print(f"ERROR: Cannot create environment '{env_id}'")
        print(f"  {e}")
        _print_env_install_hint(env_id, str(e).lower(), extras=extras, include_atari=include_atari)
        sys.exit(1)
    return gym


def main():
    # Import handler functions from submodules
    from deltatau_audit.cli._audit import (
        _run_audit,
        _run_audit_cleanrl,
        _run_audit_hf,
        _run_audit_sb3,
        _run_demo,
    )
    from deltatau_audit.cli._bench import _run_bench, _run_bench_table
    from deltatau_audit.cli._fix import _run_fix_cleanrl, _run_fix_sb3
    from deltatau_audit.cli._research import (
        _run_audit_deliberative,
        _run_audit_horizon,
        _run_badge,
        _run_certify,
        _run_diff,
        _run_research_full,
    )
    from deltatau_audit.cli._stress import (
        _run_stress_ablate,
        _run_stress_analyze,
        _run_stress_train_sb3,
    )

    parser = argparse.ArgumentParser(
        prog="deltatau-audit",
        description="Time Robustness Audit for RL agents",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── audit subcommand ──────────────────────────────────────────
    audit_parser = subparsers.add_parser("audit", help="Run audit on a checkpoint")
    audit_parser.add_argument("--checkpoint", type=str, required=True, help="Path to agent checkpoint (.pt file)")
    audit_parser.add_argument(
        "--agent-type",
        type=str,
        default="internal_time",
        choices=["internal_time", "internal_time_discount", "baseline", "skip_rnn", "ltc"],
        help="Type of agent architecture",
    )
    audit_parser.add_argument("--env", type=str, default="chain", help="Environment type (default: chain)")
    audit_parser.add_argument("--speed-hidden", action="store_true", default=True)
    audit_parser.add_argument("--speeds", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    audit_parser.add_argument("--interventions", type=str, nargs="+", default=["none", "clamp_1", "reverse", "random"])
    audit_parser.add_argument("--episodes", type=int, default=50)
    audit_parser.add_argument("--sensitivity-episodes", type=int, default=20)
    audit_parser.add_argument("--out", type=str, default="audit_report")
    audit_parser.add_argument("--device", type=str, default="cpu")
    audit_parser.add_argument("--chain-length", type=int, default=20)
    audit_parser.add_argument("--title", type=str, default="Time Robustness Audit")
    _add_ci_args(audit_parser)
    _add_seed_arg(audit_parser)
    _add_workers_arg(audit_parser)
    _add_format_arg(audit_parser)
    _add_quiet_arg(audit_parser)
    _add_threshold_args(audit_parser)
    _add_adaptive_args(audit_parser)
    _add_stats_args(audit_parser)
    _add_protocol_args(audit_parser)
    _add_explain_fail_arg(audit_parser)
    _add_tracker_args(audit_parser)

    # ── audit-sb3 subcommand ─────────────────────────────────────
    sb3_parser = subparsers.add_parser("audit-sb3", help="Audit a Stable-Baselines3 model (.zip) on any Gymnasium env")
    sb3_parser.add_argument("--model", type=str, required=True, help="Path to SB3 model (.zip file)")
    sb3_parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["ppo", "sac", "td3", "a2c"],
        help="SB3 algorithm (ppo, sac, td3, a2c)",
    )
    sb3_parser.add_argument(
        "--env", type=str, required=True, help="Gymnasium environment ID (e.g. HalfCheetah-v5, CartPole-v1)"
    )
    sb3_parser.add_argument(
        "--vec-normalize",
        metavar="PATH",
        help="Path to VecNormalize stats .pkl (if model was trained with VecNormalize)",
    )
    sb3_parser.add_argument("--out", type=str, default="audit_report", help="Output directory (default: audit_report/)")
    sb3_parser.add_argument("--episodes", type=int, default=30, help="Episodes per condition (default: 30)")
    sb3_parser.add_argument(
        "--speeds", type=int, nargs="+", default=[1, 2, 3, 5, 8], help="Speed multipliers (default: 1 2 3 5 8)"
    )
    sb3_parser.add_argument("--device", type=str, default="cpu", help="Device (default: cpu)")
    sb3_parser.add_argument("--title", type=str, default=None, help="Report title (default: auto)")
    _add_ci_args(sb3_parser)
    _add_seed_arg(sb3_parser)
    _add_seeds_arg(sb3_parser)
    _add_workers_arg(sb3_parser)
    _add_compare_arg(sb3_parser)
    _add_format_arg(sb3_parser)
    _add_quiet_arg(sb3_parser)
    _add_threshold_args(sb3_parser)
    _add_adaptive_args(sb3_parser)
    _add_stats_args(sb3_parser)
    _add_protocol_args(sb3_parser)
    _add_eval_env_wrap_args(sb3_parser)
    _add_explain_fail_arg(sb3_parser)
    _add_tracker_args(sb3_parser)

    # ── fix-sb3 subcommand ────────────────────────────────────────
    fix_parser = subparsers.add_parser("fix-sb3", help="Fix a timing-fragile SB3 model: audit -> retrain -> re-audit")
    fix_parser.add_argument("--model", type=str, required=True, help="Path to SB3 model (.zip file)")
    fix_parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["ppo", "sac", "td3", "a2c"],
        help="SB3 algorithm (ppo, sac, td3, a2c)",
    )
    fix_parser.add_argument(
        "--env", type=str, required=True, help="Gymnasium environment ID (e.g. HalfCheetah-v5, CartPole-v1)"
    )
    fix_parser.add_argument(
        "--vec-normalize",
        metavar="PATH",
        help="Path to VecNormalize stats .pkl (if model was trained with VecNormalize)",
    )
    fix_parser.add_argument("--out", type=str, default="fix_output", help="Output directory (default: fix_output/)")
    fix_parser.add_argument("--timesteps", type=int, default=None, help="Training timesteps (default: auto)")
    fix_parser.add_argument("--speed-min", type=int, default=1, help="Min speed during training (default: 1)")
    fix_parser.add_argument("--speed-max", type=int, default=5, help="Max speed during training (default: 5)")
    fix_parser.add_argument("--episodes", type=int, default=30, help="Audit episodes per condition (default: 30)")
    fix_parser.add_argument("--device", type=str, default="cpu", help="Device (default: cpu)")
    _add_ci_args(fix_parser)
    _add_seed_arg(fix_parser)
    _add_workers_arg(fix_parser)

    # ── audit-cleanrl subcommand ──────────────────────────────────
    cleanrl_parser = subparsers.add_parser(
        "audit-cleanrl", help="Audit a CleanRL agent (.pt checkpoint) on any Gymnasium env"
    )
    cleanrl_parser.add_argument("--checkpoint", type=str, required=True, help="Path to CleanRL checkpoint (.pt file)")
    cleanrl_parser.add_argument(
        "--agent-module", type=str, required=True, help="Path to Python file containing the Agent class"
    )
    cleanrl_parser.add_argument("--agent-class", type=str, default="Agent", help="Agent class name (default: Agent)")
    cleanrl_parser.add_argument(
        "--agent-kwargs",
        type=str,
        default=None,
        help="Agent constructor kwargs: key=val,key=val (e.g. obs_dim=4,act_dim=2)",
    )
    cleanrl_parser.add_argument(
        "--lstm", action="store_true", default=False, help="Agent uses LSTM (get_action_and_value takes lstm_state)"
    )
    cleanrl_parser.add_argument("--env", type=str, required=True, help="Gymnasium environment ID")
    cleanrl_parser.add_argument(
        "--out", type=str, default="audit_report", help="Output directory (default: audit_report/)"
    )
    cleanrl_parser.add_argument("--episodes", type=int, default=30, help="Episodes per condition (default: 30)")
    cleanrl_parser.add_argument(
        "--speeds", type=int, nargs="+", default=[1, 2, 3, 5, 8], help="Speed multipliers (default: 1 2 3 5 8)"
    )
    cleanrl_parser.add_argument("--device", type=str, default="cpu", help="Device (default: cpu)")
    cleanrl_parser.add_argument("--title", type=str, default=None, help="Report title (default: auto)")
    _add_ci_args(cleanrl_parser)
    _add_seed_arg(cleanrl_parser)
    _add_seeds_arg(cleanrl_parser)
    _add_workers_arg(cleanrl_parser)
    _add_compare_arg(cleanrl_parser)
    _add_format_arg(cleanrl_parser)
    _add_quiet_arg(cleanrl_parser)
    _add_threshold_args(cleanrl_parser)
    _add_adaptive_args(cleanrl_parser)
    _add_stats_args(cleanrl_parser)
    _add_protocol_args(cleanrl_parser)
    _add_explain_fail_arg(cleanrl_parser)
    _add_tracker_args(cleanrl_parser)

    # ── fix-cleanrl subcommand ────────────────────────────────────
    fix_cleanrl_parser = subparsers.add_parser(
        "fix-cleanrl", help="Fix a timing-fragile CleanRL agent: audit -> retrain -> re-audit"
    )
    fix_cleanrl_parser.add_argument(
        "--agent-module", type=str, required=True, help="Path to Python file with Agent class"
    )
    fix_cleanrl_parser.add_argument(
        "--agent-class", type=str, default="Agent", help="Agent class name (default: Agent)"
    )
    fix_cleanrl_parser.add_argument("--agent-kwargs", type=str, default=None, help="Agent kwargs: obs_dim=4,act_dim=2")
    fix_cleanrl_parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to original .pt checkpoint (optional, enables Before audit)"
    )
    fix_cleanrl_parser.add_argument("--env", type=str, required=True, help="Gymnasium environment ID")
    fix_cleanrl_parser.add_argument(
        "--out", type=str, default="fix_output", help="Output directory (default: fix_output/)"
    )
    fix_cleanrl_parser.add_argument("--timesteps", type=int, default=None, help="Training timesteps (default: auto)")
    fix_cleanrl_parser.add_argument("--speed-min", type=int, default=1, help="Min speed during training (default: 1)")
    fix_cleanrl_parser.add_argument("--speed-max", type=int, default=5, help="Max speed during training (default: 5)")
    fix_cleanrl_parser.add_argument(
        "--episodes", type=int, default=30, help="Audit episodes per condition (default: 30)"
    )
    fix_cleanrl_parser.add_argument("--device", type=str, default="cpu", help="Device (default: cpu)")
    _add_ci_args(fix_cleanrl_parser)
    _add_seed_arg(fix_cleanrl_parser)
    _add_workers_arg(fix_cleanrl_parser)

    # ── audit-hf subcommand ───────────────────────────────────────
    hf_parser = subparsers.add_parser("audit-hf", help="Audit an SB3 model downloaded directly from HuggingFace Hub")
    hf_parser.add_argument(
        "--repo", type=str, required=True, metavar="REPO_ID", help="HuggingFace repo ID (e.g. sb3/ppo-CartPole-v1)"
    )
    hf_parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["ppo", "sac", "td3", "a2c"],
        help="SB3 algorithm (ppo, sac, td3, a2c)",
    )
    hf_parser.add_argument("--env", type=str, required=True, help="Gymnasium environment ID (e.g. CartPole-v1)")
    hf_parser.add_argument(
        "--filename", type=str, default=None, help="Model filename in the repo (auto-detected if not provided)"
    )
    hf_parser.add_argument(
        "--hf-token", type=str, default=None, metavar="TOKEN", help="HuggingFace token for private repos"
    )
    hf_parser.add_argument("--out", type=str, default="audit_report", help="Output directory (default: audit_report/)")
    hf_parser.add_argument("--episodes", type=int, default=30, help="Episodes per condition (default: 30)")
    hf_parser.add_argument(
        "--speeds", type=int, nargs="+", default=[1, 2, 3, 5, 8], help="Speed multipliers (default: 1 2 3 5 8)"
    )
    hf_parser.add_argument("--device", type=str, default="cpu", help="Device (default: cpu)")
    hf_parser.add_argument("--title", type=str, default=None, help="Report title (default: auto)")
    _add_ci_args(hf_parser)
    _add_seed_arg(hf_parser)
    _add_seeds_arg(hf_parser)
    _add_workers_arg(hf_parser)
    _add_compare_arg(hf_parser)
    _add_format_arg(hf_parser)
    _add_quiet_arg(hf_parser)
    _add_threshold_args(hf_parser)
    _add_adaptive_args(hf_parser)
    _add_stats_args(hf_parser)
    _add_protocol_args(hf_parser)
    _add_eval_env_wrap_args(hf_parser)
    _add_explain_fail_arg(hf_parser)
    _add_tracker_args(hf_parser)

    # ── demo subcommand ───────────────────────────────────────────
    demo_parser = subparsers.add_parser("demo", help="Run a bundled demo (Before/After comparison)")
    demo_parser.add_argument("demo_name", type=str, nargs="?", default="cartpole", help="Demo name (default: cartpole)")
    demo_parser.add_argument("--out", type=str, default="demo_report", help="Output directory (default: demo_report/)")
    demo_parser.add_argument("--episodes", type=int, default=20, help="Episodes per condition (default: 20)")
    _add_ci_args(demo_parser)
    _add_seed_arg(demo_parser)
    _add_workers_arg(demo_parser)

    # ── bench subcommand ──────────────────────────────────────────
    bench_parser = subparsers.add_parser(
        "bench",
        help="Run matrix benchmarks from a manifest (resume supported)",
    )
    bench_sub = bench_parser.add_subparsers(dest="bench_command")
    bench_run_parser = bench_sub.add_parser(
        "run",
        help="Execute benchmark jobs from YAML/JSON manifest",
    )
    bench_run_parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to benchmark manifest (.yaml/.yml/.json).",
    )
    bench_run_parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output root directory (default: manifest output_dir or bench_runs/).",
    )
    bench_run_parser.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help="Disable resume mode and rerun all jobs.",
    )
    bench_run_parser.add_argument(
        "--fail-fast",
        action="store_true",
        default=False,
        help="Stop at first failed job.",
    )
    bench_run_parser.add_argument(
        "--protocol",
        type=str,
        choices=["custom", "ci", "research", "paper"],
        default="research",
        help="Protocol enforced for benchmark jobs (default: research).",
    )
    bench_run_parser.add_argument(
        "--allow-protocol-override",
        action="store_true",
        default=False,
        help="Allow job-level protocol fields in manifest to override --protocol.",
    )
    bench_table_parser = bench_sub.add_parser(
        "table",
        help="Generate submission table artifacts from existing bench_summary.json",
    )
    bench_table_parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="Path to bench_summary.json or benchmark output directory.",
    )
    bench_table_parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory for submission_table.* (default: summary output root).",
    )

    # ── stress subcommand ──────────────────────────────────────────
    stress_parser = subparsers.add_parser(
        "stress",
        help="Stress failure mechanism analysis and ablation planning",
    )
    stress_sub = stress_parser.add_subparsers(dest="stress_command")
    stress_analyze_parser = stress_sub.add_parser(
        "analyze",
        help="Analyze worst stress scenario and infer failure mechanism",
    )
    stress_analyze_parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="Path to audit summary.json.",
    )
    stress_analyze_parser.add_argument(
        "--out",
        type=str,
        default="stress_artifacts",
        help="Output directory for stress analysis artifacts.",
    )
    stress_analyze_parser.add_argument(
        "--stress-threshold",
        type=float,
        default=0.50,
        help="Stress gate threshold used for pass/fail evaluation (default: 0.50).",
    )
    stress_analyze_parser.add_argument(
        "--include-intervention3",
        action="store_true",
        default=False,
        help="Include intervention-3 memory upgrade in recommended ablation variants.",
    )
    stress_ablate_parser = stress_sub.add_parser(
        "ablate",
        help="Generate ablation manifest for intervention1/2/(+3) benchmark audits",
    )
    stress_ablate_parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="Path to audit summary.json used to condition the ablation plan.",
    )
    stress_ablate_parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="Gymnasium env id for ablation benchmark runs.",
    )
    stress_ablate_parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["ppo", "sac", "td3", "a2c"],
        help="SB3 algo for ablation benchmark runs.",
    )
    stress_ablate_parser.add_argument(
        "--model-template",
        type=str,
        required=True,
        help="Model path template with placeholders, e.g. checkpoints/{variant}/seed_{seed}/model.zip",
    )
    stress_ablate_parser.add_argument(
        "--out",
        type=str,
        default="stress_artifacts",
        help="Output directory for generated ablation plan artifacts.",
    )
    stress_ablate_parser.add_argument(
        "--output-dir",
        type=str,
        default="ablation_runs",
        help="output_dir field embedded into generated benchmark manifest.",
    )
    stress_ablate_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Seed list for ablation matrix (default: 0 1 2 3 4).",
    )
    stress_ablate_parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Episodes per condition in generated ablation benchmark manifest.",
    )
    stress_ablate_parser.add_argument(
        "--speeds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 5, 8],
        help="Speed list for generated ablation benchmark manifest.",
    )
    stress_ablate_parser.add_argument(
        "--protocol",
        type=str,
        choices=["custom", "ci", "research", "paper"],
        default="research",
        help="Protocol written into generated ablation manifest (default: research).",
    )
    stress_ablate_parser.add_argument(
        "--ci-gate-mode",
        type=str,
        choices=["score", "pass_rate", "worst_ci_lower"],
        default="worst_ci_lower",
        help="CI gate mode written into generated ablation manifest.",
    )
    stress_ablate_parser.add_argument(
        "--stress-threshold",
        type=float,
        default=0.50,
        help="Stress threshold used for analysis before manifest generation.",
    )
    stress_ablate_parser.add_argument(
        "--include-intervention3",
        action="store_true",
        default=False,
        help="Include intervention-3 variant in generated ablation matrix.",
    )
    stress_train_parser = stress_sub.add_parser(
        "train-sb3",
        help="Train SB3 checkpoints for stress ablation variants",
    )
    stress_train_parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="Gymnasium env id for SB3 training.",
    )
    stress_train_parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["ppo", "sac", "td3", "a2c"],
        help="SB3 algorithm.",
    )
    stress_train_parser.add_argument(
        "--out-root",
        type=str,
        default="checkpoints",
        help="Checkpoint root directory; saves as {out_root}/{variant}/seed_{seed}/model.zip.",
    )
    stress_train_parser.add_argument(
        "--out",
        type=str,
        default="stress_artifacts",
        help="Output directory for training summary artifacts.",
    )
    stress_train_parser.add_argument(
        "--timesteps",
        type=int,
        default=30000,
        help="Training timesteps per variant/seed model.",
    )
    stress_train_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Seed list for training (default: 0 1 2 3 4).",
    )
    stress_train_parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=[
            "baseline",
            "intervention1_curriculum",
            "intervention2_time_feature",
            "intervention1_plus_2",
        ],
        choices=[
            "baseline",
            "intervention1_curriculum",
            "intervention2_time_feature",
            "intervention1_plus_2",
            "intervention3_memory",
        ],
        help="Training variants to produce.",
    )
    stress_train_parser.add_argument(
        "--include-intervention3",
        action="store_true",
        default=False,
        help="Include intervention3_memory in training variants.",
    )
    stress_train_parser.add_argument(
        "--base-speed",
        type=int,
        default=3,
        help="Base speed for intervention1 jitter curriculum wrapper.",
    )
    stress_train_parser.add_argument(
        "--jitter",
        type=int,
        default=2,
        help="Jitter range for intervention1 jitter wrapper.",
    )
    stress_train_parser.add_argument(
        "--phase-period",
        type=int,
        default=200,
        help="Phase period for time-feature wrapper.",
    )
    stress_train_parser.add_argument(
        "--frame-stack",
        type=int,
        default=4,
        help="Stack size for intervention3 memory variant.",
    )
    stress_train_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Training device passed to SB3.",
    )
    stress_train_parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Retrain even when model.zip already exists.",
    )
    stress_train_parser.add_argument(
        "--fail-fast",
        action="store_true",
        default=False,
        help="Stop immediately when one training job fails.",
    )
    stress_train_parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable SB3 training logs.",
    )

    # ── badge subcommand ───────────────────────────────────────────
    badge_parser = subparsers.add_parser("badge", help="Generate SVG badge images from a summary.json")
    badge_parser.add_argument("summary_json", type=str, help="Path to summary.json from an audit run")
    badge_parser.add_argument("--out", type=str, default=".", help="Output directory for SVG files (default: .)")
    badge_parser.add_argument("--prefix", type=str, default="badge", help="Filename prefix (default: badge)")

    # ── diff subcommand ────────────────────────────────────────────
    diff_parser = subparsers.add_parser("diff", help="Compare two audit summary.json files")
    diff_parser.add_argument("before", type=str, help="Path to 'before' summary.json")
    diff_parser.add_argument("after", type=str, help="Path to 'after' summary.json")
    diff_parser.add_argument("--out", type=str, default="comparison.md", help="Output path (default: comparison.md)")

    # ── certify subcommand ────────────────────────────────────────
    certify_parser = subparsers.add_parser("certify", help="Generate a formal Safety Certificate from audit results")
    certify_parser.add_argument("summary_json", type=str, help="Path to audit summary.json")
    certify_parser.add_argument("--out", type=str, default="certificate.html", help="Output path for the certificate")

    # ── research-full subcommand ──────────────────────────────────
    research_parser = subparsers.add_parser(
        "research-full", help="Run the complete research suite: VLA, LTC, and Deliberative Audits"
    )
    research_parser.add_argument("--env", type=str, default="CartPole-v1", help="Gymnasium environment ID")
    research_parser.add_argument("--out", type=str, default="research_full_report", help="Output directory")
    research_parser.add_argument("--episodes", type=int, default=10, help="Episodes per condition")
    research_parser.add_argument(
        "--speeds", type=int, nargs="+", default=[1, 2, 5], help="Speed multipliers used for staged audits"
    )
    research_parser.add_argument(
        "--deliberative-max-thinking-steps",
        type=int,
        default=5,
        help="Max internal pondering steps for deliberative stage",
    )
    research_parser.add_argument(
        "--bridge-delay-ms", type=float, default=30.0, help="Mean transport delay for bridge stage (ms)"
    )
    research_parser.add_argument(
        "--bridge-delay-std-ms", type=float, default=10.0, help="Std dev of transport delay for bridge stage (ms)"
    )
    research_parser.add_argument(
        "--bridge-dt-ms", type=float, default=10.0, help="Nominal env step duration used in delay conversion (ms)"
    )
    research_parser.add_argument(
        "--bridge-actuator-alpha", type=float, default=0.3, help="First-order actuator lag coefficient in bridge stage"
    )
    research_parser.add_argument(
        "--no-resume", action="store_true", default=False, help="Disable resume and rerun all stages"
    )
    research_parser.add_argument(
        "--fail-fast", action="store_true", default=False, help="Stop pipeline after first failed stage"
    )
    _add_workers_arg(research_parser)
    _add_seed_arg(research_parser)

    # ── audit-deliberative subcommand ──────────────────────────────
    deliberative_parser = subparsers.add_parser(
        "audit-deliberative", help="Audit a deliberative (ACT-based) agent: measure ponder depth vs timing stress"
    )
    deliberative_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to a DeliberativeInternalTimeAgent checkpoint (.pt)"
    )
    deliberative_parser.add_argument(
        "--env", type=str, default="CartPole-v1", help="Gymnasium environment ID (default: CartPole-v1)"
    )
    deliberative_parser.add_argument("--obs-dim", type=int, default=4, help="Observation dimensionality")
    deliberative_parser.add_argument("--act-dim", type=int, default=2, help="Action space size")
    deliberative_parser.add_argument(
        "--speeds", type=int, nargs="+", default=[1, 2, 5], help="Speed multipliers for stress testing"
    )
    deliberative_parser.add_argument("--episodes", type=int, default=20, help="Episodes per speed condition")
    deliberative_parser.add_argument("--out", type=str, default="deliberative_report", help="Output directory")
    _add_seed_arg(deliberative_parser)

    # ── audit-horizon subcommand ────────────────────────────────────
    horizon_parser = subparsers.add_parser(
        "audit-horizon", help="Audit agents on cascading multi-step timing scenarios (long-horizon)"
    )
    horizon_parser.add_argument("--checkpoint", type=str, required=True, help="Path to agent checkpoint (.pt or .zip)")
    horizon_parser.add_argument("--env", type=str, default="CartPole-v1", help="Gymnasium environment ID")
    horizon_parser.add_argument(
        "--horizon", type=int, default=50, help="Number of timesteps for cascade audit (default: 50)"
    )
    horizon_parser.add_argument("--episodes", type=int, default=20, help="Episodes per condition")
    horizon_parser.add_argument("--out", type=str, default="horizon_report", help="Output directory")
    _add_seed_arg(horizon_parser)

    args = parser.parse_args()

    if args.command == "audit":
        _run_audit(args)
    elif args.command == "audit-sb3":
        _run_audit_sb3(args)
    elif args.command == "fix-sb3":
        _run_fix_sb3(args)
    elif args.command == "audit-cleanrl":
        _run_audit_cleanrl(args)
    elif args.command == "fix-cleanrl":
        _run_fix_cleanrl(args)
    elif args.command == "audit-hf":
        _run_audit_hf(args)
    elif args.command == "demo":
        _run_demo(args)
    elif args.command == "bench":
        bench_command = getattr(args, "bench_command", None)
        if bench_command == "run":
            _run_bench(args)
        elif bench_command == "table":
            _run_bench_table(args)
        else:
            bench_parser.print_help()
    elif args.command == "stress":
        stress_command = getattr(args, "stress_command", None)
        if stress_command == "analyze":
            _run_stress_analyze(args)
        elif stress_command == "ablate":
            _run_stress_ablate(args)
        elif stress_command == "train-sb3":
            _run_stress_train_sb3(args)
        else:
            stress_parser.print_help()
    elif args.command == "badge":
        _run_badge(args)
    elif args.command == "diff":
        _run_diff(args)
    elif args.command == "research-full":
        _run_research_full(args)
    elif args.command == "certify":
        _run_certify(args)
    elif args.command == "audit-deliberative":
        _run_audit_deliberative(args)
    elif args.command == "audit-horizon":
        _run_audit_horizon(args)
    else:
        # No subcommand — check if legacy args present
        if "--checkpoint" in sys.argv:
            sys.argv.insert(1, "audit")
            args = parser.parse_args()
            _run_audit(args)
        else:
            parser.print_help()
            print("\nExamples:")
            print("  python -m deltatau_audit demo cartpole")
            print("  python -m deltatau_audit audit-sb3 --algo ppo --model my_model.zip --env HalfCheetah-v5")
            print("  python -m deltatau_audit fix-sb3 --algo ppo --model my_model.zip --env HalfCheetah-v5")
            print("  python -m deltatau_audit audit-sb3 --algo ppo --model my_model.zip --env CartPole-v1 --ci")
            print(
                "  python -m deltatau_audit audit-cleanrl "
                "--checkpoint runs/CartPole/agent.pt "
                "--agent-module ppo_cartpole.py --env CartPole-v1"
            )
            print("  python -m deltatau_audit diff before/summary.json after/summary.json")
            print("  python -m deltatau_audit bench run --manifest bench/manifest.yaml")
            print("  python -m deltatau_audit bench table --summary bench_runs/")
            print(
                "  python -m deltatau_audit stress analyze --summary audit_report/summary.json --out stress_artifacts/"
            )
            print("  python -m deltatau_audit stress train-sb3 --env CartPole-v1 --algo ppo --out-root checkpoints/")
            print("  python -m deltatau_audit badge audit_report/summary.json --out badges/")


if __name__ == "__main__":
    main()
