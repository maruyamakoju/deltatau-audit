"""Fix subcommand handlers: fix-sb3, fix-cleanrl, and _parse_kwargs helper."""
import importlib.util
import os
import sys
from pathlib import Path

from deltatau_audit.cli import (
    _resolve_workers, _require_module, _validate_gym_env_or_exit,
)


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
        # Try int, then float, then string
        try:
            result[k.strip()] = int(v.strip())
        except ValueError:
            try:
                result[k.strip()] = float(v.strip())
            except ValueError:
                result[k.strip()] = v.strip()
    return result




def _run_fix_sb3(args):
    """Fix a timing-fragile SB3 model via speed-randomized retraining."""
    # (1) Model file existence check
    if not os.path.isfile(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        if not args.model.endswith(".zip"):
            print("  SB3 models are saved as .zip files. "
                  "Did you mean: {}.zip?".format(args.model))
        sys.exit(1)

    # (2) Dependency/environment checks
    _require_module(
        "stable_baselines3",
        error="ERROR: stable-baselines3 is required.",
        hint='pip install "deltatau-audit[sb3]"',
    )
    _validate_gym_env_or_exit(args.env, extras="sb3", include_atari=True)

    # (4) Run the fix pipeline
    from deltatau_audit.fixer import fix_sb3_model

    result = fix_sb3_model(
        model_path=args.model,
        algo=args.algo,
        env_id=args.env,
        output_dir=args.out,
        timesteps=args.timesteps,
        speed_min=args.speed_min,
        speed_max=args.speed_max,
        n_audit_episodes=args.episodes,
        device=args.device,
        n_workers=_resolve_workers(args),
        seed=getattr(args, "seed", None),
    )

    # CI mode: check the "After" model
    if args.ci and result.get("after"):
        after_dir = os.path.join(args.out, "after")
        from deltatau_audit.ci import write_ci_summary
        exit_code = write_ci_summary(
            result["after"]["summary"],
            result["after"]["robustness"],
            after_dir,
            deploy_threshold=args.ci_deploy_threshold,
            stress_threshold=args.ci_stress_threshold,
            gate_mode=getattr(args, "ci_gate_mode", "score"),
        )
        status = {0: "pass", 1: "warn", 2: "fail"}[exit_code]
        print(f"\n  CI (fixed model): {status.upper()}")
        sys.exit(exit_code)
    elif args.ci and result.get("skipped"):
        print("\n  CI: PASS (original model already robust)")
        sys.exit(0)




def _run_fix_cleanrl(args):
    """Fix a timing-fragile CleanRL agent via speed-randomized retraining."""
    # (1) Dependency check
    _require_module(
        "torch",
        error="ERROR: PyTorch is required.",
        hint="pip install torch",
    )

    # (2) Load agent class
    if not args.agent_module:
        print("ERROR: --agent-module is required.")
        print("  Provide the path to the Python file containing your Agent class.")
        sys.exit(1)

    import importlib.util
    from pathlib import Path
    module_path = Path(args.agent_module).resolve()
    if not module_path.exists():
        print(f"ERROR: Agent module not found: {module_path}")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("_fix_cleanrl_agent", str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    agent_class_name = args.agent_class
    if not hasattr(module, agent_class_name):
        print(f"ERROR: Class '{agent_class_name}' not found in {module_path}")
        sys.exit(1)
    agent_class = getattr(module, agent_class_name)

    # (3) Environment check
    _validate_gym_env_or_exit(args.env)

    # (4) Run fix pipeline
    from deltatau_audit.fixer_cleanrl import fix_cleanrl_agent

    result = fix_cleanrl_agent(
        agent_class=agent_class,
        agent_kwargs=_parse_kwargs(args.agent_kwargs),
        env_id=args.env,
        output_dir=args.out,
        checkpoint_path=args.checkpoint if (
            args.checkpoint and os.path.isfile(args.checkpoint)) else None,
        timesteps=args.timesteps,
        speed_min=args.speed_min,
        speed_max=args.speed_max,
        n_audit_episodes=args.episodes,
        device=args.device,
        n_workers=_resolve_workers(args),
        seed=getattr(args, "seed", None),
    )

    # CI mode
    if args.ci and result.get("after"):
        after_dir = os.path.join(args.out, "after")
        from deltatau_audit.ci import write_ci_summary
        exit_code = write_ci_summary(
            result["after"]["summary"],
            result["after"]["robustness"],
            after_dir,
            deploy_threshold=args.ci_deploy_threshold,
            stress_threshold=args.ci_stress_threshold,
            gate_mode=getattr(args, "ci_gate_mode", "score"),
        )
        status = {0: "pass", 1: "warn", 2: "fail"}[exit_code]
        print(f"\n  CI (fixed agent): {status.upper()}")
        sys.exit(exit_code)
    elif args.ci and result.get("skipped"):
        print("\n  CI: PASS (original agent already robust)")
        sys.exit(0)


