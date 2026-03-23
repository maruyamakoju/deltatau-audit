"""Audit subcommand handlers: audit, audit-sb3, audit-hf, audit-cleanrl, demo."""
import os
import sys
import time

from deltatau_audit.cli._utils import _run_audit_pipeline
from deltatau_audit.cli import (
    _activate_protocol, _json_redirect, _resolve_workers, _handle_ci,
    make_env_factory, _require_module, _validate_gym_env_or_exit,
    _wrap_external_eval_env,
)


def _run_audit(args):
    """Run audit on a checkpoint."""
    protocol_meta = _activate_protocol(args)
    _json_mode = getattr(args, "output_format", "text") == "json"

    with _json_redirect(args):
        from deltatau_audit import __version__
        print(f"deltatau-audit v{__version__}")
        print(f"  Checkpoint: {args.checkpoint}")
        print(f"  Agent type: {args.agent_type}")
        print(f"  Environment: {args.env}")
        print(f"  Speeds: {args.speeds}")
        print(f"  Episodes: {args.episodes}")
        print(f"  Output: {args.out}")
        if args.ci:
            print(f"  CI mode: ON (deploy>={args.ci_deploy_threshold}, "
                  f"stress>={args.ci_stress_threshold})")
        print()

        env_factory = make_env_factory(args.env, args.speed_hidden,
                                       args.chain_length)
        sample_env = env_factory()
        obs_dim = sample_env.observation_space.shape[0]
        act_dim = sample_env.action_space.n
        sample_env.close()

        from deltatau_audit.adapters.internal_time import InternalTimeAdapter

        adapter = InternalTimeAdapter.from_checkpoint(
            args.checkpoint, obs_dim, act_dim,
            agent_type=args.agent_type, device=args.device,
        )
        print(f"  Agent loaded ({obs_dim}D obs, {act_dim} actions)")
        print(f"  Intervention support: {adapter.supports_intervention}")
        print()

    _run_audit_pipeline(
        adapter, env_factory, args,
        title=args.title,
        extra_audit_kwargs=dict(
            interventions=args.interventions,
            sensitivity_episodes=args.sensitivity_episodes,
        ),
        compare=False,
        protocol_meta=protocol_meta,
    )



def _run_demo(args):
    """Run the CartPole Before/After demo with bundled checkpoints."""
    import gymnasium as gym

    demo_name = args.demo_name
    if demo_name != "cartpole":
        print(f"Unknown demo: {demo_name}")
        print("Available demos: cartpole")
        sys.exit(1)

    # Find bundled checkpoints
    # __file__ is in deltatau_audit/cli/, go up one level to get deltatau_audit/
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    demo_dir = os.path.join(pkg_dir, "demo_data", "cartpole")

    baseline_ckpt = os.path.join(demo_dir, "baseline.pt")
    robust_ckpt = os.path.join(demo_dir, "robust_wide.pt")

    # Fallback to demo_external_env/ (development mode)
    if not os.path.exists(baseline_ckpt):
        project_root = os.path.dirname(pkg_dir)
        baseline_ckpt = os.path.join(
            project_root, "demo_external_env", "checkpoints",
            "baseline", "final.pt")
        robust_ckpt = os.path.join(
            project_root, "demo_external_env", "checkpoints",
            "robust_wide", "final.pt")

    for path, name in [(baseline_ckpt, "baseline"), (robust_ckpt, "robust")]:
        if not os.path.exists(path):
            print(f"Checkpoint not found: {path}")
            print("Run training first or install the package with demo data.")
            sys.exit(1)

    from deltatau_audit.adapters.simple_gru import SimpleGRUAdapter
    from deltatau_audit.auditor import run_full_audit
    from deltatau_audit.report import generate_report

    def cartpole_factory():
        return gym.make("CartPole-v1")

    out_dir = args.out
    n_episodes = args.episodes

    models = [
        ("baseline", baseline_ckpt,
         "CartPole Baseline GRU (Before Fix)"),
        ("robust_wide", robust_ckpt,
         "CartPole Speed-Randomized GRU (After Fix)"),
    ]

    from deltatau_audit import __version__
    n_workers = _resolve_workers(args)
    print(f"deltatau-audit v{__version__} - CartPole Demo")
    print(f"  Episodes per condition: {n_episodes}")
    print(f"  Workers: {n_workers}"
          + (" (parallel)" if n_workers > 1 else
             "  - tip: use --workers auto for faster auditing"))
    print(f"  Output: {out_dir}/")
    if args.ci:
        print(f"  CI mode: ON (deploy>={args.ci_deploy_threshold}, "
              f"stress>={args.ci_stress_threshold})")
    print()

    results = {}
    for name, ckpt, title in models:
        print(f"\n{'=' * 60}")
        print(f"AUDITING: {title}")
        print(f"{'=' * 60}\n")

        adapter = SimpleGRUAdapter.from_checkpoint(
            ckpt, obs_dim=4, act_dim=2, hidden_dim=64)

        t0 = time.time()
        result = run_full_audit(
            adapter, cartpole_factory,
            speeds=[1, 2, 3, 5, 8],
            n_episodes=n_episodes,
            sensitivity_episodes=0,
            seed=getattr(args, "seed", None),
            n_workers=_resolve_workers(args),
        )
        elapsed = time.time() - t0
        print(f"\n  Audit completed in {elapsed:.1f}s")

        report_dir = os.path.join(out_dir, name)
        print()
        generate_report(result, report_dir, title=title)
        results[name] = result

    # Before/After comparison
    if len(results) >= 2:
        b_res = results.get("baseline", {})
        a_res = results.get("robust_wide", {})
        b_sum = b_res.get("summary", {})
        a_sum = a_res.get("summary", {})
        b_rob = b_res.get("robustness", {}).get("per_scenario_scores", {})
        a_rob = a_res.get("robustness", {}).get("per_scenario_scores", {})

        print(f"\n{'=' * 60}")
        print("BEFORE vs AFTER COMPARISON")
        print(f"{'=' * 60}\n")

        print(f"  {'Scenario':12s}  {'Before':>10s}  {'After':>10s}  "
              f"{'Change':>10s}")
        print(f"  {'-' * 12}  {'-' * 10}  {'-' * 10}  {'-' * 10}")

        for sc in b_rob:
            b_pct = b_rob[sc]["return_ratio"] * 100
            a_pct = a_rob.get(sc, {}).get("return_ratio", 0) * 100
            delta = a_pct - b_pct
            sign = "+" if delta >= 0 else ""
            print(f"  {sc:12s}  {b_pct:9.1f}%  {a_pct:9.1f}%  "
                  f"{sign}{delta:8.1f}%")

        if b_sum and a_sum:
            print(f"\n  Deployment: {b_sum['deployment_rating']} "
                  f"({b_sum['deployment_score']:.2f}) -> "
                  f"{a_sum['deployment_rating']} "
                  f"({a_sum['deployment_score']:.2f})")
            print(f"  Stress:     {b_sum['stress_rating']} "
                  f"({b_sum['stress_score']:.2f}) -> "
                  f"{a_sum['stress_rating']} "
                  f"({a_sum['stress_score']:.2f})")

    # Auto-generate comparison.md
    if len(results) >= 2:
        before_json = os.path.join(out_dir, "baseline", "summary.json")
        after_json = os.path.join(out_dir, "robust_wide", "summary.json")
        if os.path.exists(before_json) and os.path.exists(after_json):
            from deltatau_audit.diff import generate_comparison
            comp_path = os.path.join(out_dir, "comparison.md")
            generate_comparison(before_json, after_json, comp_path)
            print(f"\n  comparison.md -> {comp_path}")

    # CI mode: check the "After" model
    if args.ci and "robust_wide" in results:
        after_dir = os.path.join(out_dir, "robust_wide")
        exit_code = _handle_ci(results["robust_wide"], after_dir, args)
        sys.exit(exit_code)



def _run_audit_sb3(args):
    """Audit an SB3 model (.zip) on a Gymnasium environment."""
    protocol_meta = _activate_protocol(args)
    # (1) Model file existence check — before any imports
    if not os.path.isfile(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        if not args.model.endswith(".zip"):
            print("  SB3 models are saved as .zip files. "
                  "Did you mean: {}.zip?".format(args.model))
        print("\n  To try with a sample model:")
        print("  gh release download assets -R maruyamakoju/deltatau-audit "
              "-p cartpole_ppo_sb3.zip")
        print("  deltatau-audit audit-sb3 --algo ppo "
              "--model cartpole_ppo_sb3.zip --env CartPole-v1")
        sys.exit(1)

    # (2) Dependency/environment checks
    _require_module(
        "stable_baselines3",
        error="ERROR: stable-baselines3 is required.",
        hint='pip install "deltatau-audit[sb3]"',
    )
    gym = _validate_gym_env_or_exit(args.env, extras="sb3", include_atari=True)

    from deltatau_audit import __version__
    from deltatau_audit.adapters.sb3 import SB3Adapter

    def _load_adapter():
        return SB3Adapter.from_path(
            args.model,
            algo=args.algo,
            device=args.device,
            vec_normalize_path=getattr(args, "vec_normalize", None),
        )

    _json_mode = getattr(args, "output_format", "text") == "json"

    with _json_redirect(args):
        _n_workers = _resolve_workers(args)
        print(f"deltatau-audit v{__version__} - SB3 Audit")
        print(f"  Model: {args.model}")
        print(f"  Algo:  {args.algo}")
        print(f"  Env:   {args.env}")
        print(f"  Speeds: {args.speeds}")
        print(f"  Episodes: {args.episodes}")
        print(f"  Workers: {_n_workers}"
              + ("" if _n_workers > 1 else
                 "  - tip: --workers auto for faster auditing"))
        print(f"  Device: {args.device}")
        print(f"  Output: {args.out}")
        if args.ci:
            print(f"  CI mode: ON (deploy>={args.ci_deploy_threshold}, "
                  f"stress>={args.ci_stress_threshold})")
        print()

        # (4) Load model with friendly error
        try:
            adapter = _load_adapter()
        except Exception as e:
            print(f"ERROR: Failed to load model: {e}")
            print(f"\n  Make sure the file was saved with "
                  f"{args.algo.upper()}.save() from stable-baselines3.")
            sys.exit(1)

        print(f"  Model loaded ({args.algo.upper()} on {args.env})")
        print()

        env_factory = lambda: _wrap_external_eval_env(gym.make(args.env), args)
        title = args.title or f"{args.algo.upper()} on {args.env}"

    _run_audit_pipeline(
        adapter, env_factory, args,
        title=title,
        extra_audit_kwargs=dict(sensitivity_episodes=0),
        adapter_factory=lambda _seed: _load_adapter(),
        protocol_meta=protocol_meta,
    )



def _run_audit_cleanrl(args):
    """Audit a CleanRL agent (.pt checkpoint) on a Gymnasium environment."""
    protocol_meta = _activate_protocol(args)
    # (1) Dependency check: torch
    _require_module(
        "torch",
        error="ERROR: PyTorch is required for CleanRL auditing.",
        hint="pip install torch",
    )

    # (2) Checkpoint existence check
    if not os.path.isfile(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # (3) Load agent class dynamically
    try:
        from deltatau_audit.adapters.cleanrl import CleanRLAdapter
    except ImportError as e:
        print(f"ERROR: CleanRL adapter not available: {e}")
        sys.exit(1)

    if not args.agent_module:
        print("ERROR: --agent-module is required for CleanRL auditing.")
        print("  Provide the path to the Python file containing your Agent class.")
        print("  Example: --agent-module ppo_cartpole.py")
        sys.exit(1)

    agent_kwargs = _parse_kwargs(args.agent_kwargs)

    def _load_adapter():
        return CleanRLAdapter.from_module_path(
            checkpoint_path=args.checkpoint,
            agent_module_path=args.agent_module,
            agent_class_name=args.agent_class,
            agent_kwargs=agent_kwargs,
            lstm=args.lstm,
            device=args.device,
        )

    try:
        adapter = _load_adapter()
    except (FileNotFoundError, AttributeError, RuntimeError) as e:
        print(f"ERROR: Failed to load agent: {e}")
        sys.exit(1)

    # (4) Environment check
    gym = _validate_gym_env_or_exit(args.env)

    from deltatau_audit import __version__

    _json_mode = getattr(args, "output_format", "text") == "json"

    with _json_redirect(args):
        print(f"deltatau-audit v{__version__} - CleanRL Audit")
        print(f"  Checkpoint: {args.checkpoint}")
        print(f"  Agent module: {args.agent_module}")
        print(f"  Agent class: {args.agent_class}")
        print(f"  Env: {args.env}")
        print(f"  Speeds: {args.speeds}")
        print(f"  Episodes: {args.episodes}")
        print(f"  Output: {args.out}")
        if args.ci:
            print(f"  CI mode: ON (deploy>={args.ci_deploy_threshold}, "
                  f"stress>={args.ci_stress_threshold})")
        print()

        env_factory = lambda: gym.make(args.env)
        title = args.title or f"CleanRL on {args.env}"

    _run_audit_pipeline(
        adapter, env_factory, args,
        title=title,
        extra_audit_kwargs=dict(sensitivity_episodes=0),
        adapter_factory=lambda _seed: _load_adapter(),
        protocol_meta=protocol_meta,
    )



def _run_audit_hf(args):
    """Audit an SB3 model downloaded directly from HuggingFace Hub."""
    protocol_meta = _activate_protocol(args)
    # (1) Dependencies/environment
    _require_module(
        "huggingface_hub",
        error="ERROR: huggingface_hub is required for audit-hf.",
        hint='pip install "deltatau-audit[hf]"',
    )
    _require_module(
        "stable_baselines3",
        error="ERROR: stable-baselines3 is required.",
        hint='pip install "deltatau-audit[hf]"',
    )
    gym = _validate_gym_env_or_exit(args.env, extras="hf", include_atari=False)

    from deltatau_audit import __version__
    from deltatau_audit.adapters.sb3 import SB3Adapter

    def _load_adapter():
        return SB3Adapter.from_hub(
            repo_id=args.repo,
            algo=args.algo,
            filename=getattr(args, "filename", None),
            token=getattr(args, "hf_token", None),
            device=args.device,
        )

    _json_mode = getattr(args, "output_format", "text") == "json"

    with _json_redirect(args):
        _n_workers = _resolve_workers(args)
        print(f"deltatau-audit v{__version__} - HuggingFace Hub Audit")
        print(f"  Repo:    {args.repo}")
        print(f"  Algo:    {args.algo}")
        print(f"  Env:     {args.env}")
        print(f"  Speeds:  {args.speeds}")
        print(f"  Episodes: {args.episodes}")
        print(f"  Workers: {_n_workers}")
        print(f"  Device:  {args.device}")
        print(f"  Output:  {args.out}")
        if args.ci:
            print(f"  CI mode: ON (deploy>={args.ci_deploy_threshold}, "
                  f"stress>={args.ci_stress_threshold})")
        print()

        print(f"  Downloading from HuggingFace Hub: {args.repo} ...")
        try:
            adapter = _load_adapter()
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: Failed to load model from Hub: {e}")
            sys.exit(1)

        print(f"  Model loaded ({args.algo.upper()} on {args.env})")
        print()

        env_factory = lambda: _wrap_external_eval_env(gym.make(args.env), args)
        title = args.title or f"{args.algo.upper()} on {args.env} (from {args.repo})"

    _run_audit_pipeline(
        adapter, env_factory, args,
        title=title,
        extra_audit_kwargs=dict(sensitivity_episodes=0),
        adapter_factory=lambda _seed: _load_adapter(),
        protocol_meta=protocol_meta,
    )


