"""CLI entry point for deltatau-audit (v0.3).

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
import os
import sys
import time


def make_env_factory(env_type: str, speed_hidden: bool = True,
                     chain_length: int = 20):
    """Create an env factory based on env type string."""
    if env_type == "chain":
        from internal_time_rl.envs.variable_frequency import \
            VariableFrequencyChainEnv

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
            f"Unknown env type: {env_type}. "
            f"Currently supported: 'chain'. "
            f"For custom envs, use the Python API directly."
        )


def _add_ci_args(parser):
    """Add CI-related arguments to a subparser."""
    parser.add_argument("--ci", action="store_true", default=False,
                        help="CI mode: write ci_summary.json/md, exit code "
                             "based on thresholds")
    parser.add_argument("--ci-deploy-threshold", type=float, default=0.80,
                        help="Deployment return ratio threshold (default: 0.80)")
    parser.add_argument("--ci-stress-threshold", type=float, default=0.50,
                        help="Stress return ratio threshold (default: 0.50)")


def _handle_ci(result, out_dir, args):
    """Write CI summary and return exit code if --ci is set."""
    if not args.ci:
        return 0

    from .ci import write_ci_summary

    exit_code = write_ci_summary(
        result["summary"], result["robustness"], out_dir,
        deploy_threshold=args.ci_deploy_threshold,
        stress_threshold=args.ci_stress_threshold,
    )

    status = {0: "pass", 1: "warn", 2: "fail"}[exit_code]
    dep = result["summary"]["deployment_score"]
    stress = result["summary"]["stress_score"]

    print(f"\n  CI: {status.upper()} "
          f"(deployment={dep:.2f}, stress={stress:.2f})")
    print(f"  ci_summary.json -> {out_dir}/ci_summary.json")
    print(f"  ci_summary.md   -> {out_dir}/ci_summary.md")

    return exit_code


def _run_audit(args):
    """Run audit on a checkpoint."""
    print(f"deltatau-audit v0.3")
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

    from .adapters.internal_time import InternalTimeAdapter

    adapter = InternalTimeAdapter.from_checkpoint(
        args.checkpoint, obs_dim, act_dim,
        agent_type=args.agent_type, device=args.device,
    )
    print(f"  Agent loaded ({obs_dim}D obs, {act_dim} actions)")
    print(f"  Intervention support: {adapter.supports_intervention}")
    print()

    from .auditor import run_full_audit

    t0 = time.time()
    result = run_full_audit(
        adapter, env_factory,
        speeds=args.speeds,
        n_episodes=args.episodes,
        interventions=args.interventions,
        sensitivity_episodes=args.sensitivity_episodes,
        device=args.device,
    )
    elapsed = time.time() - t0
    print(f"\n  Audit completed in {elapsed:.1f}s")

    from .report import generate_report

    print()
    generate_report(result, args.out, title=args.title)

    exit_code = _handle_ci(result, args.out, args)
    if args.ci:
        sys.exit(exit_code)


def _run_demo(args):
    """Run the CartPole Before/After demo with bundled checkpoints."""
    import gymnasium as gym

    demo_name = args.demo_name
    if demo_name != "cartpole":
        print(f"Unknown demo: {demo_name}")
        print("Available demos: cartpole")
        sys.exit(1)

    # Find bundled checkpoints
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
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
            print(f"Run training first or install the package with demo data.")
            sys.exit(1)

    from .adapters.simple_gru import SimpleGRUAdapter
    from .auditor import run_full_audit
    from .report import generate_report

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

    print(f"deltatau-audit v0.3 — CartPole Demo")
    print(f"  Episodes per condition: {n_episodes}")
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
        )
        elapsed = time.time() - t0
        print(f"\n  Audit completed in {elapsed:.1f}s")

        report_dir = os.path.join(out_dir, name)
        print()
        generate_report(result, report_dir, title=title)
        results[name] = result

    # Before/After comparison
    if len(results) >= 2:
        print("\n" + "=" * 60)
        print("BEFORE vs AFTER COMPARISON")
        print("=" * 60)

        for name in ["baseline", "robust_wide"]:
            if name not in results:
                continue
            s = results[name]["summary"]
            rob = results[name]["robustness"]
            print(f"\n  {name.upper()}:")
            if s["reliance_rating"] != "N/A":
                print(f"    Reliance:    {s['reliance_rating']:>10s} "
                      f"({s['reliance_score']:.2f}x)")
            else:
                print(f"    Reliance:           N/A")
            print(f"    Deployment:  {s['deployment_rating']:>10s} "
                  f"(return ratio: {s['deployment_score']:.2f})")
            print(f"    Stress:      {s['stress_rating']:>10s} "
                  f"(return ratio: {s['stress_score']:.2f})")

            for sc, scores in rob["per_scenario_scores"].items():
                ret = scores["return_ratio"] * 100
                rmse = scores["rmse_ratio"]
                marker = " <<<" if ret < 80 else ""
                print(f"      {sc:>10s}: return={ret:5.0f}%, "
                      f"RMSE={rmse:.2f}x{marker}")

        b = results.get("baseline", {}).get("summary", {})
        a = results.get("robust_wide", {}).get("summary", {})
        if b and a:
            print(f"\n  {'=' * 50}")
            print(f"  Deployment: {b['deployment_rating']} -> "
                  f"{a['deployment_rating']} "
                  f"({b['deployment_score']:.2f} -> "
                  f"{a['deployment_score']:.2f})")
            print(f"  Stress:     {b['stress_rating']} -> "
                  f"{a['stress_rating']} "
                  f"({b['stress_score']:.2f} -> "
                  f"{a['stress_score']:.2f})")
            print(f"  {'=' * 50}")

    # CI mode: check the "After" model
    if args.ci and "robust_wide" in results:
        after_dir = os.path.join(out_dir, "robust_wide")
        exit_code = _handle_ci(results["robust_wide"], after_dir, args)
        sys.exit(exit_code)


def main():
    parser = argparse.ArgumentParser(
        prog="deltatau-audit",
        description="Time Robustness Audit for RL agents",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── audit subcommand ──────────────────────────────────────────
    audit_parser = subparsers.add_parser(
        "audit", help="Run audit on a checkpoint")
    audit_parser.add_argument("--checkpoint", type=str, required=True,
                              help="Path to agent checkpoint (.pt file)")
    audit_parser.add_argument("--agent-type", type=str, default="internal_time",
                              choices=["internal_time", "internal_time_discount",
                                       "baseline", "skip_rnn"],
                              help="Type of agent architecture")
    audit_parser.add_argument("--env", type=str, default="chain",
                              help="Environment type (default: chain)")
    audit_parser.add_argument("--speed-hidden", action="store_true",
                              default=True)
    audit_parser.add_argument("--speeds", type=int, nargs="+",
                              default=[1, 2, 3, 5, 8])
    audit_parser.add_argument("--interventions", type=str, nargs="+",
                              default=["none", "clamp_1", "reverse", "random"])
    audit_parser.add_argument("--episodes", type=int, default=50)
    audit_parser.add_argument("--sensitivity-episodes", type=int, default=20)
    audit_parser.add_argument("--out", type=str, default="audit_report")
    audit_parser.add_argument("--device", type=str, default="cpu")
    audit_parser.add_argument("--chain-length", type=int, default=20)
    audit_parser.add_argument("--title", type=str,
                              default="Time Robustness Audit")
    _add_ci_args(audit_parser)

    # ── demo subcommand ───────────────────────────────────────────
    demo_parser = subparsers.add_parser(
        "demo", help="Run a bundled demo (Before/After comparison)")
    demo_parser.add_argument("demo_name", type=str, nargs="?",
                             default="cartpole",
                             help="Demo name (default: cartpole)")
    demo_parser.add_argument("--out", type=str, default="demo_report",
                             help="Output directory (default: demo_report/)")
    demo_parser.add_argument("--episodes", type=int, default=30,
                             help="Episodes per condition (default: 30)")
    _add_ci_args(demo_parser)

    args = parser.parse_args()

    if args.command == "audit":
        _run_audit(args)
    elif args.command == "demo":
        _run_demo(args)
    else:
        # No subcommand — check if legacy args present
        if "--checkpoint" in sys.argv:
            sys.argv.insert(1, "audit")
            args = parser.parse_args()
            _run_audit(args)
        else:
            parser.print_help()
            print("\nExamples:")
            print("  python -m deltatau_audit demo cartpole --out demo_report/")
            print("  python -m deltatau_audit audit --checkpoint model.pt "
                  "--agent-type internal_time --env chain")
            print("  python -m deltatau_audit demo cartpole --ci  "
                  "# CI gate mode")


if __name__ == "__main__":
    main()
