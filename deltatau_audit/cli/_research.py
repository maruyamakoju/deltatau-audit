"""Research, diff, certify, and badge subcommand handlers."""
import os
import sys


def _run_diff(args):
    """Compare two summary.json files and generate comparison.md + .html."""
    import pathlib

    from deltatau_audit.diff import generate_comparison, generate_comparison_html

    out = pathlib.Path(args.out) if args.out else None

    md = generate_comparison(args.before, args.after, output_path=out)
    print(md)

    if out:
        html_path = out.with_suffix(".html")
        generate_comparison_html(args.before, args.after, output_path=html_path)
        print(f"Markdown: {out}")
        print(f"HTML:     {html_path}")
    else:
        # No output file — just print the markdown (already done above)
        pass


def _run_research_full(args):
    """Orchestrate the full research suite audit with staged resume support."""
    from deltatau_audit.research_suite import ResearchSuiteConfig, run_research_suite

    config = ResearchSuiteConfig(
        env=args.env,
        out=args.out,
        episodes=int(args.episodes),
        seed=args.seed,
        speeds=list(args.speeds),
        deliberative_max_thinking_steps=int(args.deliberative_max_thinking_steps),
        bridge_delay_ms=float(args.bridge_delay_ms),
        bridge_delay_std_ms=float(args.bridge_delay_std_ms),
        bridge_dt_ms=float(args.bridge_dt_ms),
        bridge_actuator_alpha=float(args.bridge_actuator_alpha),
        resume=not bool(args.no_resume),
        fail_fast=bool(args.fail_fast),
    )

    print("=" * 64)
    print("RUNNING RESEARCH-FULL AUDIT SUITE")
    print("=" * 64)
    print(f"  Env:       {config.env}")
    print(f"  Episodes:  {config.episodes}")
    print(f"  Speeds:    {config.speeds}")
    print(f"  Seed:      {config.seed}")
    print(f"  Resume:    {config.resume}")
    print(f"  Fail-fast: {config.fail_fast}")
    print(f"  Output:    {config.out}")
    print()

    result = run_research_suite(config)
    outcomes = result.get("outcomes", [])
    for idx, out in enumerate(outcomes, start=1):
        dep = "n/a" if out.deployment_score is None else f"{out.deployment_score:.3f}"
        stress = "n/a" if out.stress_score is None else f"{out.stress_score:.3f}"
        msg = out.reason or "-"
        print(
            f"[Stage {idx}] {out.name:<12} status={out.status:<7} "
            f"deployment={dep:<6} stress={stress:<6} reason={msg}"
        )

    print("\nResearch suite complete")
    print(f"  Summary JSON: {result.get('summary_path')}")
    print(f"  Summary MD:   {result.get('summary_md_path')}")
    if result.get("dashboard_path"):
        print(f"  Dashboard:    {result['dashboard_path']}")


def _run_certify(args):
    """Generate a formal safety certificate from audit results."""
    import json

    from deltatau_audit.report.certification import generate_safety_certificate

    print(f"Generating Safety Certificate from {args.summary_json}...")

    if not os.path.exists(args.summary_json):
        print(f"ERROR: Summary file not found: {args.summary_json}")
        return

    with open(args.summary_json, 'r') as f:
        result = json.load(f)

    status, fingerprint = generate_safety_certificate(result, args.out)

    print(f"  Status:      {status}")
    print(f"  Registry ID: DT-{fingerprint}")
    print(f"  Certificate: {args.out}")


def _run_badge(args):
    """Generate SVG badges from a summary.json."""
    from deltatau_audit.badge import generate_badges

    paths = generate_badges(args.summary_json, args.out, args.prefix)
    for name, path in paths.items():
        print(f"  {name}: {path}")


def _run_audit_deliberative(args):
    """CLI handler for audit-deliberative command.

    Audits a DeliberativeInternalTimeAgent: measures how ponder depth
    correlates with timing stress across speed conditions.
    """
    import os
    import json
    import gymnasium as gym

    from internal_time_rl.models.deliberative import DeliberativeInternalTimeAgent
    from deltatau_audit.adapters.deliberative_adapter import DeliberativeAgentAdapter
    from deltatau_audit.auditor import run_deliberative_audit

    os.makedirs(args.out, exist_ok=True)
    print(f"Deliberative Audit: {args.checkpoint}")
    print(f"  Env: {args.env}")
    print(f"  Speeds: {args.speeds}")

    # Load model
    import torch
    obs_dim = getattr(args, "obs_dim", 4)
    act_dim = getattr(args, "act_dim", 2)
    agent = DeliberativeInternalTimeAgent(obs_dim=obs_dim, act_dim=act_dim)
    try:
        state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        agent.load_state_dict(state_dict)
        print(f"  Loaded checkpoint: {args.checkpoint}")
    except Exception as e:
        print(f"  WARNING: Could not load checkpoint ({e}). Using untrained model.")

    agent.eval()
    adapter = DeliberativeAgentAdapter(agent)

    def env_factory():
        return gym.make(args.env)

    result = run_deliberative_audit(
        adapter=adapter,
        env_factory=env_factory,
        speeds=args.speeds,
        n_episodes=getattr(args, "episodes", 20),
        verbose=True,
        seed=getattr(args, "seed", None),
    )

    out_path = os.path.join(args.out, "deliberative_summary.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")
    print(f"  Deliberative score: {result['deliberative_score']:.3f} ({result['rating']})")


def _run_audit_horizon(args):
    """CLI handler for audit-horizon command.

    Tests an agent on cascading multi-step timing scenarios.
    """
    import os
    import json
    import gymnasium as gym

    from deltatau_audit.auditors.horizon_auditor import TemporalHorizonAuditor

    os.makedirs(args.out, exist_ok=True)
    print(f"Horizon Audit: {args.checkpoint}")
    print(f"  Env: {args.env}")

    # Try to load as SB3 model, fall back to generic dummy adapter
    try:
        from stable_baselines3 import PPO
        from deltatau_audit.adapters.sb3 import SB3Adapter
        model = PPO.load(args.checkpoint)
        adapter = SB3Adapter(model)
        print("  Loaded SB3 model")
    except Exception:
        try:
            import torch
            from deltatau_audit.adapters.internal_time import InternalTimeAdapter
            state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            # Try to use as internal time adapter
            adapter = InternalTimeAdapter(state)
            print("  Loaded InternalTime model")
        except Exception as e2:
            print(f"  WARNING: Could not load model ({e2}). Using dummy adapter.")
            from tests.conftest import _DummyAdapter
            adapter = _DummyAdapter()

    def env_factory():
        return gym.make(args.env)

    auditor = TemporalHorizonAuditor(verbose=True)
    result = auditor.run_cascade_audit(
        adapter=adapter,
        env_factory=env_factory,
        horizon=getattr(args, "horizon", 50),
        n_episodes=getattr(args, "episodes", 20),
        seed=getattr(args, "seed", None),
    )

    out_path = os.path.join(args.out, "horizon_summary.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")
    print(f"  Horizon robustness: {result['horizon_robustness_score']:.3f}")


