"""Stress subcommand handlers: stress analyze, ablate, train-sb3."""

import sys


def _run_stress_analyze(args):
    """Analyze stress failures from an audit summary and write artifacts."""
    from deltatau_audit.stress_lab import write_stress_analysis_artifacts

    artifacts = write_stress_analysis_artifacts(
        args.summary,
        out_dir=args.out,
        stress_threshold=args.stress_threshold,
        include_intervention3=args.include_intervention3,
    )
    print("Stress analysis complete")
    print(f"  JSON: {artifacts.get('analysis_json')}")
    print(f"  MD:   {artifacts.get('analysis_md')}")


def _run_stress_ablate(args):
    """Generate stress ablation manifest/plan artifacts."""
    from deltatau_audit.stress_lab import (
        analyze_stress_summary,
        build_ablation_manifest,
        write_ablation_plan_artifacts,
    )

    analysis = analyze_stress_summary(
        args.summary,
        stress_threshold=args.stress_threshold,
        include_intervention3=args.include_intervention3,
    )
    manifest = build_ablation_manifest(
        env=args.env,
        algo=args.algo,
        model_template=args.model_template,
        seeds=list(args.seeds),
        episodes=args.episodes,
        speeds=list(args.speeds),
        include_intervention3=args.include_intervention3,
        protocol=args.protocol,
        ci_gate_mode=args.ci_gate_mode,
        output_dir=args.output_dir,
    )
    artifacts = write_ablation_plan_artifacts(
        analysis=analysis,
        manifest=manifest,
        out_dir=args.out,
    )
    print("Stress ablation plan generated")
    print(f"  Manifest: {artifacts.get('ablation_manifest')}")
    print(f"  Plan MD:  {artifacts.get('ablation_plan_md')}")


def _run_stress_train_sb3(args):
    """Train SB3 models for stress ablation variants."""
    from deltatau_audit.stress_lab import train_sb3_ablation_models, write_training_summary

    variants = list(args.variants)
    if args.include_intervention3 and "intervention3_memory" not in variants:
        variants.append("intervention3_memory")

    summary = train_sb3_ablation_models(
        env=args.env,
        algo=args.algo,
        out_root=args.out_root,
        seeds=list(args.seeds),
        variants=variants,
        timesteps=args.timesteps,
        device=args.device,
        base_speed=args.base_speed,
        jitter=args.jitter,
        phase_period=args.phase_period,
        frame_stack=args.frame_stack,
        force=args.force,
        fail_fast=args.fail_fast,
        verbose=1 if args.verbose else 0,
    )
    artifacts = write_training_summary(summary, out_dir=args.out)

    counts = summary.get("counts", {})
    print("Stress SB3 training complete")
    print(f"  Status:  {summary.get('status')}")
    print(f"  Trained: {counts.get('trained', 0)}")
    print(f"  Skipped: {counts.get('skipped', 0)}")
    print(f"  Failed:  {counts.get('failed', 0)}")
    print(f"  Out root: {summary.get('out_root')}")
    print(f"  JSON: {artifacts.get('training_json')}")
    print(f"  MD:   {artifacts.get('training_md')}")
    if int(counts.get("failed", 0)) > 0:
        sys.exit(1)
