#!/usr/bin/env python3
"""HalfCheetah Before/After Case Study — Full Pipeline.

Runs the complete deltatau-audit workflow:
  1. Audit a standard PPO model (Before)
  2. Retrain with speed randomization (JitterWrapper)
  3. Audit the fixed model (After)
  4. Generate Before/After comparison (MD + HTML)
  5. Generate SVG badges
  6. Print summary table

Usage:
    python run_case_study.py --device cuda --workers auto --seed 42
    python run_case_study.py --device cuda --quick   # fast test (~3 min)

Requirements:
    pip install "deltatau-audit[sb3,mujoco]"
    python train.py --device cuda  # must run first
"""

import argparse
import os
import sys
import time
from pathlib import Path


def _require(pkg: str, extra: str = ""):
    try:
        __import__(pkg)
    except ImportError:
        tip = f'pip install "deltatau-audit[{extra}]"' if extra else f"pip install {pkg}"
        print(f"ERROR: {pkg} is required.\n  {tip}")
        sys.exit(1)


def _print_comparison(before: dict, after: dict):
    """Print a side-by-side Before / After table."""
    b_scores = before["robustness"].get("per_scenario_scores", {})
    a_scores = after["robustness"].get("per_scenario_scores", {})
    b_sum = before["summary"]
    a_sum = after["summary"]

    scenario_labels = {
        "jitter":    "Speed jitter",
        "delay":     "Obs. delay",
        "spike":     "Speed spike",
        "obs_noise": "Obs. noise",
        "speed_5x":  "5x speed [STRESS]",
    }

    print(f"\n{'=' * 62}")
    print("BEFORE vs AFTER - Timing Robustness Comparison")
    print(f"{'=' * 62}")
    print(f"  {'Scenario':16s}  {'Before':>8s}  {'After':>8s}  {'Change':>9s}")
    print(f"  {'-' * 16}  {'-' * 8}  {'-' * 8}  {'-' * 9}")

    for sc_key, label in scenario_labels.items():
        b = b_scores.get(sc_key, {})
        a = a_scores.get(sc_key, {})
        b_pct = b.get("return_ratio", 0) * 100
        a_pct = a.get("return_ratio", 0) * 100
        delta = a_pct - b_pct
        sign = "+" if delta >= 0 else ""
        print(f"  {label:16s}  {b_pct:7.1f}%  {a_pct:7.1f}%  "
              f"{sign}{delta:7.1f}%")

    print()
    b_d = b_sum["deployment_rating"]
    a_d = a_sum["deployment_rating"]
    b_ds = b_sum["deployment_score"]
    a_ds = a_sum["deployment_score"]
    b_s = b_sum["stress_rating"]
    a_s = a_sum["stress_rating"]
    b_ss = b_sum["stress_score"]
    a_ss = a_sum["stress_score"]

    print(f"  Deployment: {b_d} ({b_ds:.2f}) -> {a_d} ({a_ds:.2f})")
    print(f"  Stress:     {b_s} ({b_ss:.2f}) -> {a_s} ({a_ss:.2f})")
    print(f"{'=' * 62}")

    deploy_improved = a_ds > b_ds
    stress_improved = a_ss > b_ss
    if deploy_improved and stress_improved:
        print("\n  Both Deployment and Stress robustness improved.")
    elif deploy_improved:
        print("\n  Deployment robustness improved.")
        print("    Stress robustness unchanged (5x speed is an extreme condition).")
    else:
        print("\n  ? No clear improvement detected - try more training timesteps.")


def _save_summary_table(before: dict, after: dict, path: str):
    """Save a text summary table."""
    b_scores = before["robustness"].get("per_scenario_scores", {})
    a_scores = after["robustness"].get("per_scenario_scores", {})
    b_sum = before["summary"]
    a_sum = after["summary"]

    lines = []
    lines.append("HalfCheetah PPO - Before/After Timing Robustness")
    lines.append("=" * 62)
    lines.append(f"  {'Scenario':16s}  {'Before':>8s}  {'After':>8s}  {'Change':>9s}")
    lines.append(f"  {'-' * 16}  {'-' * 8}  {'-' * 8}  {'-' * 9}")

    scenario_labels = {
        "jitter": "Speed jitter", "delay": "Obs. delay",
        "spike": "Speed spike", "obs_noise": "Obs. noise",
        "speed_5x": "5x speed [STRESS]",
    }
    for sc_key, label in scenario_labels.items():
        b_pct = b_scores.get(sc_key, {}).get("return_ratio", 0) * 100
        a_pct = a_scores.get(sc_key, {}).get("return_ratio", 0) * 100
        delta = a_pct - b_pct
        sign = "+" if delta >= 0 else ""
        lines.append(f"  {label:16s}  {b_pct:7.1f}%  {a_pct:7.1f}%  "
                      f"{sign}{delta:7.1f}%")

    lines.append("")
    lines.append(f"  Deployment: {b_sum['deployment_rating']} "
                 f"({b_sum['deployment_score']:.2f}) -> "
                 f"{a_sum['deployment_rating']} "
                 f"({a_sum['deployment_score']:.2f})")
    lines.append(f"  Stress:     {b_sum['stress_rating']} "
                 f"({b_sum['stress_score']:.2f}) -> "
                 f"{a_sum['stress_rating']} "
                 f"({a_sum['stress_score']:.2f})")
    lines.append("=" * 62)

    text = "\n".join(lines) + "\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    return text


def main():
    parser = argparse.ArgumentParser(
        description="HalfCheetah Before/After case study")
    parser.add_argument("--model", type=str,
                        default="outputs/models/halfcheetah_ppo.zip",
                        help="Path to trained PPO model")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu or cuda (default: cpu)")
    parser.add_argument("--workers", type=str, default="1",
                        help="Parallel workers (int or 'auto')")
    parser.add_argument("--fix-timesteps", type=int, default=500_000,
                        help="Speed-randomized retraining steps (default: 500K)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default="outputs",
                        help="Output directory (default: outputs)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 10 episodes, no adaptive (for testing)")
    args = parser.parse_args()

    # Parse workers
    if args.workers == "auto":
        n_workers = os.cpu_count() or 1
    else:
        n_workers = int(args.workers)

    _require("stable_baselines3", "sb3,mujoco")
    _require("gymnasium")

    import gymnasium as gym

    # Verify env
    try:
        test_env = gym.make("HalfCheetah-v5")
        test_env.close()
    except Exception as e:
        print(f"ERROR: Cannot create HalfCheetah-v5: {e}")
        print('  pip install "deltatau-audit[sb3,mujoco]"')
        sys.exit(1)

    from stable_baselines3 import PPO
    from deltatau_audit.adapters.sb3 import SB3Adapter
    from deltatau_audit.auditor import run_full_audit
    from deltatau_audit.badge import generate_badges
    from deltatau_audit.callback import create_timing_audit_callback
    from deltatau_audit.diff import generate_comparison, generate_comparison_html
    from deltatau_audit.report import generate_report
    from deltatau_audit.wrappers.speed import JitterWrapper

    model_path = Path(args.model)
    out_dir = Path(args.output)

    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        print("  Run train.py first:")
        print(f"    python train.py --device {args.device}")
        sys.exit(1)

    # Audit parameters
    if args.quick:
        n_episodes = 10
        adaptive = False
        target_ci_width = 0.10
        max_episodes = 10
    else:
        n_episodes = 50
        adaptive = True
        target_ci_width = 0.05
        max_episodes = 200

    env_factory = lambda: gym.make("HalfCheetah-v5")

    # ====================================================================
    # STEP 1: Before Audit
    # ====================================================================
    print(f"\n{'=' * 60}")
    print("STEP 1/6: Before Audit (standard training)")
    print(f"{'=' * 60}\n")

    adapter_before = SB3Adapter.from_path(
        str(model_path), algo="ppo", device=args.device)

    t0 = time.time()
    result_before = run_full_audit(
        adapter_before, env_factory,
        speeds=[1, 2, 3, 5, 8],
        n_episodes=n_episodes,
        sensitivity_episodes=0,
        device=args.device,
        seed=args.seed,
        n_workers=n_workers,
        adaptive=adaptive,
        target_ci_width=target_ci_width,
        max_episodes=max_episodes,
    )
    print(f"\n  Before audit done in {time.time() - t0:.1f}s")

    before_dir = str(out_dir / "before")
    generate_report(result_before, before_dir,
                    title="HalfCheetah PPO - Before Fix")

    # ====================================================================
    # STEP 2: Speed-Randomized Retraining
    # ====================================================================
    print(f"\n{'=' * 60}")
    print(f"STEP 2/6: Speed-Randomized Retraining ({args.fix_timesteps:,} steps)")
    print(f"{'=' * 60}\n")

    def make_jitter_env():
        env = gym.make("HalfCheetah-v5")
        return JitterWrapper(env, base_speed=3, jitter=2)

    # Set up training callback for periodic audit during training
    audit_output_dir = str(out_dir / "training_audits")
    callback = create_timing_audit_callback(
        env_id="HalfCheetah-v5",
        audit_freq=100_000,
        n_episodes=10,
        output_dir=audit_output_dir,
        device=args.device,
        seed=args.seed,
        verbose=1,
    )

    t0 = time.time()
    model_fix = PPO(
        "MlpPolicy", make_jitter_env(),
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        verbose=1,
        seed=args.seed,
        device=args.device,
    )
    model_fix.learn(total_timesteps=args.fix_timesteps,
                    callback=callback,
                    progress_bar=True)
    train_time = time.time() - t0

    # Save fixed model
    fixed_model_dir = out_dir / "models"
    fixed_model_dir.mkdir(parents=True, exist_ok=True)
    fixed_model_path = fixed_model_dir / "halfcheetah_ppo_fixed.zip"
    model_fix.save(str(fixed_model_path))

    print(f"\n  Retraining done in {train_time:.0f}s ({train_time / 60:.1f} min)")
    print(f"  Fixed model saved: {fixed_model_path}")

    # ====================================================================
    # STEP 3: After Audit
    # ====================================================================
    print(f"\n{'=' * 60}")
    print("STEP 3/6: After Audit (speed-randomized model)")
    print(f"{'=' * 60}\n")

    adapter_after = SB3Adapter(model_fix, device=args.device)

    t0 = time.time()
    result_after = run_full_audit(
        adapter_after, env_factory,
        speeds=[1, 2, 3, 5, 8],
        n_episodes=n_episodes,
        sensitivity_episodes=0,
        device=args.device,
        seed=args.seed,
        n_workers=n_workers,
        adaptive=adaptive,
        target_ci_width=target_ci_width,
        max_episodes=max_episodes,
    )
    print(f"\n  After audit done in {time.time() - t0:.1f}s")

    after_dir = str(out_dir / "after")
    generate_report(result_after, after_dir,
                    title="HalfCheetah PPO - After Fix")

    # ====================================================================
    # STEP 4: Comparison
    # ====================================================================
    print(f"\n{'=' * 60}")
    print("STEP 4/6: Generating Comparison Reports")
    print(f"{'=' * 60}\n")

    before_json = str(Path(before_dir) / "summary.json")
    after_json = str(Path(after_dir) / "summary.json")

    comparison_md = str(out_dir / "comparison.md")
    comparison_html = str(out_dir / "comparison.html")

    generate_comparison(before_json, after_json, output_path=comparison_md)
    print(f"  comparison.md  -> {comparison_md}")

    generate_comparison_html(before_json, after_json, output_path=comparison_html)
    print(f"  comparison.html -> {comparison_html}")

    # ====================================================================
    # STEP 5: Badges
    # ====================================================================
    print(f"\n{'=' * 60}")
    print("STEP 5/6: Generating SVG Badges")
    print(f"{'=' * 60}\n")

    badges_dir = str(out_dir / "badges")

    before_badges = generate_badges(before_json, badges_dir, prefix="before")
    for name, path in before_badges.items():
        print(f"  before-{name}.svg -> {path}")

    after_badges = generate_badges(after_json, badges_dir, prefix="after")
    for name, path in after_badges.items():
        print(f"  after-{name}.svg  -> {path}")

    # ====================================================================
    # STEP 6: Summary
    # ====================================================================
    print(f"\n{'=' * 60}")
    print("STEP 6/6: Summary")
    print(f"{'=' * 60}")

    _print_comparison(result_before, result_after)

    # Save text summary
    summary_path = str(out_dir / "summary_table.txt")
    _save_summary_table(result_before, result_after, summary_path)
    print(f"\n  Summary saved: {summary_path}")

    # Print output file listing
    print(f"\n  Output files:")
    print(f"    {out_dir}/before/index.html          Before audit report")
    print(f"    {out_dir}/after/index.html           After audit report")
    print(f"    {out_dir}/comparison.html            Side-by-side comparison")
    print(f"    {out_dir}/comparison.md              Markdown comparison")
    print(f"    {out_dir}/badges/                    SVG badges (6 files)")
    print(f"    {out_dir}/training_audits/           Training-time audit logs")
    print(f"    {out_dir}/models/halfcheetah_ppo_fixed.zip")
    print(f"    {out_dir}/summary_table.txt          Text summary")

    # Print training audit history if available
    if hasattr(callback, 'audit_history') and callback.audit_history:
        print(f"\n  Training audit history:")
        print(f"    {'Step':>10s}  {'Deploy':>8s}  {'Stress':>8s}  {'Rating':>10s}")
        print(f"    {'-' * 10}  {'-' * 8}  {'-' * 8}  {'-' * 10}")
        for entry in callback.audit_history:
            print(f"    {entry['timestep']:>10,}  "
                  f"{entry['deployment_score']:>7.2f}  "
                  f"{entry['stress_score']:>7.2f}  "
                  f"{entry['deployment_rating']:>10s}")

    print(f"\n  Total pipeline time: "
          f"{train_time:.0f}s training + audit")
    print(f"\n  Done!")


if __name__ == "__main__":
    main()
