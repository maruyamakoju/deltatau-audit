#!/usr/bin/env python3
"""End-to-end validation: fix-sb3 on HalfCheetah-v5.

Proves the core product promise:
    "Audit → Fix → Re-audit: timing robustness improves."

This script:
  1. Loads (or trains) a standard PPO on HalfCheetah-v5
  2. Runs a full timing audit  → expected: DEGRADED or FAIL
  3. Fixes with speed-randomized retraining (fix-sb3 pipeline)
  4. Re-audits the fixed model    → expected: MILD or PASS
  5. Prints a Before / After comparison table

Requirements:
    pip install "deltatau-audit[sb3,mujoco]"
    # MuJoCo system libraries must be installed

Usage:
    python examples/validate_mujoco.py
    python examples/validate_mujoco.py --timesteps 200000 --episodes 20
    python examples/validate_mujoco.py --skip-train  # load existing model

Expected outcome (representative numbers):
    Scenario        Before   After    Change
    jitter          ~55%     ~85%    +30%
    delay           ~60%     ~88%    +28%
    spike           ~50%     ~82%    +32%
    obs_noise       ~70%     ~90%    +20%
    speed_5x        ~25%     ~50%    +25%

    Deployment: FAIL → MILD   (deployment_score 0.50 → 0.82)
    Stress:     FAIL → FAIL   (stress is harder; even fixed agents struggle at 5x)

Note: exact numbers vary by seed and hardware. The key signal is the
      direction (Before < After) and the rating improvement.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import gymnasium as gym


def _require(pkg: str, extra: str = None):
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
    print("BEFORE vs AFTER — Timing Robustness Comparison")
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
        sep = "  [STRESS]" if sc_key == "speed_5x" else ""
        print(f"  {label:16s}  {b_pct:7.1f}%  {a_pct:7.1f}%  "
              f"{sign}{delta:7.1f}%{sep}")

    print()
    b_d = b_sum["deployment_rating"]
    a_d = a_sum["deployment_rating"]
    b_ds = b_sum["deployment_score"]
    a_ds = a_sum["deployment_score"]
    b_s = b_sum["stress_rating"]
    a_s = a_sum["stress_rating"]
    b_ss = b_sum["stress_score"]
    a_ss = a_sum["stress_score"]

    print(f"  Deployment: {b_d} ({b_ds:.2f}) → {a_d} ({a_ds:.2f})")
    print(f"  Stress:     {b_s} ({b_ss:.2f}) → {a_s} ({a_ss:.2f})")
    print(f"{'=' * 62}")

    deploy_improved = a_ds > b_ds
    stress_improved = a_ss > b_ss
    if deploy_improved and stress_improved:
        print("\n  ✓ Both Deployment and Stress robustness improved.")
    elif deploy_improved:
        print("\n  ✓ Deployment robustness improved.")
        print("    Stress robustness unchanged (5x speed is an extreme condition).")
    else:
        print("\n  ? No clear improvement detected — try more training timesteps.")


def main():
    parser = argparse.ArgumentParser(
        description="Validate fix-sb3 on HalfCheetah-v5")
    parser.add_argument("--model", type=str,
                        default="runs/halfcheetah_ppo.zip",
                        help="Path to existing SB3 PPO model "
                             "(trained if not found)")
    parser.add_argument("--out", type=str, default="validate_output",
                        help="Output directory")
    parser.add_argument("--timesteps", type=int, default=300_000,
                        help="Training timesteps for initial model "
                             "(default: 300K)")
    parser.add_argument("--fix-timesteps", type=int, default=None,
                        help="Speed-randomized fine-tuning steps "
                             "(default: auto = 30%% of initial)")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Episodes per audit condition (default: 20)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for audit (default: 1)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip initial training (model must exist)")
    args = parser.parse_args()

    _require("stable_baselines3", "sb3,mujoco")
    _require("gymnasium")

    # Verify MuJoCo env is available
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
    from deltatau_audit.report import generate_report
    from deltatau_audit.fixer import fix_sb3_model

    model_path = Path(args.model)
    out_dir = Path(args.out)

    # ── Step 1: Train initial model ───────────────────────────────
    if not model_path.exists() and not args.skip_train:
        print(f"\n{'=' * 60}")
        print(f"STEP 1: Training PPO on HalfCheetah-v5 "
              f"({args.timesteps:,} steps)")
        print(f"{'=' * 60}")
        model_path.parent.mkdir(parents=True, exist_ok=True)

        env = gym.make("HalfCheetah-v5")
        model = PPO("MlpPolicy", env, verbose=1, seed=args.seed,
                    device=args.device)
        model.learn(total_timesteps=args.timesteps, progress_bar=True)
        model.save(str(model_path))
        env.close()
        print(f"  Model saved to: {model_path}")
    elif model_path.exists():
        print(f"\nLoading model from: {model_path}")
    else:
        print(f"ERROR: Model not found: {model_path}")
        print("  Remove --skip-train to train a fresh model.")
        sys.exit(1)

    # ── Step 2: Before audit ──────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 2: Before Audit (standard training)")
    print(f"{'=' * 60}\n")

    adapter_before = SB3Adapter.from_path(str(model_path), algo="ppo",
                                          device=args.device)
    env_factory = lambda: gym.make("HalfCheetah-v5")

    t0 = time.time()
    result_before = run_full_audit(
        adapter_before, env_factory,
        speeds=[1, 2, 3, 5, 8],
        n_episodes=args.episodes,
        sensitivity_episodes=0,
        device=args.device,
        seed=args.seed,
        n_workers=args.workers,
    )
    print(f"\n  Before audit done in {time.time() - t0:.1f}s")

    before_dir = str(out_dir / "before")
    generate_report(result_before, before_dir, title="HalfCheetah PPO — Before Fix")

    # ── Step 3: Speed-randomized retraining (fix-sb3) ────────────
    print(f"\n{'=' * 60}")
    print("STEP 3: Speed-Randomized Retraining (fix-sb3 pipeline)")
    print(f"{'=' * 60}\n")

    fix_timesteps = args.fix_timesteps or max(50_000, args.timesteps // 5)
    print(f"  Fine-tuning for {fix_timesteps:,} steps with speed ∈ [1, 5]...")

    result_fix = fix_sb3_model(
        model_path=str(model_path),
        algo="ppo",
        env_id="HalfCheetah-v5",
        output_dir=str(out_dir / "fix"),
        timesteps=fix_timesteps,
        speed_min=1,
        speed_max=5,
        n_audit_episodes=args.episodes,
        device=args.device,
    )

    # ── Step 4: Re-audit the fixed model ─────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 4: After Audit (speed-randomized model)")
    print(f"{'=' * 60}\n")

    fixed_model_path = str(out_dir / "fix" / "robust_model.zip")
    if not os.path.exists(fixed_model_path):
        # fix_sb3_model writes to after/robust_model.zip
        fixed_model_path = str(out_dir / "fix" / "after" / "robust_model.zip")

    if not os.path.exists(fixed_model_path):
        # Fallback: use result from fix pipeline
        if result_fix.get("after"):
            print("  Using audit result embedded in fix pipeline output.")
            result_after = result_fix["after"]
            after_dir = str(out_dir / "fix" / "after")
        else:
            print("ERROR: Could not find fixed model. "
                  "Check fix pipeline output in:", str(out_dir / "fix"))
            sys.exit(1)
    else:
        adapter_after = SB3Adapter.from_path(fixed_model_path, algo="ppo",
                                             device=args.device)
        t0 = time.time()
        result_after = run_full_audit(
            adapter_after, env_factory,
            speeds=[1, 2, 3, 5, 8],
            n_episodes=args.episodes,
            sensitivity_episodes=0,
            device=args.device,
            seed=args.seed,
            n_workers=args.workers,
        )
        print(f"\n  After audit done in {time.time() - t0:.1f}s")
        after_dir = str(out_dir / "after")
        generate_report(result_after, after_dir,
                        title="HalfCheetah PPO — After Fix")

    # ── Step 5: Print comparison ──────────────────────────────────
    _print_comparison(result_before, result_after)

    print(f"\n  Reports saved to:")
    print(f"    {out_dir}/before/index.html")
    print(f"    {out_dir}/after/index.html")


if __name__ == "__main__":
    main()
