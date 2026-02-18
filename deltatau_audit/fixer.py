"""Fix timing-fragile SB3 models via speed-randomized retraining.

The "fix" is simple: retrain with JitterWrapper so the agent experiences
variable action-repeat during training. This produces agents that survive
timing perturbations in deployment.

Complete pipeline: audit original -> retrain -> re-audit -> compare.

Usage (Python API):
    from deltatau_audit.fixer import fix_sb3_model
    result = fix_sb3_model("my_model.zip", "ppo", "HalfCheetah-v5")

Usage (CLI):
    deltatau-audit fix-sb3 --model my_model.zip --algo ppo --env HalfCheetah-v5
"""

import os
import time
from typing import Dict, Optional

import gymnasium as gym


def _make_robust_env(env_id: str, base_speed: int = 3, jitter: int = 2):
    """Create env with speed randomization for robust training."""
    from .wrappers.speed import JitterWrapper
    env = gym.make(env_id)
    return JitterWrapper(env, base_speed=base_speed, jitter=jitter)


def _estimate_timesteps(env_id: str, algo: str) -> int:
    """Estimate reasonable training timesteps based on env complexity."""
    env_lower = env_id.lower()

    # Simple envs
    if any(k in env_lower for k in ("cartpole", "mountaincar", "acrobot",
                                     "lunarlander", "pendulum")):
        return 100_000

    # MuJoCo
    if any(k in env_lower for k in ("cheetah", "hopper", "walker",
                                     "ant", "humanoid", "swimmer",
                                     "reacher", "pusher", "inverted")):
        return 500_000

    # Default
    return 200_000


def fix_sb3_model(
    model_path: str,
    algo: str,
    env_id: str,
    output_dir: str = "fix_output",
    timesteps: Optional[int] = None,
    speed_min: int = 1,
    speed_max: int = 5,
    n_audit_episodes: int = 30,
    audit_speeds: list = None,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict:
    """Fix a timing-fragile SB3 model via speed-randomized retraining.

    Pipeline:
        1. Audit the original model (Before)
        2. Retrain from scratch with speed randomization
        3. Audit the fixed model (After)
        4. Generate Before/After comparison report

    Args:
        model_path: Path to SB3 model (.zip)
        algo: Algorithm name (ppo, sac, td3, a2c)
        env_id: Gymnasium environment ID
        output_dir: Output directory for reports and fixed model
        timesteps: Training timesteps (None = auto-estimate)
        speed_min: Minimum speed during training (default: 1)
        speed_max: Maximum speed during training (default: 5)
        n_audit_episodes: Episodes per audit condition
        audit_speeds: Speed multipliers for audit (default: [1, 2, 3, 5, 8])
        device: Device string (cpu, cuda)
        verbose: Print progress

    Returns:
        Dict with before/after results, fixed model path, comparison.
    """
    if audit_speeds is None:
        audit_speeds = [1, 2, 3, 5, 8]

    if timesteps is None:
        timesteps = _estimate_timesteps(env_id, algo)

    # Compute jitter params from speed range
    base_speed = (speed_min + speed_max) // 2
    jitter = base_speed - speed_min
    jitter = max(jitter, 1)
    # Ensure we cover the full range
    effective_min = max(1, base_speed - jitter)
    effective_max = base_speed + jitter

    os.makedirs(output_dir, exist_ok=True)

    from .adapters.sb3 import SB3Adapter
    from .auditor import run_full_audit
    from .report import generate_report
    from .diff import generate_comparison
    from .ci import write_ci_summary
    from . import __version__

    if verbose:
        print(f"deltatau-audit v{__version__} -- fix-sb3")
        print(f"  Model:     {model_path}")
        print(f"  Algo:      {algo.upper()}")
        print(f"  Env:       {env_id}")
        print(f"  Timesteps: {timesteps:,}")
        print(f"  Speed range: {effective_min}-{effective_max} "
              f"(base={base_speed}, jitter={jitter})")
        print(f"  Device:    {device}")
        print(f"  Output:    {output_dir}/")
        print()

    env_factory = lambda: gym.make(env_id)

    # ═══════════════════════════════════════════════════════════════
    # STEP 1: Audit original model (Before)
    # ═══════════════════════════════════════════════════════════════
    if verbose:
        print("=" * 60)
        print("STEP 1/3: Auditing original model")
        print("=" * 60)
        print()

    adapter_before = SB3Adapter.from_path(model_path, algo=algo, device=device)

    t0 = time.time()
    before_result = run_full_audit(
        adapter_before, env_factory,
        speeds=audit_speeds,
        n_episodes=n_audit_episodes,
        sensitivity_episodes=0,
        device=device,
        verbose=verbose,
    )
    audit_time_before = time.time() - t0

    before_dir = os.path.join(output_dir, "before")
    generate_report(before_result, before_dir,
                    title=f"{algo.upper()} on {env_id} — Before Fix")
    write_ci_summary(before_result["summary"],
                     before_result["robustness"], before_dir)

    before_dep = before_result["summary"]["deployment_score"]
    before_rating = before_result["summary"]["deployment_rating"]

    if verbose:
        print(f"\n  Audit time: {audit_time_before:.0f}s")
        print(f"  Result: {before_rating} (deployment={before_dep:.2f})")

    # Check if fix is needed
    if before_dep >= 0.95:
        if verbose:
            print()
            print("  Model already PASSES deployment robustness (>=0.95)!")
            print("  No fix needed.")
            print(f"\n  Report: {before_dir}/index.html")

        return {
            "before": before_result,
            "after": None,
            "fixed_model_path": None,
            "skipped": True,
            "reason": "Already passes (deployment >= 0.95)",
        }

    # ═══════════════════════════════════════════════════════════════
    # STEP 2: Retrain with speed randomization
    # ═══════════════════════════════════════════════════════════════
    if verbose:
        print()
        print("=" * 60)
        print(f"STEP 2/3: Retraining with speed randomization")
        print(f"          {timesteps:,} timesteps, speed {effective_min}-{effective_max}")
        print("=" * 60)
        print()

    try:
        import stable_baselines3
    except ImportError:
        raise ImportError(
            "stable-baselines3 required. "
            'Install: pip install "deltatau-audit[sb3]"'
        )

    algo_map = {
        "ppo": stable_baselines3.PPO,
        "sac": stable_baselines3.SAC,
        "td3": stable_baselines3.TD3,
        "a2c": stable_baselines3.A2C,
    }
    algo_cls = algo_map.get(algo.lower())
    if algo_cls is None:
        raise ValueError(
            f"Unknown algo '{algo}'. Supported: {list(algo_map.keys())}")

    robust_env = _make_robust_env(env_id, base_speed, jitter)

    t0 = time.time()
    model = algo_cls("MlpPolicy", robust_env,
                     verbose=1 if verbose else 0,
                     device=device)
    model.learn(total_timesteps=timesteps)
    train_time = time.time() - t0
    robust_env.close()

    # Save fixed model
    fixed_model_stem = os.path.join(output_dir, f"{algo}_fixed")
    model.save(fixed_model_stem)
    fixed_model_path = fixed_model_stem + ".zip"

    if verbose:
        print(f"\n  Training completed in {train_time:.0f}s")
        print(f"  Saved: {fixed_model_path}")

    # ═══════════════════════════════════════════════════════════════
    # STEP 3: Audit fixed model (After)
    # ═══════════════════════════════════════════════════════════════
    if verbose:
        print()
        print("=" * 60)
        print("STEP 3/3: Auditing fixed model")
        print("=" * 60)
        print()

    fixed_adapter = SB3Adapter(model, device=device)

    t0 = time.time()
    after_result = run_full_audit(
        fixed_adapter, env_factory,
        speeds=audit_speeds,
        n_episodes=n_audit_episodes,
        sensitivity_episodes=0,
        device=device,
        verbose=verbose,
    )
    audit_time_after = time.time() - t0

    after_dir = os.path.join(output_dir, "after")
    generate_report(after_result, after_dir,
                    title=f"{algo.upper()} on {env_id} — After Fix")
    write_ci_summary(after_result["summary"],
                     after_result["robustness"], after_dir)

    after_dep = after_result["summary"]["deployment_score"]
    after_rating = after_result["summary"]["deployment_rating"]

    if verbose:
        print(f"\n  Audit time: {audit_time_after:.0f}s")
        print(f"  Result: {after_rating} (deployment={after_dep:.2f})")

    # ═══════════════════════════════════════════════════════════════
    # COMPARISON
    # ═══════════════════════════════════════════════════════════════
    comp_path = os.path.join(output_dir, "comparison.md")
    generate_comparison(
        os.path.join(before_dir, "summary.json"),
        os.path.join(after_dir, "summary.json"),
        output_path=comp_path,
    )

    if verbose:
        _print_comparison(before_result, after_result,
                          fixed_model_path, output_dir, train_time)

    return {
        "before": before_result,
        "after": after_result,
        "fixed_model_path": fixed_model_path,
        "skipped": False,
        "training_time": train_time,
        "training_timesteps": timesteps,
        "speed_range": (effective_min, effective_max),
    }


def _print_comparison(before: Dict, after: Dict,
                      fixed_model_path: str, output_dir: str,
                      train_time: float):
    """Print Before vs After comparison summary."""
    print()
    print("=" * 60)
    print("  BEFORE vs AFTER")
    print("=" * 60)
    print()

    b_rob = before["robustness"]["per_scenario_scores"]
    a_rob = after["robustness"]["per_scenario_scores"]

    print(f"  {'Scenario':12s}  {'Before':>10s}  {'After':>10s}  {'Change':>10s}")
    print(f"  {'-' * 12}  {'-' * 10}  {'-' * 10}  {'-' * 10}")

    for sc in b_rob:
        b_pct = b_rob[sc]["return_ratio"] * 100
        a_pct = a_rob.get(sc, {}).get("return_ratio", 0) * 100
        delta = a_pct - b_pct
        sign = "+" if delta >= 0 else ""
        print(f"  {sc:12s}  {b_pct:9.1f}%  {a_pct:9.1f}%  "
              f"{sign}{delta:8.1f}pp")

    b_sum = before["summary"]
    a_sum = after["summary"]

    print()
    print(f"  Deployment: {b_sum['deployment_rating']} "
          f"({b_sum['deployment_score']:.2f}) -> "
          f"{a_sum['deployment_rating']} "
          f"({a_sum['deployment_score']:.2f})")
    print(f"  Stress:     {b_sum['stress_rating']} "
          f"({b_sum['stress_score']:.2f}) -> "
          f"{a_sum['stress_rating']} "
          f"({a_sum['stress_score']:.2f})")
    print(f"  Quadrant:   {b_sum['quadrant']} -> {a_sum['quadrant']}")

    print()
    print(f"  Training time: {train_time:.0f}s")
    print()
    print(f"  Fixed model:   {fixed_model_path}")
    print(f"  Before report: {output_dir}/before/index.html")
    print(f"  After report:  {output_dir}/after/index.html")
    print(f"  Comparison:    {output_dir}/comparison.md")
