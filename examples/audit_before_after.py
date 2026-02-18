#!/usr/bin/env python3
"""Before/After Timing Robustness Audit — HalfCheetah PPO.

Demonstrates the complete deltatau-audit workflow:
1. Audit a standard PPO agent → FAIL
2. Audit a speed-randomized PPO agent → improvement
3. Generate comparison showing the fix works

Requirements:
    pip install "deltatau-audit[sb3,mujoco]"

Usage:
    python examples/audit_before_after.py

Pre-trained models (skip training):
    Download from https://github.com/maruyamakoju/deltatau-audit/releases/latest
    Place in runs/halfcheetah_ppo_500k.zip and runs/halfcheetah_ppo_robust_500k.zip
"""

import json
import gymnasium as gym
from pathlib import Path

from stable_baselines3 import PPO

from deltatau_audit.adapters.sb3 import SB3Adapter
from deltatau_audit.auditor import run_full_audit
from deltatau_audit.report import generate_report
from deltatau_audit.ci import write_ci_summary
from deltatau_audit.diff import generate_comparison


def audit_model(model_path, output_dir, title):
    """Load model, run full audit, generate report."""
    print(f"\n{'='*60}")
    print(f"AUDIT: {title}")
    print(f"{'='*60}\n")

    model = PPO.load(str(model_path), device="cpu")
    adapter = SB3Adapter(model)
    env_factory = lambda: gym.make("HalfCheetah-v5")

    result = run_full_audit(
        adapter,
        env_factory,
        speeds=[1, 2, 3, 5, 8],
        n_episodes=30,
    )

    out = str(output_dir)
    generate_report(result, out, title=title)
    write_ci_summary(result["summary"], result["robustness"], out)

    return result


def print_comparison(before, after):
    """Print side-by-side comparison."""
    print(f"\n{'='*60}")
    print("BEFORE vs AFTER COMPARISON")
    print(f"{'='*60}\n")

    b_rob = before["robustness"]["per_scenario_scores"]
    a_rob = after["robustness"]["per_scenario_scores"]

    print(f"  {'Scenario':12s}  {'Before':>10s}  {'After':>10s}  {'Change':>10s}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")

    for sc in b_rob:
        b_pct = b_rob[sc]["return_ratio"] * 100
        a_pct = a_rob[sc]["return_ratio"] * 100
        delta = a_pct - b_pct
        sign = "+" if delta >= 0 else ""
        print(f"  {sc:12s}  {b_pct:9.1f}%  {a_pct:9.1f}%  {sign}{delta:8.1f}%")

    b_dep = before["summary"]["deployment_score"]
    a_dep = after["summary"]["deployment_score"]
    b_str = before["summary"]["stress_score"]
    a_str = after["summary"]["stress_score"]

    print(f"\n  Deployment: {before['summary']['deployment_rating']} "
          f"({b_dep:.2f}) → {after['summary']['deployment_rating']} "
          f"({a_dep:.2f})")
    print(f"  Stress:     {before['summary']['stress_rating']} "
          f"({b_str:.2f}) → {after['summary']['stress_rating']} "
          f"({a_str:.2f})")


RELEASE_URL = (
    "https://github.com/maruyamakoju/deltatau-audit/releases/download/assets/"
)
ASSETS = {
    "runs/halfcheetah_ppo_500k.zip": "halfcheetah_ppo_500k.zip",
    "runs/halfcheetah_ppo_robust_500k.zip": "halfcheetah_ppo_robust_500k.zip",
}


def _try_download(local_path: Path, asset_name: str) -> bool:
    """Try to download a model from GitHub Releases. Returns True on success."""
    url = RELEASE_URL + asset_name
    try:
        import urllib.request
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading {asset_name} from GitHub Releases...")
        urllib.request.urlretrieve(url, str(local_path))
        print(f"  Saved: {local_path}")
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def main():
    standard_model = Path("runs/halfcheetah_ppo_500k.zip")
    robust_model = Path("runs/halfcheetah_ppo_robust_500k.zip")

    for path, asset in ASSETS.items():
        p = Path(path)
        if not p.exists():
            print(f"Model not found locally: {p}")
            ok = _try_download(p, asset)
            if not ok:
                if path.endswith("500k.zip"):
                    print(f"  Fallback: python examples/audit_halfcheetah.py")
                else:
                    print(f"  Fallback: python examples/train_robust_halfcheetah.py")
                return 1

    output = Path("halfcheetah_before_after")

    # Audit both models
    before = audit_model(
        standard_model,
        output / "before",
        "HalfCheetah PPO — Standard Training (Before)",
    )
    after = audit_model(
        robust_model,
        output / "after",
        "HalfCheetah PPO — Speed-Randomized Training (After)",
    )

    # Generate diff (from saved summary.json files)
    generate_comparison(
        output / "before" / "summary.json",
        output / "after" / "summary.json",
        output_path=str(output / "comparison.md"),
    )

    # Print results
    print_comparison(before, after)

    print(f"\n  Reports: {output}/before/index.html")
    print(f"           {output}/after/index.html")
    print(f"  Comparison: {output}/comparison.md")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
