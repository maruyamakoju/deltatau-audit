"""Run audit on CartPole agents + Before/After comparison.

Usage:
    python demo_external_env/run_audit.py
"""

import json
import os
import sys

import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deltatau_audit.adapters.simple_gru import SimpleGRUAdapter
from deltatau_audit.auditor import run_full_audit
from deltatau_audit.report import generate_report


def cartpole_factory():
    return gym.make("CartPole-v1")


def main():
    models = [
        ("baseline", "demo_external_env/checkpoints/baseline/final.pt",
         "CartPole Baseline GRU (Before Fix)"),
        ("robust_wide", "demo_external_env/checkpoints/robust_wide/final.pt",
         "CartPole Speed-Randomized GRU (After Fix)"),
    ]

    results = {}
    for name, ckpt, title in models:
        if not os.path.exists(ckpt):
            print(f"Skipping {name}: {ckpt} not found")
            continue

        print(f"\n{'=' * 60}")
        print(f"AUDITING: {title}")
        print(f"{'=' * 60}\n")

        adapter = SimpleGRUAdapter.from_checkpoint(
            ckpt, obs_dim=4, act_dim=2, hidden_dim=64)

        result = run_full_audit(
            adapter, cartpole_factory,
            speeds=[1, 2, 3, 5, 8],
            n_episodes=30,
            sensitivity_episodes=0,
        )

        out_dir = f"demo_external_env/audit_report_{name}"
        generate_report(result, out_dir, title=title)
        results[name] = result

    # ── Before/After comparison ───────────────────────────────────
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


if __name__ == "__main__":
    main()
