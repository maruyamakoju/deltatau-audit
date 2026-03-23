"""Generate LaTeX tables for the paper from existing experimental results.

Tables generated:
  Table 1 (main): Δτ slope, Spearman ρ, monotonicity, gen gap — all experiments
  Table 2 (ablation): PPOTimeV2 variant comparison
  Table 3 (robustness): Deployment/stress scores across environments
  Table 4 (comparison): Our method vs baselines

Usage:
    python scripts/generate_latex_tables.py
    python scripts/generate_latex_tables.py --out runs/paper_figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT_DIR_DEFAULT = ROOT / "runs" / "paper_figures"


def load_statistical_metrics() -> dict:
    """Load existing statistical metrics from paper figures output."""
    path = ROOT / "runs" / "paper_figures" / "statistical_metrics.json"
    if not path.exists():
        print(f"Warning: {path} not found. Run generate_paper.py first.")
        return {}
    with open(path) as f:
        return json.load(f)


# ── Table 1: Main results ─────────────────────────────────────────────────────

def table_main_results(metrics: dict, out_dir: Path) -> None:
    """Table 1: Δτ tracking across experiments (main result)."""

    model_display = {
        "baseline": r"Baseline GRU",
        "internal_time": r"\textbf{Internal Time (ours)}",
        "internal_time_discount": r"Internal Time $+ \gamma^{\Delta\tau}$",
        "skip_rnn": r"Skip-RNN (ACT)",
        "external_dt": r"ODE-RNN (oracle $\Delta t$)",
    }

    exp_display = {
        "speed_gen_hidden_5seed": "Hidden Speed\n(5 seeds)",
        "speed_gen_chain": "Visible Speed",
        "hard_chain": "Hard Chain",
    }

    # Use speed_gen_hidden_5seed as the main experiment
    main_exp = "speed_gen_hidden_5seed"
    if main_exp not in metrics:
        print(f"Warning: {main_exp} not in metrics")
        return

    exp_data = metrics[main_exp]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Main results: $\Delta\tau$ tracking across environment speeds. "
        r"Spearman $\rho$ measures rank correlation between $\Delta\tau$ and speed. "
        r"Slope is from linear regression (higher = better). "
        r"$*$ = unseen during training (generalization).}",
        r"\label{tab:main_results}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & $\Delta\tau$ Slope & 95\% CI & Spearman $\rho$ & Monotonicity \\",
        r"\midrule",
    ]

    for model_key, display_name in model_display.items():
        rho_data = exp_data.get("spearman_rho", {}).get(model_key, {})
        mono_data = exp_data.get("monotonicity_rate", {}).get(model_key, None)
        slope_data = exp_data.get("dt_slope_bootstrap_ci", {}).get(model_key, {})

        rho = rho_data.get("rho", None)
        p_val = rho_data.get("p", None)
        mono = mono_data

        slope_mean = slope_data.get("mean", None)
        slope_lo = slope_data.get("lo", None)
        slope_hi = slope_data.get("hi", None)

        # Format rho
        if rho is None or (isinstance(rho, float) and
                            rho != rho):  # NaN check
            rho_str = r"N/A"
        else:
            rho_str = f"{rho:+.3f}"
            if p_val is not None and p_val < 1e-10:
                rho_str += r"$^{***}$"

        # Format slope
        if slope_mean is None:
            slope_str = "—"
            ci_str = "—"
        else:
            slope_str = f"{slope_mean:+.4f}"
            ci_str = f"[{slope_lo:+.4f}, {slope_hi:+.4f}]"

        # Format monotonicity
        if mono is None:
            mono_str = "—"
        else:
            mono_str = f"{mono * 100:.0f}\\%"

        lines.append(
            f"  {display_name} & {slope_str} & {ci_str} & "
            f"{rho_str} & {mono_str} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{-1em}",
        r"\end{table}",
    ]

    out = "\n".join(lines)
    path = out_dir / "table_main_results.tex"
    path.write_text(out, encoding="utf-8")
    print(f"  Saved: {path}")
    return out


# ── Table 2: PPOTimeV2 ablation ───────────────────────────────────────────────

def table_ppo_ablation(out_dir: Path) -> None:
    """Table 2: PPOTimeV2 variant comparison."""

    # From experimental results (speed_gen_hidden_5seed)
    variants = [
        ("Baseline GRU (Variant 0)", 0.0, "N/A", "—", 0.0, "N/A"),
        (r"\textbf{Internal Time (Variant 1)}", 0.0245, r"[+0.003, +0.028]",
         "1.000***", 100.0, "-0.130"),
        (r"IT + $\gamma^{\Delta\tau}$ (Variant 2)", 0.0255, r"[+0.005, +0.026]",
         "1.000***", 100.0, "-0.130"),
        ("Discount-only (Variant 3)", 0.0, "—", "N/A", "—", "-0.130"),
    ]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{PPOTimeV2 variant ablation on the speed-generalization task "
        r"(5 seeds, hidden speed). Variant 1 modulates hidden state; "
        r"Variant 2 additionally applies time-dependent discount and GAE.}",
        r"\label{tab:ablation}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Variant & $\Delta\tau$ Slope & 95\% CI & Spearman $\rho$ & "
        r"Monotonicity & Gen.\ Gap \\",
        r"\midrule",
    ]

    for variant, slope, ci, rho, mono, gap in variants:
        mono_str = f"{mono:.0f}\\%" if isinstance(mono, float) else mono
        lines.append(
            f"  {variant} & {slope:+.4f} & {ci} & {rho} & "
            f"{mono_str} & {gap} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\multicolumn{6}{l}{\small $^{***}p < 10^{-24}$, "
        r"Gen. Gap = mean(seen) $-$ mean(unseen), lower = better.} \\",
        r"\end{tabular}",
        r"\vspace{-1em}",
        r"\end{table}",
    ]

    out = "\n".join(lines)
    path = out_dir / "table_ablation.tex"
    path.write_text(out, encoding="utf-8")
    print(f"  Saved: {path}")


# ── Table 3: Robustness scores ────────────────────────────────────────────────

def table_robustness(out_dir: Path) -> None:
    """Table 3: Deployment/stress robustness scores across environments."""

    # From audit reports (actual numbers)
    rows = [
        # env, agent, obs_delay, jitter, spike, speed_5x, deploy, stress, quadrant
        ("CartPole",
         "Standard GRU", 82, 66, 23, 12, "FAIL (0.23)", "FAIL (0.12)",
         "fragile"),
        ("CartPole",
         "Speed-Randomized", 95, 115, 62, 49, "DEGRADED (0.62)", "FAIL (0.48)",
         "fragile"),
        ("HalfCheetah",
         "Standard PPO", 4, 25, 91, -9, "FAIL (0.02)", "FAIL (-0.09)",
         "fragile"),
        ("HalfCheetah",
         "Speed-Randomized PPO", 148, 121, 113, 38, "PASS (1.00)", "FAIL (0.38)",
         "ready"),
    ]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Timing robustness scores (\% of nominal return) across "
        r"environments and perturbation types. "
        r"Bold = deployment-ready (score $\geq 0.80$). "
        r"Negative values indicate policy collapse.}",
        r"\label{tab:robustness}",
        r"\small",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Environment & Agent & Delay & Jitter & Spike & 5$\times$ Speed & "
        r"Deploy & Stress \\",
        r"\midrule",
    ]

    prev_env = None
    for env, agent, delay, jitter, spike, speed5x, deploy, stress, quad in rows:
        env_str = env if env != prev_env else ""
        prev_env = env

        deploy_fmt = (r"\textbf{" + deploy + r"}") if quad == "ready" else deploy
        lines.append(
            f"  {env_str} & {agent} & "
            f"{delay:+d}\\% & {jitter:+d}\\% & {spike:+d}\\% & "
            f"{speed5x:+d}\\% & {deploy_fmt} & {stress} \\\\"
        )
        if env_str:
            lines[-1] = "  " + lines[-1].strip() + "  \\addlinespace[2pt]"

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{-1em}",
        r"\end{table}",
    ]

    out = "\n".join(lines)
    path = out_dir / "table_robustness.tex"
    path.write_text(out, encoding="utf-8")
    print(f"  Saved: {path}")


# ── Table 4: Method comparison ────────────────────────────────────────────────

def table_comparison(out_dir: Path) -> None:
    """Table 4: Capability comparison vs related methods."""

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Comparison with related methods. "
        r"Our method requires no oracle $\Delta t$, learns timing internally, "
        r"and provides both measurement and repair of timing robustness.}",
        r"\label{tab:comparison}",
        r"\small",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Method & No oracle & Variable $\Delta t$ & Unseen speeds & "
        r"Internal time & Audit & Fix & Cont.\ gate \\",
        r"\midrule",
        r"ODE-RNN \citep{rubanova2019latent} & \xmark & \cmark & \pmmark & "
        r"\xmark & \xmark & \xmark & \cmark \\",
        r"Skip-RNN \citep{campos2018skip} & \cmark & \xmark & \xmark & "
        r"\pmmark & \xmark & \xmark & \xmark \\",
        r"CARE \citep{lim2022care} & \xmark & \xmark & \pmmark & "
        r"\xmark & \xmark & \xmark & \xmark \\",
        r"Frame-stack (baseline) & \cmark & \xmark & \xmark & "
        r"\xmark & \xmark & \xmark & \xmark \\",
        r"TimeFeature (dt inject.) & \xmark & \cmark & \pmmark & "
        r"\xmark & \xmark & \xmark & \xmark \\",
        r"\midrule",
        r"\textbf{InternalTime (ours)} & \cmark & \cmark & \cmark & "
        r"\cmark & \cmark & \cmark & \cmark \\",
        r"\bottomrule",
        r"\multicolumn{8}{l}{\small "
        r"\cmark=Yes, \xmark=No, \pmmark=Partial.} \\",
        r"\end{tabular}",
        r"\vspace{-1em}",
        r"\end{table}",
    ]

    # Note: requires \newcommand{\cmark}{\ding{51}}, \newcommand{\xmark}{\ding{55}}
    # and \usepackage{pifont} in paper preamble

    out = "\n".join(lines)
    path = out_dir / "table_comparison.tex"
    path.write_text(out, encoding="utf-8")
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables for the paper"
    )
    parser.add_argument(
        "--out", type=Path, default=OUT_DIR_DEFAULT,
        help="Output directory"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Generating LaTeX tables → {args.out}")

    metrics = load_statistical_metrics()

    print("\n[1/4] Main results table...")
    table_main_results(metrics, args.out)

    print("[2/4] PPOTimeV2 ablation table...")
    table_ppo_ablation(args.out)

    print("[3/4] Robustness scores table...")
    table_robustness(args.out)

    print("[4/4] Method comparison table...")
    table_comparison(args.out)

    print(f"\nDone. Tables saved to {args.out}")
    print("\nInclude in paper with:")
    print(r"  \input{table_main_results.tex}")
    print(r"  \input{table_ablation.tex}")
    print(r"  \input{table_robustness.tex}")
    print(r"  \input{table_comparison.tex}")


if __name__ == "__main__":
    main()
