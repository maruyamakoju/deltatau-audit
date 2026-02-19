"""Report generator: HTML + PNG + JSON audit report.

Produces a self-contained HTML report with:
- Badge header: Reliance (or N/A) + Deployment Robustness + Stress Robustness
- Quadrant scatter (only when reliance data available)
- Reliance section (skipped when N/A)
- Robustness section with Deployment vs Stress grouping
- Temporal sensitivity (if available)
- Actionable prescription
"""

import base64
import io
import json
import os
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..metrics import (
    reliance_color, robustness_color, severity_color,
)


def _get_report_version() -> str:
    from .. import __version__
    return __version__


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 PNG for HTML embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ── Reliance figures ──────────────────────────────────────────────────

def _plot_return_vs_speed(reliance: Dict, speeds: list) -> str:
    """Plot return vs speed (no intervention). Returns base64 PNG."""
    per_speed = reliance["per_speed"]

    fig, ax = plt.subplots(figsize=(7, 4))
    means = [per_speed[str(s)]["none"]["total_reward_mean"] for s in speeds]
    stds = [per_speed[str(s)]["none"].get("total_reward_se", 0) for s in speeds]
    ax.errorbar(speeds, means, yerr=stds, marker="o", linewidth=2,
                capsize=4, color="#2196F3", markersize=7)
    ax.set_xlabel("Environment Speed", fontsize=12)
    ax.set_ylabel("Episode Return", fontsize=12)
    ax.set_title("Return vs Speed (nominal)", fontsize=13)
    ax.set_xticks(speeds)
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _plot_reliance_rmse(reliance: Dict, speeds: list) -> str:
    """Plot value RMSE under intervention at each speed. Returns base64 PNG."""
    per_speed = reliance["per_speed"]

    fig, ax = plt.subplots(figsize=(7, 4))

    styles = {
        "none": {"label": "Normal", "color": "#2196F3", "marker": "o", "ls": "-"},
        "clamp_1": {"label": "Dt=1.0", "color": "#9E9E9E", "marker": "s", "ls": "--"},
        "reverse": {"label": "Dt reversed", "color": "#E91E63", "marker": "^", "ls": "-."},
        "random": {"label": "Dt random", "color": "#FF9800", "marker": "D", "ls": ":"},
    }

    for interv, style in styles.items():
        rmses = []
        ses = []
        valid = True
        for s in speeds:
            data = per_speed[str(s)].get(interv)
            if data is None:
                valid = False
                break
            rmses.append(data.get("rmse_mean", 0))
            ses.append(data.get("rmse_se", 0))
        if valid:
            ax.errorbar(speeds, rmses, yerr=ses, label=style["label"],
                        color=style["color"], marker=style["marker"],
                        linewidth=2, capsize=4, linestyle=style["ls"],
                        markersize=7)

    ax.set_xlabel("Environment Speed", fontsize=12)
    ax.set_ylabel("Value RMSE", fontsize=12)
    ax.set_title("Reliance Test: Value Error Under Dt Intervention", fontsize=13)
    ax.set_xticks(speeds)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _plot_reliance_bars(reliance: Dict) -> str:
    """Bar chart of worst-speed degradation per intervention. Returns base64 PNG."""
    degradation = reliance.get("degradation", {})
    if not degradation:
        return ""

    fig, ax = plt.subplots(figsize=(6, 4))

    intervs = list(degradation.keys())
    worst_pcts = []
    colors = []
    for interv in intervs:
        by_speed = degradation[interv]
        worst = max(by_speed.values(), key=lambda d: d["percent_increase"])
        worst_pcts.append(worst["percent_increase"])
        colors.append(severity_color(worst["severity"]))

    labels = {
        "clamp_1": "Dt=1.0",
        "reverse": "Dt reversed",
        "random": "Dt random",
    }
    x_labels = [labels.get(i, i) for i in intervs]

    ax.bar(range(len(intervs)), worst_pcts, color=colors, alpha=0.85,
           edgecolor="white", linewidth=1.5)
    ax.set_xticks(range(len(intervs)))
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylabel("Worst-Speed RMSE Increase (%)", fontsize=11)
    ax.set_title("Reliance Impact (Worst Case per Intervention)", fontsize=13)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    for i, pct in enumerate(worst_pcts):
        ax.text(i, pct + 2, f"+{pct:.0f}%", ha="center", fontsize=11,
                fontweight="bold")

    return _fig_to_base64(fig)


# ── Robustness figures ────────────────────────────────────────────────

def _plot_robustness_bars(robustness: Dict) -> str:
    """Bar chart of return ratio + RMSE ratio per scenario. Returns base64 PNG."""
    scores = robustness.get("per_scenario_scores", {})
    if not scores:
        return ""

    # Order: deployment scenarios first, then stress
    from ..auditor import DEPLOYMENT_SCENARIOS, STRESS_SCENARIOS
    ordered = [s for s in DEPLOYMENT_SCENARIOS if s in scores]
    ordered += [s for s in STRESS_SCENARIOS if s in scores]
    # Add any remaining
    ordered += [s for s in scores if s not in ordered]
    n = len(ordered)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: Return ratio
    ret_ratios = [scores[s]["return_ratio"] * 100 for s in ordered]
    ret_colors = []
    for r in ret_ratios:
        if r >= 95:
            ret_colors.append("#28a745")
        elif r >= 80:
            ret_colors.append("#ffc107")
        elif r >= 50:
            ret_colors.append("#fd7e14")
        else:
            ret_colors.append("#dc3545")

    scenario_labels = {
        "speed_5x": "5x Speed [STRESS]",
        "jitter": "Jitter",
        "delay": "Delay",
        "spike": "Spike",
        "obs_noise": "Obs. Noise",
    }
    labels = [scenario_labels.get(s, s) for s in ordered]

    # Visual separator between deployment and stress
    deploy_n = len([s for s in ordered if s in DEPLOYMENT_SCENARIOS])

    ax1.barh(range(n), ret_ratios, color=ret_colors, alpha=0.85,
             edgecolor="white", linewidth=1.5)
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.set_xlabel("Return (% of nominal)", fontsize=11)
    ax1.set_title("Robustness: Return", fontsize=13)
    ax1.axvline(x=100, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax1.axvline(x=80, color="#fd7e14", linewidth=0.8, linestyle=":", alpha=0.5)
    ax1.set_xlim(0, max(max(ret_ratios) * 1.1, 110))
    ax1.grid(True, alpha=0.3, axis="x")
    if deploy_n < n:
        ax1.axhline(y=deploy_n - 0.5, color="#999", linewidth=1,
                     linestyle="--", alpha=0.5)

    for i, r in enumerate(ret_ratios):
        ax1.text(r + 1, i, f"{r:.0f}%", va="center", fontsize=10,
                 fontweight="bold")

    # Right: RMSE ratio
    rmse_ratios = [scores[s]["rmse_ratio"] for s in ordered]
    rmse_colors = []
    for r in rmse_ratios:
        if r < 1.2:
            rmse_colors.append("#28a745")
        elif r < 1.5:
            rmse_colors.append("#ffc107")
        elif r < 2.0:
            rmse_colors.append("#fd7e14")
        else:
            rmse_colors.append("#dc3545")

    ax2.barh(range(n), rmse_ratios, color=rmse_colors, alpha=0.85,
             edgecolor="white", linewidth=1.5)
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(labels, fontsize=10)
    ax2.set_xlabel("RMSE ratio (vs nominal)", fontsize=11)
    ax2.set_title("Robustness: Value Calibration", fontsize=13)
    ax2.axvline(x=1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.grid(True, alpha=0.3, axis="x")
    if deploy_n < n:
        ax2.axhline(y=deploy_n - 0.5, color="#999", linewidth=1,
                     linestyle="--", alpha=0.5)

    for i, r in enumerate(rmse_ratios):
        ax2.text(r + 0.02, i, f"{r:.2f}x", va="center", fontsize=10,
                 fontweight="bold")

    fig.tight_layout()
    return _fig_to_base64(fig)


# ── Quadrant scatter ──────────────────────────────────────────────────

def _plot_quadrant(summary: Dict, comparison: list = None) -> str:
    """2D quadrant scatter: Reliance vs Deployment Robustness.

    Returns base64 PNG, or empty string if reliance is N/A.
    """
    if summary["reliance_rating"] == "N/A":
        return ""

    fig, ax = plt.subplots(figsize=(6, 5))

    # Quadrant boundaries
    rel_threshold = 2.0
    rob_threshold = 0.80

    # Shade quadrants
    ax.axhspan(rob_threshold, 1.15, xmin=0, xmax=0.5,
               alpha=0.06, color="#28a745")  # blind+robust
    ax.axhspan(rob_threshold, 1.15, xmin=0.5, xmax=1.0,
               alpha=0.08, color="#2196F3")  # aware+robust
    ax.axhspan(0, rob_threshold, xmin=0, xmax=0.5,
               alpha=0.06, color="#dc3545")  # blind+fragile
    ax.axhspan(0, rob_threshold, xmin=0.5, xmax=1.0,
               alpha=0.06, color="#FF9800")  # aware+fragile

    # Quadrant labels
    ax.text(1.0, 1.08, "Time-Blind\n& Robust", ha="center", va="center",
            fontsize=8, color="#28a745", alpha=0.7)
    ax.text(3.5, 1.08, "Time-Aware\n& Robust", ha="center", va="center",
            fontsize=8, color="#1565C0", alpha=0.7, fontweight="bold")
    ax.text(1.0, 0.25, "Time-Blind\n& Fragile", ha="center", va="center",
            fontsize=8, color="#dc3545", alpha=0.7)
    ax.text(3.5, 0.25, "Time-Aware\n but Fragile", ha="center", va="center",
            fontsize=8, color="#FF9800", alpha=0.7)

    # Threshold lines
    ax.axvline(x=rel_threshold, color="#666", linewidth=1, linestyle="--",
               alpha=0.4)
    ax.axhline(y=rob_threshold, color="#666", linewidth=1, linestyle="--",
               alpha=0.4)

    # Plot current model (y = deployment score, not overall)
    rel_score = summary["reliance_score"]
    rob_score = summary.get("deployment_score", summary["robustness_score"])
    ax.scatter([rel_score], [rob_score], s=200, c="#1565C0", zorder=5,
               edgecolors="white", linewidths=2)
    ax.annotate("This Agent", (rel_score, rob_score),
                textcoords="offset points", xytext=(12, 8),
                fontsize=10, fontweight="bold", color="#1565C0")

    # Plot comparison models if provided
    if comparison:
        colors = ["#E91E63", "#4CAF50", "#FF9800", "#9C27B0"]
        for i, (name, r_score, b_score) in enumerate(comparison):
            c = colors[i % len(colors)]
            ax.scatter([r_score], [b_score], s=120, c=c, zorder=4,
                       edgecolors="white", linewidths=1.5, marker="D")
            ax.annotate(name, (r_score, b_score),
                        textcoords="offset points", xytext=(10, -10),
                        fontsize=9, color=c)

    ax.set_xlabel("Timing Reliance (RMSE ratio)", fontsize=11)
    ax.set_ylabel("Deployment Robustness (return ratio)", fontsize=11)
    ax.set_title("Audit Quadrant", fontsize=13)
    ax.set_xlim(0.5, max(rel_score * 1.3, 4.0))
    ax.set_ylim(0, max(rob_score * 1.2, 1.15))
    ax.grid(True, alpha=0.2)

    return _fig_to_base64(fig)


# ── HTML generation ───────────────────────────────────────────────────

def generate_report(audit_result: Dict, output_dir: str,
                    title: str = "Time Robustness Audit"):
    """Generate audit report: HTML + PNGs + summary JSON.

    Adapts layout based on available data:
    - External models (no intervention): 2 badges, no reliance section
    - Internal time models: 3 badges + quadrant + reliance section
    """
    os.makedirs(output_dir, exist_ok=True)

    speeds = audit_result["speeds"]
    reliance = audit_result["reliance"]
    robustness = audit_result["robustness"]
    sensitivity = audit_result.get("sensitivity")
    summary = audit_result["summary"]

    has_reliance = summary["reliance_rating"] != "N/A"

    # Generate figures
    robustness_bars_img = _plot_robustness_bars(robustness)
    quadrant_img = ""

    if has_reliance:
        return_img = _plot_return_vs_speed(reliance, speeds)
        reliance_rmse_img = _plot_reliance_rmse(reliance, speeds)
        reliance_bars_img = _plot_reliance_bars(reliance)
        quadrant_img = _plot_quadrant(summary)
    else:
        return_img = ""
        reliance_rmse_img = ""
        reliance_bars_img = ""

    # Save individual PNGs
    png_list = [("robustness_bars", robustness_bars_img)]
    if has_reliance:
        png_list += [
            ("return_vs_speed", return_img),
            ("reliance_rmse", reliance_rmse_img),
            ("reliance_bars", reliance_bars_img),
            ("quadrant", quadrant_img),
        ]

    for name, b64 in png_list:
        if b64:
            path = os.path.join(output_dir, f"{name}.png")
            with open(path, "wb") as f:
                f.write(base64.b64decode(b64))

    # Save JSON — include version + timestamp for traceability
    import datetime
    json_data = dict(audit_result)
    json_data["_version"] = _get_report_version()
    json_data["_timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
    json_path = os.path.join(output_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)

    # ── Build HTML ────────────────────────────────────────────────
    dep_rating = summary["deployment_rating"]
    dep_color = robustness_color(dep_rating)
    str_rating = summary["stress_rating"]
    str_color = robustness_color(str_rating)
    quadrant = summary["quadrant"]

    quadrant_labels = {
        "time_aware_robust": "Time-Aware & Robust",
        "time_aware_fragile": "Time-Aware but Fragile",
        "time_blind_fragile": "Time-Blind & Fragile",
        "time_blind_robust": "Time-Blind but Robust",
        "deployment_ready": "Deployment Ready",
        "deployment_fragile": "Deployment Fragile",
    }
    quadrant_label = quadrant_labels.get(quadrant, quadrant)

    # Verdict pill color: green for good quadrants, orange/red for fragile
    _good_quadrants = {"time_aware_robust", "time_blind_robust", "deployment_ready"}
    _warn_quadrants = {"time_aware_fragile", "time_blind_robust"}
    if quadrant in _good_quadrants:
        verdict_color = "#28a745"
    elif quadrant in {"time_blind_fragile", "deployment_fragile"}:
        verdict_color = "#dc3545"
    else:
        verdict_color = "#fd7e14"
    verdict_pill = (f'<div style="text-align:center">'
                    f'<span class="verdict-pill" style="background:{verdict_color}">'
                    f'{quadrant_label}</span></div>')

    # ── Badge HTML ─────────────────────────────────────────────────
    # Helper: colored score meter bar (0–100%)
    def _meter(score_01: float, color: str, label_pct: str) -> str:
        pct = min(100, max(0, score_01 * 100))
        return (
            f'<div class="meter-wrap">'
            f'<div class="meter-fill" style="width:{pct:.0f}%;background:{color}"></div>'
            f'</div>'
            f'<div class="badge-score" style="color:{color}">{label_pct}</div>'
        )

    dep_pct = f"{summary['deployment_score']*100:.0f}%"
    str_pct = f"{summary['stress_score']*100:.0f}%"

    if has_reliance:
        rel_rating = summary["reliance_rating"]
        rel_color = reliance_color(rel_rating)
        rel_score = summary["reliance_score"]
        # Reliance meter: cap at 5x for display (>2x = deep blue)
        rel_norm = min(1.0, (rel_score - 1.0) / 4.0)
        rel_meter = (
            f'<div class="meter-wrap">'
            f'<div class="meter-fill" style="width:{rel_norm*100:.0f}%;background:{rel_color}"></div>'
            f'</div>'
            f'<div class="badge-score" style="color:{rel_color}">{rel_score:.1f}x RMSE</div>'
        )
        badge_html = f"""
    <div class="badge">
      <div class="badge-label">Timing Reliance</div>
      <div class="badge-value" style="color:{rel_color}">{rel_rating}</div>
      {rel_meter}
      <div class="badge-detail">Value RMSE ratio under Δτ intervention</div>
    </div>
    <div class="badge">
      <div class="badge-label">Deployment Robustness</div>
      <div class="badge-value" style="color:{dep_color}">{dep_rating}</div>
      {_meter(summary['deployment_score'], dep_color, dep_pct)}
      <div class="badge-detail">Return under jitter / delay / spike / noise</div>
    </div>
    <div class="badge">
      <div class="badge-label">Stress Robustness</div>
      <div class="badge-value" style="color:{str_color}">{str_rating}</div>
      {_meter(summary['stress_score'], str_color, str_pct)}
      <div class="badge-detail">Return at 5× speed (extreme)</div>
    </div>"""
    else:
        badge_html = f"""
    <div class="badge">
      <div class="badge-label">Deployment Robustness</div>
      <div class="badge-value" style="color:{dep_color}">{dep_rating}</div>
      {_meter(summary['deployment_score'], dep_color, dep_pct)}
      <div class="badge-detail">Return under jitter / delay / spike / noise</div>
    </div>
    <div class="badge">
      <div class="badge-label">Stress Robustness</div>
      <div class="badge-value" style="color:{str_color}">{str_rating}</div>
      {_meter(summary['stress_score'], str_color, str_pct)}
      <div class="badge-detail">Return at 5× speed (extreme)</div>
    </div>"""

    # ── Quadrant + reliance section (only if has_reliance) ────────
    quadrant_html = ""
    if quadrant_img:
        quadrant_html = f"""
<div class="fig-card" style="text-align:center">
  <img src="data:image/png;base64,{quadrant_img}" alt="Audit Quadrant" style="max-width:450px">
</div>"""

    reliance_html = ""
    if has_reliance:
        # Degradation table
        deg_rows = ""
        degradation = reliance.get("degradation", {})
        interv_labels = {
            "clamp_1": "Dt = 1.0 (clamped)",
            "reverse": "Dt reversed",
            "random": "Dt random",
        }
        for interv, by_speed in degradation.items():
            for s_str, deg in by_speed.items():
                pct = deg["percent_increase"]
                sev_i = deg["severity"]
                col_i = severity_color(sev_i)
                label = interv_labels.get(interv, interv)
                deg_rows += (
                    f'<tr><td>{label}</td><td>{s_str}</td>'
                    f'<td>{deg["baseline_rmse"]:.4f}</td>'
                    f'<td>{deg["intervention_rmse"]:.4f}</td>'
                    f'<td>+{pct:.0f}%</td>'
                    f'<td style="color:{col_i};font-weight:bold">{sev_i}</td></tr>\n'
                )

        reliance_html = f"""
<!-- ═══ Reliance Test ═══ -->
<h2>1. Reliance Test &mdash; Intervention Ablation</h2>
<p class="section-desc">
  Tampers with the agent's internal &Delta;&tau; to test causal dependence
  on the timing representation. <strong>High reliance = the timing channel
  is actively used by the value function.</strong>
  This is evidence of mechanism, not a failure mode.
</p>

<div class="figures">
  <div class="fig-card">
    <h3>Return vs Speed</h3>
    <img src="data:image/png;base64,{return_img}" alt="Return vs Speed">
  </div>
  <div class="fig-card">
    <h3>Value RMSE Under &Delta;&tau; Intervention</h3>
    <img src="data:image/png;base64,{reliance_rmse_img}" alt="Reliance RMSE">
  </div>
  {"<div class='fig-card'><h3>Reliance Impact (Worst Case)</h3><img src='data:image/png;base64," + reliance_bars_img + "' alt='Reliance Bars'></div>" if reliance_bars_img else ""}
</div>

<h3>Reliance Detail</h3>
<table>
  <tr><th>Intervention</th><th>Speed</th><th>Baseline RMSE</th>
      <th>Intervention RMSE</th><th>Change</th><th>Severity</th></tr>
  {deg_rows}
</table>
"""

    # ── Robustness section ────────────────────────────────────────
    from ..auditor import DEPLOYMENT_SCENARIOS, STRESS_SCENARIOS

    rob_scores = robustness.get("per_scenario_scores", {})
    scenario_labels = {
        "speed_5x": "5x Speed (unseen frequency)",
        "jitter": "Speed jitter (2 +/- 1)",
        "delay": "Observation delay (1 step)",
        "spike": "Mid-episode spike (1-5-1)",
        "obs_noise": "Observation noise (σ=0.1)",
    }

    def _make_rob_rows(scenario_list, category_label):
        rows = ""
        for scenario in scenario_list:
            sc = rob_scores.get(scenario)
            if sc is None:
                continue
            label = scenario_labels.get(scenario, scenario)
            ret_r = sc["return_ratio"] * 100
            rmse_r = sc["rmse_ratio"]
            if ret_r >= 95:
                r_col = "#28a745"
            elif ret_r >= 80:
                r_col = "#ffc107"
            else:
                r_col = "#dc3545"
            # Bootstrap CI
            ci_lo = sc.get("ci_lower")
            ci_hi = sc.get("ci_upper")
            sig = sc.get("significant", False)
            if ci_lo is not None and ci_hi is not None:
                ci_str = f"{ci_lo*100:.0f}%–{ci_hi*100:.0f}%"
                sig_str = " ***" if sig else ""
            else:
                ci_str = "–"
                sig_str = ""
            rows += (
                f'<tr><td>{category_label}</td><td>{label}</td>'
                f'<td style="color:{r_col};font-weight:bold">{ret_r:.0f}%</td>'
                f'<td>{ci_str}{sig_str}</td>'
                f'<td>{rmse_r:.2f}x</td>'
                f'<td>{sc["return_drop_pct"]:+.1f}%</td></tr>\n'
            )
        return rows

    dep_rows = _make_rob_rows(DEPLOYMENT_SCENARIOS, "Deployment")
    str_rows = _make_rob_rows(STRESS_SCENARIOS, "Stress")

    section_num = "2" if has_reliance else "1"

    # Sensitivity section
    sensitivity_html = ""
    if sensitivity:
        sens_num = "3" if has_reliance else "2"
        sens_per_speed = sensitivity.get("per_speed", {})
        sens_rows = ""
        for s_str, sv in sens_per_speed.items():
            sens_rows += (
                f'<tr><td>{s_str}</td>'
                f'<td>{sv["mean"]:.4f}</td>'
                f'<td>{sv["std"]:.4f}</td>'
                f'<td>{sv["n_samples"]}</td></tr>\n'
            )
        sensitivity_html = f"""
<h2>{sens_num}. Temporal Sensitivity &mdash; |dV/d&tau;|</h2>
<p class="section-desc">
  Finite-difference approximation of how much the value function changes
  per unit change in &Delta;&tau;. High sensitivity at unseen speeds indicates
  the timing channel is <strong>actively adapting</strong>.
</p>
<div class="metric-card">
  <div class="metric-value">{sensitivity['mean']:.4f}</div>
  <div class="metric-label">Mean |dV/d&tau;| (timing Jacobian)</div>
</div>
<table>
  <tr><th>Speed</th><th>Mean |dV/d&tau;|</th><th>Std</th><th>Samples</th></tr>
  {sens_rows}
</table>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 950px; margin: 0 auto; padding: 20px; background: #fafafa; color: #333; }}
  h1 {{ border-bottom: 3px solid #2196F3; padding-bottom: 10px; }}
  h2 {{ color: #1565C0; margin-top: 40px; }}

  .headline {{ background: white; border-radius: 12px; padding: 28px; margin: 20px 0;
               box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .badge-row {{ display: flex; gap: 16px; justify-content: center; flex-wrap: wrap; }}
  .badge {{ text-align: center; padding: 18px 24px; border-radius: 10px;
            background: #f8f9fa; min-width: 180px; flex: 1; max-width: 260px; }}
  .badge-label {{ font-size: 0.8em; color: #666; text-transform: uppercase;
                  letter-spacing: 1px; margin-bottom: 6px; }}
  .badge-value {{ font-size: 2em; font-weight: bold; }}
  .badge-detail {{ font-size: 0.78em; color: #999; margin-top: 4px; }}
  .badge-score {{ font-size: 1.05em; font-weight: 600; margin-top: 2px; }}
  .meter-wrap {{ height: 6px; background: #e0e0e0; border-radius: 3px;
                 margin: 8px 0 2px 0; overflow: hidden; }}
  .meter-fill {{ height: 100%; border-radius: 3px; transition: width 0.3s; }}
  .verdict-pill {{ display: inline-block; margin: 14px auto 0;
                   padding: 6px 20px; border-radius: 20px;
                   font-size: 0.9em; font-weight: 600; letter-spacing: 0.5px;
                   color: white; }}
  .quadrant-label {{ text-align: center; margin-top: 18px; font-size: 1.1em;
                     color: #555; font-weight: 500; }}
  .interpretation {{ text-align: center; margin-top: 12px; color: #666;
                     font-size: 0.95em; max-width: 600px; margin-left: auto;
                     margin-right: auto; }}

  .section-desc {{ color: #666; font-size: 0.95em; margin: 8px 0 16px 0;
                   border-left: 3px solid #ddd; padding-left: 12px; }}

  .figures {{ display: grid; gap: 20px; margin: 20px 0; }}
  .fig-card {{ background: white; border-radius: 8px; padding: 16px;
               box-shadow: 0 1px 4px rgba(0,0,0,0.06); }}
  .fig-card img {{ width: 100%; height: auto; }}

  table {{ width: 100%; border-collapse: collapse; background: white;
           border-radius: 8px; overflow: hidden;
           box-shadow: 0 1px 4px rgba(0,0,0,0.06); margin: 16px 0; }}
  th {{ background: #2196F3; color: white; padding: 12px 16px; text-align: left; }}
  td {{ padding: 10px 16px; border-bottom: 1px solid #eee; }}
  tr:hover {{ background: #f5f5f5; }}
  .cat-deploy {{ border-left: 3px solid #2196F3; }}
  .cat-stress {{ border-left: 3px solid #FF9800; }}

  .metric-card {{ background: white; border-radius: 10px; padding: 20px;
                  text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
                  margin: 16px 0; display: inline-block; }}
  .metric-value {{ font-size: 2em; font-weight: bold; color: #1565C0; }}
  .metric-label {{ font-size: 0.85em; color: #888; margin-top: 4px; }}

  .prescription {{ background: #e3f2fd; border-left: 4px solid #1565C0;
                   padding: 16px; margin: 20px 0; border-radius: 0 8px 8px 0; }}
  .prescription h3 {{ margin-top: 0; color: #1565C0; }}

  .meta {{ color: #999; font-size: 0.85em; margin-top: 30px; }}
</style>
</head>
<body>
<h1>{title}</h1>

<!-- ═══ Badge Header ═══ -->
<div class="headline">
  <div class="badge-row">
    {badge_html}
  </div>
  {verdict_pill}
  <div class="interpretation">{summary['prescription']}</div>
</div>

{quadrant_html}

{reliance_html}

<!-- ═══ Robustness Test ═══ -->
<h2>{section_num}. Robustness Test &mdash; Timing Perturbations</h2>
<p class="section-desc">
  Wraps the environment with timing perturbations. The agent runs
  <strong>normally</strong> &mdash; no internal intervention.
  <strong>Deployment</strong> scenarios (jitter, delay, spike) model
  realistic conditions. <strong>Stress</strong> scenarios (5x speed)
  test extreme resilience.
</p>

{"<div class='fig-card'><h3>Robustness Under Timing Perturbations</h3><img src='data:image/png;base64," + robustness_bars_img + "' alt='Robustness Bars'></div>" if robustness_bars_img else ""}

<h3>Robustness Detail</h3>
<table>
  <tr><th>Category</th><th>Scenario</th><th>Return (% nominal)</th><th>95% CI</th><th>RMSE ratio</th><th>Return Change</th></tr>
  {dep_rows}
  {str_rows}
</table>

{sensitivity_html}

<!-- ═══ Prescription ═══ -->
<div class="prescription">
  <h3>Recommendation</h3>
  <p>{summary['prescription']}</p>
</div>

<div class="meta">
  <p>Speeds tested: {speeds} |
     Episodes per condition: {audit_result['n_episodes']} |
     Intervention support: {audit_result['supports_intervention']}</p>
  <p>Generated by <code>deltatau-audit</code> v{_get_report_version()}</p>
</div>
</body>
</html>"""

    html_path = os.path.join(output_dir, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Report saved to {output_dir}/")
    print(f"  index.html           -- Full audit report")
    print(f"  summary.json         -- Machine-readable results")
    if has_reliance:
        print(f"  return_vs_speed.png  -- Speed generalization")
        print(f"  reliance_rmse.png    -- Reliance test figure")
        if reliance_bars_img:
            print(f"  reliance_bars.png    -- Reliance impact bars")
        if quadrant_img:
            print(f"  quadrant.png         -- 2D audit quadrant")
    if robustness_bars_img:
        print(f"  robustness_bars.png  -- Robustness scenario bars")

    return html_path
