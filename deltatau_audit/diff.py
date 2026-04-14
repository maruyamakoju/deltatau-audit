"""Diff two audit summary.json files and generate comparison reports."""

from __future__ import annotations

import html
import json
import pathlib
from pathlib import Path
from typing import Any, Mapping

from ._constants import DEPLOYMENT_SCENARIOS as _DEPLOY_SCENARIOS_LIST
from ._theme import quadrant_label as _quadrant_label

_DEPLOY_SCENARIOS = set(_DEPLOY_SCENARIOS_LIST)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return data


def _summary(data: Mapping[str, Any]) -> dict[str, Any]:
    summary_data = data.get("summary")
    if isinstance(summary_data, dict):
        return dict(summary_data)
    return dict(data)


def _robustness(data: Mapping[str, Any]) -> dict[str, Any]:
    robust_data = data.get("robustness")
    if isinstance(robust_data, dict):
        return dict(robust_data)
    return {}


def _scenario_key(scenario: str) -> tuple[int, str]:
    return (0 if scenario in _DEPLOY_SCENARIOS else 1, scenario)


def _ordered_scenarios(before_scores: Mapping[str, Any], after_scores: Mapping[str, Any]) -> list[str]:
    return sorted(set(before_scores) | set(after_scores), key=_scenario_key)


def _md_cell(value: Any) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _html_text(value: Any) -> str:
    return html.escape(str(value))


def _make_comparison_chart(before_data: dict, after_data: dict) -> str | None:
    """Generate a base64-encoded grouped bar chart of before/after robustness scores.

    Returns base64 PNG string or None if matplotlib is unavailable.
    """
    try:
        import base64
        import io

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    b_scores = before_data.get("per_scenario_scores", {})
    a_scores = after_data.get("per_scenario_scores", {})

    scenarios = _ordered_scenarios(b_scores, a_scores)
    if not scenarios:
        return None

    before_ratios = [b_scores.get(sc, {}).get("return_ratio", 0.0) * 100 for sc in scenarios]
    after_ratios = [a_scores.get(sc, {}).get("return_ratio", 0.0) * 100 for sc in scenarios]

    # Scenario display names
    display_names = {
        "jitter": "Speed\nJitter",
        "delay": "Obs.\nDelay",
        "spike": "Speed\nSpike",
        "obs_noise": "Obs.\nNoise",
        "speed_5x": "5x Speed\n[STRESS]",
    }
    labels = [display_names.get(sc, sc) for sc in scenarios]

    x = np.arange(len(scenarios))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(7, len(scenarios) * 1.5), 4.5))

    def _bar_color(ratio_pct):
        r = ratio_pct / 100.0
        if r > 0.95:
            return "#28a745"
        if r > 0.80:
            return "#ffc107"
        if r > 0.50:
            return "#fd7e14"
        return "#dc3545"

    bars1 = ax.bar(
        x - width / 2,
        before_ratios,
        width,
        label="Before",
        color=[_bar_color(v) for v in before_ratios],
        alpha=0.6,
        edgecolor="white",
        linewidth=1.2,
    )
    bars2 = ax.bar(
        x + width / 2,
        after_ratios,
        width,
        label="After",
        color=[_bar_color(v) for v in after_ratios],
        alpha=0.95,
        edgecolor="white",
        linewidth=1.2,
    )

    # Reference lines
    ax.axhline(y=100, color="#28a745", linestyle="--", linewidth=1, alpha=0.5, label="Nominal (100%)")
    ax.axhline(y=80, color="#ffc107", linestyle=":", linewidth=1, alpha=0.5, label="MILD threshold (80%)")
    ax.axhline(y=0, color="#999", linestyle="-", linewidth=0.5, alpha=0.3)

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        y_off = 1 if h >= 0 else -1
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, h + y_off, f"{h:.0f}%", ha="center", va=va, fontsize=7, color="#333"
        )
    for bar in bars2:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        y_off = 1 if h >= 0 else -1
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + y_off,
            f"{h:.0f}%",
            ha="center",
            va=va,
            fontsize=7,
            fontweight="bold",
            color="#111",
        )

    ax.set_xlabel("Scenario", fontsize=11)
    ax.set_ylabel("Performance vs Nominal (%)", fontsize=11)
    ax.set_title("Timing Robustness: Before vs After", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    # Y-axis range: include some space below 0 if there are negative values
    y_min = min(min(before_ratios), min(after_ratios), 0)
    y_max = max(max(before_ratios), max(after_ratios), 100) * 1.15
    ax.set_ylim(y_min - 10, y_max)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _seed_sweep_payload(data: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return embedded seed-sweep payload if available."""
    payload = data.get("seed_sweep")
    if isinstance(payload, dict):
        return dict(payload)
    if isinstance(data.get("aggregate"), dict) and isinstance(data.get("per_seed"), list):
        return dict(data)
    return None


def _seed_metric_values(seed_payload: Mapping[str, Any], key: str) -> list[float]:
    """Return per-seed values for a scalar summary metric (0-1 scale)."""
    vals: list[float] = []
    per_seed = seed_payload.get("per_seed", [])
    if not isinstance(per_seed, list):
        return vals
    for row in per_seed:
        if not isinstance(row, dict):
            continue
        v = row.get(key)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return vals


def _seed_metric_stats(seed_payload: Mapping[str, Any], key: str) -> tuple[float, float, float] | None:
    """Return (mean, ci_lower, ci_upper) for a metric, or None."""
    aggregate = seed_payload.get("aggregate", {})
    if not isinstance(aggregate, dict):
        return None
    metrics = aggregate.get("metrics", {})
    if not isinstance(metrics, dict):
        return None
    stat = metrics.get(key, {})
    if not isinstance(stat, dict):
        return None
    mean = stat.get("mean")
    ci_lo = stat.get("ci_lower")
    ci_hi = stat.get("ci_upper")
    if isinstance(mean, (int, float)) and isinstance(ci_lo, (int, float)) and isinstance(ci_hi, (int, float)):
        return float(mean), float(ci_lo), float(ci_hi)
    return None


def _make_seed_variance_chart(before_seed: Mapping[str, Any], after_seed: Mapping[str, Any]) -> str | None:
    """Generate a base64 PNG for multi-seed mean/CI and seed scatter."""
    try:
        import base64
        import io

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    metrics = [
        ("deployment_score", "Deployment"),
        ("stress_score", "Stress"),
    ]

    b_stats: list[tuple[float, float, float]] = []
    a_stats: list[tuple[float, float, float]] = []
    for key, _ in metrics:
        b_stat = _seed_metric_stats(before_seed, key)
        a_stat = _seed_metric_stats(after_seed, key)
        if b_stat is None or a_stat is None:
            return None
        b_stats.append(b_stat)
        a_stats.append(a_stat)

    x: Any = np.arange(len(metrics), dtype=float)

    b_means = np.array([s[0] * 100 for s in b_stats], dtype=float)
    b_lows = np.array([s[1] * 100 for s in b_stats], dtype=float)
    b_highs = np.array([s[2] * 100 for s in b_stats], dtype=float)
    a_means = np.array([s[0] * 100 for s in a_stats], dtype=float)
    a_lows = np.array([s[1] * 100 for s in a_stats], dtype=float)
    a_highs = np.array([s[2] * 100 for s in a_stats], dtype=float)

    b_err = np.vstack([b_means - b_lows, b_highs - b_means])
    a_err = np.vstack([a_means - a_lows, a_highs - a_means])

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    ax.errorbar(
        x - 0.08,
        b_means,
        yerr=b_err,
        fmt="o",
        color="#6c757d",
        elinewidth=1.7,
        capsize=4,
        label="Before mean ±95% CI",
    )
    ax.errorbar(
        x + 0.08,
        a_means,
        yerr=a_err,
        fmt="o",
        color="#198754",
        elinewidth=1.7,
        capsize=4,
        label="After mean ±95% CI",
    )

    rng = np.random.RandomState(7)
    for idx, (key, _) in enumerate(metrics):
        b_vals = np.array(_seed_metric_values(before_seed, key), dtype=float) * 100
        a_vals = np.array(_seed_metric_values(after_seed, key), dtype=float) * 100
        if b_vals.size > 0:
            xj = idx + rng.uniform(-0.16, -0.03, size=b_vals.size)
            ax.scatter(
                xj,
                b_vals,
                s=22,
                color="#adb5bd",
                alpha=0.8,
                edgecolors="white",
                linewidths=0.4,
                label="Before seeds" if idx == 0 else None,
            )
        if a_vals.size > 0:
            xj = idx + rng.uniform(0.03, 0.16, size=a_vals.size)
            ax.scatter(
                xj,
                a_vals,
                s=22,
                color="#51cf66",
                alpha=0.85,
                edgecolors="white",
                linewidths=0.4,
                label="After seeds" if idx == 0 else None,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics], fontsize=10)
    ax.set_ylabel("Score (% of nominal)", fontsize=11)
    ax.set_title("Multi-Seed Variance: Mean ±95% CI", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _arrow(before, after):
    """Return an arrow indicating direction of change."""
    if after > before:
        return "^"  # improved
    elif after < before:
        return "v"  # regressed
    return "="


def _rating_change(before, after):
    """Format a rating transition."""
    if before == after:
        return before
    return f"{before} -> {after}"


def _rating_color(rating: str) -> str:
    from ._theme import rating_color

    return rating_color(rating)


def generate_comparison(
    before_path: str | pathlib.Path,
    after_path: str | pathlib.Path,
    output_path: str | pathlib.Path | None = None,
) -> str:
    """Generate a comparison.md from two summary.json files.

    Args:
        before_path: Path to the 'before' summary.json
        after_path: Path to the 'after' summary.json
        output_path: Path to write comparison.md (optional)

    Returns:
        The comparison markdown string.
    """
    before_path = pathlib.Path(before_path)
    after_path = pathlib.Path(after_path)

    before = _load_json(before_path)
    after = _load_json(after_path)

    bs = _summary(before)
    as_ = _summary(after)

    lines = []
    lines.append("# Audit Comparison")
    lines.append("")
    lines.append("| | Before | After | Change |")
    lines.append("|---|---|---|---|")

    # Reliance
    br = bs.get("reliance_rating", "N/A")
    ar = as_.get("reliance_rating", "N/A")
    brs = bs.get("reliance_score")
    ars = as_.get("reliance_score")
    if br == "N/A" and ar == "N/A":
        lines.append("| Reliance | N/A | N/A | - |")
    elif br == "N/A" or ar == "N/A" or brs is None or ars is None:
        b_val = f"{brs:.2f}x ({br})" if brs is not None else "N/A"
        a_val = f"{ars:.2f}x ({ar})" if ars is not None else "N/A"
        lines.append(f"| Reliance | {b_val} | {a_val} | - |")
    else:
        lines.append(f"| Reliance | {brs:.2f}x ({br}) | {ars:.2f}x ({ar}) | {_rating_change(br, ar)} |")

    # Deployment
    bd = bs.get("deployment_score", 0)
    ad = as_.get("deployment_score", 0)
    bdr = bs.get("deployment_rating", "?")
    adr = as_.get("deployment_rating", "?")
    delta_d = ad - bd
    sign_d = "+" if delta_d >= 0 else ""
    lines.append(
        f"| **Deployment** | {bd:.2f} ({bdr}) | {ad:.2f} ({adr}) | {sign_d}{delta_d:.2f} {_rating_change(bdr, adr)} |"
    )

    # Stress
    bst = bs.get("stress_score", 0)
    ast = as_.get("stress_score", 0)
    bstr = bs.get("stress_rating", "?")
    astr = as_.get("stress_rating", "?")
    delta_s = ast - bst
    sign_s = "+" if delta_s >= 0 else ""
    lines.append(
        f"| **Stress** | {bst:.2f} ({bstr}) | {ast:.2f} ({astr}) | {sign_s}{delta_s:.2f} {_rating_change(bstr, astr)} |"
    )

    # Quadrant
    bq = bs.get("quadrant", "?")
    aq = as_.get("quadrant", "?")
    lines.append(f"| Quadrant | {_md_cell(bq)} | {_md_cell(aq)} | {_md_cell(_rating_change(bq, aq))} |")

    # Per-scenario breakdown
    b_rob = _robustness(before)
    a_rob = _robustness(after)
    b_scores = b_rob.get("per_scenario_scores", {})
    a_scores = a_rob.get("per_scenario_scores", {})

    all_scenarios = _ordered_scenarios(b_scores, a_scores)

    if all_scenarios:
        lines.append("")
        lines.append("## Per-Scenario Detail")
        lines.append("")
        lines.append("| Scenario | Category | Before | After | Change |")
        lines.append("|---|---|---|---|---|")

        for sc in all_scenarios:
            cat = "Deployment" if sc in _DEPLOY_SCENARIOS else "Stress"
            b_ret = b_scores.get(sc, {}).get("return_ratio")
            a_ret = a_scores.get(sc, {}).get("return_ratio")
            b_rmse = b_scores.get(sc, {}).get("rmse_ratio")
            a_rmse = a_scores.get(sc, {}).get("rmse_ratio")
            b_d = b_scores.get(sc, {}).get("cohens_d")
            a_d = a_scores.get(sc, {}).get("cohens_d")
            sc_name = _md_cell(sc)
            cat_name = _md_cell(cat)

            if b_ret is not None and a_ret is not None:
                delta = a_ret - b_ret
                sign = "+" if delta >= 0 else ""
                b_pct = b_ret * 100
                a_pct = a_ret * 100
                b_eff = f", d={b_d:+.2f}" if isinstance(b_d, (int, float)) else ""
                a_eff = f", d={a_d:+.2f}" if isinstance(a_d, (int, float)) else ""
                lines.append(
                    f"| {sc_name} | {cat_name} | {b_pct:.0f}% (RMSE {b_rmse:.2f}x{b_eff}) "
                    f"| {a_pct:.0f}% (RMSE {a_rmse:.2f}x{a_eff}) "
                    f"| {sign}{delta * 100:.0f}pp |"
                )
            else:
                lines.append(f"| {sc_name} | {cat_name} | - | - | - |")

    # Worst scenario changes
    b_dep_worst = b_rob.get("deployment", {}).get("worst_case", {})
    a_dep_worst = a_rob.get("deployment", {}).get("worst_case", {})
    b_str_worst = b_rob.get("stress", {}).get("worst_case", {})
    a_str_worst = a_rob.get("stress", {}).get("worst_case", {})

    lines.append("")
    lines.append("## Worst Scenarios")
    lines.append("")
    lines.append("| Category | Before | After |")
    lines.append("|---|---|---|")

    def _worst_str(worst):
        sc = worst.get("scenario")
        drop = worst.get("return_drop_pct", 0)
        if sc is None or drop <= 0:
            return "none (no drop)"
        return f"{sc} (drop {drop:.1f}%)"

    if b_dep_worst and a_dep_worst:
        lines.append(f"| Deployment | {_worst_str(b_dep_worst)} | {_worst_str(a_dep_worst)} |")

    if b_str_worst and a_str_worst:
        lines.append(f"| Stress | {_worst_str(b_str_worst)} | {_worst_str(a_str_worst)} |")

    md = "\n".join(lines) + "\n"

    if output_path is not None:
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(md, encoding="utf-8")

    return md


def generate_comparison_html(
    before_path: str | pathlib.Path,
    after_path: str | pathlib.Path,
    output_path: str | pathlib.Path | None = None,
) -> str:
    """Generate a rich HTML comparison report from two summary.json files.

    Args:
        before_path: Path to the 'before' summary.json
        after_path: Path to the 'after' summary.json
        output_path: Path to write comparison.html (optional)

    Returns:
        The HTML string.
    """
    before_path = pathlib.Path(before_path)
    after_path = pathlib.Path(after_path)

    before = _load_json(before_path)
    after = _load_json(after_path)

    bs = _summary(before)
    as_ = _summary(after)

    b_ver = before.get("_version", "")
    a_ver = after.get("_version", "")
    b_ts = before.get("_timestamp", "")
    a_ts = after.get("_timestamp", "")

    b_rob = _robustness(before)
    a_rob = _robustness(after)
    b_scores = b_rob.get("per_scenario_scores", {})
    a_scores = a_rob.get("per_scenario_scores", {})

    all_scenarios = _ordered_scenarios(b_scores, a_scores)

    scenario_labels = {
        "jitter": "Speed jitter",
        "delay": "Obs delay",
        "spike": "Mid-ep spike",
        "obs_noise": "Obs noise",
        "speed_5x": "5× speed",
    }

    def _rating_pill(rating, score=None):
        color = _rating_color(str(rating))
        label = str(rating) if rating != "N/A" else "N/A"
        score_str = f" ({score:.2f})" if score is not None and rating != "N/A" else ""
        return (
            f'<span style="background:{color};color:#fff;padding:2px 10px;'
            f'border-radius:12px;font-size:12px;font-weight:bold">'
            f"{_html_text(label)}{score_str}</span>"
        )

    def _delta_cell(b_val, a_val, is_pct=True):
        """HTML for a delta value with colored bar."""
        if b_val is None or a_val is None:
            return "—"
        delta = a_val - b_val
        if abs(delta) < 0.001:
            return '<span style="color:#999">±0</span>'
        color = "#28a745" if delta > 0 else "#dc3545"
        sign = "+" if delta >= 0 else ""
        if is_pct:
            val_str = f"{sign}{delta * 100:.0f}pp"
        else:
            val_str = f"{sign}{delta:.2f}"
        bar_w = max(3, min(60, abs(delta) * 200))
        return (
            f'<span style="display:inline-block;width:{bar_w:.0f}px;height:8px;'
            f"background:{color};border-radius:2px;vertical-align:middle;"
            f'margin-right:6px"></span>'
            f'<span style="color:{color};font-weight:bold">{val_str}</span>'
        )

    # Build scenario rows
    sc_rows = ""
    for sc in all_scenarios:
        cat = "Deployment" if sc in _DEPLOY_SCENARIOS else "Stress"
        cat_color = "#0d6efd" if cat == "Deployment" else "#6f42c1"
        b_ret = b_scores.get(sc, {}).get("return_ratio")
        a_ret = a_scores.get(sc, {}).get("return_ratio")
        b_d = b_scores.get(sc, {}).get("cohens_d")
        a_d = a_scores.get(sc, {}).get("cohens_d")
        b_pct = f"{b_ret * 100:.0f}%" if b_ret is not None else "—"
        a_pct = f"{a_ret * 100:.0f}%" if a_ret is not None else "—"
        if isinstance(b_d, (int, float)):
            b_pct += f' <span style="color:#888;font-size:12px">(d={b_d:+.2f})</span>'
        if isinstance(a_d, (int, float)):
            a_pct += f' <span style="color:#888;font-size:12px">(d={a_d:+.2f})</span>'
        delta_html = _delta_cell(b_ret, a_ret, is_pct=True)
        label = _html_text(scenario_labels.get(sc, sc))
        cat_label = _html_text(cat)
        b_sig = b_scores.get(sc, {}).get("significant", False)
        a_sig = a_scores.get(sc, {}).get("significant", False)
        sig_mark = " *" if (b_sig or a_sig) else ""
        sc_rows += (
            f"<tr>"
            f"<td><strong>{label}</strong>{sig_mark}</td>"
            f'<td><span style="background:{cat_color};color:#fff;padding:1px 8px;'
            f'border-radius:10px;font-size:11px">{cat_label}</span></td>'
            f"<td>{b_pct}</td>"
            f"<td>{a_pct}</td>"
            f"<td>{delta_html}</td>"
            f"</tr>\n"
        )

    # Badge cards helper
    def _badge_card(summary, side_label, ver, ts, card_class):
        dep_r = summary.get("deployment_rating", "?")
        dep_s = summary.get("deployment_score")
        str_r = summary.get("stress_rating", "?")
        str_s = summary.get("stress_score")
        rel_r = summary.get("reliance_rating", "N/A")
        rel_s = summary.get("reliance_score")
        quad = summary.get("quadrant", "?")
        meta = ""
        if ver:
            meta += f"v{_html_text(ver)}"
        if ts:
            meta += (" · " if meta else "") + _html_text(ts[:19].replace("T", " "))
        meta_html = f'<div style="font-size:11px;color:#999;margin-bottom:10px">{meta}</div>' if meta else ""
        rel_html = _rating_pill(rel_r) if rel_r == "N/A" else _rating_pill(rel_r, rel_s)
        side_label_safe = _html_text(side_label)
        quadrant_safe = _html_text(_quadrant_label(str(quad)))
        return f"""
        <div class="card {card_class}">
          <div class="card-title">{side_label_safe}</div>
          {meta_html}
          <div style="margin:6px 0">
            <span style="font-size:12px;color:#666;display:inline-block;width:100px">Reliance</span>
            {rel_html}
          </div>
          <div style="margin:6px 0">
            <span style="font-size:12px;color:#666;display:inline-block;width:100px">Deployment</span>
            {_rating_pill(dep_r, dep_s)}
          </div>
          <div style="margin:6px 0">
            <span style="font-size:12px;color:#666;display:inline-block;width:100px">Stress</span>
            {_rating_pill(str_r, str_s)}
          </div>
          <div style="margin-top:10px;font-size:12px;color:#555">
            Quadrant: <strong>{quadrant_safe}</strong>
          </div>
        </div>"""

    before_card = _badge_card(bs, "BEFORE", b_ver, b_ts, "before")
    after_card = _badge_card(as_, "AFTER", a_ver, a_ts, "after")

    # Generate the comparison chart
    chart_b64 = _make_comparison_chart(b_rob, a_rob)
    chart_html = ""
    if chart_b64:
        chart_html = (
            f'<div class="chart-section">'
            f'<img src="data:image/png;base64,{chart_b64}" '
            f'alt="Before vs After Comparison" '
            f'style="max-width:100%;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1);">'
            f"</div>"
        )

    before_seed = _seed_sweep_payload(before)
    after_seed = _seed_sweep_payload(after)
    seed_chart_html = ""
    if before_seed and after_seed:
        seed_chart_b64 = _make_seed_variance_chart(before_seed, after_seed)
        if seed_chart_b64:
            b_n = int(before_seed.get("n_seeds", 0) or 0)
            a_n = int(after_seed.get("n_seeds", 0) or 0)
            b_pass = before_seed.get("aggregate", {}).get("pass_rates", {})
            a_pass = after_seed.get("aggregate", {}).get("pass_rates", {})
            b_dep_rate = float(b_pass.get("deployment", 0.0))
            b_str_rate = float(b_pass.get("stress", 0.0))
            a_dep_rate = float(a_pass.get("deployment", 0.0))
            a_str_rate = float(a_pass.get("stress", 0.0))
            seed_chart_html = (
                "<h2>Multi-Seed Variance</h2>"
                '<div class="summary-line">'
                f"Before: n={b_n}, deployment pass-rate {b_dep_rate:.0%}, stress pass-rate {b_str_rate:.0%}"
                "&nbsp;→&nbsp;"
                f"After: n={a_n}, deployment pass-rate {a_dep_rate:.0%}, stress pass-rate {a_str_rate:.0%}"
                "</div>"
                '<div class="chart-section">'
                f'<img src="data:image/png;base64,{seed_chart_b64}" '
                'alt="Multi-Seed Variance" '
                'style="max-width:100%;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1);">'
                "</div>"
            )

    # Overall delta line
    dep_delta = (as_.get("deployment_score", 0) or 0) - (bs.get("deployment_score", 0) or 0)
    dep_sign = "+" if dep_delta >= 0 else ""
    dep_color = "#28a745" if dep_delta > 0.05 else ("#dc3545" if dep_delta < -0.05 else "#999")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Audit Comparison — deltatau-audit</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      max-width: 860px; margin: 40px auto; padding: 0 20px; color: #333;
      line-height: 1.5;
    }}
    h1 {{ font-size: 22px; border-bottom: 2px solid #e0e0e0; padding-bottom: 8px;
           margin-bottom: 4px; }}
    h2 {{ font-size: 16px; color: #444; margin-top: 32px; margin-bottom: 12px; }}
    .meta {{ font-size: 12px; color: #999; margin-bottom: 20px; }}
    .cards {{ display: flex; gap: 16px; margin: 20px 0; align-items: stretch; }}
    .arrow {{ align-self: center; font-size: 28px; color: #bbb; padding: 0 4px; }}
    .card {{ flex: 1; border: 1px solid #ddd; border-radius: 8px; padding: 16px 20px; }}
    .card.before {{ background: #fafafa; }}
    .card.after  {{ background: #f0fff4; border-color: #b2dfdb; }}
    .card-title  {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px;
                    color: #888; font-weight: bold; margin-bottom: 10px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; margin-top: 12px; }}
    th {{ background: #f5f5f5; text-align: left; padding: 9px 12px;
          border-bottom: 2px solid #ddd; font-size: 13px; }}
    td {{ padding: 9px 12px; border-bottom: 1px solid #eee; vertical-align: middle; }}
    tr:hover td {{ background: #fafafa; }}
    .summary-line {{ background: #fffde7; border-left: 4px solid #f9a825;
                     padding: 10px 16px; border-radius: 0 4px 4px 0;
                     margin: 16px 0; font-size: 14px; }}
    .chart-section {{ margin: 20px 0; text-align: center; }}
    .footer {{ margin-top: 40px; font-size: 11px; color: #bbb;
               border-top: 1px solid #eee; padding-top: 12px; }}
    * {{ font-size: inherit; }}
  </style>
</head>
<body>
  <h1>Audit Comparison</h1>
  <div class="meta">
    Generated by <a href="https://github.com/maruyamakoju/deltatau-audit">deltatau-audit</a>
    &nbsp;·&nbsp;
    Deployment change: <span style="color:{dep_color};font-weight:bold">{dep_sign}{dep_delta:.2f}</span>
  </div>

  <h2>Summary</h2>
  <div class="cards">
    {before_card}
    <div class="arrow">→</div>
    {after_card}
  </div>

  {chart_html}

  {seed_chart_html}

  <h2>Per-Scenario Breakdown</h2>
  <p style="font-size:12px;color:#888">* = statistically significant drop (95% bootstrap CI)</p>
  <table>
    <thead>
      <tr>
        <th>Scenario</th><th>Category</th>
        <th>Before</th><th>After</th><th>Change</th>
      </tr>
    </thead>
    <tbody>
      {sc_rows}
    </tbody>
  </table>

  <div class="footer">
    deltatau-audit — Time Robustness Audit for RL agents
  </div>
</body>
</html>
"""

    if output_path is not None:
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")

    return html
