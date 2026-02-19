"""Diff two audit summary.json files and generate comparison reports."""

import json
import pathlib


# Scenarios that count toward the Deployment badge
_DEPLOY_SCENARIOS = {"jitter", "delay", "spike", "obs_noise"}


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
    return {
        "PASS": "#28a745",
        "MILD": "#5cb85c",
        "DEGRADED": "#f0ad4e",
        "FAIL": "#dc3545",
        "N/A": "#999999",
    }.get(rating, "#666666")


def generate_comparison(before_path, after_path, output_path=None):
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

    with open(before_path) as f:
        before = json.load(f)
    with open(after_path) as f:
        after = json.load(f)

    bs = before["summary"]
    as_ = after["summary"]

    lines = []
    lines.append("# Audit Comparison")
    lines.append("")
    lines.append(f"| | Before | After | Change |")
    lines.append(f"|---|---|---|---|")

    # Reliance
    br = bs.get("reliance_rating", "N/A")
    ar = as_.get("reliance_rating", "N/A")
    brs = bs.get("reliance_score")
    ars = as_.get("reliance_score")
    if br == "N/A" and ar == "N/A":
        lines.append(f"| Reliance | N/A | N/A | - |")
    elif br == "N/A" or ar == "N/A":
        b_val = f"{brs:.2f}x ({br})" if brs is not None else "N/A"
        a_val = f"{ars:.2f}x ({ar})" if ars is not None else "N/A"
        lines.append(f"| Reliance | {b_val} | {a_val} | - |")
    else:
        lines.append(f"| Reliance | {brs:.2f}x ({br}) | {ars:.2f}x ({ar}) "
                      f"| {_rating_change(br, ar)} |")

    # Deployment
    bd = bs.get("deployment_score", 0)
    ad = as_.get("deployment_score", 0)
    bdr = bs.get("deployment_rating", "?")
    adr = as_.get("deployment_rating", "?")
    delta_d = ad - bd
    sign_d = "+" if delta_d >= 0 else ""
    lines.append(f"| **Deployment** | {bd:.2f} ({bdr}) | {ad:.2f} ({adr}) "
                 f"| {sign_d}{delta_d:.2f} {_rating_change(bdr, adr)} |")

    # Stress
    bst = bs.get("stress_score", 0)
    ast = as_.get("stress_score", 0)
    bstr = bs.get("stress_rating", "?")
    astr = as_.get("stress_rating", "?")
    delta_s = ast - bst
    sign_s = "+" if delta_s >= 0 else ""
    lines.append(f"| **Stress** | {bst:.2f} ({bstr}) | {ast:.2f} ({astr}) "
                 f"| {sign_s}{delta_s:.2f} {_rating_change(bstr, astr)} |")

    # Quadrant
    bq = bs.get("quadrant", "?")
    aq = as_.get("quadrant", "?")
    lines.append(f"| Quadrant | {bq} | {aq} | {_rating_change(bq, aq)} |")

    # Per-scenario breakdown
    b_rob = before.get("robustness", {})
    a_rob = after.get("robustness", {})
    b_scores = b_rob.get("per_scenario_scores", {})
    a_scores = a_rob.get("per_scenario_scores", {})

    all_scenarios = sorted(
        set(list(b_scores.keys()) + list(a_scores.keys())),
        key=lambda s: (0 if s in _DEPLOY_SCENARIOS else 1, s),
    )

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

            if b_ret is not None and a_ret is not None:
                delta = a_ret - b_ret
                sign = "+" if delta >= 0 else ""
                b_pct = b_ret * 100
                a_pct = a_ret * 100
                lines.append(
                    f"| {sc} | {cat} | {b_pct:.0f}% (RMSE {b_rmse:.2f}x) "
                    f"| {a_pct:.0f}% (RMSE {a_rmse:.2f}x) "
                    f"| {sign}{delta * 100:.0f}pp |"
                )
            else:
                lines.append(f"| {sc} | {cat} | - | - | - |")

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
        lines.append(
            f"| Deployment | {_worst_str(b_dep_worst)} "
            f"| {_worst_str(a_dep_worst)} |"
        )

    if b_str_worst and a_str_worst:
        lines.append(
            f"| Stress | {_worst_str(b_str_worst)} "
            f"| {_worst_str(a_str_worst)} |"
        )

    md = "\n".join(lines) + "\n"

    if output_path:
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(md, encoding="utf-8")

    return md


def generate_comparison_html(before_path, after_path, output_path=None):
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

    with open(before_path) as f:
        before = json.load(f)
    with open(after_path) as f:
        after = json.load(f)

    bs = before["summary"]
    as_ = after["summary"]

    b_ver = before.get("_version", "")
    a_ver = after.get("_version", "")
    b_ts = before.get("_timestamp", "")
    a_ts = after.get("_timestamp", "")

    b_rob = before.get("robustness", {})
    a_rob = after.get("robustness", {})
    b_scores = b_rob.get("per_scenario_scores", {})
    a_scores = a_rob.get("per_scenario_scores", {})

    all_scenarios = sorted(
        set(list(b_scores.keys()) + list(a_scores.keys())),
        key=lambda s: (0 if s in _DEPLOY_SCENARIOS else 1, s),
    )

    scenario_labels = {
        "jitter": "Speed jitter",
        "delay": "Obs delay",
        "spike": "Mid-ep spike",
        "obs_noise": "Obs noise",
        "speed_5x": "5× speed",
    }

    def _rating_pill(rating, score=None):
        color = _rating_color(rating)
        label = rating if rating != "N/A" else "N/A"
        score_str = f" ({score:.2f})" if score is not None and rating != "N/A" else ""
        return (f'<span style="background:{color};color:#fff;padding:2px 10px;'
                f'border-radius:12px;font-size:12px;font-weight:bold">'
                f'{label}{score_str}</span>')

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
        bar_w = min(60, abs(delta) * 200)
        return (
            f'<span style="display:inline-block;width:{bar_w:.0f}px;height:8px;'
            f'background:{color};border-radius:2px;vertical-align:middle;'
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
        b_pct = f"{b_ret * 100:.0f}%" if b_ret is not None else "—"
        a_pct = f"{a_ret * 100:.0f}%" if a_ret is not None else "—"
        delta_html = _delta_cell(b_ret, a_ret, is_pct=True)
        label = scenario_labels.get(sc, sc)
        b_sig = b_scores.get(sc, {}).get("significant", False)
        a_sig = a_scores.get(sc, {}).get("significant", False)
        sig_mark = (" *" if (b_sig or a_sig) else "")
        sc_rows += (
            f'<tr>'
            f'<td><strong>{label}</strong>{sig_mark}</td>'
            f'<td><span style="background:{cat_color};color:#fff;padding:1px 8px;'
            f'border-radius:10px;font-size:11px">{cat}</span></td>'
            f'<td>{b_pct}</td>'
            f'<td>{a_pct}</td>'
            f'<td>{delta_html}</td>'
            f'</tr>\n'
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
            meta += f"v{ver}"
        if ts:
            meta += (" · " if meta else "") + ts[:19].replace("T", " ")
        meta_html = f'<div style="font-size:11px;color:#999;margin-bottom:10px">{meta}</div>' if meta else ""
        rel_html = (
            _rating_pill(rel_r) if rel_r == "N/A"
            else _rating_pill(rel_r, rel_s)
        )
        return f"""
        <div class="card {card_class}">
          <div class="card-title">{side_label}</div>
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
            Quadrant: <strong>{quad}</strong>
          </div>
        </div>"""

    before_card = _badge_card(bs, "BEFORE", b_ver, b_ts, "before")
    after_card = _badge_card(as_, "AFTER", a_ver, a_ts, "after")

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

    if output_path:
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")

    return html
