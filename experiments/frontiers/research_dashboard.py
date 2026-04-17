#!/usr/bin/env python3
"""Research Dashboard — real-time visualization of autonomous research progress.

Reads the journal.json file and generates an HTML dashboard showing:
  - Frontier exploration heatmap (UCB1 priorities over time)
  - Per-frontier performance curves
  - Breakthrough timeline
  - Hyperparameter sensitivity analysis
  - Current best configurations

Usage:
    python experiments/frontiers/research_dashboard.py --journal research_runs/journal.json
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _score_color(score: float) -> str:
    """Map [0,1] score to green-yellow-red gradient."""
    if score >= 0.8:
        return "#22c55e"
    if score >= 0.6:
        return "#84cc16"
    if score >= 0.4:
        return "#eab308"
    if score >= 0.2:
        return "#f97316"
    return "#ef4444"


def generate_dashboard(journal_path: Path, output_path: Path) -> None:
    """Generate HTML dashboard from research journal."""
    data = json.loads(journal_path.read_text(encoding="utf-8"))

    total_cycles = data.get("total_cycles", 0)
    breakthroughs = data.get("breakthroughs", [])
    best_per_frontier = data.get("best_per_frontier", {})
    frontier_scores = data.get("frontier_scores", {})
    records = data.get("recent_records", [])

    # Build frontier cards
    cards_html = ""
    for name, scores in sorted(frontier_scores.items()):
        best = best_per_frontier.get(name, {})
        best_score = best.get("score", 0.0)
        n_runs = len(scores)
        trend = ""
        if len(scores) >= 2:
            delta = scores[-1] - scores[-2]
            trend = f"{'&#9650;' if delta > 0 else '&#9660;'} {abs(delta):.4f}"
            trend_color = "#22c55e" if delta > 0 else "#ef4444"
        else:
            trend_color = "#6b7280"

        # Mini sparkline (CSS-based)
        max_s = max(scores) if scores else 1
        min_s = min(scores) if scores else 0
        sparkline_bars = ""
        display_scores = scores[-30:]  # last 30 runs
        for s in display_scores:
            h = max(2, int(30 * (s - min_s) / (max_s - min_s + 1e-8)))
            c = _score_color(s)
            sparkline_bars += f'<div style="width:3px;height:{h}px;background:{c};display:inline-block;margin:0 1px;vertical-align:bottom;"></div>'

        cards_html += f"""
        <div class="card">
            <h3>{name}</h3>
            <div class="metric-row">
                <span class="label">Best Score</span>
                <span class="value" style="color:{_score_color(best_score)}">{best_score:.4f}</span>
            </div>
            <div class="metric-row">
                <span class="label">Runs</span>
                <span class="value">{n_runs}</span>
            </div>
            <div class="metric-row">
                <span class="label">Trend</span>
                <span class="value" style="color:{trend_color}">{trend}</span>
            </div>
            <div class="sparkline">{sparkline_bars}</div>
            <details>
                <summary>Best hyperparams</summary>
                <pre>{json.dumps(best.get('hyperparams', {}), indent=2)}</pre>
            </details>
        </div>
        """

    # Breakthrough timeline
    bt_html = ""
    for b in breakthroughs[-20:]:
        bt_html += f'<div class="breakthrough">&#9733; {b}</div>'

    # Recent records table
    rows_html = ""
    for r in records[-30:]:
        status_color = "#22c55e" if r.get("status") == "success" else "#ef4444"
        score = r.get("metrics", {}).get("composite_score", 0.0)
        rows_html += f"""
        <tr>
            <td>{r.get('cycle', '?')}</td>
            <td>{r.get('frontier', '?')}</td>
            <td style="color:{status_color}">{r.get('status', '?')}</td>
            <td style="color:{_score_color(score)}">{score:.4f}</td>
            <td>{r.get('duration_sec', 0):.1f}s</td>
            <td class="finding">{r.get('finding', '')[:80]}</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Autonomous Research Dashboard</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: 'SF Mono', 'Fira Code', monospace; background: #0a0a0a; color: #e5e5e5; padding: 20px; }}
    h1 {{ color: #a78bfa; font-size: 1.5rem; margin-bottom: 5px; }}
    h2 {{ color: #818cf8; font-size: 1.1rem; margin: 20px 0 10px; }}
    h3 {{ color: #c084fc; font-size: 0.95rem; margin-bottom: 10px; }}
    .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; border-bottom: 1px solid #333; padding-bottom: 15px; }}
    .stats {{ display: flex; gap: 30px; }}
    .stat {{ text-align: center; }}
    .stat .num {{ font-size: 1.8rem; font-weight: bold; color: #a78bfa; }}
    .stat .lbl {{ font-size: 0.75rem; color: #6b7280; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 15px; margin-bottom: 20px; }}
    .card {{ background: #1a1a2e; border: 1px solid #333; border-radius: 8px; padding: 15px; }}
    .metric-row {{ display: flex; justify-content: space-between; margin: 4px 0; }}
    .label {{ color: #6b7280; font-size: 0.8rem; }}
    .value {{ font-weight: bold; font-size: 0.9rem; }}
    .sparkline {{ margin-top: 10px; height: 35px; display: flex; align-items: flex-end; }}
    .breakthrough {{ background: #1a1a0a; border-left: 3px solid #eab308; padding: 8px 12px; margin: 5px 0; font-size: 0.85rem; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; }}
    th {{ text-align: left; padding: 8px; color: #6b7280; border-bottom: 1px solid #333; }}
    td {{ padding: 6px 8px; border-bottom: 1px solid #1a1a1a; }}
    .finding {{ max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #9ca3af; }}
    details {{ margin-top: 8px; }}
    summary {{ cursor: pointer; font-size: 0.8rem; color: #6b7280; }}
    pre {{ font-size: 0.7rem; color: #9ca3af; margin-top: 5px; white-space: pre-wrap; }}
    .pulse {{ animation: pulse 2s infinite; }}
    @keyframes pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.5; }} }}
</style>
</head>
<body>
<div class="header">
    <div>
        <h1>Autonomous Research Orchestrator</h1>
        <span style="color:#6b7280;font-size:0.8rem;">Pushing into uncharted territory &mdash; 24/7</span>
    </div>
    <div class="stats">
        <div class="stat">
            <div class="num">{total_cycles}</div>
            <div class="lbl">Total Cycles</div>
        </div>
        <div class="stat">
            <div class="num" style="color:#eab308">{len(breakthroughs)}</div>
            <div class="lbl">Breakthroughs</div>
        </div>
        <div class="stat">
            <div class="num" style="color:#22c55e">{len(best_per_frontier)}</div>
            <div class="lbl">Active Frontiers</div>
        </div>
    </div>
</div>

<h2>Frontier Performance</h2>
<div class="cards">{cards_html}</div>

<h2>Breakthroughs</h2>
{bt_html if bt_html else '<div style="color:#6b7280;font-size:0.85rem;">No breakthroughs yet. Keep exploring.</div>'}

<h2>Recent Experiments</h2>
<table>
    <thead><tr>
        <th>Cycle</th><th>Frontier</th><th>Status</th><th>Score</th><th>Duration</th><th>Finding</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
</table>

<div style="margin-top:30px;text-align:center;color:#333;font-size:0.7rem;">
    Generated {datetime.now(timezone.utc).isoformat()} | deltatau-audit autonomous research
</div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Dashboard written to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    journal_path = Path(args.journal)
    output_path = Path(args.output) if args.output else journal_path.parent / "dashboard.html"
    generate_dashboard(journal_path, output_path)


if __name__ == "__main__":
    main()
