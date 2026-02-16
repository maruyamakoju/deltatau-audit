"""Diff two audit summary.json files and generate comparison.md."""

import json
import pathlib


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
        key=lambda s: (0 if s in ("jitter", "delay", "spike") else 1, s),
    )

    if all_scenarios:
        lines.append("")
        lines.append("## Per-Scenario Detail")
        lines.append("")
        lines.append("| Scenario | Category | Before | After | Change |")
        lines.append("|---|---|---|---|---|")

        deploy_scenarios = {"jitter", "delay", "spike"}
        for sc in all_scenarios:
            cat = "Deployment" if sc in deploy_scenarios else "Stress"
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
