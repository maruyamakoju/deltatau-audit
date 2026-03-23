"""Summarize bench failure distributions and protocol gaps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submission_health import bench_quality_analysis, compare_bench_quality


def _metric_text(metric: dict[str, Any], *, precision: int = 4) -> str:
    count = int(metric.get("count") or 0)
    if count <= 0:
        return "n=0"
    mean = metric.get("mean")
    median = metric.get("median")
    min_value = metric.get("min")
    max_value = metric.get("max")
    return (
        f"n={count} mean={mean:.{precision}f} median={median:.{precision}f} "
        f"min={min_value:.{precision}f} max={max_value:.{precision}f}"
    )


def _render_analysis_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = ["# Bench Failure Analysis", ""]

    benches = report.get("benches")
    if isinstance(benches, list):
        for item in benches:
            if not isinstance(item, dict):
                continue
            lines.append(f"## {Path(str(item.get('source', 'bench'))).name}")
            lines.append(f"- Summary: `{item.get('summary_path')}`")
            lines.append(f"- Protocol: `{item.get('protocol')}`")
            lines.append(f"- Status: `{item.get('status')}`")
            counts = item.get("counts") if isinstance(item.get("counts"), dict) else {}
            lines.append(
                "- Counts: "
                f"passed={int(counts.get('passed') or 0)} "
                f"failed={int(counts.get('failed') or 0)} "
                f"skipped={int(counts.get('skipped') or 0)}"
            )
            lines.append(
                "- Failures: "
                f"ci_gate={int(item.get('ci_gate_failures') or 0)} "
                f"deployment={int(item.get('deployment_failures') or 0)} "
                f"stress={int(item.get('stress_failures') or 0)}"
            )
            signals = item.get("signals")
            if isinstance(signals, list) and signals:
                lines.append("- Signals:")
                for signal in signals:
                    lines.append(f"  - {signal}")
            variants = item.get("variants")
            if isinstance(variants, list) and variants:
                lines.append("")
                lines.append("| variant | jobs | ci_gate_failures | deployment_score | stress_score | worst_ci_lower | top diagnosis | top scenario |")
                lines.append("| --- | ---: | ---: | --- | --- | --- | --- | --- |")
                for variant in variants:
                    if not isinstance(variant, dict):
                        continue
                    top_diag = ""
                    diag_rows = variant.get("top_diagnosis_patterns")
                    if isinstance(diag_rows, list) and diag_rows:
                        top = diag_rows[0]
                        if isinstance(top, dict):
                            top_diag = f"{top.get('label')} ({top.get('count')})"
                    top_scenario = ""
                    scenario_rows = variant.get("top_worst_scenarios")
                    if isinstance(scenario_rows, list) and scenario_rows:
                        top = scenario_rows[0]
                        if isinstance(top, dict):
                            top_scenario = f"{top.get('label')} ({top.get('count')})"
                    lines.append(
                        "| "
                        f"{variant.get('variant')} | "
                        f"{int(variant.get('job_count') or 0)} | "
                        f"{int(variant.get('ci_gate_failures') or 0)} | "
                        f"{_metric_text(variant.get('deployment_score') or {}, precision=3)} | "
                        f"{_metric_text(variant.get('stress_score') or {}, precision=3)} | "
                        f"{_metric_text(variant.get('stress_worst_ci_lower') or {}, precision=3)} | "
                        f"{top_diag} | "
                        f"{top_scenario} |"
                    )
            lines.append("")

    comparisons = report.get("comparisons")
    if isinstance(comparisons, list) and comparisons:
        lines.append("## Comparisons")
        lines.append("")
        for comp in comparisons:
            if not isinstance(comp, dict):
                continue
            base_name = Path(str(comp.get("base_source", "base"))).name
            other_name = Path(str(comp.get("other_source", "other"))).name
            lines.append(f"### {base_name} -> {other_name}")
            lines.append(f"- Common jobs: {int(comp.get('common_jobs') or 0)}")
            lines.append(f"- Delta convention: `{comp.get('delta_direction')}`")
            lines.append(
                f"- Deployment delta: {_metric_text(comp.get('deployment_score_delta') or {}, precision=4)}"
            )
            lines.append(
                f"- Stress delta: {_metric_text(comp.get('stress_score_delta') or {}, precision=4)}"
            )
            lines.append(
                f"- Worst CI delta: {_metric_text(comp.get('stress_worst_ci_lower_delta') or {}, precision=4)}"
            )
            flips = comp.get("ci_gate_flips") if isinstance(comp.get("ci_gate_flips"), dict) else {}
            lines.append(
                "- CI gate flips: "
                f"improved={int(flips.get('improved') or 0)} "
                f"regressed={int(flips.get('regressed') or 0)} "
                f"unchanged_failed={int(flips.get('unchanged_failed') or 0)} "
                f"unchanged_passed={int(flips.get('unchanged_passed') or 0)}"
            )
            lines.append(
                "- Pattern changes: "
                f"diagnosis={int(comp.get('diagnosis_pattern_changes') or 0)} "
                f"scenario={int(comp.get('worst_scenario_changes') or 0)}"
            )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze bench failure distributions and protocol gaps.")
    parser.add_argument(
        "--bench",
        action="append",
        required=True,
        help="Bench output root or bench_summary.json path. Repeat to compare multiple benches.",
    )
    parser.add_argument("--json-out", type=str, default="", help="Optional path to write JSON analysis.")
    parser.add_argument("--markdown-out", type=str, default="", help="Optional path to write Markdown analysis.")
    args = parser.parse_args()

    bench_sources = [Path(item).expanduser().resolve() for item in args.bench]
    benches = [bench_quality_analysis(source) for source in bench_sources]
    comparisons = [
        compare_bench_quality(bench_sources[0], other_source)
        for other_source in bench_sources[1:]
    ]
    report = {
        "bench_count": len(bench_sources),
        "benches": benches,
        "comparisons": comparisons,
    }

    markdown = _render_analysis_markdown(report)
    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.markdown_out:
        _write_text(Path(args.markdown_out).expanduser().resolve(), markdown)
    if not args.markdown_out:
        print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
