"""Quick A/B comparator for two autonomous-lab runs.

Reads experiment journals + LLM-call journals from two --out dirs and
prints a side-by-side summary: mean/best composite, frontier selection
distribution, token usage, wall-clock per cycle.

Usage:
    python experiments/compare_ab_labs.py \
        --a research_runs_claude_ab --a-name claude \
        --b research_runs_codex_ab  --b-name codex
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  ! failed to load {path}: {exc}")
        return None


def _extract_cycle_composites(experiment_journal: Dict[str, Any], lab_start_cycle: int) -> List[float]:
    """Pull composite scores from records at or after the lab's start cycle."""
    scores: List[float] = []
    for rec in experiment_journal.get("records", []):
        if rec.get("cycle", -1) < lab_start_cycle:
            continue
        m = rec.get("metrics") or {}
        c = m.get("composite")
        if isinstance(c, (int, float)):
            scores.append(float(c))
    return scores


def _frontier_counts(experiment_journal: Dict[str, Any], lab_start_cycle: int) -> Counter:
    counter: Counter = Counter()
    for rec in experiment_journal.get("records", []):
        if rec.get("cycle", -1) < lab_start_cycle:
            continue
        counter[str(rec.get("frontier", "?"))] += 1
    return counter


def _summarize(name: str, out_dir: Path) -> Dict[str, Any]:
    exp_journal = _load(out_dir / "journal.json") or {}
    # Lab journal could be codex_lab_journal.json or claude_lab_journal.json
    lab_journal = None
    for candidate in ("claude_lab_journal.json", "codex_lab_journal.json"):
        candidate_path = out_dir / candidate
        lab_journal = _load(candidate_path)
        if lab_journal is not None:
            break

    lab_start_cycle = 0
    if lab_journal:
        recent_cycles = lab_journal.get("recent_cycles") or []
        if recent_cycles:
            lab_start_cycle = min(int(c.get("cycle", 0)) for c in recent_cycles)

    scores = _extract_cycle_composites(exp_journal, lab_start_cycle)
    frontiers = _frontier_counts(exp_journal, lab_start_cycle)

    total_calls = int((lab_journal or {}).get("total_codex_calls", 0))
    total_usage = (lab_journal or {}).get("total_usage") or {}

    recent = (lab_journal or {}).get("recent_cycles") or []
    strategy_durs: List[float] = []
    critique_durs: List[float] = []
    # Durations live in recent_calls keyed by label; best effort
    for call in (lab_journal or {}).get("recent_calls") or []:
        dur = call.get("duration_sec")
        if isinstance(dur, (int, float)):
            lbl = str(call.get("label", ""))
            if "strategy" in lbl:
                strategy_durs.append(float(dur))
            elif "critique" in lbl:
                critique_durs.append(float(dur))

    return {
        "name": name,
        "out_dir": str(out_dir),
        "lab_start_cycle": lab_start_cycle,
        "cycles": len(recent),
        "composites": scores,
        "composite_mean": statistics.mean(scores) if scores else None,
        "composite_max": max(scores) if scores else None,
        "composite_min": min(scores) if scores else None,
        "composite_stdev": statistics.stdev(scores) if len(scores) > 1 else None,
        "frontier_counts": dict(frontiers),
        "total_llm_calls": total_calls,
        "total_usage": total_usage,
        "strategy_mean_sec": statistics.mean(strategy_durs) if strategy_durs else None,
        "critique_mean_sec": statistics.mean(critique_durs) if critique_durs else None,
    }


def _fmt(val: Any, spec: str = "") -> str:
    if val is None:
        return "—"
    if spec and isinstance(val, (int, float)):
        return format(val, spec)
    return str(val)


def _print_report(a: Dict[str, Any], b: Dict[str, Any]) -> None:
    name_a, name_b = a["name"], b["name"]
    print(f"\n{'=' * 68}")
    print(f"  A/B comparison: {name_a}  vs  {name_b}")
    print(f"{'=' * 68}")

    rows = [
        ("cycles (this session)", _fmt(a["cycles"]), _fmt(b["cycles"])),
        ("composite mean", _fmt(a["composite_mean"], ".4f"), _fmt(b["composite_mean"], ".4f")),
        ("composite max", _fmt(a["composite_max"], ".4f"), _fmt(b["composite_max"], ".4f")),
        ("composite min", _fmt(a["composite_min"], ".4f"), _fmt(b["composite_min"], ".4f")),
        ("composite stdev", _fmt(a["composite_stdev"], ".4f"), _fmt(b["composite_stdev"], ".4f")),
        ("LLM calls", _fmt(a["total_llm_calls"]), _fmt(b["total_llm_calls"])),
        ("strategy mean (s)", _fmt(a["strategy_mean_sec"], ".1f"), _fmt(b["strategy_mean_sec"], ".1f")),
        ("critique mean (s)", _fmt(a["critique_mean_sec"], ".1f"), _fmt(b["critique_mean_sec"], ".1f")),
        ("total input tokens", _fmt(a["total_usage"].get("input_tokens"), ",d"),
                                _fmt(b["total_usage"].get("input_tokens"), ",d")),
        ("total cached tokens", _fmt(a["total_usage"].get("cached_input_tokens"), ",d"),
                                 _fmt(b["total_usage"].get("cached_input_tokens"), ",d")),
        ("total output tokens", _fmt(a["total_usage"].get("output_tokens"), ",d"),
                                 _fmt(b["total_usage"].get("output_tokens"), ",d")),
    ]
    col1 = max(len(r[0]) for r in rows)
    col2 = max(len(r[1]) for r in rows + [("_", name_a, name_b)])
    col3 = max(len(r[2]) for r in rows + [("_", name_a, name_b)])
    print(f"  {'metric':<{col1}}  {name_a:>{col2}}  {name_b:>{col3}}")
    print(f"  {'-' * col1}  {'-' * col2}  {'-' * col3}")
    for label, va, vb in rows:
        print(f"  {label:<{col1}}  {va:>{col2}}  {vb:>{col3}}")

    print("\n  Frontier selection:")
    all_frontiers = sorted(set(a["frontier_counts"]) | set(b["frontier_counts"]))
    for f in all_frontiers:
        ca = a["frontier_counts"].get(f, 0)
        cb = b["frontier_counts"].get(f, 0)
        print(f"    {f:<35}  {name_a}={ca}  {name_b}={cb}")

    if a["composites"] and b["composites"]:
        delta = (a["composite_mean"] or 0) - (b["composite_mean"] or 0)
        print(f"\n  Delta (mean): {name_a} - {name_b} = {delta:+.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, help="Output dir for run A")
    parser.add_argument("--b", required=True, help="Output dir for run B")
    parser.add_argument("--a-name", default="A")
    parser.add_argument("--b-name", default="B")
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    a = _summarize(args.a_name, Path(args.a))
    b = _summarize(args.b_name, Path(args.b))

    _print_report(a, b)

    if args.json_out:
        # Drop the raw composites list to keep the artifact compact.
        for d in (a, b):
            d.pop("composites", None)
        Path(args.json_out).write_text(
            json.dumps({"a": a, "b": b}, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
