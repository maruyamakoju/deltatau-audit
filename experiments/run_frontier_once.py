#!/usr/bin/env python3
"""Run a single frontier experiment in an isolated Python process."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import autonomous_research as base


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one frontier experiment in isolation")
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--frontier", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--journal", type=str, required=True)
    parser.add_argument("--params-json", type=str, required=True)
    parser.add_argument("--result-json", type=str, required=True)
    args = parser.parse_args(argv)

    params_path = Path(args.params_json).resolve()
    result_path = Path(args.result_json).resolve()
    out_root = Path(args.out).resolve()
    journal_path = Path(args.journal).resolve()

    params = json.loads(params_path.read_text(encoding="utf-8"))
    journal = base.ResearchJournal.load(journal_path)
    record = base.run_frontier_once(
        cycle=args.cycle,
        frontier_name=args.frontier,
        params=params,
        journal=journal,
        out_root=out_root,
    )

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(asdict(record), indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
