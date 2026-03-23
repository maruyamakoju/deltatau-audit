#!/usr/bin/env python3
"""Monitor progress of a running bench manifest execution."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import yaml


def _load_manifest(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Manifest top-level must be a mapping")
    return data


def _expand_jobs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = manifest.get("jobs", [])
    if not isinstance(jobs, list):
        return []

    expanded: list[dict[str, Any]] = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        args = job.get("args", {})
        if not isinstance(args, dict):
            args = {}
        matrix = job.get("matrix", {})
        if not isinstance(matrix, dict) or not matrix:
            expanded.append({"name": str(job.get("name", "job")), "args": dict(args), "vars": {}})
            continue

        keys = list(matrix.keys())
        values = []
        for key in keys:
            raw = matrix.get(key, [])
            if not isinstance(raw, list) or not raw:
                values.append([None])
            else:
                values.append(raw)
        for combo in itertools.product(*values):
            vars_map = {keys[idx]: combo[idx] for idx in range(len(keys))}
            combo_args: dict[str, Any] = {}
            for arg_k, arg_v in args.items():
                if isinstance(arg_v, str):
                    try:
                        combo_args[arg_k] = arg_v.format(**vars_map)
                    except Exception:
                        combo_args[arg_k] = arg_v
                else:
                    combo_args[arg_k] = arg_v
            expanded.append(
                {"name": str(job.get("name", "job")), "args": combo_args, "vars": vars_map}
            )
    return expanded


def _collect_summary_paths(expanded_jobs: list[dict[str, Any]], repo_root: Path) -> list[Path]:
    paths: list[Path] = []
    for job in expanded_jobs:
        args = job.get("args", {})
        if not isinstance(args, dict):
            continue
        out = args.get("out")
        if isinstance(out, str) and out.strip():
            out_path = Path(out)
            if not out_path.is_absolute():
                out_path = repo_root / out_path
            paths.append(out_path / "summary.json")
    return paths


def _read_bench_counts(output_root: Path) -> dict[str, int] | None:
    summary_path = output_root / "bench_summary.json"
    if not summary_path.exists():
        return None
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    counts = data.get("counts")
    if isinstance(counts, dict):
        out: dict[str, int] = {}
        for key in ("passed", "failed", "skipped"):
            value = counts.get(key, 0)
            out[key] = int(value) if isinstance(value, (int, float)) else 0
        return out
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Show bench manifest progress from artifact files")
    parser.add_argument("--manifest", type=str, required=True, help="Path to benchmark manifest")
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Bench output root (default: manifest output_dir, relative to repo root)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    manifest = _load_manifest(manifest_path)
    manifest_out = manifest.get("output_dir")
    if args.output_root:
        output_root = Path(args.output_root).resolve()
    elif isinstance(manifest_out, str) and manifest_out.strip():
        output_root = (manifest_path.parent.parent / manifest_out).resolve()
    else:
        output_root = (manifest_path.parent.parent / "bench_runs").resolve()

    expanded = _expand_jobs(manifest)
    repo_root = manifest_path.parent.parent
    summary_paths = _collect_summary_paths(expanded, repo_root)

    done = sum(1 for p in summary_paths if p.exists())
    total = len(summary_paths)
    pct = (100.0 * done / total) if total else 0.0

    print(f"manifest:    {manifest_path}")
    print(f"output_root: {output_root}")
    print(f"jobs_total:  {total}")
    print(f"jobs_done:   {done}")
    print(f"progress:    {pct:.1f}%")

    counts = _read_bench_counts(output_root)
    if counts is not None:
        print("bench_summary counts:")
        print(f"  passed:  {counts.get('passed', 0)}")
        print(f"  failed:  {counts.get('failed', 0)}")
        print(f"  skipped: {counts.get('skipped', 0)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
