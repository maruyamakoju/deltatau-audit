#!/usr/bin/env python3
"""Merge a focused bench_summary into a preserved full bench_summary."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deltatau_audit.bench import write_submission_tables_for_summary


def _resolve_repo_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _load_summary(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"invalid bench summary: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"bench summary is not a JSON object: {path}")
    return payload


def _recompute_counts(jobs: list[dict[str, object]]) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "skipped": 0}
    for job in jobs:
        status = str(job.get("status", "")).strip().lower()
        if status == "passed":
            counts["passed"] += 1
        elif status == "failed":
            counts["failed"] += 1
        else:
            counts["skipped"] += 1
    return counts


def merge_bench_summaries(
    *,
    base_summary: dict[str, object],
    patch_summary: dict[str, object],
    output_root: Path,
    base_summary_path: Path,
    patch_summary_path: Path,
) -> dict[str, object]:
    base_jobs_raw = base_summary.get("jobs")
    patch_jobs_raw = patch_summary.get("jobs")
    if not isinstance(base_jobs_raw, list):
        raise ValueError("base summary missing jobs list")
    if not isinstance(patch_jobs_raw, list):
        raise ValueError("patch summary missing jobs list")

    merged = copy.deepcopy(base_summary)
    base_jobs = [job for job in base_jobs_raw if isinstance(job, dict)]
    patch_jobs = [job for job in patch_jobs_raw if isinstance(job, dict)]
    index: dict[str, int] = {}
    merged_jobs: list[dict[str, object]] = []
    for idx, job in enumerate(base_jobs):
        job_id = job.get("id")
        if isinstance(job_id, str) and job_id.strip():
            index[job_id] = idx
        merged_jobs.append(copy.deepcopy(job))

    appended = 0
    replaced = 0
    for patch_job in patch_jobs:
        job_id = patch_job.get("id")
        if isinstance(job_id, str) and job_id.strip() and job_id in index:
            merged_jobs[index[job_id]] = copy.deepcopy(patch_job)
            replaced += 1
        else:
            merged_jobs.append(copy.deepcopy(patch_job))
            appended += 1

    counts = _recompute_counts(merged_jobs)
    merged["jobs"] = merged_jobs
    merged["output_root"] = str(output_root)
    merged["counts"] = counts
    merged["status"] = "failed" if counts["failed"] > 0 else "passed"
    merged["finished_at"] = patch_summary.get("finished_at", time.time())
    merged["merged_patch"] = {
        "base_summary": str(base_summary_path),
        "patch_summary": str(patch_summary_path),
        "merged_at_utc": datetime.now(timezone.utc).isoformat(),
        "replaced_jobs": replaced,
        "appended_jobs": appended,
    }
    artifacts = write_submission_tables_for_summary(merged, output_root)
    merged["artifacts"] = artifacts
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge a focused bench summary into a preserved full bench summary.")
    parser.add_argument("--base-summary", required=True, help="Full bench_summary.json path to update.")
    parser.add_argument("--patch-summary", required=True, help="Focused bench_summary.json path with updated jobs.")
    parser.add_argument("--output-root", required=True, help="Final output root for merged bench_summary.json and submission tables.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    base_summary_path = _resolve_repo_path(args.base_summary)
    patch_summary_path = _resolve_repo_path(args.patch_summary)
    output_root = _resolve_repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not base_summary_path.exists():
        print(f"failed: base summary missing: {base_summary_path}", file=sys.stderr)
        return 1
    if not patch_summary_path.exists():
        print(f"failed: patch summary missing: {patch_summary_path}", file=sys.stderr)
        return 1

    merged = merge_bench_summaries(
        base_summary=_load_summary(base_summary_path),
        patch_summary=_load_summary(patch_summary_path),
        output_root=output_root,
        base_summary_path=base_summary_path,
        patch_summary_path=patch_summary_path,
    )
    out_path = output_root / "bench_summary.json"
    out_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    counts = merged.get("counts", {})
    print(f"base_summary: {base_summary_path}")
    print(f"patch_summary: {patch_summary_path}")
    print(f"output_summary: {out_path}")
    print(f"status: {merged.get('status')}")
    print(
        "counts: "
        f"passed={counts.get('passed', 0)} "
        f"failed={counts.get('failed', 0)} "
        f"skipped={counts.get('skipped', 0)}"
    )
    patch_meta = merged.get("merged_patch", {})
    if isinstance(patch_meta, dict):
        print(
            "merged_jobs: "
            f"replaced={patch_meta.get('replaced_jobs', 0)} "
            f"appended={patch_meta.get('appended_jobs', 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
