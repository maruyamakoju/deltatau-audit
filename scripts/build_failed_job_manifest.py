#!/usr/bin/env python3
"""Build a focused bench manifest for failed or selected job IDs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from submission_health import build_failed_job_subset_manifest


def _resolve_repo_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a focused bench manifest for failed or selected job IDs.",
    )
    parser.add_argument("--manifest", required=True, help="Source bench manifest path.")
    parser.add_argument("--output-root", required=True, help="Bench output root with bench_summary.json.")
    parser.add_argument("--out-manifest", required=True, help="Destination manifest path.")
    parser.add_argument(
        "--job-id",
        action="append",
        default=[],
        help="Specific failed job id to include. Repeat to select multiple jobs. If omitted, failed job ids are read from bench_summary.json.",
    )
    parser.add_argument(
        "--manifest-name",
        default="",
        help="Optional manifest name override.",
    )
    parser.add_argument(
        "--description",
        default="",
        help="Optional manifest description override.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output_dir override for the focused manifest.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    manifest_path = _resolve_repo_path(args.manifest)
    output_root = _resolve_repo_path(args.output_root)
    out_manifest = _resolve_repo_path(args.out_manifest)

    result = build_failed_job_subset_manifest(
        manifest_path,
        output_root,
        manifest_name=args.manifest_name or None,
        description=args.description or None,
        output_dir=args.output_dir or None,
        job_ids=args.job_id or None,
    )

    selected_count = int(result.get("selected_count", 0))
    missing_ids = result.get("missing_job_ids", [])
    manifest = result.get("manifest")
    if not isinstance(manifest, dict):
        print("failed: helper did not return a manifest payload", file=sys.stderr)
        return 1
    if selected_count <= 0:
        print("failed: no job ids matched the source manifest", file=sys.stderr)
        return 1
    if isinstance(missing_ids, list) and missing_ids:
        print(f"failed: {len(missing_ids)} requested job ids were not found", file=sys.stderr)
        for job_id in missing_ids:
            print(f"  missing_job_id: {job_id}", file=sys.stderr)
        return 1

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    print(f"source_manifest: {manifest_path}")
    print(f"output_root: {output_root}")
    print(f"out_manifest: {out_manifest}")
    print(f"selected_jobs: {selected_count}")
    for job_id in result.get("selected_job_ids", []):
        print(f"  job_id: {job_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
