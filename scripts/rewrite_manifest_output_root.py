#!/usr/bin/env python3
"""Rewrite a bench manifest so args.out and output_dir point at a fresh root."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml


def _resolve_repo_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _rewrite_out_path(text: str, *, source_output_dir: str, target_output_dir: str) -> str:
    if text == source_output_dir:
        return target_output_dir
    prefix = source_output_dir.rstrip("/\\")
    if text.startswith(prefix + "/") or text.startswith(prefix + "\\"):
        return target_output_dir.rstrip("/\\") + text[len(prefix) :]
    return text


def rewrite_manifest_output_root(
    manifest: dict[str, object],
    *,
    target_output_dir: str,
    source_output_dir: str | None = None,
) -> tuple[dict[str, object], int]:
    rewritten = copy.deepcopy(manifest)
    if source_output_dir is None:
        raw = rewritten.get("output_dir")
        source_output_dir = raw.strip() if isinstance(raw, str) else ""
    if not source_output_dir:
        raise ValueError("source output_dir missing; pass --source-output-dir")

    job_count = 0
    jobs = rewritten.get("jobs")
    if isinstance(jobs, list):
        for job in jobs:
            if not isinstance(job, dict):
                continue
            args = job.get("args")
            if not isinstance(args, dict):
                continue
            out_value = args.get("out")
            if isinstance(out_value, str):
                new_out = _rewrite_out_path(
                    out_value,
                    source_output_dir=source_output_dir,
                    target_output_dir=target_output_dir,
                )
                if new_out != out_value:
                    args["out"] = new_out
                    job_count += 1

    rewritten["output_dir"] = target_output_dir
    rewritten["source_output_dir"] = source_output_dir
    return rewritten, job_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rewrite a bench manifest to a fresh output root.")
    parser.add_argument("--manifest", required=True, help="Source YAML/JSON manifest.")
    parser.add_argument("--out-manifest", required=True, help="Destination manifest path.")
    parser.add_argument("--output-dir", required=True, help="New output_dir and args.out root.")
    parser.add_argument(
        "--source-output-dir",
        default="",
        help="Optional source output_dir override. Defaults to manifest output_dir.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    manifest_path = _resolve_repo_path(args.manifest)
    out_manifest = _resolve_repo_path(args.out_manifest)
    try:
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"failed: could not read manifest: {exc}", file=sys.stderr)
        return 1
    if not isinstance(payload, dict):
        print("failed: manifest is not a mapping", file=sys.stderr)
        return 1

    try:
        rewritten, rewrites = rewrite_manifest_output_root(
            payload,
            target_output_dir=str(args.output_dir),
            source_output_dir=args.source_output_dir or None,
        )
    except Exception as exc:
        print(f"failed: {exc}", file=sys.stderr)
        return 1

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(yaml.safe_dump(rewritten, sort_keys=False), encoding="utf-8")
    print(f"source_manifest: {manifest_path}")
    print(f"out_manifest: {out_manifest}")
    print(f"output_dir: {args.output_dir}")
    print(f"rewritten_jobs: {rewrites}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
