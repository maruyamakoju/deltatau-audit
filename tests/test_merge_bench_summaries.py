"""Tests for scripts/merge_bench_summaries.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "merge_bench_summaries.py"
    spec = importlib.util.spec_from_file_location("merge_bench_summaries", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_main_merges_patch_jobs_and_rewrites_counts(tmp_path: Path):
    m = _load_module()

    base_summary = tmp_path / "base" / "bench_summary.json"
    patch_summary = tmp_path / "patch" / "bench_summary.json"
    output_root = tmp_path / "merged"
    base_summary.parent.mkdir(parents=True, exist_ok=True)
    patch_summary.parent.mkdir(parents=True, exist_ok=True)

    base_summary.write_text(
        json.dumps(
            {
                "manifest": "bench/high_rigor_10seed_manifest.yaml",
                "output_root": str(output_root),
                "jobs": [
                    {"id": "job_a", "status": "passed", "summary_path": "a.json", "result": {"stress_ci_gate_pass": True}},
                    {"id": "job_b", "status": "failed", "summary_path": "b.json", "result": {"stress_ci_gate_pass": False}},
                    {"id": "job_c", "status": "passed", "summary_path": "c.json", "result": {"stress_ci_gate_pass": True}},
                ],
                "counts": {"passed": 2, "failed": 1, "skipped": 0},
                "status": "failed",
            }
        ),
        encoding="utf-8",
    )
    patch_summary.write_text(
        json.dumps(
            {
                "manifest": "_status_demo/repair_manifests/job_b.yaml",
                "output_root": str(tmp_path / "patch"),
                "jobs": [
                    {"id": "job_b", "status": "passed", "summary_path": "b_new.json", "result": {"stress_ci_gate_pass": True}},
                ],
                "counts": {"passed": 1, "failed": 0, "skipped": 0},
                "status": "passed",
                "finished_at": 123.0,
            }
        ),
        encoding="utf-8",
    )

    rc = m.main(
        [
            "--base-summary",
            str(base_summary),
            "--patch-summary",
            str(patch_summary),
            "--output-root",
            str(output_root),
        ]
    )

    assert rc == 0
    merged = json.loads((output_root / "bench_summary.json").read_text(encoding="utf-8"))
    assert merged["status"] == "passed"
    assert merged["counts"] == {"passed": 3, "failed": 0, "skipped": 0}
    assert [job["id"] for job in merged["jobs"]] == ["job_a", "job_b", "job_c"]
    assert merged["jobs"][1]["summary_path"] == "b_new.json"
    assert merged["artifacts"]["submission_rows"] == "3"
    assert (output_root / "submission_table.csv").exists()
    assert (output_root / "submission_table.md").exists()
