"""Tests for scripts/rewrite_manifest_output_root.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "rewrite_manifest_output_root.py"
    spec = importlib.util.spec_from_file_location("rewrite_manifest_output_root", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_main_rewrites_output_dir_and_job_outs(tmp_path: Path):
    m = _load_module()

    manifest = tmp_path / "bench" / "mini.yaml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "name: mini\n"
        "output_dir: bench_runs/mini\n"
        "jobs:\n"
        "  - name: job_a\n"
        "    command: audit-sb3\n"
        "    args:\n"
        "      out: bench_runs/mini/a\n"
        "  - name: job_b\n"
        "    command: audit-sb3\n"
        "    args:\n"
        "      out: bench_runs/mini/sub/b\n",
        encoding="utf-8",
    )
    out_manifest = tmp_path / "rewritten.yaml"

    rc = m.main(
        [
            "--manifest",
            str(manifest),
            "--out-manifest",
            str(out_manifest),
            "--output-dir",
            "bench_runs/mini_fresh",
        ]
    )

    assert rc == 0
    rewritten = yaml.safe_load(out_manifest.read_text(encoding="utf-8"))
    assert rewritten["output_dir"] == "bench_runs/mini_fresh"
    assert rewritten["source_output_dir"] == "bench_runs/mini"
    assert rewritten["jobs"][0]["args"]["out"] == "bench_runs/mini_fresh/a"
    assert rewritten["jobs"][1]["args"]["out"] == "bench_runs/mini_fresh/sub/b"
