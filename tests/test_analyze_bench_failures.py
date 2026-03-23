"""Tests for scripts/analyze_bench_failures.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "analyze_bench_failures.py"
    spec = importlib.util.spec_from_file_location("analyze_bench_failures", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_main_writes_json_and_markdown(monkeypatch, tmp_path: Path):
    m = _load_module()
    base_root = tmp_path / "bench_runs" / "base"
    other_root = tmp_path / "bench_runs" / "other"
    base_root.mkdir(parents=True, exist_ok=True)
    other_root.mkdir(parents=True, exist_ok=True)

    base_payload = {
        "status": "failed",
        "counts": {"passed": 0, "failed": 1, "skipped": 0},
        "protocol": {"forced": "research"},
        "jobs": [
            {
                "id": "job_a",
                "name": "baseline",
                "status": "failed",
                "result": {
                    "deployment_score": 0.2,
                    "deployment_rating": "FAIL",
                    "stress_score": 0.1,
                    "stress_rating": "FAIL",
                    "stress_worst_ci_lower": 0.09,
                    "stress_ci_gate_pass": False,
                    "diagnosis_pattern": "Pattern A",
                    "stress_worst_scenario": "speed_5x",
                },
            }
        ],
    }
    other_payload = {
        "status": "failed",
        "counts": {"passed": 0, "failed": 1, "skipped": 0},
        "protocol": {"forced": "paper"},
        "jobs": [
            {
                "id": "job_a",
                "name": "baseline",
                "status": "failed",
                "result": {
                    "deployment_score": 0.1,
                    "deployment_rating": "FAIL",
                    "stress_score": 0.05,
                    "stress_rating": "FAIL",
                    "stress_worst_ci_lower": 0.04,
                    "stress_ci_gate_pass": False,
                    "diagnosis_pattern": "Pattern B",
                    "stress_worst_scenario": "adversarial_jitter",
                },
            }
        ],
    }
    (base_root / "bench_summary.json").write_text(json.dumps(base_payload), encoding="utf-8")
    (other_root / "bench_summary.json").write_text(json.dumps(other_payload), encoding="utf-8")

    json_out = tmp_path / "analysis" / "report.json"
    md_out = tmp_path / "analysis" / "report.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_bench_failures.py",
            "--bench",
            str(base_root),
            "--bench",
            str(other_root),
            "--json-out",
            str(json_out),
            "--markdown-out",
            str(md_out),
        ],
    )

    assert m.main() == 0
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["bench_count"] == 2
    assert len(payload["benches"]) == 2
    assert payload["comparisons"][0]["common_jobs"] == 1
    markdown = md_out.read_text(encoding="utf-8")
    assert "# Bench Failure Analysis" in markdown
    assert "## base" in markdown
    assert "### base -> other" in markdown
