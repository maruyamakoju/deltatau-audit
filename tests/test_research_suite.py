"""Tests for research-suite orchestration helpers."""

from __future__ import annotations

import json
from pathlib import Path

from deltatau_audit.research_suite import (
    StageOutcome,
    _cached_stage_outcome,
    derive_recommendations,
)


def test_cached_stage_outcome_returns_none_without_summary(tmp_path: Path):
    stage_dir = tmp_path / "deliberative"
    stage_dir.mkdir()

    assert _cached_stage_outcome("deliberative", stage_dir) is None


def test_cached_stage_outcome_reads_scores(tmp_path: Path):
    stage_dir = tmp_path / "ltc"
    stage_dir.mkdir()
    (stage_dir / "summary.json").write_text(
        json.dumps({"summary": {"deployment_score": 0.91, "stress_score": 0.62}}),
        encoding="utf-8",
    )

    out = _cached_stage_outcome("ltc", stage_dir)

    assert out is not None
    assert out.status == "cached"
    assert out.deployment_score == 0.91
    assert out.stress_score == 0.62


def test_derive_recommendations_detects_failures_and_low_scores():
    outcomes = [
        StageOutcome(
            name="deliberative",
            status="failed",
            reason="RuntimeError: boom",
            deployment_score=None,
            stress_score=None,
            output_dir="x",
            duration_sec=1.0,
        ),
        StageOutcome(
            name="ltc",
            status="success",
            reason=None,
            deployment_score=0.50,
            stress_score=0.20,
            output_dir="y",
            duration_sec=1.0,
        ),
        StageOutcome(
            name="bridge",
            status="skipped",
            reason="missing prereq",
            deployment_score=None,
            stress_score=None,
            output_dir="z",
            duration_sec=0.0,
        ),
    ]

    recs = derive_recommendations(outcomes)
    joined = " ".join(recs)

    assert "Fix failed stages first" in joined
    assert "Deployment robustness below threshold" in joined
    assert "Stress robustness below threshold" in joined
    assert "Resolve skipped stages" in joined


def test_derive_recommendations_reports_ready_when_all_good():
    outcomes = [
        StageOutcome(
            name="deliberative",
            status="success",
            reason=None,
            deployment_score=0.93,
            stress_score=0.71,
            output_dir="a",
            duration_sec=1.0,
        ),
        StageOutcome(
            name="ltc",
            status="cached",
            reason="resume",
            deployment_score=0.88,
            stress_score=0.65,
            output_dir="b",
            duration_sec=0.0,
        ),
    ]

    recs = derive_recommendations(outcomes)
    assert any("All stages passed thresholds" in r for r in recs)
