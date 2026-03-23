"""Tests for audit result JSON schema and validation."""

from __future__ import annotations

import json
from pathlib import Path

from deltatau_audit.report import generate_report
from deltatau_audit.schema import (
    SCHEMA_VERSION,
    get_audit_result_schema_path,
    load_audit_result_schema,
    prepare_audit_result,
    validate_audit_result,
)
from tests.conftest import make_result


def test_schema_file_exists_and_loads():
    schema_path = get_audit_result_schema_path()
    assert schema_path.exists()
    schema = load_audit_result_schema()
    assert schema.get("type") == "object"
    assert "schema_version" in schema.get("required", [])
    assert "manifest" in schema.get("required", [])


def test_generate_report_output_validates_against_schema(tmp_path: Path):
    result = make_result()
    # Intentionally omit metadata to ensure generator backfills them.
    result.pop("schema_version", None)
    result.pop("manifest", None)

    generate_report(result, str(tmp_path), title="Schema Validation Test")
    summary_path = tmp_path / "summary.json"
    data = json.loads(summary_path.read_text(encoding="utf-8"))

    assert data["schema_version"] == SCHEMA_VERSION
    assert isinstance(data.get("manifest"), dict)
    validate_audit_result(data)


def test_prepare_audit_result_backfills_manifest_without_clobbering_fields():
    result = make_result()
    result["manifest"] = {
        "title": "Atlas Certification",
        "experiment": {"env": "CartPole-v1"},
    }

    prepared = prepare_audit_result(
        result,
        report_version="0.8.0",
        report_timestamp="2026-03-14T00:00:00+00:00",
    )

    assert prepared["schema_version"] == SCHEMA_VERSION
    assert prepared["_version"] == "0.8.0"
    assert prepared["_timestamp"] == "2026-03-14T00:00:00+00:00"
    assert prepared["manifest"]["title"] == "Atlas Certification"
    assert prepared["manifest"]["experiment"]["env"] == "CartPole-v1"
    assert "protocol" in prepared["manifest"]
    assert "runtime" in prepared["manifest"]


def test_prepare_audit_result_backfills_diagnosis_without_clobbering_fields():
    result = make_result()
    result.pop("diagnosis", None)

    prepared = prepare_audit_result(result)

    assert isinstance(prepared["diagnosis"]["status"], str)
    assert isinstance(prepared["diagnosis"]["issues"], list)
    assert prepared["diagnosis"]["summary_line"]

    result["diagnosis"] = {
        "summary_line": "Custom diagnosis summary.",
        "issues": [{"scenario": "jitter", "rating": "FAIL"}],
    }
    prepared = prepare_audit_result(result)

    assert prepared["diagnosis"]["summary_line"] == "Custom diagnosis summary."
    assert prepared["diagnosis"]["issues"] == [{"scenario": "jitter", "rating": "FAIL"}]
    assert "status" in prepared["diagnosis"]
