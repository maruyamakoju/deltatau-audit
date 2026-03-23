from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .provenance import minimal_manifest

SCHEMA_VERSION = "1.0.0"


def get_audit_result_schema_path() -> Path:
    """Return the bundled JSON schema path for audit results."""
    return Path(__file__).resolve().parent / "schemas" / "audit_result.schema.json"


@lru_cache(maxsize=1)
def load_audit_result_schema() -> dict[str, Any]:
    """Load and cache the bundled audit-result JSON schema."""
    schema_path = get_audit_result_schema_path()
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _deep_merge_mappings(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = deepcopy(dict(base))
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[str(key)] = _deep_merge_mappings(existing, value)
        else:
            merged[str(key)] = deepcopy(value)
    return merged


def _normalized_manifest(manifest: Any) -> dict[str, Any]:
    base = minimal_manifest(protocol_name="custom")
    if not isinstance(manifest, Mapping):
        return base
    return _deep_merge_mappings(base, manifest)


def _infer_diagnosis_status(summary: Any) -> str:
    if not isinstance(summary, Mapping):
        return "pass"

    deployment_rating = str(summary.get("deployment_rating", "N/A")).upper()
    stress_rating = str(summary.get("stress_rating", "N/A")).upper()
    if "FAIL" in (deployment_rating, stress_rating):
        return "fail"
    if deployment_rating in ("DEGRADED", "MILD") or stress_rating in ("DEGRADED", "MILD"):
        return "warn"
    return "pass"


def _default_diagnosis(summary: Any, robustness: Any) -> dict[str, Any]:
    if isinstance(summary, Mapping) and isinstance(robustness, Mapping):
        try:
            from .diagnose import generate_diagnosis
        except Exception:
            pass
        else:
            try:
                generated = generate_diagnosis(dict(summary), dict(robustness))
            except Exception:
                generated = None
            if isinstance(generated, Mapping):
                return deepcopy(dict(generated))

    status = _infer_diagnosis_status(summary)
    if status == "pass":
        summary_line = "No significant timing failures detected."
    else:
        summary_line = "Timing issues detected, but detailed diagnosis is unavailable."
    return {
        "status": status,
        "failing_scenarios": [],
        "issues": [],
        "primary_pattern": None,
        "root_cause": None,
        "fix_recommendation": None,
        "summary_line": summary_line,
    }


def _normalized_diagnosis(diagnosis: Any, summary: Any, robustness: Any) -> dict[str, Any]:
    base = _default_diagnosis(summary, robustness)
    if not isinstance(diagnosis, Mapping):
        return base
    return _deep_merge_mappings(base, diagnosis)


def prepare_audit_result(
    audit_result: Mapping[str, Any],
    *,
    report_version: str | None = None,
    report_timestamp: str | None = None,
) -> dict[str, Any]:
    """Return a normalized audit result ready for serialization/reporting."""
    normalized = deepcopy(dict(audit_result))
    normalized["schema_version"] = SCHEMA_VERSION
    normalized["manifest"] = _normalized_manifest(normalized.get("manifest"))
    normalized["diagnosis"] = _normalized_diagnosis(
        normalized.get("diagnosis"),
        normalized.get("summary"),
        normalized.get("robustness"),
    )
    if report_version is not None:
        normalized["_version"] = str(report_version)
    if report_timestamp is not None:
        normalized["_timestamp"] = str(report_timestamp)
    return normalized


def _basic_validate_audit_result(
    audit_result: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> None:
    required = schema.get("required", [])
    if not isinstance(required, list):
        raise ValueError("audit result schema is malformed: top-level required must be a list")

    missing = [field for field in required if field not in audit_result]
    if missing:
        joined = ", ".join(sorted(str(field) for field in missing))
        raise ValueError(f"audit result is missing required fields: {joined}")

    if audit_result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"audit result schema_version must be {SCHEMA_VERSION}, got {audit_result.get('schema_version')!r}"
        )

    manifest = audit_result.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("audit result manifest must be an object")

    manifest_required = (
        schema.get("properties", {})
        .get("manifest", {})
        .get("required", [])
    )
    if not isinstance(manifest_required, list):
        raise ValueError("audit result schema is malformed: manifest.required must be a list")
    missing_manifest = [field for field in manifest_required if field not in manifest]
    if missing_manifest:
        joined = ", ".join(sorted(str(field) for field in missing_manifest))
        raise ValueError(f"audit result manifest is missing required fields: {joined}")

    summary = audit_result.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("audit result summary must be an object")

    summary_required = (
        schema.get("properties", {})
        .get("summary", {})
        .get("required", [])
    )
    if not isinstance(summary_required, list):
        raise ValueError("audit result schema is malformed: summary.required must be a list")
    missing_summary = [field for field in summary_required if field not in summary]
    if missing_summary:
        joined = ", ".join(sorted(str(field) for field in missing_summary))
        raise ValueError(f"audit result summary is missing required fields: {joined}")

    robustness = audit_result.get("robustness")
    if not isinstance(robustness, Mapping):
        raise ValueError("audit result robustness must be an object")

    robustness_required = (
        schema.get("properties", {})
        .get("robustness", {})
        .get("required", [])
    )
    if not isinstance(robustness_required, list):
        raise ValueError("audit result schema is malformed: robustness.required must be a list")
    missing_robustness = [field for field in robustness_required if field not in robustness]
    if missing_robustness:
        joined = ", ".join(sorted(str(field) for field in missing_robustness))
        raise ValueError(f"audit result robustness is missing required fields: {joined}")

    diagnosis = audit_result.get("diagnosis")
    if not isinstance(diagnosis, Mapping):
        raise ValueError("audit result diagnosis must be an object")


def validate_audit_result(audit_result: Mapping[str, Any]) -> None:
    """Validate an audit result against the bundled JSON schema."""
    schema = load_audit_result_schema()
    try:
        from jsonschema import Draft202012Validator
    except Exception:
        _basic_validate_audit_result(audit_result, schema)
        return

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(dict(audit_result))


class ReliabilityLevel(Enum):
    UNSET = 0
    UNRELIABLE = 1
    DEGRADED = 2
    ROBUST = 3
    CERTIFIED = 4


@dataclass(frozen=True)
class MetricValue:
    """A strictly typed metric value with optional confidence intervals."""

    value: float
    lower_ci: Optional[float] = None
    upper_ci: Optional[float] = None
    unit: str = "scalar"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "lower_ci": self.lower_ci,
            "upper_ci": self.upper_ci,
            "unit": self.unit,
            **self.metadata,
        }


@dataclass(frozen=True)
class TemporalCapability:
    """Represents the agent's temporal intelligence capabilities."""

    can_ponder: bool = False
    max_lookahead_steps: int = 0
    supports_variable_dt: bool = False
    has_internal_clock: bool = False


@dataclass
class AuditStageResult:
    """Result of a specific audit stage (e.g., Jitter, Stress, Adversarial)."""

    stage_name: str
    pass_rate: float
    metrics: Dict[str, MetricValue]
    success: bool
    reasoning: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)


@dataclass
class AuditReport:
    """Professional-grade audit report container."""

    agent_id: str
    timestamp: str
    reliability_score: float
    level: ReliabilityLevel
    stages: List[AuditStageResult]
    capabilities: TemporalCapability
    summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
            "reliability_score": self.reliability_score,
            "level": self.level.name,
            "stages": [s.__dict__ for s in self.stages],
            "capabilities": self.capabilities.__dict__,
            "summary": self.summary,
            **self.metadata,
        }


__all__ = [
    "SCHEMA_VERSION",
    "AuditReport",
    "AuditStageResult",
    "MetricValue",
    "ReliabilityLevel",
    "TemporalCapability",
    "get_audit_result_schema_path",
    "load_audit_result_schema",
    "prepare_audit_result",
    "validate_audit_result",
]
