"""Shared helpers for submission manifest expansion and bench health checks."""

from __future__ import annotations

from collections import Counter
import itertools
import json
import os
import re
import shlex
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import yaml


def _to_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text and text.lstrip("+-").isdigit():
            return int(text)
    return default


def _coerce_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    return str(value)


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _normalize_job_ids(job_ids: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in job_ids:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def load_manifest(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Manifest is not a mapping: {path}")
    return data


def expand_manifest_jobs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list):
        return []

    expanded: list[dict[str, Any]] = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        args = job.get("args", {})
        explicit_id = job.get("id")
        if not isinstance(args, dict):
            args = {}
        matrix = job.get("matrix", {})
        if not isinstance(matrix, dict) or not matrix:
            item: dict[str, Any] = {"args": dict(args), "vars": {}}
            if isinstance(explicit_id, str) and explicit_id:
                item["id"] = explicit_id
            expanded.append(item)
            continue

        keys = list(matrix.keys())
        value_lists: list[list[Any]] = []
        for key in keys:
            values = matrix.get(key, [])
            if not isinstance(values, list) or not values:
                values = [None]
            value_lists.append(values)

        for combo in itertools.product(*value_lists):
            vars_map = {keys[idx]: combo[idx] for idx in range(len(keys))}
            combo_args: dict[str, Any] = {}
            for arg_key, arg_value in args.items():
                if isinstance(arg_value, str):
                    try:
                        combo_args[arg_key] = arg_value.format(**vars_map)
                    except Exception:
                        combo_args[arg_key] = arg_value
                elif isinstance(arg_value, list):
                    formatted_list: list[Any] = []
                    for item in arg_value:
                        if isinstance(item, str):
                            try:
                                formatted_list.append(item.format(**vars_map))
                            except Exception:
                                formatted_list.append(item)
                        else:
                            formatted_list.append(item)
                    combo_args[arg_key] = formatted_list
                else:
                    combo_args[arg_key] = arg_value
            item = {"args": combo_args, "vars": vars_map}
            if isinstance(explicit_id, str) and explicit_id:
                try:
                    item["id"] = explicit_id.format(**vars_map)
                except Exception:
                    item["id"] = explicit_id
            expanded.append(item)
    return expanded


def _manifest_job_id(base_name: str, vars_map: dict[str, Any], index: int) -> str:
    if not vars_map:
        return f"{base_name}_{index:03d}"
    parts = [f"{key}-{vars_map[key]}" for key in sorted(vars_map.keys())]
    safe = "_".join(_coerce_str(part).replace(os.sep, "-") for part in parts)
    return f"{base_name}_{safe}"


def manifest_job_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list):
        return []

    rows: list[dict[str, Any]] = []
    job_counter = 0
    for raw_job in jobs:
        if not isinstance(raw_job, dict):
            continue
        name = str(raw_job.get("name", "job"))
        command = raw_job.get("command")
        if not isinstance(command, str) or not command:
            continue
        for item in expand_manifest_jobs({"jobs": [raw_job]}):
            job_counter += 1
            vars_map = item.get("vars", {})
            args = item.get("args", {})
            if not isinstance(vars_map, dict) or not isinstance(args, dict):
                continue
            explicit_id = item.get("id")
            if isinstance(explicit_id, str) and explicit_id.strip():
                job_id = explicit_id.strip()
            else:
                job_id = _manifest_job_id(name, vars_map, job_counter)
            rows.append(
                {
                    "id": job_id,
                    "name": name,
                    "command": command,
                    "vars": dict(vars_map),
                    "args": dict(args),
                }
            )
    return rows


def build_job_subset_manifest(
    manifest_path: Path,
    *,
    job_ids: Iterable[Any],
    manifest_name: str | None = None,
    description: str | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    requested_ids = _normalize_job_ids(job_ids)
    requested_set = set(requested_ids)
    rows = manifest_job_rows(manifest)

    selected_rows = [row for row in rows if row["id"] in requested_set]
    selected_ids = [row["id"] for row in selected_rows]
    missing_ids = [job_id for job_id in requested_ids if job_id not in set(selected_ids)]

    source_name = manifest.get("name")
    base_name = source_name if isinstance(source_name, str) and source_name.strip() else manifest_path.stem
    subset_name = manifest_name or f"{base_name}_failed_subset"
    subset_description = description or f"Focused subset of {manifest_path.name} ({len(selected_rows)} jobs)"

    subset: dict[str, Any] = {}
    for key, value in manifest.items():
        if key == "jobs":
            continue
        subset[key] = value
    subset["name"] = subset_name
    subset["description"] = subset_description
    if output_dir:
        subset["output_dir"] = output_dir
    subset["source_manifest"] = str(manifest_path)
    subset["selected_job_ids"] = selected_ids
    subset["jobs"] = [
        {
            "name": row["name"],
            "id": row["id"],
            "command": row["command"],
            "args": row["args"],
        }
        for row in selected_rows
    ]

    return {
        "manifest": subset,
        "requested_job_ids": requested_ids,
        "selected_job_ids": selected_ids,
        "missing_job_ids": missing_ids,
        "selected_count": len(selected_rows),
        "available_count": len(rows),
    }


def build_failed_job_subset_manifest(
    manifest_path: Path,
    output_root: Path,
    *,
    manifest_name: str | None = None,
    description: str | None = None,
    output_dir: str | None = None,
    job_ids: Iterable[Any] | None = None,
) -> dict[str, Any]:
    selected_ids = (
        _normalize_job_ids(job_ids)
        if job_ids is not None
        else _normalize_job_ids(bench_failure_breakdown(output_root).get("failed_job_ids", []))
    )
    return build_job_subset_manifest(
        manifest_path,
        job_ids=selected_ids,
        manifest_name=manifest_name,
        description=description,
        output_dir=output_dir,
    )


def summary_targets_from_manifest(manifest_path: Path, output_root: Path) -> list[Path]:
    manifest = load_manifest(manifest_path)
    expanded = expand_manifest_jobs(manifest)
    repo_root = manifest_path.parent.parent

    targets: list[Path] = []
    for row in expanded:
        args = row.get("args", {})
        if not isinstance(args, dict):
            continue
        out_value = args.get("out")
        if isinstance(out_value, str) and out_value.strip():
            out_path = Path(out_value)
            if not out_path.is_absolute():
                out_path = repo_root / out_path
            targets.append((out_path / "summary.json").resolve())

    # Fallback when manifest has no explicit out fields.
    if not targets:
        targets = list(output_root.resolve().rglob("summary.json"))

    # De-duplicate while preserving order.
    seen: set[str] = set()
    unique: list[Path] = []
    for target in targets:
        key = str(target)
        if key in seen:
            continue
        seen.add(key)
        unique.append(target)
    return unique


def bench_counts(output_root: Path) -> tuple[dict[str, int] | None, datetime | None, str | None]:
    summary = resolve_bench_summary_path(output_root)
    if not summary.exists():
        return None, None, None
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
    except Exception:
        return None, None, None
    counts = payload.get("counts")
    if not isinstance(counts, dict):
        return None, None, None
    parsed_counts = {key: _to_int(counts.get(key, 0)) for key in ("passed", "failed", "skipped")}
    updated = datetime.fromtimestamp(summary.stat().st_mtime, tz=timezone.utc)
    status = payload.get("status")
    if not isinstance(status, str):
        status = None
    return parsed_counts, updated, status


def resolve_bench_summary_path(source: Path) -> Path:
    path = Path(source)
    if path.suffix.lower() == ".json":
        return path
    return path / "bench_summary.json"


def _read_bench_summary_payload(source: Path) -> dict[str, Any] | None:
    summary = resolve_bench_summary_path(source)
    if not summary.exists():
        return None
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _empty_failure_breakdown() -> dict[str, Any]:
    return {
        "failed_total": 0,
        "ci_gate_failures": 0,
        "runtime_failures": 0,
        "other_failures": 0,
        "failed_job_ids": [],
        "ci_gate_summary_paths": [],
    }


def _job_result_ci_gate_failed(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    gate = result.get("stress_ci_gate_pass")
    return gate is False


def _metric_summary(values: Iterable[Any]) -> dict[str, Any]:
    numbers = [value for value in (_to_float(item) for item in values) if value is not None]
    if not numbers:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(numbers),
        "mean": sum(numbers) / len(numbers),
        "median": median(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }


def _top_counter_rows(counter: Counter[str], *, limit: int = 3) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, count in counter.most_common(max(1, int(limit))):
        rows.append({"label": label, "count": count})
    return rows


def _job_metric_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        return {}

    index: dict[str, dict[str, Any]] = {}
    for raw_job in jobs:
        if not isinstance(raw_job, dict):
            continue
        job_id = raw_job.get("id")
        if not isinstance(job_id, str) or not job_id.strip():
            continue
        result = raw_job.get("result")
        result_dict = result if isinstance(result, dict) else {}
        index[job_id.strip()] = {
            "status": str(raw_job.get("status", "")).strip().lower() or "unknown",
            "deployment_score": _to_float(result_dict.get("deployment_score")),
            "stress_score": _to_float(result_dict.get("stress_score")),
            "stress_worst_ci_lower": _to_float(result_dict.get("stress_worst_ci_lower")),
            "ci_gate_failed": result_dict.get("stress_ci_gate_pass") is False,
            "diagnosis_pattern": str(result_dict.get("diagnosis_pattern", "")).strip(),
            "stress_worst_scenario": str(result_dict.get("stress_worst_scenario", "")).strip(),
        }
    return index


def bench_quality_analysis(source: Path) -> dict[str, Any]:
    summary_path = resolve_bench_summary_path(source)
    payload = _read_bench_summary_payload(summary_path)
    if not payload:
        return {
            "source": str(source),
            "summary_path": str(summary_path),
            "exists": False,
            "job_count": 0,
            "variants": [],
            "signals": [f"bench_summary missing: {summary_path}"],
        }

    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        jobs = []

    protocol_meta = payload.get("protocol")
    protocol_name = ""
    if isinstance(protocol_meta, dict):
        forced = protocol_meta.get("forced")
        if isinstance(forced, str):
            protocol_name = forced
    if not protocol_name and jobs:
        first_result = jobs[0].get("result") if isinstance(jobs[0], dict) else {}
        if isinstance(first_result, dict):
            protocol_name = str(first_result.get("protocol", "")).strip()

    overall_diagnosis = Counter[str]()
    overall_scenarios = Counter[str]()
    variant_rollups: dict[str, dict[str, Any]] = {}
    ci_gate_failures = 0
    deployment_failures = 0
    stress_failures = 0

    for raw_job in jobs:
        if not isinstance(raw_job, dict):
            continue
        variant = str(raw_job.get("name") or raw_job.get("id") or "unknown").strip() or "unknown"
        status_text = str(raw_job.get("status", "")).strip().lower() or "unknown"
        result = raw_job.get("result")
        result_dict = result if isinstance(result, dict) else {}
        deployment_score = _to_float(result_dict.get("deployment_score"))
        stress_score = _to_float(result_dict.get("stress_score"))
        ci_lower = _to_float(result_dict.get("stress_worst_ci_lower"))
        deployment_failed = str(result_dict.get("deployment_rating", "")).upper() == "FAIL"
        stress_failed = str(result_dict.get("stress_rating", "")).upper() == "FAIL"
        ci_gate_failed = result_dict.get("stress_ci_gate_pass") is False
        diagnosis_pattern = str(result_dict.get("diagnosis_pattern", "")).strip()
        worst_scenario = str(result_dict.get("stress_worst_scenario", "")).strip()

        rollup = variant_rollups.setdefault(
            variant,
            {
                "variant": variant,
                "job_count": 0,
                "status_counts": Counter[str](),
                "ci_gate_failures": 0,
                "deployment_failures": 0,
                "stress_failures": 0,
                "_deployment_scores": [],
                "_stress_scores": [],
                "_ci_lowers": [],
                "_diagnosis": Counter[str](),
                "_scenarios": Counter[str](),
            },
        )
        rollup["job_count"] += 1
        rollup["status_counts"][status_text] += 1
        if ci_gate_failed:
            ci_gate_failures += 1
            rollup["ci_gate_failures"] += 1
        if deployment_failed:
            deployment_failures += 1
            rollup["deployment_failures"] += 1
        if stress_failed:
            stress_failures += 1
            rollup["stress_failures"] += 1
        if deployment_score is not None:
            rollup["_deployment_scores"].append(deployment_score)
        if stress_score is not None:
            rollup["_stress_scores"].append(stress_score)
        if ci_lower is not None:
            rollup["_ci_lowers"].append(ci_lower)
        if diagnosis_pattern:
            overall_diagnosis[diagnosis_pattern] += 1
            rollup["_diagnosis"][diagnosis_pattern] += 1
        if worst_scenario:
            overall_scenarios[worst_scenario] += 1
            rollup["_scenarios"][worst_scenario] += 1

    variants: list[dict[str, Any]] = []
    variants_all_failed = True
    for variant, rollup in sorted(variant_rollups.items()):
        job_count = int(rollup["job_count"])
        ci_failed = int(rollup["ci_gate_failures"])
        if job_count <= 0 or ci_failed < job_count:
            variants_all_failed = False
        variants.append(
            {
                "variant": variant,
                "job_count": job_count,
                "status_counts": dict(sorted(rollup["status_counts"].items())),
                "ci_gate_failures": ci_failed,
                "deployment_failures": int(rollup["deployment_failures"]),
                "stress_failures": int(rollup["stress_failures"]),
                "deployment_score": _metric_summary(rollup["_deployment_scores"]),
                "stress_score": _metric_summary(rollup["_stress_scores"]),
                "stress_worst_ci_lower": _metric_summary(rollup["_ci_lowers"]),
                "top_diagnosis_patterns": _top_counter_rows(rollup["_diagnosis"]),
                "top_worst_scenarios": _top_counter_rows(rollup["_scenarios"]),
            }
        )

    job_count = len(jobs)
    counts = payload.get("counts")
    counts_dict = counts if isinstance(counts, dict) else {}
    signals: list[str] = []
    if job_count > 0 and ci_gate_failures >= job_count:
        signals.append(f"all {job_count} jobs failed the CI quality gate")
    if variants and variants_all_failed:
        signals.append(f"every variant failed all of its jobs ({len(variants)} variants)")
    if overall_diagnosis:
        label, count = overall_diagnosis.most_common(1)[0]
        if count >= job_count and job_count > 0:
            signals.append(f"uniform diagnosis pattern: {label}")
        elif job_count > 0 and count / job_count >= 0.7:
            signals.append(f"dominant diagnosis pattern: {label} ({count}/{job_count})")
    if overall_scenarios:
        label, count = overall_scenarios.most_common(1)[0]
        if count >= job_count and job_count > 0:
            signals.append(f"uniform worst-case scenario: {label}")
        elif job_count > 0 and count / job_count >= 0.7:
            signals.append(f"dominant worst-case scenario: {label} ({count}/{job_count})")

    return {
        "source": str(source),
        "summary_path": str(summary_path),
        "exists": True,
        "manifest": payload.get("manifest"),
        "output_root": payload.get("output_root"),
        "protocol": protocol_name or None,
        "status": payload.get("status"),
        "counts": {
            "passed": _to_int(counts_dict.get("passed")),
            "failed": _to_int(counts_dict.get("failed")),
            "skipped": _to_int(counts_dict.get("skipped")),
        },
        "job_count": job_count,
        "ci_gate_failures": ci_gate_failures,
        "deployment_failures": deployment_failures,
        "stress_failures": stress_failures,
        "top_diagnosis_patterns": _top_counter_rows(overall_diagnosis),
        "top_worst_scenarios": _top_counter_rows(overall_scenarios),
        "variants": variants,
        "signals": signals,
    }


def compare_bench_quality(base_source: Path, other_source: Path) -> dict[str, Any]:
    base_summary_path = resolve_bench_summary_path(base_source)
    other_summary_path = resolve_bench_summary_path(other_source)
    base_payload = _read_bench_summary_payload(base_summary_path)
    other_payload = _read_bench_summary_payload(other_summary_path)
    if not base_payload or not other_payload:
        missing: list[str] = []
        if not base_payload:
            missing.append(str(base_summary_path))
        if not other_payload:
            missing.append(str(other_summary_path))
        return {
            "base_source": str(base_source),
            "other_source": str(other_source),
            "common_jobs": 0,
            "missing": missing,
        }

    base_index = _job_metric_index(base_payload)
    other_index = _job_metric_index(other_payload)
    common_ids = sorted(set(base_index) & set(other_index))

    deployment_deltas: list[float] = []
    stress_deltas: list[float] = []
    ci_lower_deltas: list[float] = []
    improved_ci_gate = 0
    regressed_ci_gate = 0
    unchanged_failed = 0
    unchanged_passed = 0
    diagnosis_changes = 0
    scenario_changes = 0

    for job_id in common_ids:
        base_metrics = base_index[job_id]
        other_metrics = other_index[job_id]
        base_dep = base_metrics.get("deployment_score")
        other_dep = other_metrics.get("deployment_score")
        if isinstance(base_dep, float) and isinstance(other_dep, float):
            deployment_deltas.append(other_dep - base_dep)
        base_stress = base_metrics.get("stress_score")
        other_stress = other_metrics.get("stress_score")
        if isinstance(base_stress, float) and isinstance(other_stress, float):
            stress_deltas.append(other_stress - base_stress)
        base_ci = base_metrics.get("stress_worst_ci_lower")
        other_ci = other_metrics.get("stress_worst_ci_lower")
        if isinstance(base_ci, float) and isinstance(other_ci, float):
            ci_lower_deltas.append(other_ci - base_ci)

        base_failed = bool(base_metrics.get("ci_gate_failed"))
        other_failed = bool(other_metrics.get("ci_gate_failed"))
        if base_failed and not other_failed:
            improved_ci_gate += 1
        elif not base_failed and other_failed:
            regressed_ci_gate += 1
        elif base_failed and other_failed:
            unchanged_failed += 1
        else:
            unchanged_passed += 1

        if base_metrics.get("diagnosis_pattern") != other_metrics.get("diagnosis_pattern"):
            diagnosis_changes += 1
        if base_metrics.get("stress_worst_scenario") != other_metrics.get("stress_worst_scenario"):
            scenario_changes += 1

    base_analysis = bench_quality_analysis(base_summary_path)
    other_analysis = bench_quality_analysis(other_summary_path)
    return {
        "base_source": str(base_source),
        "other_source": str(other_source),
        "base_protocol": base_analysis.get("protocol"),
        "other_protocol": other_analysis.get("protocol"),
        "delta_direction": "other_minus_base",
        "common_jobs": len(common_ids),
        "deployment_score_delta": _metric_summary(deployment_deltas),
        "stress_score_delta": _metric_summary(stress_deltas),
        "stress_worst_ci_lower_delta": _metric_summary(ci_lower_deltas),
        "ci_gate_flips": {
            "improved": improved_ci_gate,
            "regressed": regressed_ci_gate,
            "unchanged_failed": unchanged_failed,
            "unchanged_passed": unchanged_passed,
        },
        "diagnosis_pattern_changes": diagnosis_changes,
        "worst_scenario_changes": scenario_changes,
    }


def bench_failure_breakdown(output_root: Path) -> dict[str, Any]:
    payload = _read_bench_summary_payload(output_root)
    if not payload:
        return _empty_failure_breakdown()

    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        return _empty_failure_breakdown()

    failed_total = 0
    ci_gate_failures = 0
    runtime_failures = 0
    other_failures = 0
    failed_job_ids: list[str] = []
    ci_gate_summary_paths: list[str] = []

    for raw_job in jobs:
        if not isinstance(raw_job, dict):
            continue
        status_text = str(raw_job.get("status", "")).strip().lower()
        if status_text == "failed":
            pass
        elif status_text == "skipped" and _job_result_ci_gate_failed(raw_job.get("result")):
            pass
        else:
            continue

        failed_total += 1
        job_id = raw_job.get("id")
        if isinstance(job_id, str) and job_id.strip():
            failed_job_ids.append(job_id)

        summary_path = raw_job.get("summary_path")
        summary_path_text = summary_path.strip() if isinstance(summary_path, str) else ""
        has_summary = bool(summary_path_text)
        if has_summary:
            # Failed with summary usually means CI gate failure, not runtime crash.
            ci_gate_failures += 1
            ci_gate_summary_paths.append(summary_path_text)
            continue

        returncode = _to_int(raw_job.get("returncode"), default=1)
        if returncode != 0:
            runtime_failures += 1
        else:
            other_failures += 1

    return {
        "failed_total": failed_total,
        "ci_gate_failures": ci_gate_failures,
        "runtime_failures": runtime_failures,
        "other_failures": other_failures,
        "failed_job_ids": failed_job_ids,
        "ci_gate_summary_paths": ci_gate_summary_paths,
    }


def ci_gate_failed_summary_paths(failure_breakdown: Any) -> list[str]:
    if not isinstance(failure_breakdown, dict):
        return []
    raw = failure_breakdown.get("ci_gate_summary_paths")
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


_CARTPOLE_FAILED_JOB_ID_RE = re.compile(r"^cartpole_(?P<variant>.+)_seed-(?P<seed>\d+)$")


def cartpole_failed_variant_seeds(failure_breakdown: Any) -> dict[str, list[int]]:
    if not isinstance(failure_breakdown, dict):
        return {}
    raw_ids = failure_breakdown.get("failed_job_ids")
    if not isinstance(raw_ids, list):
        return {}
    grouped: dict[str, set[int]] = {}
    for item in raw_ids:
        if not isinstance(item, str):
            continue
        match = _CARTPOLE_FAILED_JOB_ID_RE.match(item.strip())
        if not match:
            continue
        seed = _to_int(match.group("seed"), default=-1)
        if seed < 0:
            continue
        grouped.setdefault(match.group("variant"), set()).add(seed)
    return {variant: sorted(seeds) for variant, seeds in sorted(grouped.items())}


def cartpole_retrain_commands(
    variant_seeds: dict[str, list[int]],
    *,
    timesteps: int = 45000,
    force: bool = True,
    base_speed: int = 3,
    jitter: int = 2,
    phase_period: int = 200,
) -> list[str]:
    commands: list[str] = []
    for variant, seeds in sorted(variant_seeds.items()):
        unique_seeds = sorted({_to_int(seed, default=-1) for seed in seeds})
        unique_seeds = [seed for seed in unique_seeds if seed >= 0]
        if not unique_seeds:
            continue
        seed_part = " ".join(str(seed) for seed in unique_seeds)
        commands.append(
            "python -m deltatau_audit stress train-sb3 "
            "--env CartPole-v1 --algo ppo --out-root checkpoints_cartpole_ppo "
            f"--variants {shlex.quote(variant)} --seeds {seed_part} "
            f"--timesteps {int(timesteps)}"
            + (" --force" if force else "")
            + f" --base-speed {int(base_speed)}"
            + f" --jitter {int(jitter)}"
            + f" --phase-period {int(phase_period)}"
            + " --fail-fast"
        )
    return commands


def summary_cleanup_commands(paths: Iterable[str]) -> list[str]:
    normalized = ci_gate_failed_summary_paths({"ci_gate_summary_paths": list(paths)})
    commands: list[str] = []
    for path in normalized:
        path_literal = path.replace("'", "\\'")
        code = f"from pathlib import Path; Path(r'{path_literal}').unlink(missing_ok=True)"
        commands.append(f'python -c "{code}"')
    return commands


@dataclass(frozen=True)
class BenchQualityRepairPlan:
    job_name: str
    manifest: str
    output_root: str
    protocol: str
    ci_gate_failures: int
    strategy: str
    retrain_commands: tuple[str, ...]
    cleanup_summary_paths: tuple[str, ...]
    rerun_command: str
    refresh_summary_command: str
    diagnostic_commands: tuple[str, ...]
    reasons: tuple[str, ...]
    use_no_resume: bool = False
    rerun_scope: str = "full"
    repair_manifest_path: str = ""
    repair_output_dir: str = ""
    focused_job_ids: tuple[str, ...] = ()


def _repair_manifest_slug(job_name: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(job_name).strip())
    return text.strip("._-") or "bench_failed_subset"


def cartpole_failure_scope(failure_breakdown: Any) -> dict[str, Any]:
    variant_seeds = cartpole_failed_variant_seeds(failure_breakdown)
    total_cells = sum(len(seeds) for seeds in variant_seeds.values())
    return {
        "variant_seeds": variant_seeds,
        "variant_count": len(variant_seeds),
        "failed_cells": total_cells,
        "max_failed_seeds_per_variant": max((len(seeds) for seeds in variant_seeds.values()), default=0),
    }


def build_quality_repair_plan(
    *,
    job_name: str,
    manifest: str,
    output_root: str = "",
    protocol: str,
    failure_breakdown: Any,
    timesteps: int = 45000,
    force_retrain: bool = True,
    include_retrain: bool = True,
    base_speed: int = 3,
    jitter: int = 2,
    phase_period: int = 200,
    repair_manifest_path: str = "",
    expected_jobs: int = 0,
) -> BenchQualityRepairPlan | None:
    if not isinstance(failure_breakdown, dict):
        return None

    ci_gate_failures = _to_int(failure_breakdown.get("ci_gate_failures"))
    if ci_gate_failures <= 0:
        return None

    reasons: list[str] = []
    retrain_cmds: list[str] = []
    manifest_name = Path(manifest).name.lower()
    failed_job_ids = _normalize_job_ids(failure_breakdown.get("failed_job_ids", []))
    failure_scope = cartpole_failure_scope(failure_breakdown)
    failure_rate = (ci_gate_failures / expected_jobs) if expected_jobs > 0 else 0.0
    widespread_cartpole_failure = (
        "high_rigor_10seed_manifest" in manifest_name
        and failure_scope["variant_count"] >= 3
        and (
            (expected_jobs > 0 and ci_gate_failures >= expected_jobs)
            or failure_rate >= 0.6
            or ci_gate_failures >= 15
        )
    )
    if widespread_cartpole_failure:
        diagnostic_cmds: list[str] = []
        if isinstance(output_root, str) and output_root.strip():
            slug = _repair_manifest_slug(job_name)
            diagnostic_cmds.append(
                "python scripts/analyze_bench_failures.py "
                f"--bench {shlex.quote(output_root)} "
                f"--json-out {shlex.quote(f'_status_demo/diagnostics/{slug}.json')} "
                f"--markdown-out {shlex.quote(f'_status_demo/diagnostics/{slug}.md')}"
            )
        scope_variants = int(failure_scope["variant_count"])
        scope_cells = int(failure_scope["failed_cells"])
        denominator = expected_jobs if expected_jobs > 0 else scope_cells
        reasons.append(
            f"{job_name}: widespread CI-gate failure ({ci_gate_failures}/{denominator} jobs across {scope_variants} variants)"
        )
        reasons.append(f"{job_name}: diagnose protocol/claim mismatch before additional retraining")
        return BenchQualityRepairPlan(
            job_name=str(job_name),
            manifest=str(manifest),
            output_root=str(output_root),
            protocol=str(protocol),
            ci_gate_failures=ci_gate_failures,
            strategy="diagnose_protocol",
            retrain_commands=(),
            cleanup_summary_paths=(),
            rerun_command="",
            refresh_summary_command="",
            diagnostic_commands=tuple(diagnostic_cmds),
            reasons=tuple(reasons),
            use_no_resume=False,
            rerun_scope="diagnose",
            repair_manifest_path="",
            repair_output_dir="",
            focused_job_ids=tuple(failed_job_ids),
        )
    if include_retrain and "high_rigor_10seed_manifest" in manifest_name:
        variant_seeds = failure_scope["variant_seeds"]
        if variant_seeds:
            summary_parts = [
                f"{variant}: seeds {','.join(str(seed) for seed in seeds)}"
                for variant, seeds in sorted(variant_seeds.items())
                if seeds
            ]
            if summary_parts:
                reasons.append(f"{job_name} failed cells: {'; '.join(summary_parts)}")
            retrain_cmds.extend(
                cartpole_retrain_commands(
                    variant_seeds,
                    timesteps=timesteps,
                    force=force_retrain,
                    base_speed=base_speed,
                    jitter=jitter,
                    phase_period=phase_period,
                )
            )

    cleanup_paths = ci_gate_failed_summary_paths(failure_breakdown)
    if cleanup_paths:
        reasons.append(f"{job_name}: clear {len(cleanup_paths)} failed summaries before rerun")

    focused_manifest = repair_manifest_path
    focused_output_dir = ""
    rerun_scope = "full"
    use_no_resume = not bool(cleanup_paths)
    refresh_summary_command = ""
    if failed_job_ids and isinstance(output_root, str) and output_root.strip():
        if not focused_manifest:
            focused_manifest = f"_status_demo/repair_manifests/{_repair_manifest_slug(job_name)}.yaml"
        focused_output_dir = f"_status_demo/repair_bench_runs/{_repair_manifest_slug(job_name)}"
        build_cmd = (
            f"python scripts/build_failed_job_manifest.py "
            f"--manifest {shlex.quote(manifest)} "
            f"--output-root {shlex.quote(output_root)} "
            f"--out-manifest {shlex.quote(focused_manifest)} "
            f"--output-dir {shlex.quote(focused_output_dir)}"
        )
        for failed_job_id in failed_job_ids:
            build_cmd = f"{build_cmd} --job-id {shlex.quote(failed_job_id)}"
        rerun_command = (
            f"{build_cmd} && "
            f"python -m deltatau_audit bench run --manifest {shlex.quote(focused_manifest)} "
            f"--protocol {shlex.quote(protocol)}"
        )
        refresh_summary_command = (
            f"python scripts/merge_bench_summaries.py "
            f"--base-summary {shlex.quote(str(Path(output_root) / 'bench_summary.json'))} "
            f"--patch-summary {shlex.quote(str(Path(focused_output_dir) / 'bench_summary.json'))} "
            f"--output-root {shlex.quote(output_root)}"
        )
        reasons.append(f"{job_name}: rerun focused subset ({len(failed_job_ids)} failed jobs)")
        reasons.append(f"{job_name}: merge focused rerun into full bench summary")
        rerun_scope = "focused"
        use_no_resume = False
    else:
        rerun_command = (
            f"python -m deltatau_audit bench run --manifest {shlex.quote(manifest)} "
            f"--protocol {shlex.quote(protocol)}"
        )
        if use_no_resume:
            rerun_command = f"{rerun_command} --no-resume"
            reasons.append(f"{job_name}: fallback to --no-resume (failed summary paths unavailable)")

    return BenchQualityRepairPlan(
        job_name=str(job_name),
        manifest=str(manifest),
        output_root=str(output_root),
        protocol=str(protocol),
        ci_gate_failures=ci_gate_failures,
        strategy="repair",
        retrain_commands=tuple(retrain_cmds),
        cleanup_summary_paths=tuple(cleanup_paths),
        rerun_command=rerun_command,
        refresh_summary_command=refresh_summary_command,
        diagnostic_commands=(),
        reasons=tuple(reasons),
        use_no_resume=use_no_resume,
        rerun_scope=rerun_scope,
        repair_manifest_path=str(focused_manifest),
        repair_output_dir=str(focused_output_dir),
        focused_job_ids=tuple(failed_job_ids),
    )


def repair_plan_commands(
    plan: BenchQualityRepairPlan,
    *,
    post_commands: Iterable[str] | None = None,
) -> list[str]:
    commands: list[str] = []
    if plan.strategy == "diagnose_protocol":
        commands.extend(plan.diagnostic_commands)
    else:
        commands.extend(plan.retrain_commands)
        commands.extend(summary_cleanup_commands(plan.cleanup_summary_paths))
        if plan.rerun_command:
            commands.append(plan.rerun_command)
        if plan.refresh_summary_command:
            commands.append(plan.refresh_summary_command)
    if post_commands:
        commands.extend(str(command) for command in post_commands if str(command).strip())
    return commands


def repair_plan_command_chain(
    plan: BenchQualityRepairPlan,
    *,
    post_commands: Iterable[str] | None = None,
) -> str:
    return " && ".join(repair_plan_commands(plan, post_commands=post_commands))


def repair_plan_payload(plan: BenchQualityRepairPlan | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    return {
        "job_name": plan.job_name,
        "manifest": plan.manifest,
        "output_root": plan.output_root,
        "protocol": plan.protocol,
        "ci_gate_failures": plan.ci_gate_failures,
        "strategy": plan.strategy,
        "retrain_commands": list(plan.retrain_commands),
        "cleanup_summary_paths": list(plan.cleanup_summary_paths),
        "rerun_command": plan.rerun_command,
        "refresh_summary_command": plan.refresh_summary_command,
        "diagnostic_commands": list(plan.diagnostic_commands),
        "reasons": list(plan.reasons),
        "use_no_resume": plan.use_no_resume,
        "rerun_scope": plan.rerun_scope,
        "repair_manifest_path": plan.repair_manifest_path,
        "repair_output_dir": plan.repair_output_dir,
        "focused_job_ids": list(plan.focused_job_ids),
    }


def check_bench_execution(
    manifest_path: Path,
    output_root: Path,
    *,
    protocol: str = "paper",
    job_name: str | None = None,
) -> dict[str, Any]:
    manifest_exists = manifest_path.exists()
    targets = summary_targets_from_manifest(manifest_path, output_root) if manifest_exists else []
    expected_jobs = len(targets)
    completed_jobs = sum(1 for path in targets if path.exists())
    missing_jobs = max(0, expected_jobs - completed_jobs)

    counts, _, status = bench_counts(output_root)
    failure_breakdown = bench_failure_breakdown(output_root)
    bench_summary_path = output_root / "bench_summary.json"
    bench_summary_exists = counts is not None
    failed_jobs = _to_int((counts or {}).get("failed"))
    breakdown_failed_total = _to_int(failure_breakdown.get("failed_total"))
    ready = (
        manifest_exists
        and expected_jobs > 0
        and completed_jobs == expected_jobs
        and bench_summary_exists
        and status == "passed"
        and failed_jobs == 0
        and breakdown_failed_total == 0
    )

    reasons: list[str] = []
    if not manifest_exists:
        reasons.append(f"manifest missing: {manifest_path}")
    if expected_jobs == 0:
        reasons.append("manifest expands to 0 jobs")
    if missing_jobs > 0:
        reasons.append(f"{missing_jobs}/{expected_jobs} summaries missing")
    if not bench_summary_exists:
        reasons.append(f"bench_summary missing: {bench_summary_path}")
    elif status != "passed":
        reasons.append(f"bench_summary status is {status!r}")
    if failed_jobs > 0:
        reasons.append(f"{failed_jobs} failed jobs in bench_summary")
    runtime_failures = _to_int(failure_breakdown.get("runtime_failures"))
    ci_gate_failures = _to_int(failure_breakdown.get("ci_gate_failures"))
    if runtime_failures > 0:
        reasons.append(f"{runtime_failures} runtime failures (missing summary or crashed jobs)")
    if ci_gate_failures > 0:
        reasons.append(f"{ci_gate_failures} quality-gate failures (summary exists but CI gate failed)")

    normalized_job_name = str(job_name).strip() if isinstance(job_name, str) and str(job_name).strip() else output_root.name
    quality_repair_plan = build_quality_repair_plan(
        job_name=normalized_job_name,
        manifest=str(manifest_path),
        output_root=str(output_root),
        protocol=str(protocol),
        failure_breakdown=failure_breakdown,
        expected_jobs=expected_jobs,
    )

    return {
        "manifest_path": str(manifest_path),
        "output_root": str(output_root),
        "manifest_exists": manifest_exists,
        "expected_jobs": expected_jobs,
        "completed_jobs": completed_jobs,
        "missing_jobs": missing_jobs,
        "bench_summary_path": str(bench_summary_path),
        "bench_summary_exists": bench_summary_exists,
        "bench_status": status,
        "counts": counts or {},
        "failure_breakdown": failure_breakdown,
        "quality_repair_plan": repair_plan_payload(quality_repair_plan),
        "repair_command_chain": repair_plan_command_chain(quality_repair_plan) if quality_repair_plan is not None else None,
        "ready": ready,
        "reasons": reasons,
    }
