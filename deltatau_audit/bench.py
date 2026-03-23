"""Benchmark matrix runner for reproducible long-horizon experiments."""

from __future__ import annotations

import csv
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_SUBMISSION_COLUMNS: tuple[str, ...] = (
    "job_id",
    "status",
    "command",
    "env",
    "algo",
    "seed",
    "variant",
    "protocol",
    "deployment_score",
    "deployment_rating",
    "stress_score",
    "stress_rating",
    "stress_worst_scenario",
    "stress_worst_return_ratio",
    "stress_worst_ci_lower",
    "stress_ci_gate_pass",
    "quadrant",
    "diagnosis_pattern",
    "summary_path",
)


def _load_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}")

    suffix = p.suffix.lower()
    if suffix in {".json"}:
        data = json.loads(p.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency path
            raise RuntimeError(
                "YAML manifest requires PyYAML. Install: pip install pyyaml"
            ) from exc
        loaded = yaml.safe_load(p.read_text(encoding="utf-8"))
        data = loaded if isinstance(loaded, dict) else {}
    else:
        # Try JSON first then YAML as fallback.
        text = p.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
        except Exception:
            try:
                import yaml  # type: ignore
            except ImportError as exc:  # pragma: no cover - dependency path
                raise RuntimeError(
                    "Unknown manifest extension and PyYAML unavailable."
                ) from exc
            loaded = yaml.safe_load(text)
            data = loaded if isinstance(loaded, dict) else {}

    if not isinstance(data, dict):
        raise ValueError("Manifest must be a mapping/object at top level")
    return data


def _coerce_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    return str(value)


def _to_cli_flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def _expand_job_matrix(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand one benchmark job with optional matrix dimensions."""
    matrix = job.get("matrix")
    args = dict(job.get("args", {}))
    explicit_id = job.get("id")
    if not isinstance(args, dict):
        raise ValueError("job.args must be a mapping")

    if not isinstance(matrix, dict) or not matrix:
        item: dict[str, Any] = {"vars": {}, "args": args}
        if isinstance(explicit_id, str) and explicit_id:
            item["id"] = explicit_id
        return [item]

    dim_names: list[str] = []
    dim_values: list[list[Any]] = []
    for k, v in matrix.items():
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError(f"job.matrix.{k} must be a non-empty list")
        dim_names.append(str(k))
        dim_values.append(v)

    expanded: list[dict[str, Any]] = []
    for combo in itertools.product(*dim_values):
        vars_map = {dim_names[i]: combo[i] for i in range(len(dim_names))}
        combo_args: dict[str, Any] = {}
        for key, value in args.items():
            if isinstance(value, str):
                try:
                    combo_args[key] = value.format(**vars_map)
                except Exception:
                    combo_args[key] = value
            elif isinstance(value, list):
                out_list: list[Any] = []
                for item in value:
                    if isinstance(item, str):
                        try:
                            out_list.append(item.format(**vars_map))
                        except Exception:
                            out_list.append(item)
                    else:
                        out_list.append(item)
                combo_args[key] = out_list
            else:
                combo_args[key] = value
        item = {"vars": vars_map, "args": combo_args}
        if isinstance(explicit_id, str) and explicit_id:
            try:
                item["id"] = explicit_id.format(**vars_map)
            except Exception:
                item["id"] = explicit_id
        expanded.append(item)
    return expanded


def _build_command(command: str, args: dict[str, Any]) -> list[str]:
    """Translate job args into `python -m deltatau_audit ...` argv list."""
    cmd = [sys.executable, "-m", "deltatau_audit", command]

    for key, value in args.items():
        flag = _to_cli_flag(str(key))
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        if value is None:
            continue
        if isinstance(value, list):
            if not value:
                continue
            cmd.append(flag)
            cmd.extend(_coerce_str(v) for v in value)
            continue
        cmd.extend([flag, _coerce_str(value)])
    return cmd


def _job_id(base_name: str, vars_map: dict[str, Any], index: int) -> str:
    if not vars_map:
        return f"{base_name}_{index:03d}"
    parts = [f"{k}-{vars_map[k]}" for k in sorted(vars_map.keys())]
    safe = "_".join(_coerce_str(p).replace(os.sep, "-") for p in parts)
    return f"{base_name}_{safe}"


def _is_success_summary(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(data, dict) and "summary" in data


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(loaded, dict):
        return loaded
    return None


def _infer_worst_scenario(
    per_scenario_scores: dict[str, Any], fallback: str | None
) -> tuple[str | None, dict[str, Any]]:
    if isinstance(fallback, str) and fallback in per_scenario_scores:
        entry = per_scenario_scores.get(fallback)
        return fallback, entry if isinstance(entry, dict) else {}

    best_name: str | None = None
    best_entry: dict[str, Any] = {}
    best_ratio = float("inf")
    for name, raw in per_scenario_scores.items():
        if not isinstance(raw, dict):
            continue
        ratio = _safe_float(raw.get("return_ratio"))
        if ratio is None:
            continue
        if ratio < best_ratio:
            best_ratio = ratio
            best_name = str(name)
            best_entry = raw
    return best_name, best_entry


def _seed_sweep_ci_lower(result: dict[str, Any], scenario: str | None) -> float | None:
    if not scenario:
        return None
    seed_sweep = result.get("seed_sweep")
    if not isinstance(seed_sweep, dict):
        return None

    aggregate = seed_sweep.get("aggregate")
    if not isinstance(aggregate, dict):
        return None
    scenario_metrics = aggregate.get("scenario_metrics")
    if not isinstance(scenario_metrics, dict):
        return None
    sc = scenario_metrics.get(scenario)
    if not isinstance(sc, dict):
        return None
    ret = sc.get("return_ratio")
    if not isinstance(ret, dict):
        return None
    return _safe_float(ret.get("ci_lower"))


def extract_job_metrics(summary_path: Path) -> dict[str, Any]:
    """Extract stable benchmark metrics from one audit summary.json."""
    result = _read_json(summary_path)
    if not result:
        return {}

    summary = result.get("summary")
    robustness = result.get("robustness")
    diagnosis = result.get("diagnosis")
    manifest = result.get("manifest")

    if not isinstance(summary, dict) or not isinstance(robustness, dict):
        return {}

    per_scenario = robustness.get("per_scenario_scores")
    per_scores: dict[str, Any] = (
        per_scenario if isinstance(per_scenario, dict) else {}
    )
    stress = robustness.get("stress")
    stress_entry: dict[str, Any] = stress if isinstance(stress, dict) else {}
    worst_case = stress_entry.get("worst_case")
    worst: dict[str, Any] = worst_case if isinstance(worst_case, dict) else {}

    guessed_worst, guessed_entry = _infer_worst_scenario(
        per_scores,
        worst.get("scenario") if isinstance(worst.get("scenario"), str) else None,
    )
    worst_scenario = guessed_worst
    worst_ratio = _safe_float(guessed_entry.get("return_ratio"))
    if worst_ratio is None:
        worst_ratio = _safe_float(worst.get("return_ratio"))
    ci_lower = _safe_float(guessed_entry.get("ci_lower"))
    if ci_lower is None:
        ci_lower = _seed_sweep_ci_lower(result, worst_scenario)

    manifest_protocol = "custom"
    if isinstance(manifest, dict):
        protocol = manifest.get("protocol")
        if isinstance(protocol, dict):
            name = protocol.get("name")
            if name:
                manifest_protocol = str(name)

    primary_pattern = None
    if isinstance(diagnosis, dict):
        pattern = diagnosis.get("primary_pattern")
        if pattern:
            primary_pattern = str(pattern)

    stress_threshold = _safe_float(summary.get("stress_threshold"))
    stress_ci_gate_pass: bool | None = None
    if ci_lower is not None and stress_threshold is not None:
        stress_ci_gate_pass = ci_lower >= stress_threshold

    return {
        "protocol": manifest_protocol,
        "deployment_score": _safe_float(summary.get("deployment_score")),
        "deployment_rating": summary.get("deployment_rating"),
        "stress_score": _safe_float(summary.get("stress_score")),
        "stress_rating": summary.get("stress_rating"),
        "quadrant": summary.get("quadrant"),
        "stress_worst_scenario": worst_scenario,
        "stress_worst_return_ratio": worst_ratio,
        "stress_worst_ci_lower": ci_lower,
        "stress_ci_gate_pass": stress_ci_gate_pass,
        "diagnosis_pattern": primary_pattern,
    }


def _job_param(job: dict[str, Any], name: str) -> Any:
    vars_map = job.get("vars")
    if isinstance(vars_map, dict) and name in vars_map:
        return vars_map[name]
    args = job.get("args")
    if isinstance(args, dict) and name in args:
        return args[name]
    return None


def _job_metrics(job: dict[str, Any]) -> dict[str, Any]:
    maybe = job.get("result")
    if isinstance(maybe, dict) and maybe:
        return maybe
    summary_path = job.get("summary_path")
    if isinstance(summary_path, str):
        path = Path(summary_path)
        if path.exists():
            return extract_job_metrics(path)
    return {}


def build_submission_rows(run_summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten bench summary jobs into a tabular submission-friendly schema."""
    jobs = run_summary.get("jobs")
    if not isinstance(jobs, list):
        return []

    rows: list[dict[str, Any]] = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        metrics = _job_metrics(job)
        row = {
            "job_id": job.get("id"),
            "status": job.get("status"),
            "command": job.get("command"),
            "env": _job_param(job, "env"),
            "algo": _job_param(job, "algo"),
            "seed": _job_param(job, "seed"),
            "variant": _job_param(job, "variant"),
            "protocol": metrics.get("protocol", _job_param(job, "protocol")),
            "deployment_score": metrics.get("deployment_score"),
            "deployment_rating": metrics.get("deployment_rating"),
            "stress_score": metrics.get("stress_score"),
            "stress_rating": metrics.get("stress_rating"),
            "stress_worst_scenario": metrics.get("stress_worst_scenario"),
            "stress_worst_return_ratio": metrics.get("stress_worst_return_ratio"),
            "stress_worst_ci_lower": metrics.get("stress_worst_ci_lower"),
            "stress_ci_gate_pass": metrics.get("stress_ci_gate_pass"),
            "quadrant": metrics.get("quadrant"),
            "diagnosis_pattern": metrics.get("diagnosis_pattern"),
            "summary_path": job.get("summary_path"),
        }
        rows.append(row)
    return rows


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _rows_to_markdown(rows: list[dict[str, Any]]) -> str:
    header = "| " + " | ".join(_SUBMISSION_COLUMNS) + " |"
    sep = "| " + " | ".join(["---"] * len(_SUBMISSION_COLUMNS)) + " |"
    lines = [header, sep]
    for row in rows:
        cells = [_format_cell(row.get(col)) for col in _SUBMISSION_COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    if not rows:
        lines.append("| " + " | ".join([""] * len(_SUBMISSION_COLUMNS)) + " |")
    return "\n".join(lines) + "\n"


def write_submission_tables(
    rows: list[dict[str, Any]], output_root: str | os.PathLike[str]
) -> dict[str, str]:
    """Write submission table artifacts (CSV + Markdown)."""
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    csv_path = root / "submission_table.csv"
    md_path = root / "submission_table.md"

    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(_SUBMISSION_COLUMNS))
        writer.writeheader()
        for row in rows:
            payload = {k: _format_cell(row.get(k)) for k in _SUBMISSION_COLUMNS}
            writer.writerow(payload)

    md_path.write_text(_rows_to_markdown(rows), encoding="utf-8")
    return {
        "submission_csv": str(csv_path),
        "submission_md": str(md_path),
        "submission_rows": str(len(rows)),
    }


def load_run_summary(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load bench_summary.json from either a file path or output-root dir."""
    p = Path(path)
    if p.is_dir():
        p = p / "bench_summary.json"
    data = _read_json(p)
    if data is None:
        raise ValueError(f"Invalid bench summary JSON: {p}")
    return data


def write_submission_tables_for_summary(
    run_summary: dict[str, Any], output_root: str | os.PathLike[str] | None = None
) -> dict[str, str]:
    """Regenerate submission table artifacts from a loaded bench summary."""
    root = output_root or run_summary.get("output_root", "bench_runs")
    rows = build_submission_rows(run_summary)
    return write_submission_tables(rows, root)


def _finalize_run_summary(base_out: Path, run_summary: dict[str, Any]) -> dict[str, Any]:
    artifacts = write_submission_tables_for_summary(run_summary, base_out)
    run_summary["artifacts"] = artifacts
    summary_path = base_out / "bench_summary.json"
    summary_path.write_text(
        json.dumps(run_summary, indent=2, default=str),
        encoding="utf-8",
    )
    return run_summary


def _enforce_protocol(
    args: dict[str, Any], *, protocol_name: str | None, allow_override: bool
) -> None:
    if not protocol_name:
        return
    if allow_override and "protocol" in args:
        return
    args["protocol"] = str(protocol_name)


def _pick_existing_summary(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if _is_success_summary(p):
            return p
    return None


def run_manifest(
    manifest_path: str | os.PathLike[str],
    *,
    output_root: str | os.PathLike[str] | None = None,
    resume: bool = True,
    fail_fast: bool = False,
    protocol_name: str | None = None,
    allow_protocol_override: bool = False,
) -> dict[str, Any]:
    """Run benchmark jobs described in a manifest and return summary."""
    manifest = _load_manifest(manifest_path)
    jobs = manifest.get("jobs", [])
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Manifest must contain non-empty `jobs` list")

    base_out = Path(output_root or manifest.get("output_dir", "bench_runs")).resolve()
    base_out.mkdir(parents=True, exist_ok=True)

    run_summary: dict[str, Any] = {
        "manifest": str(Path(manifest_path).resolve()),
        "output_root": str(base_out),
        "started_at": time.time(),
        "resume": bool(resume),
        "fail_fast": bool(fail_fast),
        "protocol": {
            "forced": protocol_name,
            "allow_override": bool(allow_protocol_override),
        },
        "jobs": [],
        "status": "running",
    }

    total_failed = 0
    total_skipped = 0
    total_ok = 0
    job_counter = 0

    for raw_job in jobs:
        if not isinstance(raw_job, dict):
            raise ValueError("Each job in manifest.jobs must be an object")
        name = str(raw_job.get("name", "job"))
        command = raw_job.get("command")
        if not isinstance(command, str) or not command:
            raise ValueError(f"Manifest job `{name}` missing valid `command`")

        expanded = _expand_job_matrix(raw_job)
        for item in expanded:
            job_counter += 1
            vars_map = item["vars"]
            args = dict(item["args"])
            _enforce_protocol(
                args,
                protocol_name=protocol_name,
                allow_override=allow_protocol_override,
            )

            explicit_jid = item.get("id")
            if isinstance(explicit_jid, str) and explicit_jid.strip():
                jid = explicit_jid.strip()
            else:
                jid = _job_id(name, vars_map, job_counter)
            job_out = (base_out / jid).resolve()
            job_out.mkdir(parents=True, exist_ok=True)

            if "out" not in args:
                args["out"] = str(job_out)

            target_out = Path(_coerce_str(args.get("out", str(job_out)))).resolve()
            summary_candidates = [target_out / "summary.json"]
            if target_out != job_out:
                summary_candidates.append(job_out / "summary.json")
            existing_summary = _pick_existing_summary(summary_candidates)

            if resume and existing_summary is not None:
                total_skipped += 1
                run_summary["jobs"].append(
                    {
                        "id": jid,
                        "name": name,
                        "command": command,
                        "vars": vars_map,
                        "args": args,
                        "status": "skipped",
                        "reason": "resume: summary.json already exists",
                        "returncode": 0,
                        "duration_s": 0.0,
                        "summary_path": str(existing_summary),
                        "result": extract_job_metrics(existing_summary),
                    }
                )
                continue

            cmd = _build_command(command, args)
            log_path = job_out / "bench_run.log"
            err_path = job_out / "bench_run.err.log"
            t0 = time.time()
            with log_path.open("w", encoding="utf-8", errors="replace") as log_fp, err_path.open(
                "w", encoding="utf-8", errors="replace"
            ) as err_fp:
                proc = subprocess.run(cmd, stdout=log_fp, stderr=err_fp, text=True)
            elapsed = time.time() - t0

            status = "passed" if proc.returncode == 0 else "failed"
            if status == "passed":
                total_ok += 1
            else:
                total_failed += 1

            post_summary = _pick_existing_summary(summary_candidates)
            if post_summary is None and (job_out / "summary.json").exists():
                post_summary = job_out / "summary.json"

            run_summary["jobs"].append(
                {
                    "id": jid,
                    "name": name,
                    "command": command,
                    "vars": vars_map,
                    "args": args,
                    "status": status,
                    "returncode": int(proc.returncode),
                    "duration_s": elapsed,
                    "stdout_log": str(log_path),
                    "stderr_log": str(err_path),
                    "summary_path": str(post_summary) if post_summary else None,
                    "result": extract_job_metrics(post_summary)
                    if post_summary is not None
                    else {},
                }
            )

            if fail_fast and proc.returncode != 0:
                run_summary["status"] = "failed"
                run_summary["finished_at"] = time.time()
                run_summary["counts"] = {
                    "passed": total_ok,
                    "failed": total_failed,
                    "skipped": total_skipped,
                }
                return _finalize_run_summary(base_out, run_summary)

    run_summary["finished_at"] = time.time()
    run_summary["counts"] = {
        "passed": total_ok,
        "failed": total_failed,
        "skipped": total_skipped,
    }
    run_summary["status"] = "failed" if total_failed > 0 else "passed"
    return _finalize_run_summary(base_out, run_summary)
