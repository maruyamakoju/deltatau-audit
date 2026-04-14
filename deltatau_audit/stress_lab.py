"""Stress failure analysis and ablation planning utilities."""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

_SPEED_SCENARIO_RE = re.compile(r"^speed_(\d+(?:\.\d+)?)x$")

INTERVENTIONS: dict[str, dict[str, str]] = {
    "intervention1_curriculum": {
        "title": "Intervention 1: Speed-randomization curriculum",
        "short": "speed curriculum",
        "description": (
            "Train with progressive speed randomization (e.g., 1x→2x→3x→5x) "
            "to broaden timing support while preserving baseline competence."
        ),
    },
    "intervention2_time_feature": {
        "title": "Intervention 2: Time-feature observation",
        "short": "time features",
        "description": (
            "Add explicit timing features (dt / elapsed / phase) to observations "
            "so the policy can condition actions on execution rate."
        ),
    },
    "intervention3_memory": {
        "title": "Intervention 3: Recurrent or frame-stack memory",
        "short": "memory upgrade",
        "description": (
            "Use recurrence (GRU/LSTM) or stronger frame stacking to mitigate "
            "aliasing and partial observability under high-frequency control."
        ),
    },
}

_BASE_ABLATION_VARIANTS: tuple[str, ...] = (
    "baseline",
    "intervention1_curriculum",
    "intervention2_time_feature",
    "intervention1_plus_2",
)


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    loaded = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object: {p}")
    return loaded


def _is_nonincreasing(values: list[float], tol: float = 0.02) -> bool:
    for i in range(1, len(values)):
        if values[i] > values[i - 1] + tol:
            return False
    return True


def _speed_curve(per_scenario_scores: dict[str, Any]) -> list[dict[str, Any]]:
    curve: list[tuple[float, str, dict[str, Any]]] = []
    for scenario, raw in per_scenario_scores.items():
        if not isinstance(raw, dict):
            continue
        m = _SPEED_SCENARIO_RE.match(str(scenario))
        if not m:
            continue
        speed = float(m.group(1))
        curve.append((speed, str(scenario), raw))
    curve.sort(key=lambda row: row[0])

    out: list[dict[str, Any]] = []
    for speed, scenario, raw in curve:
        out.append(
            {
                "speed": speed,
                "scenario": scenario,
                "return_ratio": _safe_float(raw.get("return_ratio")),
                "ci_lower": _safe_float(raw.get("ci_lower")),
                "ci_upper": _safe_float(raw.get("ci_upper")),
                "rmse_ratio": _safe_float(raw.get("rmse_ratio")),
            }
        )
    return out


def _infer_worst_scenario(
    robustness: dict[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    per_raw = robustness.get("per_scenario_scores")
    per_scores: dict[str, Any] = per_raw if isinstance(per_raw, dict) else {}

    stress_raw = robustness.get("stress")
    stress: dict[str, Any] = stress_raw if isinstance(stress_raw, dict) else {}
    worst_raw = stress.get("worst_case")
    worst: dict[str, Any] = worst_raw if isinstance(worst_raw, dict) else {}

    scenario = worst.get("scenario")
    if isinstance(scenario, str) and scenario in per_scores:
        sc = per_scores.get(scenario)
        if isinstance(sc, dict):
            return scenario, sc
        return scenario, {}

    best_name: str | None = None
    best_ratio = float("inf")
    best_entry: dict[str, Any] = {}
    for name, raw in per_scores.items():
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


def _classify_curve(
    curve: list[dict[str, Any]],
    *,
    worst_ratio: float | None,
    worst_rmse_ratio: float | None,
) -> tuple[str, str, str]:
    """Return (pattern_id, pattern_label, rationale)."""
    ratios = [pt["return_ratio"] for pt in curve if isinstance(pt.get("return_ratio"), float)]
    speeds = [pt["speed"] for pt in curve if isinstance(pt.get("return_ratio"), float)]

    if worst_ratio is not None and (worst_ratio < 0.30 or (worst_rmse_ratio or 1.0) >= 1.60):
        return (
            "control_saturation",
            "Control-bandwidth saturation",
            "Return collapses to very low values and/or RMSE explodes at stress speed.",
        )

    if len(ratios) >= 2 and len(speeds) >= 2:
        drops = [ratios[i - 1] - ratios[i] for i in range(1, len(ratios))]
        max_drop = max(drops) if drops else 0.0
        max_idx = drops.index(max_drop) + 1 if drops else 0
        collapse_speed = speeds[max_idx] if max_idx < len(speeds) else speeds[-1]

        if max_drop >= 0.20 and collapse_speed >= 5.0:
            return (
                "threshold_collapse",
                "Threshold-like collapse",
                "Performance is stable at lower speeds then drops sharply at high speed.",
            )

        if _is_nonincreasing(ratios) and ratios[-1] <= ratios[0] - 0.20:
            return (
                "gradual_shift",
                "Gradual distribution shift",
                "Performance decays with speed in a mostly monotonic pattern.",
            )

    return (
        "mixed_shift",
        "Mixed stress degradation",
        "Stress degradation appears non-monotonic; multiple mechanisms may coexist.",
    )


def _mechanism_from_pattern(
    pattern_id: str,
) -> tuple[str, str, str]:
    if pattern_id == "threshold_collapse":
        return (
            "A",
            "Observation aliasing under high frequency",
            "High-speed execution compresses useful temporal cues, causing information loss.",
        )
    if pattern_id == "gradual_shift":
        return (
            "B",
            "Implicit step-time assumption",
            "Policy appears to rely on a fixed step-rate prior and degrades as dt shifts.",
        )
    if pattern_id == "control_saturation":
        return (
            "C",
            "Action/control saturation",
            "Controller bandwidth is insufficient under stress speeds; action dynamics break.",
        )
    return (
        "B/C",
        "Hybrid timing + control failure",
        "Observed stress curve suggests both timing mismatch and control instability.",
    )


def _recommended_variants(*, include_intervention3: bool) -> list[str]:
    variants = list(_BASE_ABLATION_VARIANTS)
    if include_intervention3:
        variants.append("intervention3_memory")
    return variants


def analyze_stress_result(
    result: dict[str, Any],
    *,
    stress_threshold: float = 0.50,
    include_intervention3: bool = True,
) -> dict[str, Any]:
    """Analyze stress failure mode from one audit summary payload."""
    summary_raw = result.get("summary")
    summary = summary_raw if isinstance(summary_raw, dict) else {}
    robustness_raw = result.get("robustness")
    robustness = robustness_raw if isinstance(robustness_raw, dict) else {}
    diagnosis_raw = result.get("diagnosis")
    diagnosis = diagnosis_raw if isinstance(diagnosis_raw, dict) else {}

    worst_scenario, worst_entry = _infer_worst_scenario(robustness)
    worst_ratio = _safe_float(worst_entry.get("return_ratio"))
    worst_rmse_ratio = _safe_float(worst_entry.get("rmse_ratio"))
    worst_ci_lower = _safe_float(worst_entry.get("ci_lower"))
    worst_ci_upper = _safe_float(worst_entry.get("ci_upper"))

    if worst_ratio is None:
        stress_score = _safe_float(summary.get("stress_score"))
        if stress_score is not None:
            worst_ratio = stress_score
    gate_value = worst_ci_lower if worst_ci_lower is not None else worst_ratio
    gate_pass = gate_value >= float(stress_threshold) if isinstance(gate_value, float) else False

    per_raw = robustness.get("per_scenario_scores")
    per_scores = per_raw if isinstance(per_raw, dict) else {}
    curve = _speed_curve(per_scores)

    pattern_id, pattern_label, pattern_rationale = _classify_curve(
        curve,
        worst_ratio=worst_ratio,
        worst_rmse_ratio=worst_rmse_ratio,
    )
    mechanism_code, mechanism_name, mechanism_rationale = _mechanism_from_pattern(pattern_id)

    variants = _recommended_variants(include_intervention3=include_intervention3)
    recommendations: list[dict[str, str]] = []
    for variant in variants:
        if variant == "baseline":
            recommendations.append(
                {
                    "variant": "baseline",
                    "title": "Baseline (no intervention)",
                    "description": "Reference checkpoint used as control condition.",
                }
            )
            continue
        if variant == "intervention1_plus_2":
            recommendations.append(
                {
                    "variant": variant,
                    "title": "Intervention 1 + 2 combined",
                    "description": (
                        "Combine speed curriculum with explicit time features; "
                        "this is the primary candidate for strong stress recovery."
                    ),
                }
            )
            continue
        base = INTERVENTIONS.get(variant, {})
        recommendations.append(
            {
                "variant": variant,
                "title": base.get("title", variant),
                "description": base.get("description", ""),
            }
        )

    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stress_threshold": float(stress_threshold),
        "stress_gate_metric": "worst_case_ci_lower_or_return_ratio",
        "stress_gate_value": gate_value,
        "stress_gate_pass": gate_pass,
        "worst_scenario": {
            "scenario": worst_scenario,
            "return_ratio": worst_ratio,
            "rmse_ratio": worst_rmse_ratio,
            "ci_lower": worst_ci_lower,
            "ci_upper": worst_ci_upper,
            "significant": bool(worst_entry.get("significant", False)),
        },
        "speed_curve": curve,
        "pattern": {
            "id": pattern_id,
            "label": pattern_label,
            "rationale": pattern_rationale,
        },
        "mechanism": {
            "code": mechanism_code,
            "name": mechanism_name,
            "rationale": mechanism_rationale,
        },
        "diagnosis_summary_line": diagnosis.get("summary_line"),
        "ablation_variants": variants,
        "recommended_interventions": recommendations,
    }


def analyze_stress_summary(
    summary_json: str | Path,
    *,
    stress_threshold: float = 0.50,
    include_intervention3: bool = True,
) -> dict[str, Any]:
    """Analyze one summary.json file and return structured stress analysis."""
    result = _load_json(summary_json)
    analysis = analyze_stress_result(
        result,
        stress_threshold=stress_threshold,
        include_intervention3=include_intervention3,
    )
    analysis["summary_json"] = str(Path(summary_json).resolve())
    return analysis


def render_stress_analysis_markdown(analysis: dict[str, Any]) -> str:
    """Render stress analysis as concise Markdown."""
    worst = analysis.get("worst_scenario", {})
    pattern = analysis.get("pattern", {})
    mechanism = analysis.get("mechanism", {})
    variants = analysis.get("ablation_variants", [])
    gate_pass = bool(analysis.get("stress_gate_pass"))

    lines: list[str] = []
    lines.append("# Stress Failure Analysis")
    lines.append("")
    lines.append(f"- summary_json: `{analysis.get('summary_json', '')}`")
    lines.append(f"- stress_threshold: `{analysis.get('stress_threshold')}`")
    lines.append(f"- stress_gate_value: `{analysis.get('stress_gate_value')}`")
    lines.append(f"- stress_gate_pass: `{gate_pass}`")
    lines.append("")
    lines.append("## Worst Scenario")
    lines.append("")
    lines.append(f"- scenario: `{worst.get('scenario')}`")
    lines.append(f"- return_ratio: `{worst.get('return_ratio')}`")
    lines.append(f"- rmse_ratio: `{worst.get('rmse_ratio')}`")
    lines.append(f"- ci_lower: `{worst.get('ci_lower')}`")
    lines.append(f"- ci_upper: `{worst.get('ci_upper')}`")
    lines.append("")
    lines.append("## Mechanistic Classification")
    lines.append("")
    lines.append(f"- pattern: `{pattern.get('label')}` ({pattern.get('id')})")
    lines.append(f"- mechanism: `{mechanism.get('code')}` {mechanism.get('name')}")
    lines.append(f"- rationale: {mechanism.get('rationale')}")
    lines.append("")
    lines.append("## Ablation Variants")
    lines.append("")
    for v in variants if isinstance(variants, list) else []:
        lines.append(f"- `{v}`")
    lines.append("")

    curve = analysis.get("speed_curve", [])
    if isinstance(curve, list) and curve:
        lines.append("## Speed Curve")
        lines.append("")
        lines.append("| scenario | speed | return_ratio | ci_lower | ci_upper |")
        lines.append("| --- | --- | --- | --- | --- |")
        for pt in curve:
            if not isinstance(pt, dict):
                continue
            lines.append(
                "| {scenario} | {speed} | {ret} | {lo} | {hi} |".format(
                    scenario=pt.get("scenario"),
                    speed=pt.get("speed"),
                    ret=pt.get("return_ratio"),
                    lo=pt.get("ci_lower"),
                    hi=pt.get("ci_upper"),
                )
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _yaml_or_json_dump(payload: dict[str, Any], path: Path) -> Path:
    """Write YAML when available; fall back to JSON."""
    suffix = path.suffix.lower()
    if suffix not in {".yaml", ".yml", ".json"}:
        path = path.with_suffix(".yaml")

    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    try:
        import yaml  # type: ignore
    except ImportError:
        json_path = path.with_suffix(".json")
        json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return json_path

    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    return path


def write_stress_analysis_artifacts(
    summary_json: str | Path,
    *,
    out_dir: str | Path,
    stress_threshold: float = 0.50,
    include_intervention3: bool = True,
) -> dict[str, str]:
    """Write stress analysis JSON + Markdown artifacts."""
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    analysis = analyze_stress_summary(
        summary_json,
        stress_threshold=stress_threshold,
        include_intervention3=include_intervention3,
    )

    json_path = out / "stress_analysis.json"
    md_path = out / "stress_analysis.md"
    json_path.write_text(json.dumps(analysis, indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_stress_analysis_markdown(analysis), encoding="utf-8")
    return {
        "analysis_json": str(json_path),
        "analysis_md": str(md_path),
    }


def build_ablation_manifest(
    *,
    env: str,
    algo: str,
    model_template: str,
    seeds: list[int] | None = None,
    episodes: int = 50,
    speeds: list[int] | None = None,
    include_intervention3: bool = False,
    protocol: str = "research",
    ci_gate_mode: str = "worst_ci_lower",
    output_dir: str = "ablation_runs",
) -> dict[str, Any]:
    """Create bench manifest for stress intervention ablation audits."""
    speed_list = speeds if speeds is not None else [1, 2, 3, 5, 8]
    seed_list = seeds if seeds is not None else [0, 1, 2, 3, 4]
    variants = _recommended_variants(include_intervention3=include_intervention3)

    jobs: list[dict[str, Any]] = []
    for variant in variants:
        args: dict[str, Any] = {
            "algo": str(algo),
            "env": str(env),
            "model": str(model_template),
            "seed": "{seed}",
            "seeds": ["{seed}"],
            "episodes": int(episodes),
            "speeds": [int(s) for s in speed_list],
            "adaptive": False,
            "protocol": str(protocol),
            "allow_protocol_override": True,
            "ci": True,
            "ci_gate_mode": str(ci_gate_mode),
            "out": f"{output_dir}/{variant}/seed_{{seed}}",
            "explain_fail": True,
        }
        if variant in {"intervention2_time_feature", "intervention1_plus_2"}:
            args["env_wrap_time_feature"] = True
        if variant == "intervention3_memory":
            args["env_wrap_frame_stack"] = 4
            args["env_wrap_flatten_obs"] = True

        jobs.append(
            {
                "name": f"stress_ablation_{variant}",
                "command": "audit-sb3",
                "matrix": {"variant": [variant], "seed": seed_list},
                "args": args,
            }
        )

    return {
        "output_dir": output_dir,
        "meta": {
            "purpose": "stress-mechanism ablation",
            "model_template_hint": (
                "Use {variant} and/or {seed} placeholders in model path, "
                "e.g. checkpoints/{variant}/seed_{seed}/model.zip"
            ),
        },
        "jobs": jobs,
    }


def write_ablation_plan_artifacts(
    *,
    analysis: dict[str, Any],
    manifest: dict[str, Any],
    out_dir: str | Path,
) -> dict[str, str]:
    """Write ablation manifest + markdown plan artifacts."""
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = _yaml_or_json_dump(manifest, out / "ablation_manifest.yaml")
    md_path = out / "ablation_plan.md"

    mechanism = analysis.get("mechanism", {})
    pattern = analysis.get("pattern", {})
    worst = analysis.get("worst_scenario", {})
    variants = analysis.get("ablation_variants", [])

    lines: list[str] = []
    lines.append("# Stress Ablation Plan")
    lines.append("")
    lines.append("## Rationale")
    lines.append("")
    lines.append(f"- pattern: `{pattern.get('label')}` ({pattern.get('id')})")
    lines.append(f"- mechanism: `{mechanism.get('code')}` {mechanism.get('name')}")
    lines.append(f"- worst_scenario: `{worst.get('scenario')}`")
    lines.append(f"- gate_value: `{analysis.get('stress_gate_value')}`")
    lines.append(f"- gate_pass: `{analysis.get('stress_gate_pass')}`")
    lines.append("")
    lines.append("## Variants")
    lines.append("")
    for v in variants if isinstance(variants, list) else []:
        if v == "baseline":
            lines.append("- `baseline`: no intervention")
            continue
        if v == "intervention1_plus_2":
            lines.append("- `intervention1_plus_2`: combined intervention")
            continue
        info = INTERVENTIONS.get(str(v), {})
        lines.append(f"- `{v}`: {info.get('description', '')}")
    lines.append("")
    lines.append("## Generated Manifest")
    lines.append("")
    lines.append(f"- `{manifest_path}`")
    lines.append(f"- Run: `python -m deltatau_audit bench run --manifest {manifest_path}`")
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "ablation_manifest": str(manifest_path),
        "ablation_plan_md": str(md_path),
    }


def _variant_training_env(
    env_id: str,
    *,
    variant: str,
    seed: int | None,
    base_speed: int,
    jitter: int,
    phase_period: int,
    frame_stack: int,
):
    import gymnasium as gym

    from .wrappers.speed import JitterWrapper
    from .wrappers.time_feature import TimeFeatureWrapper

    env = gym.make(env_id)

    if variant in {"intervention1_curriculum", "intervention1_plus_2"}:
        env = JitterWrapper(env, base_speed=base_speed, jitter=jitter, seed=seed)
    if variant in {"intervention2_time_feature", "intervention1_plus_2"}:
        env = TimeFeatureWrapper(env, phase_period=phase_period)
    if variant == "intervention3_memory":
        try:
            from gymnasium.wrappers import FlattenObservation

            try:
                from gymnasium.wrappers import FrameStackObservation
            except Exception:
                from gymnasium.wrappers import FrameStack as FrameStackObservation

            env = FrameStackObservation(env, stack_size=max(2, int(frame_stack)))
            env = FlattenObservation(env)
        except Exception:
            # Fallback if frame-stack wrappers are unavailable in local gymnasium build.
            env = TimeFeatureWrapper(env, phase_period=phase_period)

    return env


def _algo_cls(algo: str):
    try:
        import stable_baselines3 as sb3
    except ImportError as exc:
        raise ImportError(
            'stable-baselines3 is required for stress ablation training. Install: pip install "deltatau-audit[sb3]"'
        ) from exc

    mapping = {
        "ppo": sb3.PPO,
        "sac": sb3.SAC,
        "td3": sb3.TD3,
        "a2c": sb3.A2C,
    }
    algo_name = str(algo).lower()
    if algo_name not in mapping:
        raise ValueError(f"Unsupported algo: {algo}")
    return mapping[algo_name]


def train_sb3_ablation_models(
    *,
    env: str,
    algo: str,
    out_root: str | Path,
    seeds: list[int] | None = None,
    variants: list[str] | None = None,
    timesteps: int = 30_000,
    device: str = "cpu",
    base_speed: int = 3,
    jitter: int = 2,
    phase_period: int = 200,
    frame_stack: int = 4,
    force: bool = False,
    fail_fast: bool = False,
    verbose: int = 0,
) -> dict[str, Any]:
    """Train SB3 ablation variants and save checkpoints for bench manifests.

    Output path per run:
        {out_root}/{variant}/seed_{seed}/model.zip
    """
    root = Path(out_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    seed_list = list(seeds) if seeds is not None else [0, 1, 2, 3, 4]
    variant_list = list(variants) if variants is not None else list(_BASE_ABLATION_VARIANTS)
    algo_class = _algo_cls(algo)

    jobs: list[dict[str, Any]] = []
    n_ok = 0
    n_skip = 0
    n_fail = 0

    for variant in variant_list:
        for seed in seed_list:
            model_path = root / variant / f"seed_{seed}" / "model.zip"
            model_path.parent.mkdir(parents=True, exist_ok=True)

            if model_path.exists() and not force:
                jobs.append(
                    {
                        "variant": variant,
                        "seed": int(seed),
                        "status": "skipped",
                        "reason": "exists",
                        "model_path": str(model_path),
                    }
                )
                n_skip += 1
                continue

            t0 = dt.datetime.now(dt.timezone.utc).timestamp()
            try:
                env_train = _variant_training_env(
                    env,
                    variant=variant,
                    seed=int(seed),
                    base_speed=base_speed,
                    jitter=jitter,
                    phase_period=phase_period,
                    frame_stack=frame_stack,
                )
                model = algo_class(
                    "MlpPolicy",
                    env_train,
                    seed=int(seed),
                    device=device,
                    verbose=int(verbose),
                )
                model.learn(total_timesteps=int(timesteps))
                save_stem = str(model_path.with_suffix(""))
                model.save(save_stem)
                env_train.close()

                elapsed = dt.datetime.now(dt.timezone.utc).timestamp() - t0
                jobs.append(
                    {
                        "variant": variant,
                        "seed": int(seed),
                        "status": "trained",
                        "model_path": str(model_path),
                        "timesteps": int(timesteps),
                        "duration_s": float(elapsed),
                    }
                )
                n_ok += 1
            except Exception as exc:
                elapsed = dt.datetime.now(dt.timezone.utc).timestamp() - t0
                jobs.append(
                    {
                        "variant": variant,
                        "seed": int(seed),
                        "status": "failed",
                        "error": str(exc),
                        "model_path": str(model_path),
                        "duration_s": float(elapsed),
                    }
                )
                n_fail += 1
                if fail_fast:
                    return {
                        "status": "failed",
                        "env": str(env),
                        "algo": str(algo),
                        "out_root": str(root),
                        "counts": {
                            "trained": n_ok,
                            "skipped": n_skip,
                            "failed": n_fail,
                        },
                        "jobs": jobs,
                    }

    status = "failed" if n_fail > 0 else "passed"
    return {
        "status": status,
        "env": str(env),
        "algo": str(algo),
        "out_root": str(root),
        "counts": {
            "trained": n_ok,
            "skipped": n_skip,
            "failed": n_fail,
        },
        "jobs": jobs,
    }


def write_training_summary(summary: dict[str, Any], *, out_dir: str | Path) -> dict[str, str]:
    """Persist stress ablation training summary as JSON/Markdown."""
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "stress_train_summary.json"
    md_path = out / "stress_train_summary.md"

    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    counts = summary.get("counts", {})
    lines = [
        "# Stress Ablation Training Summary",
        "",
        f"- env: `{summary.get('env')}`",
        f"- algo: `{summary.get('algo')}`",
        f"- out_root: `{summary.get('out_root')}`",
        f"- status: `{summary.get('status')}`",
        f"- trained: `{counts.get('trained', 0)}`",
        f"- skipped: `{counts.get('skipped', 0)}`",
        f"- failed: `{counts.get('failed', 0)}`",
        "",
        "## Jobs",
        "",
        "| variant | seed | status | model_path |",
        "| --- | --- | --- | --- |",
    ]
    jobs = summary.get("jobs", [])
    if isinstance(jobs, list):
        for job in jobs:
            if not isinstance(job, dict):
                continue
            lines.append(
                "| {variant} | {seed} | {status} | {model} |".format(
                    variant=job.get("variant"),
                    seed=job.get("seed"),
                    status=job.get("status"),
                    model=job.get("model_path"),
                )
            )
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "training_json": str(json_path),
        "training_md": str(md_path),
    }
