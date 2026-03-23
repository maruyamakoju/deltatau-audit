"""Master submission preparation script.

Orchestrates all steps needed to prepare the paper for arxiv/NeurIPS submission:
1. Run 10-seed Chain experiments
2. Run 10-seed CartPole ablation
3. Run dm_control experiments (if available)
4. Generate all paper figures
5. Compute full statistical table
6. Generate LaTeX tables
7. Create submission package

Usage:
    python scripts/prepare_submission.py [--skip-training] [--skip-dm-control]
    python scripts/prepare_submission.py --check-only   # check missing artifacts only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submission_health import (
    cartpole_failed_variant_seeds as _shared_cartpole_failed_variant_seeds,
    cartpole_retrain_commands as _shared_cartpole_retrain_commands,
    check_bench_execution as _shared_check_bench_execution,
    expand_manifest_jobs as _shared_expand_manifest_jobs,
    summary_targets_from_manifest as _shared_summary_targets_from_manifest,
)

RUNS_DIR = ROOT / "runs"
DOCS_DIR = ROOT / "docs"
BENCH_DIR = ROOT / "bench"
BENCH_RUNS_DIR = ROOT / "bench_runs"
CARTPOLE_MANIFEST = BENCH_DIR / "high_rigor_10seed_manifest.yaml"
DM_CONTROL_MANIFEST = BENCH_DIR / "dm_control_research_manifest.yaml"
CARTPOLE_BENCH_OUT = BENCH_RUNS_DIR / "cartpole_high_rigor_10seed"
DM_CONTROL_BENCH_OUT = BENCH_RUNS_DIR / "dm_control"


# ── Colour helpers ────────────────────────────────────────────────────────────
class C:
    G = "\033[92m"  # green
    Y = "\033[93m"  # yellow
    R = "\033[91m"  # red
    B = "\033[94m"  # blue
    E = "\033[0m"   # reset


def ok(msg: str) -> None:
    print(f"  {C.G}[OK]{C.E} {msg}")


def warn(msg: str) -> None:
    print(f"  {C.Y}[WARN]{C.E} {msg}")


def err(msg: str) -> None:
    print(f"  {C.R}[ERR]{C.E} {msg}")


def info(msg: str) -> None:
    print(f"  {C.B}[INFO]{C.E} {msg}")


def header(msg: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {msg}")
    print(f"{'='*70}")


def _repo_env() -> dict[str, str]:
    env = os.environ.copy()
    current = env.get("PYTHONPATH", "")
    root_str = str(ROOT)
    env["PYTHONPATH"] = root_str if not current else f"{root_str}{os.pathsep}{current}"
    return env


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


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


def _prioritized_reasons(reasons: Any, *, max_items: int = 4) -> list[str]:
    if not isinstance(reasons, list):
        return []
    normalized: list[str] = [str(reason) for reason in reasons if str(reason).strip()]
    if not normalized:
        return []

    priority_patterns = (
        "runtime failures",
        "quality-gate failures",
        "failed jobs",
        "summaries missing",
        "bench_summary status",
    )

    ranked: list[tuple[int, int, str]] = []
    for idx, reason in enumerate(normalized):
        rank = len(priority_patterns)
        lower = reason.lower()
        for p_idx, pattern in enumerate(priority_patterns):
            if pattern in lower:
                rank = p_idx
                break
        ranked.append((rank, idx, reason))

    ranked.sort(key=lambda row: (row[0], row[1]))
    limit = max(1, int(max_items))
    return [row[2] for row in ranked[:limit]]


def _bench_repair_hint(
    *,
    label: str,
    manifest: str,
    bench: dict[str, Any],
) -> str:
    base_cmd = f"python -m deltatau_audit bench run --manifest {manifest} --protocol paper"
    breakdown = bench.get("failure_breakdown")
    breakdown_dict = breakdown if isinstance(breakdown, dict) else {}
    runtime_failures = _to_int(breakdown_dict.get("runtime_failures"))
    ci_gate_failures = _to_int(breakdown_dict.get("ci_gate_failures"))
    expected_jobs = _to_int(bench.get("expected_jobs"))
    completed_jobs = _to_int(bench.get("completed_jobs"))
    output_root = str(bench.get("output_root", "")).strip()

    if runtime_failures > 0:
        if expected_jobs > 0 and completed_jobs < expected_jobs:
            return f"Resume missing/crashed {label} jobs: {base_cmd}"
        return f"Rerun {label} runtime failures: {base_cmd}"
    if ci_gate_failures > 0:
        if expected_jobs > 0 and ci_gate_failures >= expected_jobs:
            if output_root:
                return (
                    f"{label} all jobs failed the quality gate; diagnose protocol/claim mismatch first: "
                    f"python scripts/analyze_bench_failures.py --bench {output_root}"
                )
            return f"{label} all jobs failed the quality gate; diagnose protocol/claim mismatch before retraining"
        return f"{label} quality gate failed; retrain failing variants then rerun: {base_cmd}"
    return f"Run/repair {label} bench: {base_cmd}"


def _cartpole_failed_variant_seeds(bench: dict[str, Any]) -> dict[str, list[int]]:
    breakdown = bench.get("failure_breakdown")
    return _shared_cartpole_failed_variant_seeds(breakdown)


def _cartpole_retrain_commands(
    variant_seeds: dict[str, list[int]],
    *,
    timesteps: int = 45000,
    force: bool = True,
    base_speed: int = 3,
    jitter: int = 2,
    phase_period: int = 200,
) -> list[str]:
    return _shared_cartpole_retrain_commands(
        variant_seeds,
        timesteps=timesteps,
        force=force,
        base_speed=base_speed,
        jitter=jitter,
        phase_period=phase_period,
    )


def _extract_scores(summary_payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(summary_payload, dict):
        return {
            "deployment_score": None,
            "deployment_rating": None,
            "stress_score": None,
            "stress_rating": None,
        }
    robustness = summary_payload.get("robustness", {})
    if not isinstance(robustness, dict):
        robustness = {}
    deployment = robustness.get("deployment", {})
    if not isinstance(deployment, dict):
        deployment = {}
    stress = robustness.get("stress", {})
    if not isinstance(stress, dict):
        stress = {}
    return {
        "deployment_score": _to_float(deployment.get("return_score")),
        "deployment_rating": deployment.get("rating"),
        "stress_score": _to_float(stress.get("return_score")),
        "stress_rating": stress.get("rating"),
    }


def _expand_manifest_jobs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return _shared_expand_manifest_jobs(manifest)


def _summary_targets_from_manifest(manifest_path: Path, output_root: Path) -> list[Path]:
    return _shared_summary_targets_from_manifest(manifest_path, output_root)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _quality_repair_plan(bench: dict[str, Any]) -> dict[str, Any] | None:
    plan = bench.get("quality_repair_plan")
    return plan if isinstance(plan, dict) else None


def _quality_repair_chain(bench: dict[str, Any]) -> str | None:
    chain = bench.get("repair_command_chain")
    return chain if isinstance(chain, str) and chain.strip() else None


def check_bench_execution(
    manifest_path: Path,
    output_root: Path,
    *,
    protocol: str = "paper",
    job_name: str | None = None,
) -> dict[str, Any]:
    return _shared_check_bench_execution(
        manifest_path,
        output_root,
        protocol=protocol,
        job_name=job_name,
    )


# ── Status checks ─────────────────────────────────────────────────────────────

def check_chain_experiments() -> dict:
    """Check which chain experiments are complete and how many seeds."""
    status = {}
    specs = [
        ("speed_gen_hidden_10seed", True),
        ("speed_gen_chain", False),
        ("hard_chain", False),
    ]
    for exp, required in specs:
        exp_dir = RUNS_DIR / exp
        results_path = exp_dir / "results.json"
        if results_path.exists():
            with open(results_path) as f:
                data = json.load(f)
            # Prefer explicit metadata; fallback to model/seed directories.
            n_seeds = 1
            if isinstance(data, dict) and data:
                first_model = list(data.values())[0]
                if isinstance(first_model, dict):
                    n_seeds = int(first_model.get("n_seeds", 1))
            if n_seeds <= 1:
                model_seed_counts = []
                for model_dir in exp_dir.iterdir() if exp_dir.exists() else []:
                    if not model_dir.is_dir():
                        continue
                    seeds = [
                        d for d in model_dir.iterdir()
                        if d.is_dir() and d.name.startswith("seed_")
                    ]
                    if seeds:
                        model_seed_counts.append(len(seeds))
                if model_seed_counts:
                    n_seeds = max(model_seed_counts)
            status[exp] = {
                "complete": True,
                "seeds": n_seeds,
                "path": str(results_path),
                "required": required,
            }
        else:
            status[exp] = {
                "complete": False,
                "seeds": 0,
                "path": None,
                "required": required,
            }
    return status


def check_halfcheetah() -> dict:
    """Check HalfCheetah audit results and measured improvement."""
    before_dir = ROOT / "halfcheetah_before_after" / "before"
    after_dir = ROOT / "halfcheetah_before_after" / "after"
    before_path = before_dir / "summary.json"
    after_path = after_dir / "summary.json"
    before_payload = _load_json(before_path)
    after_payload = _load_json(after_path)
    before_scores = _extract_scores(before_payload)
    after_scores = _extract_scores(after_payload)

    before_dep = before_scores["deployment_score"]
    after_dep = after_scores["deployment_score"]
    before_stress = before_scores["stress_score"]
    after_stress = after_scores["stress_score"]
    dep_improved = (
        isinstance(before_dep, float)
        and isinstance(after_dep, float)
        and after_dep > before_dep
    )
    stress_improved = (
        isinstance(before_stress, float)
        and isinstance(after_stress, float)
        and after_stress > before_stress
    )
    robust_not_fail = after_scores["deployment_rating"] != "FAIL"
    ready = (
        before_payload is not None
        and after_payload is not None
        and dep_improved
        and robust_not_fail
    )

    return {
        "before_exists": before_payload is not None,
        "after_exists": after_payload is not None,
        "before_path": str(before_path),
        "after_path": str(after_path),
        "before_scores": before_scores,
        "after_scores": after_scores,
        "deployment_improved": dep_improved,
        "stress_improved": stress_improved,
        "ready": ready,
    }


def check_cartpole(*, bench_output_root: Path = CARTPOLE_BENCH_OUT) -> dict:
    """Check CartPole ablation checkpoints and high-rigor bench execution."""
    variants = [
        "baseline", "intervention1_curriculum", "intervention2_time_feature",
        "intervention1_plus_2", "intervention3_memory"
    ]
    ckpt_dir = ROOT / "checkpoints_cartpole_ppo"
    status = {}
    for v in variants:
        v_dir = ckpt_dir / v
        if v_dir.exists():
            # Count seeds
            seeds = [d for d in v_dir.iterdir()
                     if d.is_dir() and d.name.startswith("seed_")]
            status[v] = {"exists": True, "seeds": len(seeds)}
        else:
            status[v] = {"exists": False, "seeds": 0}

    checkpoints_ready = all(
        row["exists"] and row["seeds"] >= 10
        for row in status.values()
    )
    bench = check_bench_execution(
        CARTPOLE_MANIFEST,
        bench_output_root,
        protocol="paper",
        job_name="cartpole_high_rigor_bench",
    )

    return {
        "variants": status,
        "checkpoints_ready": checkpoints_ready,
        "bench": bench,
        "ready": checkpoints_ready and bench["ready"],
    }


def check_dm_control(*, bench_output_root: Path = DM_CONTROL_BENCH_OUT) -> dict:
    """Check dm_control availability, checkpoints, and bench execution health."""
    try:
        import shimmy  # noqa
        has_shimmy = True
    except ImportError:
        has_shimmy = False

    try:
        import dm_control  # noqa
        has_dm_control = True
    except ImportError:
        has_dm_control = False

    ckpt_dir = ROOT / "checkpoints" / "dm_control"
    envs = ["walker_walk", "cheetah_run", "reacher_easy", "humanoid_stand"]
    env_status = {}
    for env in envs:
        for variant in ["standard", "robust"]:
            key = f"{env}_{variant}"
            ckpt_path = ckpt_dir / f"{key}.zip"
            env_status[key] = ckpt_path.exists()

    checkpoints_ready = all(env_status.values())
    bench = check_bench_execution(
        DM_CONTROL_MANIFEST,
        bench_output_root,
        protocol="paper",
        job_name="dm_control_bench",
    )
    env_ready = has_shimmy and has_dm_control and checkpoints_ready

    return {
        "shimmy_available": has_shimmy,
        "dm_control_available": has_dm_control,
        "checkpoints": env_status,
        "checkpoints_ready": checkpoints_ready,
        "environment_ready": env_ready,
        "bench": bench,
        "ready": env_ready and bench["ready"],
    }


def check_paper_figures() -> dict:
    """Check which paper figures exist."""
    fig_dir = RUNS_DIR / "paper_figures"
    required_figs = [
        "fig_hero.png", "fig_main_result.png", "fig_ablation.png",
        "fig_dt_tracking_detail.png", "fig_killer.png",
        "results_table.tex", "statistical_metrics.json",
    ]
    optional_figs = [
        "fig_dm_control.png", "fig_quadrant.png", "fig_theory.png",
        "ablation_table.tex", "robustness_table.tex",
    ]
    status = {}
    for fig in required_figs:
        status[fig] = {"exists": (fig_dir / fig).exists(), "required": True}
    for fig in optional_figs:
        status[fig] = {"exists": (fig_dir / fig).exists(), "required": False}
    return status


def check_paper_draft() -> dict:
    """Check paper LaTeX draft."""
    paper_dir = DOCS_DIR / "paper"
    return {
        "paper_tex": (paper_dir / "paper.tex").exists(),
        "references_bib": (paper_dir / "references.bib").exists(),
    }


def check_theory_docs() -> dict:
    """Check theory and related work documents."""
    return {
        "theory_md": (DOCS_DIR / "theory.md").exists(),
        "related_work_md": (DOCS_DIR / "related_work.md").exists(),
    }


# ── Full status report ────────────────────────────────────────────────────────

def print_status_report(
    *,
    cartpole_bench_out: Path = CARTPOLE_BENCH_OUT,
    dm_control_bench_out: Path = DM_CONTROL_BENCH_OUT,
) -> dict[str, Any]:
    """Print complete submission status report and return structured readiness."""
    header("SUBMISSION STATUS REPORT")

    # Chain experiments
    print(f"\n{C.B}1. Chain Experiments{C.E}")
    chain_status = check_chain_experiments()
    for exp, s in chain_status.items():
        if s["complete"]:
            n = s["seeds"]
            if n >= 10:
                ok(f"{exp}: {n} seeds (paper-ready)")
            elif n >= 5:
                level = "need 10 for paper" if s.get("required", False) else "optional experiment"
                warn(f"{exp}: {n} seeds ({level})")
            else:
                if s.get("required", False):
                    err(f"{exp}: {n} seeds (too few)")
                else:
                    warn(f"{exp}: {n} seeds (optional experiment)")
        else:
            if s.get("required", False):
                err(f"{exp}: NOT FOUND")
            else:
                warn(f"{exp}: NOT FOUND (optional experiment)")
    chain_required_ready = all(
        s["complete"] and s["seeds"] >= 10
        for s in chain_status.values()
        if s.get("required", False)
    )

    # HalfCheetah
    print(f"\n{C.B}2. HalfCheetah{C.E}")
    hc = check_halfcheetah()
    if hc["before_exists"]:
        ok("HalfCheetah baseline audit: done")
    else:
        err("HalfCheetah baseline audit: MISSING")
    if hc["after_exists"]:
        ok("HalfCheetah robust audit: done")
    else:
        err("HalfCheetah robust audit: MISSING")
    before_scores = hc["before_scores"]
    after_scores = hc["after_scores"]
    if hc["before_exists"] and hc["after_exists"]:
        info(
            "  Deployment score: "
            f"{before_scores['deployment_score']} ({before_scores['deployment_rating']}) -> "
            f"{after_scores['deployment_score']} ({after_scores['deployment_rating']})"
        )
        info(
            "  Stress score: "
            f"{before_scores['stress_score']} ({before_scores['stress_rating']}) -> "
            f"{after_scores['stress_score']} ({after_scores['stress_rating']})"
        )
        if hc["deployment_improved"]:
            ok("Deployment improved from baseline")
        else:
            err("Deployment did not improve from baseline")
        if hc["stress_improved"]:
            ok("Stress robustness improved from baseline")
        else:
            warn("Stress robustness did not improve from baseline")
    halfcheetah_ready = bool(hc["ready"])

    # CartPole
    print(f"\n{C.B}3. CartPole Ablation{C.E}")
    cp = check_cartpole(bench_output_root=cartpole_bench_out)
    for variant, s in cp["variants"].items():
        if s["exists"]:
            n = s["seeds"]
            if n >= 10:
                ok(f"{variant}: {n} seeds (paper-ready)")
            elif n >= 5:
                warn(f"{variant}: {n} seeds (need 10 for paper)")
            else:
                warn(f"{variant}: {n} seeds (found)")
        else:
            err(f"{variant}: NOT FOUND (run: python -m deltatau_audit bench run --manifest bench/high_rigor_10seed_manifest.yaml --protocol paper)")
    if cp["checkpoints_ready"]:
        ok("CartPole checkpoints: complete 10-seed matrix")
    else:
        err("CartPole checkpoints: incomplete matrix")
    cp_bench = cp["bench"]
    info(
        "  Bench completion: "
        f"{cp_bench['completed_jobs']}/{cp_bench['expected_jobs']} summaries"
    )
    info(f"  Bench output root: {cp_bench['output_root']}")
    if cp_bench["bench_summary_exists"]:
        counts = cp_bench["counts"]
        info(
            "  Bench status: "
            f"{cp_bench['bench_status']} "
            f"(passed={_to_int(counts.get('passed'))}, "
            f"failed={_to_int(counts.get('failed'))}, "
            f"skipped={_to_int(counts.get('skipped'))})"
        )
    if cp_bench["ready"]:
        ok("CartPole high-rigor bench: passed")
    else:
        err("CartPole high-rigor bench: NOT ready")
        for reason in _prioritized_reasons(cp_bench.get("reasons")):
            warn(f"  {reason}")
        cp_plan = _quality_repair_plan(cp_bench)
        if cp_plan:
            for reason in cp_plan.get("reasons", []):
                info(str(reason))
            if cp_plan.get("strategy") == "diagnose_protocol":
                for cmd in cp_plan.get("diagnostic_commands", []):
                    info(f"Diagnostic command: {cmd}")
            else:
                for cmd in cp_plan.get("retrain_commands", []):
                    info(f"Targeted retrain: {cmd}")
                cleanup_paths = cp_plan.get("cleanup_summary_paths", [])
                if isinstance(cleanup_paths, list) and cleanup_paths:
                    info(f"Clear {len(cleanup_paths)} failed summaries before rerun")
                rerun_command = cp_plan.get("rerun_command")
                if isinstance(rerun_command, str) and rerun_command:
                    info(f"Rerun command: {rerun_command}")
                refresh_command = cp_plan.get("refresh_summary_command")
                if isinstance(refresh_command, str) and refresh_command:
                    info(f"Post-rerun command: {refresh_command}")
    cartpole_ready = bool(cp["ready"])

    # dm_control
    print(f"\n{C.B}4. dm_control Suite{C.E}")
    dm = check_dm_control(bench_output_root=dm_control_bench_out)
    if dm["shimmy_available"] and dm["dm_control_available"]:
        ok("shimmy + dm_control: available")
    elif not dm["shimmy_available"]:
        warn("shimmy: NOT installed (pip install shimmy[dm-control])")
    elif not dm["dm_control_available"]:
        warn("dm_control: NOT installed")

    missing_ckpts = [k for k, v in dm["checkpoints"].items() if not v]
    if not missing_ckpts:
        ok("All dm_control checkpoints: found")
    else:
        for k in missing_ckpts:
            warn(f"  Missing: checkpoints/dm_control/{k}.zip")
    dm_bench = dm["bench"]
    info(
        "  Bench completion: "
        f"{dm_bench['completed_jobs']}/{dm_bench['expected_jobs']} summaries"
    )
    info(f"  Bench output root: {dm_bench['output_root']}")
    if dm_bench["bench_summary_exists"]:
        counts = dm_bench["counts"]
        info(
            "  Bench status: "
            f"{dm_bench['bench_status']} "
            f"(passed={_to_int(counts.get('passed'))}, "
            f"failed={_to_int(counts.get('failed'))}, "
            f"skipped={_to_int(counts.get('skipped'))})"
        )
    if dm["ready"]:
        ok("dm_control bench: passed")
    else:
        err("dm_control bench: NOT ready")
        for reason in _prioritized_reasons(dm_bench.get("reasons")):
            warn(f"  {reason}")
    dm_ready = bool(dm["ready"])

    # Paper figures
    print(f"\n{C.B}5. Paper Figures{C.E}")
    figs = check_paper_figures()
    for fig, s in figs.items():
        if s["exists"]:
            ok(f"{fig}")
        elif s["required"]:
            err(f"{fig}: MISSING (required)")
        else:
            warn(f"{fig}: missing (optional)")
    figures_ready = all(s["exists"] for s in figs.values() if s["required"])

    # Paper draft
    print(f"\n{C.B}6. Paper LaTeX{C.E}")
    paper = check_paper_draft()
    if paper["paper_tex"]:
        ok("docs/paper/paper.tex: exists")
        # Check size
        tex_size = (DOCS_DIR / "paper" / "paper.tex").stat().st_size
        info(f"  Size: {tex_size:,} bytes (~{tex_size//40} lines)")
    else:
        err("docs/paper/paper.tex: MISSING")
    if paper["references_bib"]:
        ok("docs/paper/references.bib: exists")
    else:
        warn("docs/paper/references.bib: missing")
    paper_ready = paper["paper_tex"] and paper["references_bib"]

    # Theory docs
    print(f"\n{C.B}7. Theory Documents{C.E}")
    theory = check_theory_docs()
    if theory["theory_md"]:
        ok("docs/theory.md: exists")
    else:
        err("docs/theory.md: MISSING")
    if theory["related_work_md"]:
        ok("docs/related_work.md: exists")
    else:
        err("docs/related_work.md: MISSING")
    theory_ready = theory["theory_md"] and theory["related_work_md"]

    # Summary
    header("SUMMARY")
    category_checks = [
        ("chain_required", chain_required_ready),
        ("halfcheetah", halfcheetah_ready),
        ("cartpole", cartpole_ready),
        ("dm_control", dm_ready),
        ("paper_figures", figures_ready),
        ("paper_latex", paper_ready),
        ("theory_docs", theory_ready),
    ]
    n_ready = sum(1 for _, ready in category_checks if ready)
    total = len(category_checks)
    print(f"\n  Readiness: {n_ready}/{total} categories complete")
    if n_ready == total:
        print(f"\n  {C.G}[OK] READY FOR SUBMISSION{C.E}")
    else:
        remaining = total - n_ready
        print(f"\n  {C.Y}[WARN] {remaining} categories need work before submission{C.E}")
        print("\n  Next steps:")
        if not chain_required_ready:
            info("Run 10-seed chain experiments: PYTHONPATH=. python experiments/run_speed_generalization.py --seeds 10 --speed-hidden --output-dir runs/speed_gen_hidden_10seed")
        if not cartpole_ready:
            info(
                _bench_repair_hint(
                    label="CartPole high-rigor",
                    manifest="bench/high_rigor_10seed_manifest.yaml",
                    bench=cp_bench,
                )
            )
            repair_chain = _quality_repair_chain(cp_bench)
            if repair_chain:
                info(f"Executable repair chain: {repair_chain}")
        if not dm_ready:
            info(
                _bench_repair_hint(
                    label="dm_control",
                    manifest="bench/dm_control_research_manifest.yaml",
                    bench=dm_bench,
                )
            )
        if not paper_ready:
            info("Generate paper: docs/paper/paper.tex (in progress via agents)")
        if not (dm["shimmy_available"] and dm["dm_control_available"]):
            info("Install dm_control: pip install 'deltatau-audit[dm_control]'")

    return {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ready": n_ready == total,
        "n_ready": n_ready,
        "n_total": total,
        "categories": {name: ready for name, ready in category_checks},
        "chain": chain_status,
        "halfcheetah": hc,
        "cartpole": cp,
        "dm_control": dm,
        "figures": figs,
        "paper": paper,
        "theory": theory,
    }


# ── Training runners ──────────────────────────────────────────────────────────

def run_10seed_chain(dry_run: bool = False) -> None:
    """Run 10-seed chain experiments."""
    header("Running 10-seed Chain Experiments")
    cmd = [
        sys.executable, "experiments/run_speed_generalization.py",
        "--seeds", "10",
        "--speed-hidden",
        "--output-dir", "runs/speed_gen_hidden_10seed",
    ]
    info(f"Command: {' '.join(cmd)}")
    if dry_run:
        warn("DRY RUN - skipping actual execution")
        return
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    if result.returncode == 0:
        ok(f"Chain experiments complete ({elapsed:.0f}s)")
    else:
        err(f"Chain experiments FAILED (exit code {result.returncode})")


def run_cartpole_10seed(dry_run: bool = False) -> None:
    """Run 10-seed CartPole ablation via bench manifest."""
    header("Running 10-seed CartPole Ablation")
    manifest = BENCH_DIR / "high_rigor_10seed_manifest.yaml"
    if not manifest.exists():
        err(f"Manifest not found: {manifest}")
        info("Generate it first with: python scripts/prepare_submission.py")
        return
    cmd = [
        sys.executable, "-m", "deltatau_audit",
        "bench", "run",
        "--manifest", str(manifest),
        "--protocol", "paper",
    ]
    info(f"Command: {' '.join(cmd)}")
    if dry_run:
        warn("DRY RUN - skipping actual execution")
        return
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    if result.returncode == 0:
        ok(f"CartPole ablation complete ({elapsed:.0f}s)")
    else:
        err(f"CartPole ablation FAILED (exit code {result.returncode})")


def run_dm_control_experiments(dry_run: bool = False) -> None:
    """Run dm_control experiments."""
    header("Running dm_control Experiments")

    # Check prerequisites
    dm = check_dm_control()
    if not dm["shimmy_available"]:
        err("shimmy not installed. Run: pip install 'deltatau-audit[dm_control]'")
        return
    if not dm["dm_control_available"]:
        err("dm_control not installed.")
        return

    manifest = BENCH_DIR / "dm_control_research_manifest.yaml"
    if not manifest.exists():
        err(f"Manifest not found: {manifest}")
        return

    cmd = [
        sys.executable, "-m", "deltatau_audit",
        "bench", "run",
        "--manifest", str(manifest),
        "--protocol", "paper",
    ]
    info(f"Command: {' '.join(cmd)}")
    if dry_run:
        warn("DRY RUN - skipping actual execution")
        return
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode == 0:
        ok("dm_control experiments complete")
    else:
        err(f"dm_control experiments FAILED (exit code {result.returncode})")


def generate_paper_figures(dry_run: bool = False) -> None:
    """Regenerate all paper figures."""
    header("Generating Paper Figures")
    cmd = [sys.executable, "scripts/generate_paper.py"]
    info(f"Command: {' '.join(cmd)}")
    if dry_run:
        warn("DRY RUN - skipping actual execution")
        return
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode == 0:
        ok("Paper figures generated in runs/paper_figures/")
    else:
        err(f"Figure generation FAILED (exit code {result.returncode})")


def generate_latex_tables(dry_run: bool = False) -> None:
    """Generate paper tables from aggregate metrics."""
    header("Generating LaTeX Tables")
    cmd = [sys.executable, "scripts/generate_latex_tables.py"]
    info(f"Command: {' '.join(cmd)}")
    if dry_run:
        warn("DRY RUN - skipping actual execution")
        return
    result = subprocess.run(cmd, cwd=ROOT, env=_repo_env())
    if result.returncode == 0:
        ok("LaTeX tables generated in runs/paper_figures/")
    else:
        err(f"LaTeX table generation FAILED (exit code {result.returncode})")


def compile_paper(dry_run: bool = False) -> None:
    """Compile the LaTeX paper."""
    header("Compiling LaTeX Paper")
    paper_dir = DOCS_DIR / "paper"
    if not (paper_dir / "paper.tex").exists():
        err("paper.tex not found")
        return

    # Try pdflatex
    cmd = ["pdflatex", "-interaction=nonstopmode", "paper.tex"]
    info(f"Command: {' '.join(cmd)} (in docs/paper/)")
    if dry_run:
        warn("DRY RUN - skipping actual execution")
        return

    for run_num in range(2):  # Run twice for references
        result = subprocess.run(cmd, cwd=paper_dir)
        if result.returncode != 0 and run_num == 0:
            warn("First pdflatex pass failed (may be OK, running bibtex)")
            # Run bibtex
            subprocess.run(["bibtex", "paper"], cwd=paper_dir)

    pdf_path = paper_dir / "paper.pdf"
    if pdf_path.exists():
        size_kb = pdf_path.stat().st_size // 1024
        ok(f"paper.pdf generated ({size_kb} KB)")
    else:
        err("paper.pdf not generated (check LaTeX errors)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare paper submission artifacts"
    )
    parser.add_argument("--check-only", action="store_true",
                        help="Only print status report, don't run anything")
    parser.add_argument("--strict-check", action="store_true",
                        help="Return non-zero when readiness is not complete")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip all training steps")
    parser.add_argument("--skip-dm-control", action="store_true",
                        help="Skip dm_control experiments")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--compile-paper", action="store_true",
                        help="Also compile LaTeX paper to PDF")
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional path to write the structured status report as JSON",
    )
    parser.add_argument(
        "--cartpole-bench-out",
        type=str,
        default="",
        help="Optional CartPole bench output root to inspect instead of bench_runs/cartpole_high_rigor_10seed",
    )
    parser.add_argument(
        "--dm-control-bench-out",
        type=str,
        default="",
        help="Optional dm_control bench output root to inspect instead of bench_runs/dm_control",
    )
    args = parser.parse_args()

    cartpole_bench_out = (
        Path(args.cartpole_bench_out).expanduser().resolve()
        if args.cartpole_bench_out
        else CARTPOLE_BENCH_OUT
    )
    dm_control_bench_out = (
        Path(args.dm_control_bench_out).expanduser().resolve()
        if args.dm_control_bench_out
        else DM_CONTROL_BENCH_OUT
    )

    initial_report = print_status_report(
        cartpole_bench_out=cartpole_bench_out,
        dm_control_bench_out=dm_control_bench_out,
    )
    json_out = Path(args.json_out).expanduser().resolve() if args.json_out else None

    if args.check_only:
        if json_out is not None:
            _write_json(json_out, initial_report)
        if args.strict_check and not initial_report["ready"]:
            return 1
        return 0

    dry = args.dry_run

    if not args.skip_training:
        # Step 1: 10-seed chain experiments
        chain_status = check_chain_experiments()
        needs_10seed = any(
            not s["complete"] or s["seeds"] < 10
            for s in chain_status.values()
            if s.get("required", False)
        )
        if needs_10seed:
            run_10seed_chain(dry_run=dry)
        else:
            ok("10-seed chain experiments already complete - skipping")

        # Step 2: CartPole ablation
        run_cartpole_10seed(dry_run=dry)

        # Step 3: dm_control
        if not args.skip_dm_control:
            run_dm_control_experiments(dry_run=dry)

    # Step 4: Regenerate figures
    generate_paper_figures(dry_run=dry)

    # Step 5: Generate tables
    generate_latex_tables(dry_run=dry)

    # Step 6: Compile paper
    if args.compile_paper:
        compile_paper(dry_run=dry)

    # Final report
    final_report = print_status_report(
        cartpole_bench_out=cartpole_bench_out,
        dm_control_bench_out=dm_control_bench_out,
    )
    if json_out is not None:
        _write_json(json_out, final_report)
    if args.strict_check and not final_report["ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

