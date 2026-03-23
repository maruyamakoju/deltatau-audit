from __future__ import annotations

import json
from pathlib import Path

from deltatau_audit.bench import _finalize_run_summary
from deltatau_audit.research_suite import (
    ResearchSuiteConfig,
    StageOutcome,
    _finalize_suite,
)

_GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def _load_golden_text(name: str) -> str:
    return (_GOLDEN_DIR / name).read_text(encoding="utf-8")


def _load_golden_json(name: str) -> dict:
    return json.loads(_load_golden_text(name))


def _suite_fixture(tmp_path: Path) -> tuple[ResearchSuiteConfig, list[StageOutcome]]:
    suite_root = tmp_path / "suite"
    cfg = ResearchSuiteConfig(
        env="CartPole-v1",
        out=str(suite_root),
        episodes=8,
        seed=7,
        speeds=[1, 2, 5],
        deliberative_max_thinking_steps=4,
        bridge_delay_ms=120.0,
        bridge_delay_std_ms=15.0,
        bridge_dt_ms=20.0,
        bridge_actuator_alpha=0.35,
        resume=True,
        fail_fast=False,
    )
    outcomes = [
        StageOutcome(
            name="deliberative",
            status="success",
            reason=None,
            deployment_score=0.91,
            stress_score=0.64,
            output_dir=str(suite_root / "deliberative"),
            duration_sec=12.5,
        ),
        StageOutcome(
            name="ltc",
            status="cached",
            reason="existing summary.json reused (--resume)",
            deployment_score=0.88,
            stress_score=0.58,
            output_dir=str(suite_root / "ltc"),
            duration_sec=0.0,
        ),
        StageOutcome(
            name="bridge",
            status="skipped",
            reason="Bridge stage needs a successful deliberative or ltc stage.",
            deployment_score=None,
            stress_score=None,
            output_dir=str(suite_root / "bridge"),
            duration_sec=0.0,
        ),
    ]
    return cfg, outcomes


def _bench_fixture(tmp_path: Path) -> tuple[Path, dict]:
    bench_root = tmp_path / "bench"
    run_summary = {
        "status": "passed",
        "started_at_utc": "2026-03-14T00:00:00+00:00",
        "finished_at_utc": "2026-03-14T00:05:00+00:00",
        "duration_sec": 300.0,
        "output_root": str(bench_root),
        "counts": {"passed": 2, "failed": 0, "skipped": 0},
        "jobs": [
            {
                "id": "cartpole_seed0",
                "status": "passed",
                "command": "audit-sb3",
                "vars": {"env": "CartPole-v1", "seed": 0},
                "args": {"algo": "ppo", "variant": "baseline"},
                "summary_path": str(bench_root / "cartpole_seed0" / "summary.json"),
                "result": {
                    "protocol": "research",
                    "deployment_score": 0.88,
                    "deployment_rating": "MILD",
                    "stress_score": 0.56,
                    "stress_rating": "DEGRADED",
                    "quadrant": "deployment_ready",
                    "stress_worst_scenario": "speed_5x",
                    "stress_worst_return_ratio": 0.56,
                    "stress_worst_ci_lower": 0.51,
                    "stress_ci_gate_pass": True,
                    "diagnosis_pattern": "Extreme Frequency Fragility",
                },
            },
            {
                "id": "acrobot_seed1",
                "status": "passed",
                "command": "audit-cleanrl",
                "vars": {"env": "Acrobot-v1", "seed": 1},
                "args": {"algo": "ppo", "variant": "time_feature"},
                "summary_path": str(bench_root / "acrobot_seed1" / "summary.json"),
                "result": {
                    "protocol": "paper",
                    "deployment_score": 0.93,
                    "deployment_rating": "PASS",
                    "stress_score": 0.72,
                    "stress_rating": "MILD",
                    "quadrant": "time_aware_robust",
                    "stress_worst_scenario": "delay",
                    "stress_worst_return_ratio": 0.72,
                    "stress_worst_ci_lower": 0.68,
                    "stress_ci_gate_pass": True,
                    "diagnosis_pattern": "Observation Recency Dependency",
                },
            },
        ],
    }
    return bench_root, run_summary


def _canonicalize_suite_summary(data: dict) -> dict:
    canonical = json.loads(json.dumps(data))
    canonical["generated_at_utc"] = "<timestamp>"
    canonical["config"]["out"] = "<suite_root>"
    for stage in canonical["stages"]:
        stage["output_dir"] = f"<suite_root>/{Path(stage['output_dir']).name}"
    return canonical


def _canonicalize_suite_markdown(text: str, suite_root: Path) -> str:
    normalized = text.replace(str(suite_root), "<suite_root>").replace("\\", "/")
    lines: list[str] = []
    for line in normalized.splitlines():
        if line.startswith("- Generated (UTC): `"):
            lines.append("- Generated (UTC): `<timestamp>`")
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"


def _canonicalize_bench_summary(data: dict, bench_root: Path) -> dict:
    canonical = json.loads(json.dumps(data))
    canonical["output_root"] = "<bench_root>"
    canonical["artifacts"] = {
        "submission_csv": "<bench_root>/submission_table.csv",
        "submission_md": "<bench_root>/submission_table.md",
        "submission_rows": canonical["artifacts"]["submission_rows"],
    }
    for job in canonical["jobs"]:
        job["summary_path"] = f"<bench_root>/{Path(job['summary_path']).parent.name}/summary.json"
    return canonical


def _canonicalize_table_text(text: str, bench_root: Path) -> str:
    return text.replace(str(bench_root), "<bench_root>").replace("\\", "/")


def test_research_suite_artifacts_match_contract(tmp_path: Path):
    cfg, outcomes = _suite_fixture(tmp_path)
    suite_root = Path(cfg.out)

    result = _finalize_suite(cfg, outcomes, suite_root, dashboard=False)

    assert result["dashboard_path"] is None
    summary = json.loads((suite_root / "suite_summary.json").read_text(encoding="utf-8"))
    summary_md = (suite_root / "suite_summary.md").read_text(encoding="utf-8")

    assert _canonicalize_suite_summary(summary) == _load_golden_json(
        "suite_summary_contract.json"
    )
    assert _canonicalize_suite_markdown(summary_md, suite_root) == _load_golden_text(
        "suite_summary_contract.md"
    )


def test_bench_artifacts_match_contract(tmp_path: Path):
    bench_root, run_summary = _bench_fixture(tmp_path)

    _finalize_run_summary(bench_root, run_summary)

    bench_summary = json.loads((bench_root / "bench_summary.json").read_text(encoding="utf-8"))
    submission_md = (bench_root / "submission_table.md").read_text(encoding="utf-8")
    submission_csv = (bench_root / "submission_table.csv").read_text(encoding="utf-8")

    assert _canonicalize_bench_summary(bench_summary, bench_root) == _load_golden_json(
        "bench_summary_contract.json"
    )
    assert _canonicalize_table_text(submission_md, bench_root) == _load_golden_text(
        "submission_table_contract.md"
    )
    assert _canonicalize_table_text(submission_csv, bench_root) == _load_golden_text(
        "submission_table_contract.csv"
    )
