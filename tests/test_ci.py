"""Tests for deltatau_audit.ci — exit codes & CI summary generation."""

import json
import os

from deltatau_audit.ci import (
    compute_exit_code,
    compute_seed_sweep_exit_code,
    write_ci_summary,
    write_seed_sweep_ci_summary,
)

# ── Exit code logic ───────────────────────────────────────────────────

class TestExitCode:
    def test_pass(self):
        summary = {"deployment_score": 0.95, "stress_score": 0.80}
        assert compute_exit_code(summary) == 0

    def test_warn_stress_below(self):
        summary = {"deployment_score": 0.90, "stress_score": 0.40}
        assert compute_exit_code(summary) == 1

    def test_fail_deployment_below(self):
        summary = {"deployment_score": 0.50, "stress_score": 0.90}
        assert compute_exit_code(summary) == 2

    def test_fail_both_below(self):
        """Deployment failure takes priority over stress."""
        summary = {"deployment_score": 0.30, "stress_score": 0.20}
        assert compute_exit_code(summary) == 2

    def test_custom_thresholds(self):
        summary = {"deployment_score": 0.70, "stress_score": 0.40}
        # With relaxed thresholds, both pass
        assert compute_exit_code(summary, deploy_threshold=0.50,
                                 stress_threshold=0.30) == 0

    def test_boundary_deploy_exact(self):
        """At exactly the threshold, it's below (< not <=)."""
        summary = {"deployment_score": 0.80, "stress_score": 1.0}
        assert compute_exit_code(summary) == 0

    def test_boundary_deploy_just_below(self):
        summary = {"deployment_score": 0.799, "stress_score": 1.0}
        assert compute_exit_code(summary) == 2

    def test_boundary_stress_exact(self):
        summary = {"deployment_score": 0.90, "stress_score": 0.50}
        assert compute_exit_code(summary) == 0

    def test_boundary_stress_just_below(self):
        summary = {"deployment_score": 0.90, "stress_score": 0.499}
        assert compute_exit_code(summary) == 1

    def test_missing_scores_default_pass(self):
        """Missing scores default to 1.0 → pass."""
        assert compute_exit_code({}) == 0

    def test_worst_ci_lower_gate_uses_ci_lower_not_mean_scores(self):
        summary = {"deployment_score": 0.95, "stress_score": 0.95}
        robustness = {
            "per_scenario_scores": {
                "jitter": {"ci_lower": 0.90},
                "delay": {"ci_lower": 0.85},
                "spike": {"ci_lower": 0.81},
                "obs_noise": {"ci_lower": 0.84},
                "speed_5x": {"ci_lower": 0.40},
            }
        }
        # Mean scores pass, but strict gate should WARN due stress CI lower.
        assert (
            compute_exit_code(
                summary,
                deploy_threshold=0.80,
                stress_threshold=0.50,
                robustness=robustness,
                gate_mode="worst_ci_lower",
            )
            == 1
        )


class TestSeedSweepExitCode:
    def test_pass(self):
        pass_rates = {"deployment": 0.9, "stress": 0.8}
        assert compute_seed_sweep_exit_code(pass_rates) == 0

    def test_warn_stress_rate_below(self):
        pass_rates = {"deployment": 0.9, "stress": 0.4}
        assert compute_seed_sweep_exit_code(pass_rates) == 1

    def test_fail_deployment_rate_below(self):
        pass_rates = {"deployment": 0.6, "stress": 0.9}
        assert compute_seed_sweep_exit_code(pass_rates) == 2


# ── CI summary file generation ────────────────────────────────────────

class TestWriteCiSummary:
    def test_writes_json_and_md(self, tmp_path):
        summary = {
            "deployment_score": 0.92,
            "deployment_rating": "MILD",
            "stress_score": 0.60,
            "stress_rating": "DEGRADED",
        }
        robustness = {
            "deployment": {"worst_case": {"scenario": "jitter"}},
            "stress": {"worst_case": {"scenario": "speed_5x"}},
            "per_scenario_scores": {
                "jitter": {"return_ratio": 0.92},
                "speed_5x": {"return_ratio": 0.60},
            },
        }
        out = str(tmp_path / "ci_out")
        exit_code = write_ci_summary(summary, robustness, out)

        # deploy 0.92 >= 0.80 → OK, stress 0.60 >= 0.50 → OK → pass
        assert exit_code == 0

        # Check JSON
        json_path = os.path.join(out, "ci_summary.json")
        assert os.path.exists(json_path)
        with open(json_path) as f:
            data = json.load(f)
        assert data["status"] == "pass"
        assert data["exit_code"] == 0
        assert data["deployment_score"] == 0.92
        assert data["deployment_worst"] == "jitter"
        assert data["stress_worst"] == "speed_5x"

        # Check MD
        md_path = os.path.join(out, "ci_summary.md")
        assert os.path.exists(md_path)
        md = open(md_path, encoding="utf-8").read()
        assert "PASS" in md.upper() or "pass" in md.lower()
        assert "Deployment" in md
        assert "Stress" in md

    def test_fail_output(self, tmp_path):
        summary = {
            "deployment_score": 0.40,
            "deployment_rating": "FAIL",
            "stress_score": 0.20,
            "stress_rating": "FAIL",
        }
        robustness = {
            "deployment": {"worst_case": {"scenario": "delay"}},
            "stress": {"worst_case": {"scenario": "speed_5x"}},
            "per_scenario_scores": {},
        }
        out = str(tmp_path / "ci_fail")
        exit_code = write_ci_summary(summary, robustness, out)
        assert exit_code == 2

        with open(os.path.join(out, "ci_summary.json")) as f:
            data = json.load(f)
        assert data["status"] == "fail"
        assert data["exit_code"] == 2

    def test_writes_multi_seed_summary(self, tmp_path):
        summary = {
            "deployment_score": 0.88,
            "deployment_rating": "MILD",
            "stress_score": 0.66,
            "stress_rating": "DEGRADED",
        }
        aggregate = {
            "n_seeds": 5,
            "pass_rates": {
                "deployment": 0.8,
                "stress": 0.6,
            },
            "quadrant_counts": {
                "deployment_ready": 4,
                "deployment_fragile": 1,
            },
        }
        out = str(tmp_path / "ci_multi")
        exit_code = write_seed_sweep_ci_summary(
            summary,
            aggregate,
            out,
            min_deploy_pass_rate=0.8,
            min_stress_pass_rate=0.5,
        )
        assert exit_code == 0

        with open(os.path.join(out, "ci_summary.json"), encoding="utf-8") as f:
            data = json.load(f)
        assert data["mode"] == "multi_seed"
        assert data["n_seeds"] == 5
        assert data["pass_rates"]["deployment"] == 0.8
        assert data["status"] == "pass"

        md = open(os.path.join(out, "ci_summary.md"), encoding="utf-8").read()
        assert "Multi-Seed" in md
        assert "Pass Rate" in md

    def test_worst_ci_lower_gate_written_to_summary(self, tmp_path):
        summary = {
            "deployment_score": 0.92,
            "deployment_rating": "MILD",
            "stress_score": 0.70,
            "stress_rating": "DEGRADED",
        }
        robustness = {
            "deployment": {"worst_case": {"scenario": "spike"}},
            "stress": {"worst_case": {"scenario": "speed_5x"}},
            "per_scenario_scores": {
                "jitter": {"return_ratio": 0.9, "ci_lower": 0.86},
                "delay": {"return_ratio": 0.92, "ci_lower": 0.88},
                "spike": {"return_ratio": 0.85, "ci_lower": 0.81},
                "obs_noise": {"return_ratio": 0.95, "ci_lower": 0.90},
                "speed_5x": {"return_ratio": 0.70, "ci_lower": 0.45},
            },
        }
        out = str(tmp_path / "ci_worst_ci")
        exit_code = write_ci_summary(
            summary,
            robustness,
            out,
            deploy_threshold=0.80,
            stress_threshold=0.50,
            gate_mode="worst_ci_lower",
        )
        assert exit_code == 1
        with open(os.path.join(out, "ci_summary.json"), encoding="utf-8") as f:
            data = json.load(f)
        assert data["gate_mode"] == "worst_ci_lower"
        assert data["gate_scores"]["stress"] == 0.45
