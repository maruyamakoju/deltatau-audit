"""Tests for deltatau_audit.report.generator — HTML and figure generation."""

import pytest
import json
import os
import base64

from deltatau_audit.report.generator import (
    _fig_to_base64,
    _plot_return_vs_speed,
    _plot_reliance_rmse,
    _plot_reliance_bars,
    _plot_robustness_bars,
    _plot_quadrant,
    generate_report,
)


# ── Helper fixtures ───────────────────────────────────────────────────

@pytest.fixture
def mock_reliance_data():
    """Mock reliance audit data."""
    return {
        "per_speed": {
            "1": {
                "none": {"total_reward_mean": 100.0, "total_reward_se": 5.0, "rmse_mean": 0.5, "rmse_se": 0.1},
                "clamp_1": {"rmse_mean": 0.6, "rmse_se": 0.1},
                "reverse": {"rmse_mean": 0.8, "rmse_se": 0.1},
            },
            "2": {
                "none": {"total_reward_mean": 95.0, "total_reward_se": 5.0, "rmse_mean": 0.55, "rmse_se": 0.1},
                "clamp_1": {"rmse_mean": 0.7, "rmse_se": 0.1},
                "reverse": {"rmse_mean": 0.9, "rmse_se": 0.1},
            },
        },
        "degradation": {
            "clamp_1": {
                "1": {"percent_increase": 20.0, "ratio": 1.2, "baseline_rmse": 0.5, "intervention_rmse": 0.6, "severity": "MILD"},
                "2": {"percent_increase": 27.0, "ratio": 1.27, "baseline_rmse": 0.55, "intervention_rmse": 0.7, "severity": "MODERATE"},
            },
        },
        "rating": "MODERATE",
        "score": 1.27,
        "worst_case": {"speed": "2", "intervention": "clamp_1", "rmse_ratio": 1.27, "percent": 27.0},
    }


@pytest.fixture
def mock_robustness_data():
    """Mock robustness audit data."""
    return {
        "scenarios": {
            "nominal": {"total_reward_mean": 100.0, "rmse_mean": 0.5},
            "jitter": {"total_reward_mean": 90.0, "rmse_mean": 0.6},
            "speed_5x": {"total_reward_mean": 50.0, "rmse_mean": 1.0},
        },
        "per_scenario_scores": {
            "jitter": {
                "return_ratio": 0.90,
                "return_drop_pct": 10.0,
                "rmse_ratio": 1.2,
                "rmse_increase_pct": 20.0,
                "ci_lower": 0.85,
                "ci_upper": 0.95,
                "significant": False,
            },
            "speed_5x": {
                "return_ratio": 0.50,
                "return_drop_pct": 50.0,
                "rmse_ratio": 2.0,
                "rmse_increase_pct": 100.0,
                "ci_lower": 0.45,
                "ci_upper": 0.55,
                "significant": True,
            },
        },
        "deployment": {
            "return_score": 0.90,
            "rating": "MILD",
            "worst_case": {"scenario": "jitter", "return_ratio": 0.90, "return_drop_pct": 10.0},
        },
        "stress": {
            "return_score": 0.50,
            "rating": "FAIL",
            "worst_case": {"scenario": "speed_5x", "return_ratio": 0.50, "return_drop_pct": 50.0},
        },
        "rating": "FAIL",
        "return_score": 0.50,
        "rmse_score": 2.0,
        "worst_case": {"scenario": "speed_5x", "return_ratio": 0.50, "return_drop_pct": 50.0},
    }


@pytest.fixture
def mock_audit_result(mock_reliance_data, mock_robustness_data):
    """Mock full audit result."""
    return {
        "speeds": [1, 2],
        "n_episodes": 10,
        "supports_intervention": True,
        "reliance": mock_reliance_data,
        "robustness": mock_robustness_data,
        "sensitivity": {
            "mean": 0.123,
            "std": 0.045,
            "median": 0.110,
            "n_samples": 50,
            "per_speed": {
                "1": {"mean": 0.100, "std": 0.030, "n_samples": 25},
                "2": {"mean": 0.145, "std": 0.060, "n_samples": 25},
            },
        },
        "summary": {
            "reliance_rating": "MODERATE",
            "reliance_score": 1.27,
            "deployment_rating": "MILD",
            "deployment_score": 0.90,
            "stress_rating": "FAIL",
            "stress_score": 0.50,
            "robustness_rating": "FAIL",
            "robustness_score": 0.50,
            "robustness_rmse_score": 2.0,
            "sensitivity_mean": 0.123,
            "quadrant": "time_aware_fragile",
            "prescription": "Agent uses internal timing but degrades under deployment conditions.",
        },
    }


@pytest.fixture
def mock_audit_result_no_intervention():
    """Mock audit result for agent without intervention support."""
    return {
        "speeds": [1],
        "n_episodes": 10,
        "supports_intervention": False,
        "reliance": {
            "per_speed": {},
            "degradation": {},
            "rating": "N/A",
            "score": None,
            "worst_case": {"speed": None, "intervention": None, "rmse_ratio": None, "percent": None},
        },
        "robustness": {
            "scenarios": {
                "nominal": {"total_reward_mean": 100.0, "rmse_mean": 0.5},
                "jitter": {"total_reward_mean": 95.0, "rmse_mean": 0.55},
            },
            "per_scenario_scores": {
                "jitter": {
                    "return_ratio": 0.95,
                    "return_drop_pct": 5.0,
                    "rmse_ratio": 1.1,
                    "rmse_increase_pct": 10.0,
                    "ci_lower": 0.92,
                    "ci_upper": 0.98,
                    "significant": False,
                },
            },
            "deployment": {
                "return_score": 0.95,
                "rating": "PASS",
                "worst_case": {"scenario": "jitter", "return_ratio": 0.95, "return_drop_pct": 5.0},
            },
            "stress": {
                "return_score": 1.0,
                "rating": "PASS",
                "worst_case": {"scenario": None, "return_ratio": 1.0, "return_drop_pct": 0.0},
            },
            "rating": "PASS",
            "return_score": 0.95,
            "rmse_score": 1.1,
            "worst_case": {"scenario": "jitter", "return_ratio": 0.95, "return_drop_pct": 5.0},
        },
        "sensitivity": None,
        "summary": {
            "reliance_rating": "N/A",
            "reliance_score": None,
            "deployment_rating": "PASS",
            "deployment_score": 0.95,
            "stress_rating": "PASS",
            "stress_score": 1.0,
            "robustness_rating": "PASS",
            "robustness_score": 0.95,
            "robustness_rmse_score": 1.1,
            "sensitivity_mean": None,
            "quadrant": "deployment_ready",
            "prescription": "Agent maintains performance under deployment timing conditions.",
        },
    }


# ── Figure generation tests ───────────────────────────────────────────

class TestFigureGeneration:
    def test_fig_to_base64(self):
        """Test conversion of matplotlib figure to base64."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot([1, 2, 3], [1, 4, 9])

        b64 = _fig_to_base64(fig)

        assert isinstance(b64, str)
        assert len(b64) > 100  # Should be substantial
        # Should be valid base64
        try:
            decoded = base64.b64decode(b64)
            assert len(decoded) > 0
        except Exception as e:
            pytest.fail(f"Invalid base64: {e}")

    def test_plot_return_vs_speed(self, mock_reliance_data):
        """Test return vs speed plot generation."""
        b64 = _plot_return_vs_speed(mock_reliance_data, [1, 2])

        assert isinstance(b64, str)
        assert len(b64) > 100

    def test_plot_reliance_rmse(self, mock_reliance_data):
        """Test reliance RMSE plot generation."""
        b64 = _plot_reliance_rmse(mock_reliance_data, [1, 2])

        assert isinstance(b64, str)
        assert len(b64) > 100

    def test_plot_reliance_bars(self, mock_reliance_data):
        """Test reliance bars plot generation."""
        b64 = _plot_reliance_bars(mock_reliance_data)

        assert isinstance(b64, str)
        assert len(b64) > 100

    def test_plot_reliance_bars_empty(self):
        """Empty degradation should return empty string."""
        b64 = _plot_reliance_bars({"degradation": {}})
        assert b64 == ""

    def test_plot_robustness_bars(self, mock_robustness_data):
        """Test robustness bars plot generation."""
        b64 = _plot_robustness_bars(mock_robustness_data)

        assert isinstance(b64, str)
        assert len(b64) > 100

    def test_plot_robustness_bars_empty(self):
        """Empty scores should return empty string."""
        b64 = _plot_robustness_bars({"per_scenario_scores": {}})
        assert b64 == ""

    def test_plot_quadrant_with_reliance(self, mock_audit_result):
        """Test quadrant plot with reliance data."""
        b64 = _plot_quadrant(mock_audit_result["summary"])

        assert isinstance(b64, str)
        assert len(b64) > 100

    def test_plot_quadrant_without_reliance(self, mock_audit_result_no_intervention):
        """Test quadrant plot without reliance data."""
        b64 = _plot_quadrant(mock_audit_result_no_intervention["summary"])

        # Should return empty string when reliance is N/A
        assert b64 == ""


# ── Report generation tests ───────────────────────────────────────────

class TestGenerateReport:
    def test_generate_full_report(self, tmp_path, mock_audit_result):
        """Test full report generation with all components."""
        output_dir = str(tmp_path / "report")

        html_path = generate_report(mock_audit_result, output_dir, title="Test Audit")

        # Check that files were created
        assert os.path.exists(html_path)
        assert os.path.exists(os.path.join(output_dir, "summary.json"))
        assert os.path.exists(os.path.join(output_dir, "robustness_bars.png"))
        assert os.path.exists(os.path.join(output_dir, "return_vs_speed.png"))
        assert os.path.exists(os.path.join(output_dir, "reliance_rmse.png"))
        assert os.path.exists(os.path.join(output_dir, "reliance_bars.png"))
        assert os.path.exists(os.path.join(output_dir, "quadrant.png"))

        # Check HTML content
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()

        assert "Test Audit" in html
        assert "MODERATE" in html  # Reliance rating
        assert "MILD" in html  # Deployment rating
        assert "FAIL" in html  # Stress rating
        assert "time_aware_fragile" in html or "Time-Aware but Fragile" in html
        assert "data:image/png;base64," in html  # Embedded images

        # Check JSON structure
        with open(os.path.join(output_dir, "summary.json"), "r") as f:
            data = json.load(f)

        assert data["speeds"] == [1, 2]
        assert data["n_episodes"] == 10
        assert data["supports_intervention"] is True

    def test_generate_report_no_intervention(self, tmp_path, mock_audit_result_no_intervention):
        """Test report generation for agent without intervention support."""
        output_dir = str(tmp_path / "report_no_interv")

        html_path = generate_report(mock_audit_result_no_intervention, output_dir)

        # Check that basic files exist
        assert os.path.exists(html_path)
        assert os.path.exists(os.path.join(output_dir, "summary.json"))
        assert os.path.exists(os.path.join(output_dir, "robustness_bars.png"))

        # Should NOT have reliance-specific plots
        assert not os.path.exists(os.path.join(output_dir, "return_vs_speed.png"))
        assert not os.path.exists(os.path.join(output_dir, "reliance_rmse.png"))
        assert not os.path.exists(os.path.join(output_dir, "quadrant.png"))

        # Check HTML content
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()

        assert "PASS" in html  # Deployment rating
        assert "deployment_ready" in html or "Deployment Ready" in html
        # Should have 2 badges (not 3)
        assert html.count("badge-label") == 2

    def test_report_creates_directory(self, tmp_path, mock_audit_result):
        """Test that report creates output directory if it doesn't exist."""
        output_dir = str(tmp_path / "nested" / "path" / "report")

        html_path = generate_report(mock_audit_result, output_dir)

        assert os.path.exists(output_dir)
        assert os.path.exists(html_path)

    def test_report_html_structure(self, tmp_path, mock_audit_result):
        """Test HTML structure and required elements."""
        output_dir = str(tmp_path / "report")
        html_path = generate_report(mock_audit_result, output_dir)

        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()

        # Check HTML structure
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "<head>" in html
        assert "<body>" in html
        assert "</html>" in html

        # Check CSS presence
        assert "<style>" in html
        assert ".badge" in html
        assert ".prescription" in html

        # Check sections
        assert "Reliance Test" in html or "1. Reliance" in html
        assert "Robustness Test" in html or "Robustness" in html
        assert "Recommendation" in html

    def test_json_output_format(self, tmp_path, mock_audit_result):
        """Test that JSON output is valid and complete."""
        output_dir = str(tmp_path / "report")
        generate_report(mock_audit_result, output_dir)

        json_path = os.path.join(output_dir, "summary.json")
        with open(json_path, "r") as f:
            data = json.load(f)

        # Verify structure matches input
        assert data["speeds"] == mock_audit_result["speeds"]
        assert data["n_episodes"] == mock_audit_result["n_episodes"]
        assert data["summary"]["quadrant"] == mock_audit_result["summary"]["quadrant"]
        assert data["reliance"]["rating"] == mock_audit_result["reliance"]["rating"]
        assert data["robustness"]["rating"] == mock_audit_result["robustness"]["rating"]

    def test_png_files_are_valid(self, tmp_path, mock_audit_result):
        """Test that generated PNG files are valid images."""
        output_dir = str(tmp_path / "report")
        generate_report(mock_audit_result, output_dir)

        png_files = [
            "robustness_bars.png",
            "return_vs_speed.png",
            "reliance_rmse.png",
            "reliance_bars.png",
            "quadrant.png",
        ]

        for png_file in png_files:
            path = os.path.join(output_dir, png_file)
            assert os.path.exists(path)

            # Check file size (should not be empty)
            size = os.path.getsize(path)
            assert size > 1000, f"{png_file} is too small: {size} bytes"

            # Check PNG header
            with open(path, "rb") as f:
                header = f.read(8)
                assert header == b'\x89PNG\r\n\x1a\n', f"{png_file} has invalid PNG header"
