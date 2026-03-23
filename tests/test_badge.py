"""Tests for SVG badge generation (badge.py)."""
from __future__ import annotations

import json
import os
import re
import tempfile


class TestBadgeDeployment:
    def test_returns_svg_string(self):
        from deltatau_audit.badge import badge_deployment
        svg = badge_deployment({"deployment_rating": "PASS", "deployment_score": 0.95})
        assert svg.strip().startswith("<svg")
        assert "PASS" in svg

    def test_fail_rating(self):
        from deltatau_audit.badge import badge_deployment
        svg = badge_deployment({"deployment_rating": "FAIL", "deployment_score": -0.04})
        assert "FAIL" in svg

    def test_mild_color_is_amber_not_green(self):
        """MILD should be amber (#ffc107), not Bootstrap green (#5cb85c)."""
        from deltatau_audit.badge import badge_deployment
        svg = badge_deployment({"deployment_rating": "MILD", "deployment_score": 0.85})
        assert "#ffc107" in svg, f"MILD should be amber #ffc107, got: {svg[:500]}"
        assert "#5cb85c" not in svg, "MILD must NOT be Bootstrap green #5cb85c"


class TestBadgeStress:
    def test_returns_svg_string(self):
        from deltatau_audit.badge import badge_stress
        svg = badge_stress({"stress_rating": "FAIL", "stress_score": 0.25})
        assert svg.strip().startswith("<svg")
        assert "FAIL" in svg


class TestBadgeReliance:
    def test_na_case(self):
        from deltatau_audit.badge import badge_reliance
        svg = badge_reliance({"reliance_rating": "N/A", "reliance_score": None})
        assert "N/A" in svg

    def test_missing_keys_returns_na(self):
        from deltatau_audit.badge import badge_reliance
        svg = badge_reliance({})
        assert "N/A" in svg

    def test_high_reliance_blue(self):
        from deltatau_audit.badge import badge_reliance
        svg = badge_reliance({"reliance_rating": "HIGH", "reliance_score": 3.5})
        assert "HIGH" in svg
        # Should be in the blue spectrum, not red/green
        assert "#dc3545" not in svg  # not red (FAIL color)
        assert "#28a745" not in svg  # not green (PASS color)


class TestBadgeStatus:
    def test_deployment_ready(self):
        from deltatau_audit.badge import badge_status
        svg = badge_status({"quadrant": "deployment_ready"})
        assert "Deployment Ready" in svg

    def test_deployment_fragile(self):
        from deltatau_audit.badge import badge_status
        svg = badge_status({"quadrant": "deployment_fragile"})
        assert "Deployment Fragile" in svg

    def test_time_aware_robust(self):
        from deltatau_audit.badge import badge_status
        svg = badge_status({"quadrant": "time_aware_robust"})
        assert "Time-Aware" in svg


class TestSVGUniqueness:
    def test_different_badges_have_unique_ids(self):
        """When badges are embedded in the same page, their SVG IDs must not collide."""
        from deltatau_audit.badge import badge_deployment, badge_stress

        svg1 = badge_deployment({"deployment_rating": "PASS", "deployment_score": 0.95})
        svg2 = badge_stress({"stress_rating": "FAIL", "stress_score": 0.25})

        # Extract all id= attributes from each
        ids1 = set(re.findall(r'id="([^"]+)"', svg1))
        ids2 = set(re.findall(r'id="([^"]+)"', svg2))

        # They should not share any IDs
        collision = ids1 & ids2
        assert not collision, f"SVG ID collision detected: {collision}"

    def test_same_badge_has_stable_ids(self):
        """Same badge content should produce stable (deterministic) IDs."""
        from deltatau_audit.badge import badge_deployment

        summary = {"deployment_rating": "MILD", "deployment_score": 0.85}
        svg1 = badge_deployment(summary)
        svg2 = badge_deployment(summary)

        ids1 = set(re.findall(r'id="([^"]+)"', svg1))
        ids2 = set(re.findall(r'id="([^"]+)"', svg2))
        assert ids1 == ids2, "Same input should produce same IDs"

    def test_svg_has_version_attribute(self):
        from deltatau_audit.badge import badge_deployment
        svg = badge_deployment({"deployment_rating": "PASS", "deployment_score": 0.95})
        assert 'version="1.1"' in svg


class TestGenerateBadges:
    def test_writes_four_files(self):
        """generate_badges should write deployment, stress, reliance, status."""
        from deltatau_audit.badge import generate_badges

        summary_data = {
            "summary": {
                "deployment_rating": "MILD",
                "deployment_score": 0.85,
                "stress_rating": "FAIL",
                "stress_score": 0.25,
                "reliance_rating": "N/A",
                "reliance_score": None,
                "quadrant": "deployment_ready",
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = os.path.join(tmpdir, "summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary_data, f)

            paths = generate_badges(summary_path, tmpdir, prefix="test")

            assert "deployment" in paths
            assert "stress" in paths
            assert "reliance" in paths  # reliance badge
            assert "status" in paths

            for path in paths.values():
                assert os.path.exists(path), f"Expected badge file: {path}"
                content = open(path).read()
                assert content.strip().startswith("<svg")
