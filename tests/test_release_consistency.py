"""Tests for scripts/check_release_consistency.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "check_release_consistency.py"
    spec = importlib.util.spec_from_file_location("check_release_consistency", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_repo(tmp_path: Path, *, pyproject: str, init: str, changelog: str, readme: str) -> Path:
    (tmp_path / "deltatau_audit").mkdir(parents=True, exist_ok=True)
    (tmp_path / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    (tmp_path / "deltatau_audit" / "__init__.py").write_text(init, encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
    (tmp_path / "README.md").write_text(readme, encoding="utf-8")
    return tmp_path


def test_check_consistency_ok(tmp_path: Path):
    m = _load_module()
    root = _write_repo(
        tmp_path,
        pyproject='[project]\nname = "x"\nversion = "0.8.0"\n',
        init='__version__ = "0.8.0"\n',
        changelog="# Changelog\n\n## [Unreleased]\n\n## [0.8.0] - 2026-02-24\n",
        readme="version = {0.8.0}\n",
    )

    errors = m.check_consistency(root, expected_version="v0.8.0", check_readme=True)

    assert errors == []


def test_check_consistency_detects_mismatch(tmp_path: Path):
    m = _load_module()
    root = _write_repo(
        tmp_path,
        pyproject='[project]\nname = "x"\nversion = "0.8.0"\n',
        init='__version__ = "0.8.1"\n',
        changelog="# Changelog\n\n## [0.8.0] - 2026-02-24\n",
        readme="version = {0.8.0}\n",
    )

    errors = m.check_consistency(root, expected_version=None, check_readme=True)

    assert any("Version mismatch across files" in err for err in errors)


def test_check_consistency_expected_version_mismatch(tmp_path: Path):
    m = _load_module()
    root = _write_repo(
        tmp_path,
        pyproject='[project]\nname = "x"\nversion = "0.8.0"\n',
        init='__version__ = "0.8.0"\n',
        changelog="# Changelog\n\n## [0.8.0] - 2026-02-24\n",
        readme="version = {0.8.0}\n",
    )

    errors = m.check_consistency(root, expected_version="0.9.0", check_readme=True)

    assert any("Expected version 0.9.0" in err for err in errors)


def test_check_consistency_fails_when_readme_citation_missing(tmp_path: Path):
    m = _load_module()
    root = _write_repo(
        tmp_path,
        pyproject='[project]\nname = "x"\nversion = "0.8.0"\n',
        init='__version__ = "0.8.0"\n',
        changelog="# Changelog\n\n## [0.8.0] - 2026-02-24\n",
        readme="# no citation\n",
    )

    errors = m.check_consistency(root, expected_version=None, check_readme=True)

    assert any("README.md citation version field not found" in err for err in errors)
