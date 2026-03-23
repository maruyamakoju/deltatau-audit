"""Validate version and release metadata consistency.

Checks:
1) pyproject.toml [project].version
2) deltatau_audit/__init__.py __version__
3) Latest released version header in CHANGELOG.md
4) README citation version field (optional)

Exit code is non-zero when consistency checks fail.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - py39/310 fallback
    import tomli as tomllib  # type: ignore

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
INIT_VERSION_RE = re.compile(r"^__version__\s*=\s*\"([^\"]+)\"\s*$", re.MULTILINE)
CHANGELOG_HEADER_RE = re.compile(r"^##\s+\[([^\]]+)\]", re.MULTILINE)
README_CITATION_RE = re.compile(r"^\s*version\s*=\s*\{([^\}]+)\}", re.MULTILINE | re.IGNORECASE)


def _normalize_version(raw: str) -> str:
    value = raw.strip()
    if value.startswith("v"):
        value = value[1:]
    return value


def read_pyproject_version(path: Path) -> str:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    project = data.get("project")
    if not isinstance(project, dict):
        raise ValueError("pyproject.toml is missing [project] table")
    version = project.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("pyproject.toml [project].version is missing")
    return version.strip()


def read_init_version(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    m = INIT_VERSION_RE.search(text)
    if not m:
        raise ValueError("__version__ not found in deltatau_audit/__init__.py")
    return m.group(1).strip()


def read_latest_changelog_release(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    matches = CHANGELOG_HEADER_RE.findall(text)
    for entry in matches:
        name = entry.strip()
        if name.lower() == "unreleased":
            continue
        if SEMVER_RE.match(name):
            return name
    raise ValueError("No released semver header found in CHANGELOG.md")


def read_readme_citation_version(path: Path) -> str | None:
    text = path.read_text(encoding="utf-8")
    m = README_CITATION_RE.search(text)
    if not m:
        return None
    return m.group(1).strip()


def check_consistency(root: Path, expected_version: str | None, check_readme: bool) -> list[str]:
    pyproject_version = read_pyproject_version(root / "pyproject.toml")
    init_version = read_init_version(root / "deltatau_audit" / "__init__.py")
    changelog_version = read_latest_changelog_release(root / "CHANGELOG.md")
    readme_version = read_readme_citation_version(root / "README.md")

    values: list[tuple[str, str]] = [
        ("pyproject.toml", pyproject_version),
        ("deltatau_audit/__init__.py", init_version),
        ("CHANGELOG.md", changelog_version),
    ]
    if check_readme and readme_version is not None:
        values.append(("README.md citation", readme_version))

    errors: list[str] = []
    distinct = sorted({v for _, v in values})
    if len(distinct) != 1:
        detail = ", ".join(f"{src}={ver}" for src, ver in values)
        errors.append(f"Version mismatch across files: {detail}")

    if expected_version:
        expected_norm = _normalize_version(expected_version)
        if not SEMVER_RE.match(expected_norm):
            errors.append(
                f"--expected-version must be semver (x.y.z or vx.y.z), got: {expected_version}"
            )
        elif distinct and distinct[0] != expected_norm:
            errors.append(f"Expected version {expected_norm}, found {distinct[0]}")

    for source, version in values:
        if not SEMVER_RE.match(version):
            errors.append(f"{source} contains non-semver version: {version}")

    if check_readme and readme_version is None:
        errors.append("README.md citation version field not found")

    return errors


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check release metadata consistency")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root directory (default: script parent).",
    )
    parser.add_argument(
        "--expected-version",
        type=str,
        default=None,
        help="Optional expected version to enforce (e.g., 0.8.0 or v0.8.0).",
    )
    parser.add_argument(
        "--skip-readme",
        action="store_true",
        default=False,
        help="Skip README citation version check.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    root = args.root.resolve()

    errors = check_consistency(
        root=root,
        expected_version=args.expected_version,
        check_readme=not args.skip_readme,
    )
    if errors:
        print("Release consistency check: FAILED")
        for msg in errors:
            print(f"  - {msg}")
        return 1

    print("Release consistency check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
