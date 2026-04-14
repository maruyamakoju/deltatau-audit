"""Runtime provenance and reproducibility metadata helpers."""

from __future__ import annotations

import datetime
import hashlib
import os
import platform
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping


def _run_command(args: list[str], *, cwd: str | None = None, timeout_s: float = 2.0) -> str | None:
    """Run a command and return stripped stdout, else None on failure."""
    try:
        out = subprocess.check_output(
            args,
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=cwd,
            timeout=timeout_s,
        )
    except Exception:
        return None
    text = out.strip()
    return text if text else None


@lru_cache(maxsize=8)
def _collect_git_info_cached(cwd_key: str) -> dict[str, Any]:
    cwd = cwd_key or None
    """Collect git commit/branch/dirty status for reproducibility."""
    commit = _run_command(["git", "rev-parse", "HEAD"], cwd=cwd)
    if commit is None:
        return {
            "available": False,
            "commit": None,
            "short_commit": None,
            "branch": None,
            "dirty": None,
        }
    status = _run_command(["git", "status", "--porcelain"], cwd=cwd, timeout_s=3.0)
    branch = _run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
    dirty = bool(status) if status is not None else False
    return {
        "available": True,
        "commit": commit,
        "short_commit": commit[:12],
        "branch": branch,
        "dirty": dirty,
    }


def collect_git_info(cwd: str | None = None) -> dict[str, Any]:
    key = str(Path(cwd).resolve()) if cwd else ""
    return dict(_collect_git_info_cached(key))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _collect_lockfile_hashes(cwd: str | None = None) -> dict[str, str]:
    """Hash common project dependency files when present."""
    root = Path(cwd or os.getcwd())
    file_names = ("pyproject.toml", "requirements.txt", "poetry.lock", "uv.lock")
    out: dict[str, str] = {}
    for name in file_names:
        p = root / name
        if not p.exists() or not p.is_file():
            continue
        try:
            content = p.read_text(encoding="utf-8")
        except OSError:
            continue
        out[name] = _sha256_text(content)
    return out


@lru_cache(maxsize=8)
def _collect_dependency_info_cached(cwd_key: str) -> dict[str, Any]:
    cwd = cwd_key or None
    """Collect dependency snapshot metadata (hash + counts)."""
    freeze = _run_command([sys.executable, "-m", "pip", "freeze"], cwd=cwd, timeout_s=20.0)
    lock_hashes = _collect_lockfile_hashes(cwd=cwd)

    if freeze is None:
        return {
            "available": False,
            "pip_freeze_sha256": None,
            "pip_freeze_count": 0,
            "pip_freeze_sample": [],
            "lockfile_hashes": lock_hashes,
        }

    lines = [ln.strip() for ln in freeze.splitlines() if ln.strip()]
    return {
        "available": True,
        "pip_freeze_sha256": _sha256_text("\n".join(lines)),
        "pip_freeze_count": len(lines),
        "pip_freeze_sample": lines[:40],
        "lockfile_hashes": lock_hashes,
    }


def collect_dependency_info(cwd: str | None = None) -> dict[str, Any]:
    key = str(Path(cwd).resolve()) if cwd else ""
    return dict(_collect_dependency_info_cached(key))


def collect_runtime_info() -> dict[str, Any]:
    """Collect runtime environment metadata."""
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "executable": sys.executable,
        "cwd": os.getcwd(),
    }


def minimal_manifest(protocol_name: str = "custom") -> dict[str, Any]:
    """Return a minimal valid manifest placeholder."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return {
        "created_at": now,
        "protocol": {"name": protocol_name, "config": {}},
        "experiment": {},
        "cli": {"command": None, "argv": []},
        "runtime": collect_runtime_info(),
        "git": collect_git_info(),
        "dependencies": collect_dependency_info(),
    }


def _jsonable_map(data: Mapping[str, Any]) -> dict[str, Any]:
    """Convert mapping into JSON-safe primitives where possible."""
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[str(key)] = value
        elif isinstance(value, (list, tuple)):
            arr: list[Any] = []
            for item in value:
                if isinstance(item, (str, int, float, bool)) or item is None:
                    arr.append(item)
                else:
                    arr.append(str(item))
            out[str(key)] = arr
        elif isinstance(value, Mapping):
            out[str(key)] = _jsonable_map(value)
        else:
            out[str(key)] = str(value)
    return out


def build_manifest(
    *,
    command: str | None,
    argv: Iterable[str],
    protocol_name: str,
    protocol_config: Mapping[str, Any] | None,
    experiment: Mapping[str, Any],
    cwd: str | None = None,
) -> dict[str, Any]:
    """Build a reproducibility manifest for an audit run."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return {
        "created_at": now,
        "protocol": {
            "name": protocol_name,
            "config": _jsonable_map(protocol_config or {}),
        },
        "experiment": _jsonable_map(experiment),
        "cli": {
            "command": command,
            "argv": list(argv),
        },
        "runtime": collect_runtime_info(),
        "git": collect_git_info(cwd=cwd),
        "dependencies": collect_dependency_info(cwd=cwd),
    }
