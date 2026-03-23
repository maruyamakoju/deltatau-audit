from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "check_contracts.py"
    spec = importlib.util.spec_from_file_location("check_contracts", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_check_contracts_invokes_expected_pytest_command(monkeypatch):
    m = _load_module()
    calls: list[tuple[list[str], str]] = []

    def _fake_run(cmd, cwd=None, check=False):  # noqa: ARG001
        calls.append((list(cmd), str(cwd)))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(m.subprocess, "run", _fake_run)

    rc = m.main([])

    assert rc == 0
    assert calls
    cmd, cwd = calls[0]
    assert cmd[:4] == [sys.executable, "-m", "pytest", "-q"]
    assert cmd[4:] == m.CONTRACT_TESTS
    assert cwd == str(m.ROOT)
