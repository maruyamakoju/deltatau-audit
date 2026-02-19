"""Tests for v0.5.4: Type annotations + py.typed marker."""

import inspect
import pathlib
import pytest


# ─────────────────────────────────────────────────────────────────────
# 1. py.typed marker exists (PEP 561)
# ─────────────────────────────────────────────────────────────────────

def test_py_typed_marker_exists():
    """Package must ship a py.typed file (PEP 561 compliance)."""
    import deltatau_audit
    pkg_dir = pathlib.Path(deltatau_audit.__file__).parent
    assert (pkg_dir / "py.typed").exists(), \
        f"py.typed not found in {pkg_dir}"


# ─────────────────────────────────────────────────────────────────────
# 2. run_full_audit has correct annotations
# ─────────────────────────────────────────────────────────────────────

def test_run_full_audit_has_return_annotation():
    from deltatau_audit.auditor import run_full_audit
    hints = run_full_audit.__annotations__
    assert "return" in hints


def test_run_full_audit_adaptive_annotated():
    from deltatau_audit.auditor import run_full_audit
    sig = inspect.signature(run_full_audit)
    assert "adaptive" in sig.parameters
    assert "target_ci_width" in sig.parameters
    assert "max_episodes" in sig.parameters


def test_run_robustness_audit_has_return_annotation():
    from deltatau_audit.auditor import run_robustness_audit
    hints = run_robustness_audit.__annotations__
    assert "return" in hints


def test_run_robustness_audit_adaptive_annotated():
    from deltatau_audit.auditor import run_robustness_audit
    sig = inspect.signature(run_robustness_audit)
    assert "adaptive" in sig.parameters
    assert "target_ci_width" in sig.parameters
    assert "max_episodes" in sig.parameters


# ─────────────────────────────────────────────────────────────────────
# 3. generate_diagnosis has correct annotations
# ─────────────────────────────────────────────────────────────────────

def test_generate_diagnosis_has_annotations():
    from deltatau_audit.diagnose import generate_diagnosis
    hints = generate_diagnosis.__annotations__
    assert "return" in hints
    assert "summary" in hints
    assert "robustness" in hints


# ─────────────────────────────────────────────────────────────────────
# 4. AgentAdapter ABC annotations
# ─────────────────────────────────────────────────────────────────────

def test_agent_adapter_reset_hidden_annotated():
    from deltatau_audit.adapters.base import AgentAdapter
    hints = AgentAdapter.reset_hidden.__annotations__
    assert "return" in hints


def test_agent_adapter_act_annotated():
    from deltatau_audit.adapters.base import AgentAdapter
    hints = AgentAdapter.act.__annotations__
    assert "return" in hints


# ─────────────────────────────────────────────────────────────────────
# 5. Public API exports
# ─────────────────────────────────────────────────────────────────────

def test_package_has_version():
    import deltatau_audit
    assert hasattr(deltatau_audit, "__version__")
    assert isinstance(deltatau_audit.__version__, str)
    assert len(deltatau_audit.__version__) > 0


def test_auditor_exports_run_full_audit():
    from deltatau_audit.auditor import run_full_audit, run_robustness_audit
    assert callable(run_full_audit)
    assert callable(run_robustness_audit)


def test_diagnose_exports_generate_diagnosis():
    from deltatau_audit.diagnose import generate_diagnosis
    assert callable(generate_diagnosis)
