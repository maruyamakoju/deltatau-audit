from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from deltatau_audit.schema import AuditReport, TemporalCapability


@runtime_checkable
class AgentAdapter(Protocol):
    """Rigorous interface for any agent being audited.

    Every agent must conform to this protocol to be evaluated by the unified engine.
    This unified version supports both the legacy timing-ablation axis and the
    newer reasoning-aware axis.
    """

    def get_capabilities(self) -> TemporalCapability:
        """Returns metadata about what the agent can do (e.g., pondering, lookahead)."""
        ...

    def act(
        self,
        observation: Any,
        deterministic: bool = True,
        ponder_steps: Optional[int] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Core action selection method.

        Args:
            observation: Current environment observation.
            deterministic: Whether to use deterministic action selection.
            ponder_steps: Optional override for internal reasoning steps.

        Returns:
            Tuple of (action, info_dict containing value, dt, hidden_state, etc.).
        """
        ...

    def reset_internal_state(self) -> None:
        """Resets recurrent or internal reasoning states (e.g., MCTS trees, RNN hidden)."""
        ...

    def rerun_with_dt(self, observation: Any, target_dt: float) -> Dict[str, Any]:
        """Re-run the transition logic with a specific Δτ override.

        Optional — only needed for intervention ablation.
        """
        ...

    def recompute_value(self, info: Dict[str, Any]) -> float:
        """Compute value from a (possibly intervened) internal state info."""
        ...


@runtime_checkable
class EnvironmentalStressor(Protocol):
    """Interface for dynamic environment modifications (e.g., Jitter, Hunter attacks)."""

    def apply(self, env: Any, intensity: float) -> Any:
        """Applies a stress or attack to the environment.

        Returns:
            The modified environment.
        """
        ...


@runtime_checkable
class Auditor(Protocol):
    """The high-level engine that runs an audit session."""

    def run(self, agent: AgentAdapter, env_id: str, **kwargs: Any) -> AuditReport:
        """Executes a full multi-stage audit on the given agent."""
        ...


@runtime_checkable
class Fixer(Protocol):
    """Automated policy correction module."""

    def fix(self, agent: AgentAdapter, report: AuditReport) -> AgentAdapter:
        """Attempts to correct identified vulnerabilities based on the audit report."""
        ...


# Protocol presets (Backward compatibility for CLI but strictly typed)
PROTOCOL_PRESETS: Dict[str, Dict[str, Any]] = {
    "ci": {
        "episodes": 30,
        "speeds": [1, 2, 3, 5, 8],
        "adaptive": False,
        "target_ci_width": 0.10,
        "max_episodes": 200,
        "bootstrap_samples": 1000,
        "ci_gate_mode": "score",
    },
    "research": {
        "episodes": 50,
        "speeds": [1, 2, 3, 5, 8],
        "adaptive": True,
        "target_ci_width": 0.05,
        "max_episodes": 300,
        "bootstrap_samples": 2000,
        "ci_gate_mode": "worst_ci_lower",
        "seeds": [0, 1, 2, 3, 4],
    },
    "paper": {
        "episodes": 100,
        "speeds": [1, 2, 3, 5, 8],
        "adaptive": True,
        "target_ci_width": 0.03,
        "max_episodes": 500,
        "bootstrap_samples": 5000,
        "ci_gate_mode": "worst_ci_lower",
        "seeds": list(range(10)),
    },
    "deepmind": {
        "episodes": 200,
        "speeds": [1, 1.5, 2, 3, 5, 10],
        "adaptive": True,
        "target_ci_width": 0.01,
        "bootstrap_samples": 10000,
        "seeds": list(range(10)),
    },
}

_PROTOCOL_ARG_MAP: Dict[str, Tuple[str, str]] = {
    "episodes": ("episodes", "--episodes"),
    "speeds": ("speeds", "--speeds"),
    "adaptive": ("adaptive", "--adaptive"),
    "target_ci_width": ("target_ci_width", "--target-ci-width"),
    "max_episodes": ("max_episodes", "--max-episodes"),
    "bootstrap_samples": ("bootstrap_samples", "--bootstrap-samples"),
    "ci_gate_mode": ("ci_gate_mode", "--ci-gate-mode"),
    "seeds": ("seeds", "--seeds"),
}


def apply_protocol_defaults(args: Any, argv: Optional[List[str]] = None) -> Dict[str, Dict[str, Any] | str]:
    """Apply protocol preset defaults to an argparse-like namespace.

    When ``allow_protocol_override`` is false, preset values always win.
    When true, only fields explicitly set on the CLI are preserved.
    """
    name = getattr(args, "protocol", "custom") or "custom"
    preset = PROTOCOL_PRESETS.get(name, {})
    if name == "custom" or not preset:
        return {"name": "custom", "applied": {}, "ignored": {}}

    allow_override = bool(getattr(args, "allow_protocol_override", False))
    argv_tokens = set(argv or [])
    applied: Dict[str, Any] = {}
    ignored: Dict[str, Any] = {}

    for preset_key, preset_value in preset.items():
        arg_info = _PROTOCOL_ARG_MAP.get(preset_key)
        if arg_info is None:
            continue
        attr_name, cli_flag = arg_info
        if not hasattr(args, attr_name):
            continue

        current_value = getattr(args, attr_name)
        cli_explicit = cli_flag in argv_tokens
        if allow_override and cli_explicit:
            ignored[attr_name] = current_value
            continue

        if current_value != preset_value:
            setattr(args, attr_name, preset_value)
            applied[attr_name] = preset_value

    return {"name": name, "applied": applied, "ignored": ignored}
