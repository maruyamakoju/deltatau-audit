"""Tests for protocol preset application."""

from __future__ import annotations

from argparse import Namespace

from deltatau_audit.protocols import apply_protocol_defaults


def _args() -> Namespace:
    return Namespace(
        protocol="custom",
        allow_protocol_override=False,
        episodes=30,
        speeds=[1, 2],
        adaptive=False,
        target_ci_width=0.10,
        max_episodes=500,
        deploy_threshold=0.80,
        stress_threshold=0.50,
        ci_min_deployment_pass_rate=0.80,
        ci_min_stress_pass_rate=0.50,
        ci_gate_mode="score",
        seeds=None,
    )


def test_research_protocol_applies_expected_defaults():
    args = _args()
    args.protocol = "research"

    meta = apply_protocol_defaults(args, argv=["audit-sb3", "--protocol", "research"])

    assert meta["name"] == "research"
    assert args.episodes == 50
    assert args.adaptive is True
    assert abs(args.target_ci_width - 0.05) < 1e-9
    assert args.max_episodes == 300
    assert args.ci_gate_mode == "worst_ci_lower"
    assert args.seeds == [0, 1, 2, 3, 4]


def test_protocol_override_flag_keeps_user_values():
    args = _args()
    args.protocol = "research"
    args.allow_protocol_override = True
    args.episodes = 77
    args.ci_gate_mode = "score"

    meta = apply_protocol_defaults(
        args,
        argv=[
            "audit-sb3",
            "--protocol",
            "research",
            "--allow-protocol-override",
            "--episodes",
            "77",
            "--ci-gate-mode",
            "score",
        ],
    )

    assert meta["name"] == "research"
    assert args.episodes == 77
    assert args.ci_gate_mode == "score"
    assert args.adaptive is True  # still applied for unspecified fields

