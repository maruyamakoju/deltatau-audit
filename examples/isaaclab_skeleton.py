#!/usr/bin/env python3
"""IsaacLab / RSL-RL skeleton for deltatau-audit.

This shows how to plug an IsaacLab (RSL-RL) trained policy into
deltatau-audit using the TorchPolicyAdapter.

IsaacLab trains via RSL-RL and saves checkpoints as:
    {"model_state_dict": {"actor.weight": ..., "actor.bias": ...,
                          "critic.weight": ..., "critic.bias": ...}}

Usage:
    # 1. Install deltatau-audit
    pip install deltatau-audit

    # 2. Install your IsaacLab environment (outside scope of this example)

    # 3. Run this script with your checkpoint and env_factory
    python examples/isaaclab_skeleton.py --checkpoint model.pt
"""

import argparse
from typing import Callable

import torch
import torch.nn as nn

from deltatau_audit.adapters.torch_policy import TorchPolicyAdapter
from deltatau_audit.auditor import run_full_audit
from deltatau_audit.report import generate_report


# ── Replace these with your actual network architectures ─────────────────────

class ActorNetwork(nn.Module):
    """Example actor. Replace with your RSL-RL ActorCritic.actor."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x):
        return self.net(x)


class CriticNetwork(nn.Module):
    """Example critic. Replace with your RSL-RL ActorCritic.critic."""

    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


# ── Audit skeleton ─────────────────────────────────────────────────────────────

def make_env_factory(env_id: str) -> Callable:
    """Return a zero-argument callable that creates your Gymnasium env.

    For IsaacLab environments, wrap them in a Gymnasium-compatible adapter
    (IsaacLab provides gym_utils for this).

    For standard Gymnasium envs (for testing):
        import gymnasium as gym
        return lambda: gym.make(env_id)
    """
    import gymnasium as gym
    return lambda: gym.make(env_id)


def audit_isaaclab_checkpoint(
    checkpoint_path: str,
    obs_dim: int,
    act_dim: int,
    env_id: str,
    output_dir: str = "isaaclab_audit/",
    device: str = "cpu",
):
    """Load an RSL-RL checkpoint and run timing robustness audit.

    Args:
        checkpoint_path: Path to .pt checkpoint (RSL-RL format).
        obs_dim: Observation space dimension.
        act_dim: Action space dimension.
        env_id: Gymnasium environment ID.
        output_dir: Where to write the HTML audit report.
        device: "cpu" or "cuda".

    Checkpoint formats supported:
        {"model_state_dict": {"actor.*": ..., "critic.*": ...}}  # RSL-RL
        {"actor": state_dict, "critic": state_dict}               # explicit split
        raw state_dict                                             # actor-only
    """
    actor = ActorNetwork(obs_dim=obs_dim, act_dim=act_dim)
    critic = CriticNetwork(obs_dim=obs_dim)

    # Continuous action space: is_discrete=False
    adapter = TorchPolicyAdapter.from_checkpoint(
        checkpoint_path=checkpoint_path,
        actor=actor,
        critic=critic,
        is_discrete=False,
        device=device,
    )

    env_factory = make_env_factory(env_id)

    print(f"Auditing: {checkpoint_path}")
    print(f"  obs_dim={obs_dim}, act_dim={act_dim}, env={env_id}")
    print()

    result = run_full_audit(
        adapter,
        env_factory,
        speeds=[1, 2, 3, 5, 8],
        n_episodes=30,
        sensitivity_episodes=0,
        device=device,
    )

    generate_report(result, output_dir,
                    title=f"IsaacLab Timing Audit — {env_id}")

    summary = result["summary"]
    print(f"Deployment: {summary['deployment_rating']} "
          f"({summary['deployment_score']:.2f})")
    print(f"Stress:     {summary['stress_rating']} "
          f"({summary['stress_score']:.2f})")
    print(f"\nReport: {output_dir}index.html")


# ── Direct callable API (no checkpoint file needed) ───────────────────────────

def audit_from_callable(act_fn, env_id: str, output_dir: str = "audit/"):
    """Audit any PyTorch policy given as a callable.

    This is the most flexible interface — useful when your policy is part of
    a larger system (e.g., IsaacLab runner) and you don't want to extract it.

    Args:
        act_fn: Callable(obs: Tensor shape [1, obs_dim]) -> (action, value)
                where action is Tensor and value is scalar Tensor.
        env_id: Gymnasium environment ID.
        output_dir: Report output directory.

    Example:
        # Inside your IsaacLab eval script:
        def my_act(obs):
            with torch.no_grad():
                action = runner.alg.actor_critic.act(obs)
                value  = runner.alg.actor_critic.evaluate(obs)
            return action, value

        audit_from_callable(my_act, "CartPole-v1")
    """
    import gymnasium as gym

    adapter = TorchPolicyAdapter(act_fn=act_fn, device="cpu")
    env_factory = lambda: gym.make(env_id)

    result = run_full_audit(
        adapter, env_factory,
        speeds=[1, 2, 3, 5, 8],
        n_episodes=30,
    )
    generate_report(result, output_dir)
    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="IsaacLab / RSL-RL timing robustness audit skeleton")
    parser.add_argument("--checkpoint", type=str,
                        default=None,
                        help="Path to RSL-RL .pt checkpoint")
    parser.add_argument("--obs-dim", type=int, default=48,
                        help="Observation dimension (default: 48)")
    parser.add_argument("--act-dim", type=int, default=12,
                        help="Action dimension (default: 12)")
    parser.add_argument("--env", type=str, default="CartPole-v1",
                        help="Gymnasium environment ID")
    parser.add_argument("--out", type=str, default="isaaclab_audit/",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.checkpoint is None:
        print("No checkpoint provided. Running with random weights as demo.")
        print("In production: pass --checkpoint path/to/model.pt\n")

        # Demo: random weights
        actor = ActorNetwork(obs_dim=args.obs_dim, act_dim=args.act_dim)
        critic = CriticNetwork(obs_dim=args.obs_dim)

        @torch.no_grad()
        def random_act(obs):
            logits = actor(obs)
            action = logits  # continuous, deterministic
            value = critic(obs)
            return action, value

        import gymnasium as gym
        # Fall back to CartPole for the demo (obs/act dims won't match
        # unless --obs-dim 4 --act-dim 2 is passed)
        print("TIP: Use --env CartPole-v1 --obs-dim 4 --act-dim 2 for a "
              "quick demo with a standard Gymnasium env.\n")
        return

    audit_isaaclab_checkpoint(
        checkpoint_path=args.checkpoint,
        obs_dim=args.obs_dim,
        act_dim=args.act_dim,
        env_id=args.env,
        output_dir=args.out,
        device=args.device,
    )


if __name__ == "__main__":
    main()
