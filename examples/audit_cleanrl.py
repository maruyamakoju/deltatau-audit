#!/usr/bin/env python3
"""Audit a CleanRL PPO agent for timing robustness.

This script:
1. Trains a minimal CleanRL-style PPO agent on CartPole-v1
2. Saves the checkpoint as a .pt file
3. Audits it with deltatau-audit

No external CleanRL dependency needed — the Agent class is defined inline.

Usage:
    pip install "deltatau-audit[demo]"
    python examples/audit_cleanrl.py
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.distributions import Categorical

from deltatau_audit.adapters.cleanrl import CleanRLAdapter
from deltatau_audit.auditor import run_full_audit
from deltatau_audit.report import generate_report


# ── Minimal CleanRL-style Agent ──────────────────────────────────────────────

class Agent(nn.Module):
    """CleanRL-compatible actor-critic for CartPole."""

    def __init__(self, obs_dim: int = 4, act_dim: int = 2):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, act_dim),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


# ── Training (minimal PPO) ────────────────────────────────────────────────────

def train_agent(checkpoint_path: str, total_steps: int = 80_000) -> Agent:
    """Train a CleanRL PPO agent on CartPole-v1."""
    if Path(checkpoint_path).exists():
        print(f"Found existing checkpoint: {checkpoint_path}")
        agent = Agent(obs_dim=4, act_dim=2)
        agent.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        return agent

    print(f"Training CleanRL PPO on CartPole-v1 ({total_steps} steps)...")
    env = gym.make("CartPole-v1")
    agent = Agent(obs_dim=4, act_dim=2)
    optimizer = optim.Adam(agent.parameters(), lr=2.5e-4, eps=1e-5)

    num_steps = 128
    num_minibatches = 4
    update_epochs = 4
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 0.5

    obs_buf = torch.zeros(num_steps, 4)
    act_buf = torch.zeros(num_steps, dtype=torch.long)
    logp_buf = torch.zeros(num_steps)
    rew_buf = torch.zeros(num_steps)
    done_buf = torch.zeros(num_steps)
    val_buf = torch.zeros(num_steps)

    obs, _ = env.reset()
    obs_t = torch.tensor(obs, dtype=torch.float32)
    done = False
    global_step = 0
    ep_returns = []
    ep_return = 0.0

    while global_step < total_steps:
        for step in range(num_steps):
            obs_buf[step] = obs_t
            done_buf[step] = float(done)

            with torch.no_grad():
                action, logp, _, value = agent.get_action_and_value(
                    obs_t.unsqueeze(0))
            act_buf[step] = action
            logp_buf[step] = logp
            val_buf[step] = value.squeeze()

            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            rew_buf[step] = reward
            ep_return += reward
            obs_t = torch.tensor(obs, dtype=torch.float32)
            global_step += 1

            if done:
                ep_returns.append(ep_return)
                ep_return = 0.0
                obs, _ = env.reset()
                obs_t = torch.tensor(obs, dtype=torch.float32)
                done = False

        # GAE
        with torch.no_grad():
            next_val = agent.get_value(obs_t.unsqueeze(0)).squeeze()
        advantages = torch.zeros(num_steps)
        last_gae = 0.0
        for t in reversed(range(num_steps)):
            next_v = next_val if t == num_steps - 1 else val_buf[t + 1]
            next_d = 0.0 if t == num_steps - 1 else done_buf[t + 1]
            delta = rew_buf[t] + gamma * next_v * (1 - next_d) - val_buf[t]
            last_gae = delta + gamma * gae_lambda * (1 - next_d) * last_gae
            advantages[t] = last_gae
        returns = advantages + val_buf

        # PPO update
        b_inds = np.arange(num_steps)
        mb_size = num_steps // num_minibatches
        for _ in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, num_steps, mb_size):
                idx = b_inds[start:start + mb_size]
                _, new_logp, entropy, new_val = agent.get_action_and_value(
                    obs_buf[idx], act_buf[idx])
                ratio = (new_logp - logp_buf[idx]).exp()
                adv = advantages[idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg_loss = torch.max(
                    -adv * ratio,
                    -adv * ratio.clamp(1 - clip_coef, 1 + clip_coef),
                ).mean()
                vf_loss = 0.5 * (new_val.squeeze() - returns[idx]).pow(2).mean()
                loss = pg_loss - ent_coef * entropy.mean() + vf_coef * vf_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        if len(ep_returns) > 0 and global_step % 10000 < num_steps:
            recent = ep_returns[-20:] if len(ep_returns) >= 20 else ep_returns
            print(f"  step={global_step:6d}  "
                  f"mean_return={np.mean(recent):.1f}")

    env.close()
    torch.save(agent.state_dict(), checkpoint_path)
    print(f"Saved: {checkpoint_path}\n")
    return agent


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    checkpoint_path = "cleanrl_cartpole.pt"

    # Train (or load existing)
    agent = train_agent(checkpoint_path)

    # Wrap in deltatau adapter
    adapter = CleanRLAdapter(agent, lstm=False, device="cpu")

    # Audit
    print("Auditing CleanRL agent for timing robustness...")
    result = run_full_audit(
        adapter,
        env_factory=lambda: gym.make("CartPole-v1"),
        speeds=[1, 2, 3, 5, 8],
        n_episodes=30,
        sensitivity_episodes=0,
    )

    # Report
    generate_report(result, "cleanrl_audit_report/", title="CleanRL CartPole Audit")

    summary = result["summary"]
    print(f"\nDeployment: {summary['deployment_rating']} "
          f"({summary['deployment_score']:.2f})")
    print(f"Stress:     {summary['stress_rating']} "
          f"({summary['stress_score']:.2f})")
    print("\nReport: cleanrl_audit_report/index.html")
    print("\nCLI equivalent:")
    print(f"  deltatau-audit audit-cleanrl \\")
    print(f"    --checkpoint {checkpoint_path} \\")
    print(f"    --agent-module examples/audit_cleanrl.py \\")
    print(f"    --agent-class Agent \\")
    print(f"    --agent-kwargs obs_dim=4,act_dim=2 \\")
    print(f"    --env CartPole-v1")


if __name__ == "__main__":
    main()
