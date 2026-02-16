"""Train GRU agent on CartPole with self-contained PPO.

Demonstrates deltatau-audit on an external (non-InternalTimeAgent) model.

Usage:
    python demo_external_env/train.py --mode baseline
    python demo_external_env/train.py --mode robust
"""

import argparse
import json
import os
import sys

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


# ══════════════════════════════════════════════════════════════════════
# MODEL — Simple GRU policy (NOT InternalTimeAgent)
# ══════════════════════════════════════════════════════════════════════

class SimpleGRUPolicy(nn.Module):
    """Minimal GRU actor-critic for discrete control."""

    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(obs_dim, hidden_dim)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.Tanh(),
            nn.Linear(64, act_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Orthogonal init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Smaller init for policy output
        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)

    def forward(self, obs, hidden):
        h = self.gru(obs, hidden)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return Categorical(logits=logits), value, h

    def get_initial_hidden(self, batch, device="cpu"):
        return torch.zeros(batch, self.hidden_dim, device=device)


# ══════════════════════════════════════════════════════════════════════
# TRAINING WRAPPERS
# ══════════════════════════════════════════════════════════════════════

class RandomSpeedWrapper(gym.Wrapper):
    """Randomize frame-skip at each episode reset."""

    def __init__(self, env, speeds=(1, 2, 3)):
        super().__init__(env)
        self.speeds = speeds
        self.speed = 1

    def reset(self, **kwargs):
        self.speed = int(np.random.choice(self.speeds))
        return self.env.reset(**kwargs)

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        for _ in range(self.speed):
            obs, r, terminated, truncated, info = self.env.step(action)
            total_reward += r
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


# ══════════════════════════════════════════════════════════════════════
# VECTORIZED ENV
# ══════════════════════════════════════════════════════════════════════

class SyncVecEnv:
    """Minimal sync vec env with auto-reset."""

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self._ep_rew = np.zeros(self.num_envs)

    def reset(self):
        obs_list = []
        for env in self.envs:
            obs, _ = env.reset()
            obs_list.append(obs)
        self._ep_rew[:] = 0
        return np.stack(obs_list)

    def step(self, actions):
        obs_list, rew_list, done_list = [], [], []
        completed = []
        for i, (env, a) in enumerate(zip(self.envs, actions)):
            obs, reward, term, trunc, info = env.step(int(a))
            done = term or trunc
            self._ep_rew[i] += reward
            if done:
                completed.append(self._ep_rew[i])
                self._ep_rew[i] = 0
                obs, _ = env.reset()
            obs_list.append(obs)
            rew_list.append(reward)
            done_list.append(float(done))
        return (np.stack(obs_list),
                np.array(rew_list, dtype=np.float32),
                np.array(done_list, dtype=np.float32),
                completed)

    def close(self):
        for env in self.envs:
            env.close()


# ══════════════════════════════════════════════════════════════════════
# PPO (self-contained, no dependency on internal_time_rl)
# ══════════════════════════════════════════════════════════════════════

def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """Compute GAE advantages and returns."""
    num_steps = len(rewards)
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]
        next_non_term = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_term - values[t]
        last_gae = delta + gamma * lam * next_non_term * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def ppo_update(model, optimizer, obs, actions, log_probs_old, advantages,
               returns, hiddens, num_epochs=4, num_minibatches=4,
               clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5):
    """Single PPO update from collected rollout data."""
    batch_size = obs.shape[0]
    mb_size = batch_size // num_minibatches

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    metrics = {"policy_loss": 0, "value_loss": 0, "entropy": 0}
    n_updates = 0

    for _ in range(num_epochs):
        idx = torch.randperm(batch_size, device=obs.device)
        for start in range(0, batch_size, mb_size):
            mb = idx[start:start + mb_size]

            dist, values, _ = model(obs[mb], hiddens[mb])
            new_log_probs = dist.log_prob(actions[mb])
            entropy = dist.entropy().mean()

            # Policy loss (clipped)
            ratio = (new_log_probs - log_probs_old[mb]).exp()
            surr1 = ratio * advantages[mb]
            surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages[mb]
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = 0.5 * (values - returns[mb]).pow(2).mean()

            # Total
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            metrics["policy_loss"] += policy_loss.item()
            metrics["value_loss"] += value_loss.item()
            metrics["entropy"] += entropy.item()
            n_updates += 1

    return {k: v / max(n_updates, 1) for k, v in metrics.items()}


# ══════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def make_env(mode="baseline"):
    def factory():
        env = gym.make("CartPole-v1")
        if mode == "robust":
            env = RandomSpeedWrapper(env, speeds=(1, 2, 3))
        return env
    return factory


def train(mode, out_dir, total_timesteps=500_000, num_envs=8,
          num_steps=128, device="cpu"):
    os.makedirs(out_dir, exist_ok=True)

    obs_dim = 4
    act_dim = 2

    model = SimpleGRUPolicy(obs_dim, act_dim, hidden_dim=64)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4, eps=1e-5)

    vec_env = SyncVecEnv([make_env(mode) for _ in range(num_envs)])
    num_updates = total_timesteps // (num_steps * num_envs)

    obs = vec_env.reset()
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    hidden = model.get_initial_hidden(num_envs, device)

    all_rewards = []
    history = {"episode_rewards": [], "mode": mode}

    print(f"Training {mode} GRU agent on CartPole-v1")
    print(f"  Model: SimpleGRUPolicy (hidden=64)")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Updates: {num_updates}")
    if mode == "robust":
        print(f"  Speed randomization: [1, 2, 3]")
    print()

    for update in range(1, num_updates + 1):
        # Storage
        b_obs = torch.zeros(num_steps, num_envs, obs_dim, device=device)
        b_actions = torch.zeros(num_steps, num_envs, dtype=torch.long,
                                device=device)
        b_logprobs = torch.zeros(num_steps, num_envs, device=device)
        b_rewards = torch.zeros(num_steps, num_envs, device=device)
        b_dones = torch.zeros(num_steps, num_envs, device=device)
        b_values = torch.zeros(num_steps, num_envs, device=device)
        b_hiddens = torch.zeros(num_steps, num_envs, 64, device=device)

        model.eval()
        with torch.no_grad():
            for step in range(num_steps):
                b_obs[step] = obs_t
                b_hiddens[step] = hidden

                dist, value, hidden_new = model(obs_t, hidden)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                b_actions[step] = action
                b_logprobs[step] = log_prob
                b_values[step] = value

                next_obs, rewards, dones, completed = \
                    vec_env.step(action.cpu().numpy())

                b_rewards[step] = torch.tensor(rewards, device=device)
                b_dones[step] = torch.tensor(dones, device=device)

                # Reset hidden for done envs
                for i in range(num_envs):
                    if dones[i]:
                        hidden_new[i] = torch.zeros(64, device=device)

                all_rewards.extend(completed)
                obs_t = torch.tensor(
                    next_obs, dtype=torch.float32, device=device)
                hidden = hidden_new

            # Last value for GAE
            _, last_value, _ = model(obs_t, hidden)

        # Compute advantages
        advantages, returns = compute_gae(
            b_rewards, b_values, b_dones, last_value)

        # Flatten for PPO update
        flat_obs = b_obs.reshape(-1, obs_dim)
        flat_actions = b_actions.reshape(-1)
        flat_logprobs = b_logprobs.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_hiddens = b_hiddens.reshape(-1, 64)

        # PPO update
        model.train()
        metrics = ppo_update(
            model, optimizer,
            flat_obs, flat_actions, flat_logprobs,
            flat_advantages, flat_returns, flat_hiddens,
        )

        if all_rewards:
            recent = np.mean(all_rewards[-20:])
            history["episode_rewards"].append(float(recent))

        if update % 20 == 0:
            ts = update * num_steps * num_envs
            r_str = (f"R={np.mean(all_rewards[-20:]):.1f}"
                     if len(all_rewards) >= 20 else
                     (f"R={np.mean(all_rewards):.1f}"
                      if all_rewards else "R=--"))
            print(f"  U {update}/{num_updates} T={ts:,} | {r_str} "
                  f"| Ent={metrics['entropy']:.3f}")

    # Save
    torch.save({
        "model": model.state_dict(),
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "hidden_dim": 64,
        "mode": mode,
    }, os.path.join(out_dir, "final.pt"))

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f)

    vec_env.close()

    final_r = (np.mean(all_rewards[-50:]) if len(all_rewards) >= 50
               else (np.mean(all_rewards) if all_rewards else 0))
    print(f"\n  Final avg reward: {final_r:.1f}")
    print(f"  Saved to {out_dir}/final.pt")
    return final_r


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "robust"],
                        required=True)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.out is None:
        args.out = f"demo_external_env/checkpoints/{args.mode}"

    train(args.mode, args.out,
          total_timesteps=args.timesteps, device=args.device)
