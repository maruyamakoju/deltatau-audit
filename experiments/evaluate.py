"""Evaluate a trained agent and visualize internal time dynamics.

Usage:
    python evaluate.py runs/internal_time_gru/seed_0/final.pt
    python evaluate.py runs/internal_time_gru/seed_0/final.pt --episodes 5
"""

import argparse
import json
import os

import numpy as np
import torch

from internal_time_rl.models.policy import InternalTimeAgent
from internal_time_rl.envs.delayed_reward_chain import DelayedRewardChainEnv
from internal_time_rl.envs.flickering_env import FlickeringWrapper
from internal_time_rl.analysis.visualize import plot_episode_time_dynamics


def evaluate(checkpoint_path, num_episodes=5, render=True):
    """Load a trained agent and run episodes, recording time dynamics."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    env_cfg = config.get("env", {})
    model_cfg = config.get("model", {})

    # Recreate environment
    env = DelayedRewardChainEnv(
        length=env_cfg.get("length", 20),
        delay=env_cfg.get("delay", 10),
        max_steps=env_cfg.get("max_steps", 200),
        noise=env_cfg.get("noise", 0.0),
    )
    if env_cfg.get("flickering", False):
        env = FlickeringWrapper(env, flicker_prob=env_cfg.get("flicker_prob", 0.3))

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Recreate agent
    agent = InternalTimeAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=model_cfg.get("hidden_dim", 128),
        latent_dim=model_cfg.get("latent_dim", 64),
        time_hidden_dim=model_cfg.get("time_hidden_dim", 32),
        use_internal_time=model_cfg.get("use_internal_time", True),
        transition_type=model_cfg.get("transition_type", "gru"),
    )
    agent.load_state_dict(checkpoint["agent"])
    agent.eval()

    print(f"Loaded agent from {checkpoint_path}")
    print(f"Internal time: {model_cfg.get('use_internal_time', True)}")
    print(f"Update: {checkpoint.get('update', '?')}")
    print()

    all_episode_data = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        hidden = agent.get_initial_hidden(1, torch.device("cpu"))

        ep_rewards = []
        ep_delta_taus = []
        ep_positions = []
        ep_actions = []
        total_reward = 0
        done = False

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _, hidden, delta_tau = agent.get_action_and_value(
                    obs_t, hidden
                )

            action_int = action.item()
            obs, reward, terminated, truncated, info = env.step(action_int)
            done = terminated or truncated

            ep_rewards.append(reward)
            ep_delta_taus.append(delta_tau.item())
            ep_positions.append(info.get("position", 0))
            ep_actions.append(action_int)
            total_reward += reward

        ep_data = {
            "rewards": ep_rewards,
            "delta_taus": ep_delta_taus,
            "positions": ep_positions,
            "actions": ep_actions,
            "total_reward": total_reward,
            "length": len(ep_rewards),
        }
        all_episode_data.append(ep_data)

        print(
            f"Episode {ep + 1}: reward={total_reward:.2f}, "
            f"length={len(ep_rewards)}, "
            f"dt_mean={np.mean(ep_delta_taus):.3f}, "
            f"dt_std={np.std(ep_delta_taus):.3f}"
        )

        if render and model_cfg.get("use_internal_time", True):
            save_dir = os.path.dirname(checkpoint_path)
            plot_episode_time_dynamics(
                ep_delta_taus,
                ep_rewards,
                ep_positions,
                save_path=os.path.join(save_dir, f"episode_{ep}_dynamics.png"),
            )

    # Summary
    print(f"\nAverage reward: {np.mean([e['total_reward'] for e in all_episode_data]):.3f}")
    print(f"Average length: {np.mean([e['length'] for e in all_episode_data]):.1f}")

    # Save episode data
    save_dir = os.path.dirname(checkpoint_path)
    with open(os.path.join(save_dir, "eval_episodes.json"), "w") as f:
        json.dump(all_episode_data, f)

    return all_episode_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint .pt file")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    evaluate(args.checkpoint, num_episodes=args.episodes, render=not args.no_render)
