"""Main training script for Internal Time RL.

Usage:
    # Train with internal time (default)
    python train.py

    # Train baseline (no internal time)
    python train.py --use-internal-time 0

    # Custom config
    python train.py --config configs/default.yaml --seed 123

    # With flickering observations (POMDP)
    python train.py --flickering 1 --flicker-prob 0.3
"""

import argparse
import json
import os

import numpy as np
import torch

from internal_time_rl.models.policy import InternalTimeAgent
from internal_time_rl.models.agent_v2 import SelfModelAgent
from internal_time_rl.algorithms.ppo_time import PPOTime, RolloutBuffer
from internal_time_rl.algorithms.ppo_self_model import PPOSelfModel
from internal_time_rl.envs.delayed_reward_chain import DelayedRewardChainEnv
from internal_time_rl.envs.tmaze import TMazeEnv
from internal_time_rl.envs.flickering_env import FlickeringWrapper, VariableSpeedWrapper


def make_env(env_config, seed=None):
    """Create environment from config dict."""
    name = env_config.get("name", "delayed_chain")

    if name == "delayed_chain":
        env = DelayedRewardChainEnv(
            length=env_config.get("length", 20),
            delay=env_config.get("delay", 10),
            max_steps=env_config.get("max_steps", 200),
            noise=env_config.get("noise", 0.0),
        )
    elif name == "tmaze":
        env = TMazeEnv(
            corridor_length=env_config.get("corridor_length", 10),
            delay=env_config.get("delay", 0),
            max_steps=env_config.get("max_steps", 100),
            noise=env_config.get("noise", 0.0),
        )
    else:
        raise ValueError(f"Unknown environment: {name}")

    if env_config.get("flickering", False):
        env = FlickeringWrapper(env, flicker_prob=env_config.get("flicker_prob", 0.3))
    if env_config.get("variable_speed", False):
        env = VariableSpeedWrapper(env)

    return env


class SyncVectorEnv:
    """Simple synchronous vectorized environment."""

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)

    def reset(self):
        obs_list = []
        for env in self.envs:
            obs, _ = env.reset()
            obs_list.append(obs)
        return np.stack(obs_list)

    def step(self, actions):
        obs_list, rew_list, done_list = [], [], []
        infos = []
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            if done:
                info["terminal_reward"] = reward
                info["terminal_obs"] = obs.copy()
                obs, _ = env.reset()
            obs_list.append(obs)
            rew_list.append(reward)
            done_list.append(float(done))
            infos.append(info)
        return (
            np.stack(obs_list),
            np.array(rew_list, dtype=np.float32),
            np.array(done_list, dtype=np.float32),
            infos,
        )


def load_config(path):
    """Load YAML config file."""
    if os.path.exists(path):
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


def train(config):
    """Main training loop."""
    # Device
    device_str = config.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"Device: {device}")

    # Seed
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Configs
    env_cfg = config.get("env", {})
    model_cfg = config.get("model", {})
    algo_cfg = config.get("algorithm", {})
    log_cfg = config.get("logging", {})

    # Environment
    num_envs = algo_cfg.get("num_envs", 8)
    vec_env = SyncVectorEnv(
        [lambda: make_env(env_cfg, seed) for _ in range(num_envs)]
    )

    # Dimensions
    sample_env = make_env(env_cfg)
    obs_dim = sample_env.observation_space.shape[0]
    act_dim = sample_env.action_space.n
    use_time = model_cfg.get("use_internal_time", True)

    agent_type = model_cfg.get("agent_type", "standard")
    print(f"Env: {env_cfg.get('name', 'delayed_chain')} | Obs: {obs_dim} | Act: {act_dim}")
    print(f"Internal Time: {use_time} | Agent: {agent_type}")

    # Agent
    if agent_type == "self_model":
        agent = SelfModelAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=model_cfg.get("hidden_dim", 128),
            latent_dim=model_cfg.get("latent_dim", 64),
            time_hidden_dim=model_cfg.get("time_hidden_dim", 32),
            bottleneck_dim=model_cfg.get("bottleneck_dim", 64),
            transition_type=model_cfg.get("transition_type", "gru"),
            time_init_bias=model_cfg.get("time_init_bias", 0.0),
        ).to(device)
    else:
        agent = InternalTimeAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=model_cfg.get("hidden_dim", 128),
            latent_dim=model_cfg.get("latent_dim", 64),
            time_hidden_dim=model_cfg.get("time_hidden_dim", 32),
            use_internal_time=use_time,
            transition_type=model_cfg.get("transition_type", "gru"),
            time_init_bias=model_cfg.get("time_init_bias", 0.0),
        ).to(device)

    num_params = sum(p.numel() for p in agent.parameters())
    print(f"Parameters: {num_params:,}")

    # PPO (use self-model variant if applicable)
    if agent_type == "self_model":
        ppo = PPOSelfModel(
            agent=agent,
            lr=algo_cfg.get("lr", 3e-4),
            gamma=algo_cfg.get("gamma", 0.99),
            gae_lambda=algo_cfg.get("gae_lambda", 0.95),
            clip_epsilon=algo_cfg.get("clip_epsilon", 0.2),
            value_coef=algo_cfg.get("value_coef", 0.5),
            entropy_coef=algo_cfg.get("entropy_coef", 0.01),
            time_var_coef=algo_cfg.get("time_var_coef", 0.005),
            time_mean_coef=algo_cfg.get("time_mean_coef", 0.01),
            self_model_coef=algo_cfg.get("self_model_coef", 0.1),
            max_grad_norm=algo_cfg.get("max_grad_norm", 0.5),
            num_epochs=algo_cfg.get("num_epochs", 4),
            num_minibatches=algo_cfg.get("num_minibatches", 4),
        )
    else:
        ppo = PPOTime(
            agent=agent,
            lr=algo_cfg.get("lr", 3e-4),
            gamma=algo_cfg.get("gamma", 0.99),
            gae_lambda=algo_cfg.get("gae_lambda", 0.95),
            clip_epsilon=algo_cfg.get("clip_epsilon", 0.2),
            value_coef=algo_cfg.get("value_coef", 0.5),
            entropy_coef=algo_cfg.get("entropy_coef", 0.01),
            time_var_coef=algo_cfg.get("time_var_coef", 0.005),
            time_mean_coef=algo_cfg.get("time_mean_coef", 0.01),
            max_grad_norm=algo_cfg.get("max_grad_norm", 0.5),
            num_epochs=algo_cfg.get("num_epochs", 4),
            num_minibatches=algo_cfg.get("num_minibatches", 4),
        )

    # Buffer
    num_steps = algo_cfg.get("num_steps", 128)
    buffer = RolloutBuffer(num_steps, num_envs, obs_dim, agent.hidden_dim, device)

    # Training schedule
    total_timesteps = algo_cfg.get("total_timesteps", 500_000)
    num_updates = total_timesteps // (num_steps * num_envs)
    log_interval = log_cfg.get("log_interval", 10)
    save_interval = log_cfg.get("save_interval", 100)

    # Logging
    log_dir = config.get("log_dir", "runs/default")
    os.makedirs(log_dir, exist_ok=True)

    history = {
        "episode_rewards": [],
        "episode_lengths": [],
        "delta_tau_means": [],
        "delta_tau_stds": [],
        "policy_losses": [],
        "value_losses": [],
        "entropies": [],
        "time_losses": [],
    }

    # Save config
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Initialize
    obs = vec_env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    hidden = agent.get_initial_hidden(num_envs, device)

    ep_rewards = np.zeros(num_envs)
    ep_lengths = np.zeros(num_envs)
    completed = []

    print(f"\nTraining: {total_timesteps:,} timesteps, {num_updates} updates")
    print("=" * 70)

    for update in range(1, num_updates + 1):
        buffer.reset()
        dt_collection = []

        # === Collect rollout ===
        agent.eval()
        with torch.no_grad():
            for step in range(num_steps):
                action, log_prob, _, value, hidden_new, delta_tau = (
                    agent.get_action_and_value(obs, hidden)
                )

                actions_np = action.cpu().numpy()
                next_obs, rewards, dones, infos = vec_env.step(actions_np)

                buffer.add(
                    obs,
                    action,
                    torch.tensor(rewards, dtype=torch.float32, device=device),
                    torch.tensor(dones, dtype=torch.float32, device=device),
                    log_prob,
                    value,
                    hidden,
                    delta_tau,
                )

                dt_collection.append(delta_tau.cpu().numpy())

                # Track episodes
                ep_rewards += rewards
                ep_lengths += 1
                for i in range(num_envs):
                    if dones[i]:
                        completed.append(
                            {"reward": ep_rewards[i], "length": ep_lengths[i]}
                        )
                        ep_rewards[i] = 0
                        ep_lengths[i] = 0
                        hidden_new[i] = torch.zeros(agent.hidden_dim, device=device)

                obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
                hidden = hidden_new

            # Bootstrap value for GAE
            _, _, _, last_value, _, _ = agent.get_action_and_value(obs, hidden)
            buffer.compute_gae(last_value, ppo.gamma, ppo.gae_lambda)

        # === PPO Update ===
        agent.train()
        metrics = ppo.update(buffer)

        # === Logging ===
        dt_arr = np.concatenate(dt_collection)
        history["delta_tau_means"].append(float(dt_arr.mean()))
        history["delta_tau_stds"].append(float(dt_arr.std()))
        history["policy_losses"].append(metrics["policy_loss"])
        history["value_losses"].append(metrics["value_loss"])
        history["entropies"].append(metrics["entropy"])
        history["time_losses"].append(metrics["time_loss"])

        if completed:
            recent = completed[-min(20, len(completed)) :]
            avg_r = np.mean([e["reward"] for e in recent])
            avg_l = np.mean([e["length"] for e in recent])
            history["episode_rewards"].append(float(avg_r))
            history["episode_lengths"].append(float(avg_l))

        if update % log_interval == 0:
            ts = update * num_steps * num_envs
            parts = [f"U {update}/{num_updates} | T {ts:,}"]
            if completed:
                recent = completed[-min(20, len(completed)) :]
                parts.append(f"R {np.mean([e['reward'] for e in recent]):.2f}")
                parts.append(f"L {np.mean([e['length'] for e in recent]):.1f}")
            parts.append(f"dt {dt_arr.mean():.3f}+/-{dt_arr.std():.3f}")
            parts.append(f"Ent {metrics['entropy']:.3f}")
            parts.append(f"KL {metrics['approx_kl']:.4f}")
            print(" | ".join(parts))

        if update % save_interval == 0:
            torch.save(
                {
                    "update": update,
                    "agent": agent.state_dict(),
                    "optimizer": ppo.optimizer.state_dict(),
                    "config": config,
                },
                os.path.join(log_dir, f"checkpoint_{update}.pt"),
            )

    # Save final results
    torch.save(
        {
            "update": num_updates,
            "agent": agent.state_dict(),
            "optimizer": ppo.optimizer.state_dict(),
            "config": config,
        },
        os.path.join(log_dir, "final.pt"),
    )

    with open(os.path.join(log_dir, "history.json"), "w") as f:
        json.dump(history, f)

    print(f"\nDone. Results saved to {log_dir}/")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Internal Time RL Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--use-internal-time", type=int, default=1)
    parser.add_argument("--transition-type", type=str, default="gru", choices=["gru", "ode"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--delay", type=int, default=None)
    parser.add_argument("--chain-length", type=int, default=None)
    parser.add_argument("--flickering", type=int, default=None)
    parser.add_argument("--flicker-prob", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    # Apply CLI overrides
    if "model" not in config:
        config["model"] = {}
    if "env" not in config:
        config["env"] = {}
    if "algorithm" not in config:
        config["algorithm"] = {}

    config["model"]["use_internal_time"] = bool(args.use_internal_time)
    config["model"]["transition_type"] = args.transition_type

    if args.seed is not None:
        config["seed"] = args.seed
    if args.log_dir is not None:
        config["log_dir"] = args.log_dir
    if args.total_timesteps is not None:
        config["algorithm"]["total_timesteps"] = args.total_timesteps
    if args.delay is not None:
        config["env"]["delay"] = args.delay
    if args.chain_length is not None:
        config["env"]["length"] = args.chain_length
    if args.flickering is not None:
        config["env"]["flickering"] = bool(args.flickering)
    if args.flicker_prob is not None:
        config["env"]["flicker_prob"] = args.flicker_prob
    if args.device is not None:
        config["device"] = args.device

    train(config)
