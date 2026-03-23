"""Professional Training Entry Point for Internal Time RL.

Supports standard InternalTimeAgent and the new DeliberativeInternalTimeAgent 
(Level 3) with iterative state refinement.
"""

import argparse
import os
import torch
import numpy as np

from internal_time_rl.models.policy import InternalTimeAgent
from internal_time_rl.models.deliberative import DeliberativeInternalTimeAgent
from internal_time_rl.models.agent_v2 import SelfModelAgent
from internal_time_rl.algorithms.ppo_time import PPOTime, RolloutBuffer
from internal_time_rl.algorithms.ppo_self_model import PPOSelfModel
from internal_time_rl.envs.delayed_reward_chain import DelayedRewardChainEnv
from internal_time_rl.envs.tmaze import TMazeEnv
from internal_time_rl.envs.flickering_env import FlickeringWrapper, VariableSpeedWrapper
from internal_time_rl.envs.vec_env import SyncVectorEnv
from internal_time_rl.utils.trainer import Trainer


def make_env(env_config, seed=None):
    """Create environment with professional wrappers."""
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


def load_config(args):
    """Load and merge configurations from file and CLI."""
    config = {"model": {}, "env": {}, "algorithm": {}, "logging": {}}
    
    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config) as f:
            file_cfg = yaml.safe_load(f)
            for k in config:
                if k in file_cfg:
                    config[k].update(file_cfg[k])
            if "seed" in file_cfg: config["seed"] = file_cfg["seed"]
            if "device" in file_cfg: config["device"] = file_cfg["device"]

    # CLI Overrides
    config["seed"] = args.seed or config.get("seed", 42)
    config["device"] = args.device or config.get("device", "auto")
    config["log_dir"] = args.log_dir or config.get("log_dir", f"runs/{args.agent_type}_{args.env_name}")
    
    config["model"]["agent_type"] = args.agent_type
    config["model"]["use_internal_time"] = bool(args.use_internal_time)
    config["model"]["thinking_steps"] = args.thinking_steps
    
    config["env"]["name"] = args.env_name
    config["env"]["flickering"] = bool(args.flickering)
    
    config["algorithm"]["total_timesteps"] = args.total_timesteps or config["algorithm"].get("total_timesteps", 500_000)
    config["algorithm"]["num_envs"] = args.num_envs
    
    return config


def main():
    parser = argparse.ArgumentParser(description="Internal Time RL - DeepMind/Cambridge Grade Training")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--agent-type", type=str, default="deliberative", choices=["standard", "deliberative", "self_model"])
    parser.add_argument("--env-name", type=str, default="delayed_chain")
    parser.add_argument("--use-internal-time", type=int, default=1)
    parser.add_argument("--thinking-steps", type=int, default=3, help="Steps for deliberative reasoning")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--flickering", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    config = load_config(args)
    
    # Device setup
    if config["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["device"])
    
    # Seed
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Vector Env
    vec_env = SyncVectorEnv([
        lambda: make_env(config["env"], config["seed"] + i) 
        for i in range(config["algorithm"]["num_envs"])
    ])
    
    obs_dim = vec_env.observation_space.shape[0]
    act_dim = vec_env.action_space.n
    
    # Agent Factory
    agent_type = config["model"]["agent_type"]
    if agent_type == "deliberative":
        agent = DeliberativeInternalTimeAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=config["model"].get("hidden_dim", 128),
            thinking_steps=config["model"]["thinking_steps"]
        ).to(device)
    elif agent_type == "self_model":
        agent = SelfModelAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=config["model"].get("hidden_dim", 128)
        ).to(device)
    else:
        agent = InternalTimeAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=config["model"].get("hidden_dim", 128),
            use_internal_time=config["model"]["use_internal_time"]
        ).to(device)

    # Algorithm Factory
    if agent_type == "self_model":
        ppo = PPOSelfModel(agent=agent, **config["algorithm"])
    else:
        ppo = PPOTime(agent=agent, **config["algorithm"])

    # Buffer
    buffer = RolloutBuffer(
        config["algorithm"].get("num_steps", 128),
        config["algorithm"]["num_envs"],
        obs_dim, agent.hidden_dim, device
    )

    # Initialize Trainer (Professional Logic)
    trainer = Trainer(
        config=config,
        agent=agent,
        ppo=ppo,
        vec_env=vec_env,
        buffer=buffer,
        device=device
    )

    # Execute
    try:
        trainer.train()
    finally:
        vec_env.close()


if __name__ == "__main__":
    main()
