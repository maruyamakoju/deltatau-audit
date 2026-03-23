"""Speed Generalization Experiment - Core paper result.

Tests whether internal time helps agents generalize across different
temporal granularities (action repeat / control frequency).

Protocol:
  Train on speeds {1, 2, 3}
  Test on speeds  {1, 2, 3, 5, 8}  (includes unseen)
  Metric: reward at each test speed + generalization gap

Models compared:
  1. Baseline GRU (no internal time)
  2. Internal Time GRU (Δτ for state update only)
  3. Internal Time GRU + Time Discount (Δτ for state + γ/λ)
  4. Skip-RNN (binary update gate, ACT-like)
  5. External Dt (ODE-RNN, dt from environment)

Usage:
    python run_speed_generalization.py              # Full experiment
    python run_speed_generalization.py --quick      # Quick test
    python run_speed_generalization.py --env chain  # Chain environment
    python run_speed_generalization.py --env timing # Interval timing
"""

import argparse
import copy
import json
import os

import numpy as np
import torch

from internal_time_rl.models.policy import InternalTimeAgent
from internal_time_rl.models.baselines import SkipRNNAgent, ExternalDtAgent
from internal_time_rl.algorithms.ppo_time_v2 import PPOTimeV2, RolloutBufferV2
from internal_time_rl.envs.variable_frequency import (
    VariableFrequencyChainEnv,
    IntervalTimingEnv,
)


class SyncVecEnv:
    """Synchronous vectorized environment."""

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)

    def reset(self):
        obs_list, info_list = [], []
        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            info_list.append(info)
        return np.stack(obs_list), info_list

    def step(self, actions):
        obs_list, rew_list, done_list, info_list = [], [], [], []
        for env, a in zip(self.envs, actions):
            obs, reward, term, trunc, info = env.step(int(a))
            done = term or trunc
            if done:
                info["terminal_reward"] = reward
                obs, _ = env.reset()
                info["speed"] = env.current_speed  # new episode speed
            obs_list.append(obs)
            rew_list.append(reward)
            done_list.append(float(done))
            info_list.append(info)
        return (
            np.stack(obs_list),
            np.array(rew_list, dtype=np.float32),
            np.array(done_list, dtype=np.float32),
            info_list,
        )


def make_env(env_type, train_speeds, fixed_speed=None, **kwargs):
    if env_type == "chain":
        return VariableFrequencyChainEnv(
            chain_length=kwargs.get("chain_length", 20),
            delay=kwargs.get("delay", 10),
            max_agent_steps=kwargs.get("max_steps", 100),
            train_speeds=tuple(train_speeds),
            speed_in_obs=kwargs.get("speed_in_obs", True),
            noise=kwargs.get("noise", 0.05),
            fixed_speed=fixed_speed,
        )
    elif env_type == "timing":
        return IntervalTimingEnv(
            min_target=kwargs.get("min_target", 5),
            max_target=kwargs.get("max_target", 30),
            max_steps=kwargs.get("max_steps", 50),
            tolerance=kwargs.get("tolerance", 2),
            train_speeds=tuple(train_speeds),
            speed_in_obs=kwargs.get("speed_in_obs", True),
            fixed_speed=fixed_speed,
        )
    else:
        raise ValueError(f"Unknown env type: {env_type}")


def create_agent(agent_type, obs_dim, act_dim, hidden_dim=128, latent_dim=64):
    """Create agent by type string."""
    if agent_type == "baseline":
        return InternalTimeAgent(
            obs_dim, act_dim, hidden_dim, latent_dim, use_internal_time=False
        )
    elif agent_type == "internal_time":
        return InternalTimeAgent(
            obs_dim, act_dim, hidden_dim, latent_dim, use_internal_time=True
        )
    elif agent_type == "skip_rnn":
        return SkipRNNAgent(obs_dim, act_dim, hidden_dim, latent_dim)
    elif agent_type == "external_dt":
        return ExternalDtAgent(obs_dim, act_dim, hidden_dim, latent_dim)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def get_speed_from_infos(infos):
    """Extract speed values from info dicts."""
    return np.array([info.get("speed", 1) for info in infos], dtype=np.float32)


def train_agent(
    agent_type,
    env_type,
    train_speeds,
    seed,
    total_timesteps,
    use_time_discount=False,
    device="cpu",
    log_dir="runs",
    env_kwargs=None,
):
    """Train a single agent and return history."""
    if env_kwargs is None:
        env_kwargs = {}

    torch.manual_seed(seed)
    np.random.seed(seed)

    num_envs = 16
    num_steps = 128

    vec_env = SyncVecEnv(
        [lambda: make_env(env_type, train_speeds, **env_kwargs) for _ in range(num_envs)]
    )

    sample_env = make_env(env_type, train_speeds, **env_kwargs)
    obs_dim = sample_env.observation_space.shape[0]
    act_dim = sample_env.action_space.n

    agent = create_agent(agent_type, obs_dim, act_dim).to(device)

    ppo = PPOTimeV2(
        agent=agent,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        time_smooth_coef=0.02,
        time_mean_coef=0.01,
        max_grad_norm=0.5,
        num_epochs=4,
        num_minibatches=4,
        use_time_discount=use_time_discount,
        use_smoothness=True,
    )

    buffer = RolloutBufferV2(num_steps, num_envs, obs_dim, agent.hidden_dim, device)
    num_updates = total_timesteps // (num_steps * num_envs)

    os.makedirs(log_dir, exist_ok=True)

    history = {
        "episode_rewards": [],
        "delta_tau_means": [],
        "delta_tau_stds": [],
        "speed_dt_pairs": [],  # (speed, delta_tau) pairs for analysis
    }

    obs, infos = vec_env.reset()
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    hidden = agent.get_initial_hidden(num_envs, device)

    ep_rewards = np.zeros(num_envs)
    completed = []

    for update in range(1, num_updates + 1):
        buffer.reset()
        dt_all = []
        speed_dt = []

        agent.eval()
        with torch.no_grad():
            for step in range(num_steps):
                # Set external dt for ODE-RNN baseline
                speeds = get_speed_from_infos(infos)
                if agent_type == "external_dt":
                    ext_dt = torch.tensor(
                        speeds / 3.0, dtype=torch.float32, device=device
                    ).unsqueeze(-1)
                    agent.set_external_dt(ext_dt)

                action, log_prob, _, value, hidden_new, delta_tau = (
                    agent.get_action_and_value(obs_t, hidden)
                )

                actions_np = action.cpu().numpy()
                next_obs, rewards, dones, infos = vec_env.step(actions_np)

                ext_dt_val = torch.tensor(
                    speeds / 3.0, dtype=torch.float32, device=device
                ).unsqueeze(-1)

                buffer.add(
                    obs_t, action,
                    torch.tensor(rewards, dtype=torch.float32, device=device),
                    torch.tensor(dones, dtype=torch.float32, device=device),
                    log_prob, value, hidden, delta_tau, ext_dt_val,
                )

                dt_np = delta_tau.cpu().numpy().flatten()
                dt_all.append(dt_np)
                for s, d in zip(speeds, dt_np):
                    speed_dt.append((float(s), float(d)))

                ep_rewards += rewards
                for i in range(num_envs):
                    if dones[i]:
                        completed.append(ep_rewards[i])
                        ep_rewards[i] = 0
                        hidden_new[i] = torch.zeros(agent.hidden_dim, device=device)

                obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
                hidden = hidden_new

            _, _, _, last_value, _, _ = agent.get_action_and_value(obs_t, hidden)
            buffer.compute_gae(last_value, ppo.gamma, ppo.gae_lambda, use_time_discount)

        agent.train()
        metrics = ppo.update(buffer)

        dt_arr = np.concatenate(dt_all)
        history["delta_tau_means"].append(float(dt_arr.mean()))
        history["delta_tau_stds"].append(float(dt_arr.std()))
        history["speed_dt_pairs"].extend(speed_dt[-200:])  # keep recent

        if completed:
            history["episode_rewards"].append(float(np.mean(completed[-20:])))

        if update % 20 == 0:
            ts = update * num_steps * num_envs
            r_str = f"R {np.mean(completed[-20:]):.2f}" if completed else "R --"
            print(
                f"  U {update}/{num_updates} T {ts:,} | {r_str} "
                f"| dt {dt_arr.mean():.3f}+/-{dt_arr.std():.3f} "
                f"| Ent {metrics['entropy']:.3f}"
            )

    # Save
    torch.save(
        {"agent": agent.state_dict(), "config": {"agent_type": agent_type}},
        os.path.join(log_dir, "final.pt"),
    )
    with open(os.path.join(log_dir, "history.json"), "w") as f:
        # speed_dt_pairs can be huge, save only last 5000
        save_hist = {k: v for k, v in history.items()}
        save_hist["speed_dt_pairs"] = history["speed_dt_pairs"][-5000:]
        json.dump(save_hist, f)

    return agent, history


def evaluate_at_speed(agent, agent_type, env_type, speed, device, num_episodes=50,
                      env_kwargs=None):
    """Evaluate a trained agent at a specific speed."""
    if env_kwargs is None:
        env_kwargs = {}

    env = make_env(env_type, [speed], fixed_speed=speed, **env_kwargs)
    rewards = []
    delta_taus = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        hidden = agent.get_initial_hidden(1, device)
        total_reward = 0
        done = False

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            if agent_type == "external_dt":
                ext_dt = torch.tensor(
                    [[speed / 3.0]], dtype=torch.float32, device=device
                )
                agent.set_external_dt(ext_dt)

            with torch.no_grad():
                action, _, _, _, hidden, dt = agent.get_action_and_value(obs_t, hidden)

            delta_taus.append(dt.item())
            obs, reward, term, trunc, info = env.step(action.item())
            total_reward += reward
            done = term or trunc

        rewards.append(total_reward)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_dt": float(np.mean(delta_taus)),
        "std_dt": float(np.std(delta_taus)),
    }


def run_experiment(
    env_type="chain",
    num_seeds=3,
    total_timesteps=300_000,
    train_speeds=(1, 2, 3),
    test_speeds=(1, 2, 3, 5, 8),
    output_dir="runs/speed_gen",
    device_str="auto",
    env_kwargs=None,
):
    """Run full speed generalization experiment."""
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if env_kwargs is None:
        env_kwargs = {}

    MODELS = {
        "baseline": {"agent_type": "baseline", "use_time_discount": False},
        "internal_time": {"agent_type": "internal_time", "use_time_discount": False},
        "internal_time_discount": {"agent_type": "internal_time", "use_time_discount": True},
        "skip_rnn": {"agent_type": "skip_rnn", "use_time_discount": False},
        "external_dt": {"agent_type": "external_dt", "use_time_discount": False},
    }

    all_results = {}

    for model_name, model_cfg in MODELS.items():
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")

        speed_results = {s: [] for s in test_speeds}
        speed_dt_results = {s: [] for s in test_speeds}

        for seed in range(num_seeds):
            print(f"\n--- Seed {seed} ---")
            log_dir = os.path.join(output_dir, model_name, f"seed_{seed}")

            agent, history = train_agent(
                agent_type=model_cfg["agent_type"],
                env_type=env_type,
                train_speeds=train_speeds,
                seed=seed * 1000 + 42,
                total_timesteps=total_timesteps,
                use_time_discount=model_cfg["use_time_discount"],
                device=device,
                log_dir=log_dir,
                env_kwargs=env_kwargs,
            )

            # Evaluate at each test speed
            agent.eval()
            print(f"  Evaluating at speeds: {test_speeds}")
            for speed in test_speeds:
                result = evaluate_at_speed(
                    agent, model_cfg["agent_type"], env_type, speed, device,
                    env_kwargs=env_kwargs,
                )
                speed_results[speed].append(result["mean_reward"])
                speed_dt_results[speed].append(result["mean_dt"])
                print(
                    f"    Speed {speed}: reward={result['mean_reward']:.3f} "
                    f"dt={result['mean_dt']:.3f}"
                )

        # Aggregate
        all_results[model_name] = {
            "speed_rewards": {
                str(s): {
                    "mean": float(np.mean(speed_results[s])),
                    "std": float(np.std(speed_results[s])),
                }
                for s in test_speeds
            },
            "speed_dts": {
                str(s): {
                    "mean": float(np.mean(speed_dt_results[s])),
                    "std": float(np.std(speed_dt_results[s])),
                }
                for s in test_speeds
            },
        }

    # Summary
    print(f"\n{'='*70}")
    print("SPEED GENERALIZATION RESULTS")
    print(f"{'='*70}")
    print(f"{'Model':30s}", end="")
    for s in test_speeds:
        marker = "" if s in train_speeds else "*"
        print(f"  Speed {s}{marker:3s}", end="")
    print("  | Gen Gap")

    for model_name, res in all_results.items():
        print(f"{model_name:30s}", end="")
        seen_rewards = []
        unseen_rewards = []
        for s in test_speeds:
            r = res["speed_rewards"][str(s)]
            print(f"  {r['mean']:+.3f}  ", end="")
            if s in train_speeds:
                seen_rewards.append(r["mean"])
            else:
                unseen_rewards.append(r["mean"])

        if seen_rewards and unseen_rewards:
            gap = np.mean(seen_rewards) - np.mean(unseen_rewards)
            print(f"  | {gap:+.3f}")
        else:
            print(f"  | --")

    print("\n* = unseen during training")

    # Delta tau vs speed
    print(f"\nDelta Tau by Speed:")
    print(f"{'Model':30s}", end="")
    for s in test_speeds:
        print(f"  Speed {s:3d}", end="")
    print()
    for model_name, res in all_results.items():
        print(f"{model_name:30s}", end="")
        for s in test_speeds:
            d = res["speed_dts"][str(s)]
            print(f"  {d['mean']:.3f}  ", end="")
        print()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_dir}/results.json")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["chain", "timing"], default="chain")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--output-dir", type=str, default="runs/speed_gen")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--speed-hidden", action="store_true",
                        help="Hide speed from observations")
    args = parser.parse_args()

    if args.quick:
        args.timesteps = min(args.timesteps, 50_000)
        args.seeds = min(args.seeds, 2)

    env_kwargs = {}
    if args.speed_hidden:
        env_kwargs["speed_in_obs"] = False

    run_experiment(
        env_type=args.env,
        num_seeds=args.seeds,
        total_timesteps=args.timesteps,
        output_dir=args.output_dir,
        device_str=args.device,
        env_kwargs=env_kwargs,
    )
