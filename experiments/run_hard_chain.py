"""Hard Chain Experiment - designed to show reward differentiation.

The standard chain task is too easy: all models reach similar reward.
This harder variant adds:
1. Flickering observations (30% masked) - tests memory under uncertainty
2. Shorter time budget (50 steps instead of 100) - time pressure
3. Longer chain (30 instead of 20) - more challenging navigation
4. Speed hidden - forces internal clock adaptation
5. Higher noise (0.1) - harder perception

Hypothesis: Internal time should help because:
- When obs is flickered, Δτ can slow down (preserve state)
- At higher speeds with tight time budget, Δτ can speed up
- This gives a computational advantage baseline GRU doesn't have

Usage:
    python run_hard_chain.py
    python run_hard_chain.py --quick
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
from internal_time_rl.envs.variable_frequency import VariableFrequencyChainEnv
from internal_time_rl.envs.flickering_env import FlickeringWrapper


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
                # Get speed from underlying env
                base = env.env if hasattr(env, 'env') else env
                info["speed"] = base.current_speed
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


def make_hard_chain(train_speeds, fixed_speed=None, flicker_prob=0.3):
    """Create hard chain: longer, flickering, tight time budget, hidden speed."""
    env = VariableFrequencyChainEnv(
        chain_length=30,
        delay=10,
        max_agent_steps=50,
        train_speeds=tuple(train_speeds),
        speed_in_obs=False,
        noise=0.1,
        fixed_speed=fixed_speed,
    )
    if flicker_prob > 0:
        env = FlickeringWrapper(env, flicker_prob=flicker_prob)
    return env


def get_speed_from_infos(infos):
    return np.array([info.get("speed", 1) for info in infos], dtype=np.float32)


def create_agent(agent_type, obs_dim, act_dim, hidden_dim=128, latent_dim=64):
    if agent_type == "baseline":
        return InternalTimeAgent(obs_dim, act_dim, hidden_dim, latent_dim, use_internal_time=False)
    elif agent_type == "internal_time":
        return InternalTimeAgent(obs_dim, act_dim, hidden_dim, latent_dim, use_internal_time=True)
    elif agent_type == "skip_rnn":
        return SkipRNNAgent(obs_dim, act_dim, hidden_dim, latent_dim)
    elif agent_type == "external_dt":
        return ExternalDtAgent(obs_dim, act_dim, hidden_dim, latent_dim)
    else:
        raise ValueError(f"Unknown: {agent_type}")


def train_agent(agent_type, train_speeds, seed, total_timesteps, use_time_discount=False,
                device="cpu", log_dir="runs", flicker_prob=0.3):
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_envs = 16
    num_steps = 128

    vec_env = SyncVecEnv(
        [lambda: make_hard_chain(train_speeds, flicker_prob=flicker_prob)
         for _ in range(num_envs)]
    )

    sample_env = make_hard_chain(train_speeds, flicker_prob=flicker_prob)
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
        entropy_coef=0.02,  # Higher entropy for harder task
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
    history = {"episode_rewards": [], "delta_tau_means": []}

    obs, infos = vec_env.reset()
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    hidden = agent.get_initial_hidden(num_envs, device)

    ep_rewards = np.zeros(num_envs)
    completed = []

    for update in range(1, num_updates + 1):
        buffer.reset()
        dt_all = []

        agent.eval()
        with torch.no_grad():
            for step in range(num_steps):
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

                dt_all.append(delta_tau.cpu().numpy().flatten())

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

        if completed:
            history["episode_rewards"].append(float(np.mean(completed[-20:])))

        if update % 20 == 0:
            ts = update * num_steps * num_envs
            r_str = f"R {np.mean(completed[-20:]):.3f}" if completed else "R --"
            print(
                f"  U {update}/{num_updates} T {ts:,} | {r_str} "
                f"| dt {dt_arr.mean():.3f}+/-{dt_arr.std():.3f}"
            )

    torch.save(
        {"agent": agent.state_dict(), "config": {"agent_type": agent_type}},
        os.path.join(log_dir, "final.pt"),
    )
    return agent, history


def evaluate_at_speed(agent, agent_type, speed, device, num_episodes=50, flicker_prob=0.3):
    env = make_hard_chain([speed], fixed_speed=speed, flicker_prob=flicker_prob)
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
                ext_dt = torch.tensor([[speed / 3.0]], dtype=torch.float32, device=device)
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


def run_experiment(num_seeds=3, total_timesteps=400_000, output_dir="runs/hard_chain",
                   device_str="auto", flicker_prob=0.3):
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    train_speeds = (1, 2, 3)
    test_speeds = (1, 2, 3, 5, 8)

    MODELS = {
        "baseline": {"agent_type": "baseline", "use_time_discount": False},
        "internal_time": {"agent_type": "internal_time", "use_time_discount": False},
        "internal_time_discount": {"agent_type": "internal_time", "use_time_discount": True},
        "skip_rnn": {"agent_type": "skip_rnn", "use_time_discount": False},
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
                train_speeds=train_speeds,
                seed=seed * 1000 + 42,
                total_timesteps=total_timesteps,
                use_time_discount=model_cfg["use_time_discount"],
                device=device,
                log_dir=log_dir,
                flicker_prob=flicker_prob,
            )

            agent.eval()
            print(f"  Evaluating at speeds: {test_speeds}")
            for speed in test_speeds:
                result = evaluate_at_speed(
                    agent, model_cfg["agent_type"], speed, device,
                    flicker_prob=flicker_prob,
                )
                speed_results[speed].append(result["mean_reward"])
                speed_dt_results[speed].append(result["mean_dt"])
                print(
                    f"    Speed {speed}: reward={result['mean_reward']:.3f} "
                    f"dt={result['mean_dt']:.3f}"
                )

        all_results[model_name] = {
            "speed_rewards": {
                str(s): {"mean": float(np.mean(speed_results[s])),
                          "std": float(np.std(speed_results[s]))}
                for s in test_speeds
            },
            "speed_dts": {
                str(s): {"mean": float(np.mean(speed_dt_results[s])),
                          "std": float(np.std(speed_dt_results[s]))}
                for s in test_speeds
            },
        }

    # Summary
    print(f"\n{'='*70}")
    print("HARD CHAIN RESULTS (flickering + time pressure + hidden speed)")
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
        gap = np.mean(seen_rewards) - np.mean(unseen_rewards)
        print(f"  | {gap:+.3f}")

    print(f"\nΔτ by Speed:")
    for model_name, res in all_results.items():
        print(f"{model_name:30s}", end="")
        for s in test_speeds:
            d = res["speed_dts"][str(s)]
            print(f"  {d['mean']:.3f}  ", end="")
        print()

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_dir}/results.json")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=400_000)
    parser.add_argument("--output-dir", type=str, default="runs/hard_chain")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--flicker", type=float, default=0.3)
    args = parser.parse_args()

    if args.quick:
        args.timesteps = 100_000
        args.seeds = 2

    run_experiment(
        num_seeds=args.seeds,
        total_timesteps=args.timesteps,
        output_dir=args.output_dir,
        device_str=args.device,
        flicker_prob=args.flicker,
    )
