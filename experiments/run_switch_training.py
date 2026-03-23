"""Train agents with mid-episode speed switching.

This trains agents on episodes where speed changes mid-episode,
enabling DYNAMIC Δτ adaptation (not just per-episode estimation).

Then runs the switching experiment to produce the "killer figure".

Usage:
    python run_switch_training.py
"""

import json
import os

import numpy as np
import torch

from internal_time_rl.models.policy import InternalTimeAgent
from internal_time_rl.models.baselines import SkipRNNAgent
from internal_time_rl.algorithms.ppo_time_v2 import PPOTimeV2, RolloutBufferV2
from internal_time_rl.envs.variable_frequency import VariableFrequencyChainEnv


class SwitchingChainEnv(VariableFrequencyChainEnv):
    """Chain env that randomly switches speed mid-episode during training."""

    def __init__(self, switch_prob=0.5, **kwargs):
        self._switch_prob = switch_prob
        super().__init__(**kwargs)

    def reset(self, seed=None, options=None):
        result = super().reset(seed=seed, options=options)

        # Randomly decide: constant or switching episode
        if self.np_random.random() < self._switch_prob:
            self.speed_schedule = "switch"
            speeds_pool = [1, 2, 3, 5, 8]
            s1, s2 = self.np_random.choice(speeds_pool, size=2, replace=False)
            self.switch_speeds = (int(s1), int(s2))
            self.switch_step = int(self.np_random.integers(
                self.max_agent_steps * 2 // 10,
                self.max_agent_steps * 6 // 10
            ))
            self.current_speed = int(s1)
        else:
            self.speed_schedule = "constant"

        return result


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
                info["speed"] = env.current_speed
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


def make_switching_env(switch_prob=0.5):
    return SwitchingChainEnv(
        switch_prob=switch_prob,
        chain_length=20,
        delay=10,
        max_agent_steps=100,
        train_speeds=(1, 2, 3),
        speed_in_obs=False,
        noise=0.05,
    )


def get_speed_from_infos(infos):
    return np.array([info.get("speed", 1) for info in infos], dtype=np.float32)


def train_agent(agent_type, agent, seed, total_timesteps, use_time_discount=False,
                device="cpu", log_dir="runs"):
    """Train a single agent with switching curriculum."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_envs = 16
    num_steps = 128

    vec_env = SyncVecEnv(
        [lambda: make_switching_env(switch_prob=0.5) for _ in range(num_envs)]
    )

    obs_dim = vec_env.envs[0].observation_space.shape[0]

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
    }

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
        json.dump(history, f)

    return agent, history


def evaluate_at_speed(agent, agent_type, speed, device="cpu", num_episodes=50):
    """Evaluate agent at a specific constant speed."""
    env = VariableFrequencyChainEnv(
        chain_length=20, delay=10, max_agent_steps=100,
        train_speeds=(speed,), speed_in_obs=False,
        fixed_speed=speed,
    )

    rewards = []
    delta_taus = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        hidden = agent.get_initial_hidden(1, device)
        total_reward = 0
        ep_dts = []
        done = False

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _, hidden, dt = agent.get_action_and_value(obs_t, hidden)
            ep_dts.append(dt.item())
            obs, reward, term, trunc, info = env.step(action.item())
            total_reward += reward
            done = term or trunc

        rewards.append(total_reward)
        delta_taus.append(np.mean(ep_dts))

    return {
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "dt_mean": float(np.mean(delta_taus)),
        "dt_std": float(np.std(delta_taus)),
    }


def main():
    output_base = "runs/switch_trained"
    os.makedirs(output_base, exist_ok=True)

    total_timesteps = 300_000
    test_speeds = [1, 2, 3, 5, 8]
    device = "cpu"

    # Get dims from sample env
    sample = make_switching_env()
    obs_dim = sample.observation_space.shape[0]
    act_dim = sample.action_space.n

    models = {
        "internal_time": ("internal_time", False),
        "internal_time_discount": ("internal_time", True),
        "baseline": ("baseline", False),
        "skip_rnn": ("skip_rnn", False),
    }

    results = {}

    for model_name, (agent_class, use_disc) in models.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name} with switching curriculum")
        print(f"{'='*60}")

        if agent_class == "baseline":
            agent = InternalTimeAgent(obs_dim, act_dim, use_internal_time=False).to(device)
        elif agent_class == "internal_time":
            agent = InternalTimeAgent(obs_dim, act_dim, use_internal_time=True).to(device)
        elif agent_class == "skip_rnn":
            agent = SkipRNNAgent(obs_dim, act_dim).to(device)

        model_dir = os.path.join(output_base, model_name, "seed_0")
        agent, hist = train_agent(
            model_name, agent, seed=42, total_timesteps=total_timesteps,
            use_time_discount=use_disc, device=device, log_dir=model_dir,
        )

        # Evaluate at each speed
        print(f"\nEvaluating {model_name}...")
        model_results = {"speed_rewards": {}, "speed_dts": {}}
        for speed in test_speeds:
            res = evaluate_at_speed(agent, model_name, speed, device)
            model_results["speed_rewards"][str(speed)] = {
                "mean": res["mean"], "std": res["std"]
            }
            model_results["speed_dts"][str(speed)] = {
                "mean": res["dt_mean"], "std": res["dt_std"]
            }
            print(f"  Speed {speed}: R={res['mean']:.3f}±{res['std']:.3f}, "
                  f"Δτ={res['dt_mean']:.3f}±{res['dt_std']:.3f}")

        results[model_name] = model_results

    # Save results
    with open(os.path.join(output_base, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_base}/results.json")


if __name__ == "__main__":
    main()
