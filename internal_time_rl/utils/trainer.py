import os
import json
import time
import numpy as np
import torch
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

@dataclass
class TrainingConfig:
    """Structured configuration for training sessions."""
    seed: int = 42
    device: str = "auto"
    total_timesteps: int = 500_000
    num_steps: int = 128
    num_envs: int = 8
    log_interval: int = 10
    save_interval: int = 100
    log_dir: str = "runs/default"
    
    # Nested configs (simplified for this refactor)
    env_name: str = "delayed_chain"
    agent_type: str = "standard"

class Trainer:
    """Professional Reinforcement Learning Trainer.
    
    Encapsulates the PPO training loop with support for internal time dynamics
     and extensive logging.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        agent: torch.nn.Module,
        ppo: Any,
        vec_env: Any,
        buffer: Any,
        device: torch.device
    ):
        self.config = config
        self.agent = agent
        self.ppo = ppo
        self.vec_env = vec_env
        self.buffer = buffer
        self.device = device
        
        self.log_dir = config.get("log_dir", "runs/default")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Performance tracking
        self.history = {
            "episode_rewards": [],
            "episode_lengths": [],
            "delta_tau_means": [],
            "delta_tau_stds": [],
            "policy_losses": [],
            "value_losses": [],
            "entropies": [],
            "time_losses": [],
        }
        self.completed_episodes = []
        
        # State tracking
        self.obs = None
        self.hidden = None
        self.ep_rewards = np.zeros(config.get("algorithm", {}).get("num_envs", 8))
        self.ep_lengths = np.zeros(config.get("algorithm", {}).get("num_envs", 8))

    def _initialize_run(self):
        """Prepare environment and agent for training."""
        num_envs = self.vec_env.num_envs
        raw_obs = self.vec_env.reset()
        self.obs = torch.tensor(raw_obs, dtype=torch.float32, device=self.device)
        self.hidden = self.agent.get_initial_hidden(num_envs, self.device)
        
        # Save formal config
        with open(os.path.join(self.log_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)

    def collect_rollouts(self, num_steps: int):
        """Collect a trajectory from the environment."""
        self.agent.eval()
        dt_collection = []
        
        with torch.no_grad():
            for _ in range(num_steps):
                # Standard action selection
                action, log_prob, _, value, hidden_new, delta_tau = (
                    self.agent.get_action_and_value(self.obs, self.hidden)
                )

                actions_np = action.cpu().numpy()
                next_obs_np, rewards, dones, infos = self.vec_env.step(actions_np)

                self.buffer.add(
                    self.obs,
                    action,
                    torch.tensor(rewards, dtype=torch.float32, device=self.device),
                    torch.tensor(dones, dtype=torch.float32, device=self.device),
                    log_prob,
                    value,
                    self.hidden,
                    delta_tau,
                )

                dt_collection.append(delta_tau.cpu().numpy())

                # Track episode stats
                self.ep_rewards += rewards
                self.ep_lengths += 1
                for i in range(self.vec_env.num_envs):
                    if dones[i]:
                        self.completed_episodes.append({
                            "reward": self.ep_rewards[i],
                            "length": self.ep_lengths[i]
                        })
                        self.ep_rewards[i] = 0
                        self.ep_lengths[i] = 0
                        # Reset hidden state for the environment that finished
                        hidden_new[i] = torch.zeros(self.agent.hidden_dim, device=self.device)

                self.obs = torch.tensor(next_obs_np, dtype=torch.float32, device=self.device)
                self.hidden = hidden_new

            # Bootstrap last value
            _, _, _, last_value, _, _ = self.agent.get_action_and_value(self.obs, self.hidden)
            self.buffer.compute_gae(last_value, self.ppo.gamma, self.ppo.gae_lambda)
            
        return dt_collection

    def train(self):
        """Main training execution."""
        self._initialize_run()
        
        algo_cfg = self.config.get("algorithm", {})
        num_steps = algo_cfg.get("num_steps", 128)
        num_envs = self.vec_env.num_envs
        total_timesteps = algo_cfg.get("total_timesteps", 500_000)
        num_updates = total_timesteps // (num_steps * num_envs)
        
        log_interval = self.config.get("logging", {}).get("log_interval", 10)
        save_interval = self.config.get("logging", {}).get("save_interval", 100)

        print(f"
Starting training: {total_timesteps:,} steps | {num_updates} updates")
        print("-" * 50)

        for update in range(1, num_updates + 1):
            start_time = time.time()
            self.buffer.reset()
            
            # 1. Collect
            dt_batch = self.collect_rollouts(num_steps)
            
            # 2. Update
            self.agent.train()
            metrics = self.ppo.update(self.buffer)
            
            # 3. Log & Save
            self._update_history(metrics, dt_batch)
            
            if update % log_interval == 0:
                self._report_progress(update, num_updates, num_steps * num_envs, metrics, dt_batch, time.time() - start_time)
                
            if update % save_interval == 0:
                self.save_checkpoint(update)

        self.save_checkpoint("final")
        self._finalize_results()

    def _update_history(self, metrics, dt_batch):
        dt_arr = np.concatenate(dt_batch)
        self.history["delta_tau_means"].append(float(dt_arr.mean()))
        self.history["delta_tau_stds"].append(float(dt_arr.std()))
        for k in ["policy_loss", "value_loss", "entropy", "time_loss"]:
            if k in metrics:
                self.history[f"{k}es" if not k.endswith('y') else "entropies"].append(metrics[k])
        
        if self.completed_episodes:
            recent = self.completed_episodes[-20:]
            self.history["episode_rewards"].append(float(np.mean([e["reward"] for e in recent])))
            self.history["episode_lengths"].append(float(np.mean([e["length"] for e in recent])))

    def _report_progress(self, update, total_updates, step_increment, metrics, dt_batch, duration):
        ts = update * step_increment
        dt_arr = np.concatenate(dt_batch)
        fps = int(step_increment / duration)
        
        parts = [f"Update {update}/{total_updates} | Steps {ts:,} | FPS {fps}"]
        if self.completed_episodes:
            recent = self.completed_episodes[-20:]
            parts.append(f"Rew {np.mean([e['reward'] for e in recent]):.2f}")
        parts.append(f"dt {dt_arr.mean():.3f}")
        parts.append(f"Ent {metrics.get('entropy', 0):.3f}")
        print(" | ".join(parts))

    def save_checkpoint(self, tag):
        path = os.path.join(self.log_dir, f"checkpoint_{tag}.pt")
        torch.save({
            "agent": self.agent.state_dict(),
            "ppo": self.ppo.optimizer.state_dict(),
            "config": self.config,
        }, path)

    def _finalize_results(self):
        with open(os.path.join(self.log_dir, "history.json"), "w") as f:
            json.dump(self.history, f)
        print(f"
Training complete. Results saved to: {self.log_dir}")
