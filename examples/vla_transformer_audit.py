"""
Example: Auditing a Vision-Language-Action (VLA) Transformer model.
This script demonstrates how `deltatau_audit` handles high-dimensional, Dict-based
observation spaces (like pixels + language instructions + proprioception) and
autoregressive sequence models (like Decision Transformers).

This represents "Level Up 1: Architecture and Observation Space Scale-Up"
for real-world robotics and foundational models.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Tuple, Optional

from deltatau_audit.adapters.base import AgentAdapter
from deltatau_audit.auditor import run_full_audit
from deltatau_audit.report import generate_report

# ---------------------------------------------------------------------------
# 1. Mock VLA Environment (Dict Space)
# ---------------------------------------------------------------------------
class MockVLAEnv(gym.Env):
    """A mock Vision-Language-Action environment for a robotic manipulator."""
    def __init__(self):
        super().__init__()
        
        # Dict observation: images, text embedding, proprioception
        self.observation_space = spaces.Dict({
            "pixels": spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8),
            "instruction": spaces.Box(low=-1.0, high=1.0, shape=(512,), dtype=np.float32),
            "proprio": spaces.Box(low=-5.0, high=5.0, shape=(7,), dtype=np.float32)
        })
        
        # Continuous action space for 7-DoF arm
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        return self.observation_space.sample(), {}

    def step(self, action):
        self.step_count += 1
        obs = self.observation_space.sample()
        # Simulated reward
        reward = 1.0 - 0.1 * np.sum(np.abs(action))
        terminated = self.step_count >= 50
        truncated = False
        return obs, reward, terminated, truncated, {}


# ---------------------------------------------------------------------------
# 2. Mock VLA Transformer Model
# ---------------------------------------------------------------------------
class MockVLATransformer(nn.Module):
    """A mock autoregressive Decision Transformer for Dict observations."""
    def __init__(self, context_len: int = 10):
        super().__init__()
        self.context_len = context_len
        # In a real model, we would have CNN encoders, LLM embeddings, etc.
        # Here we just flatten to simulate the representation.
        obs_flat_dim = (3 * 64 * 64) + 512 + 7
        self.d_model = 256
        
        self.proj = nn.Linear(obs_flat_dim, self.d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, batch_first=True),
            num_layers=2
        )
        self.action_head = nn.Linear(self.d_model, 7)
        self.value_head = nn.Linear(self.d_model, 1)

    def forward(self, obs_seq: torch.Tensor):
        # obs_seq: (B, L, D)
        x = self.proj(obs_seq)
        out = self.transformer(x)
        # Take the last token prediction
        last_token = out[:, -1, :]
        action = torch.tanh(self.action_head(last_token))
        value = self.value_head(last_token).squeeze(-1)
        return action, value


# ---------------------------------------------------------------------------
# 3. Transformer Adapter for the Auditor
# ---------------------------------------------------------------------------
class VLAAdapter(AgentAdapter):
    """Adapter bridging the Transformer sequence requirement to step-by-step auditing."""
    def __init__(self, model: MockVLATransformer, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.context_len = model.context_len

    def reset_hidden(self, batch: int = 1, device: str = "cpu") -> Any:
        # Hidden state is a list of previous flattened observations (the context window)
        # Returns a list of lists: one list per batch element
        return [[] for _ in range(batch)]

    def _flatten_dict_obs(self, obs_dict: dict) -> torch.Tensor:
        """Flatten dict observation to a single 1D tensor."""
        pixels = torch.tensor(obs_dict["pixels"], dtype=torch.float32).flatten() / 255.0
        instruction = torch.tensor(obs_dict["instruction"], dtype=torch.float32)
        proprio = torch.tensor(obs_dict["proprio"], dtype=torch.float32)
        return torch.cat([pixels, instruction, proprio])

    def act(self, obs: Any, hidden: Any) -> Tuple[np.ndarray, float, Any, Optional[float]]:
        # obs might be a single dict or a batched dict (vectorized env)
        # For simplicity, assuming a single environment (batch size 1) in this example
        if not isinstance(obs, dict) and isinstance(obs, np.ndarray):
            # Fallback if wrapped by VecEnv converting dict to array somehow,
            # but ideally we expect the raw dict.
            pass
            
        flat_obs = self._flatten_dict_obs(obs)
        
        # Update hidden state (context window) for the first batch element
        history = hidden[0]
        history.append(flat_obs)
        if len(history) > self.context_len:
            history.pop(0)
            
        # Prepare sequence tensor: (1, L, D)
        seq_tensor = torch.stack(history).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, value = self.model(seq_tensor)
            
        return action[0].cpu().numpy(), value[0].item(), [history], None


# ---------------------------------------------------------------------------
# 4. Run the Audit
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Initializing Mock VLA Model and Environment...")
    model = MockVLATransformer()
    adapter = VLAAdapter(model)
    
    print("Running Time Robustness Audit on Transformer/Dict Space...")
    # We use small numbers for the mock run
    result = run_full_audit(
        adapter=adapter,
        env_factory=MockVLAEnv,
        speeds=[1, 2, 3],
        n_episodes=3,
        n_workers=1,
        seed=42
    )
    
    report_dir = "vla_audit_report"
    generate_report(result, report_dir, title="VLA Transformer Audit")
    print(f"Audit complete. See {report_dir}/index.html")
    print(f"Deployment Score: {result['summary']['deployment_score']:.2f}")
