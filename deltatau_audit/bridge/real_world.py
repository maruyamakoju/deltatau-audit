"""
Sim-to-Real Bridge: High-fidelity deployment wrappers.

Models the physical gap between ideal simulation and real-world hardware:
- TransportDelay: Stochastic latency in observation/action pipelines.
- ClockDrift: Modeling asynchronous execution frequencies.
- Quantization: Sensor resolution and bit-depth limits.
- ActuatorLag: First-order response dynamics for real motors.
"""

import gymnasium as gym
import numpy as np
from typing import Optional, Dict, Any, Union
from collections import deque

class TransportDelayWrapper(gym.Wrapper):
    """
    Models stochastic latency (ms) converted to environment steps.
    Ideal for simulating network jitter or ROS2 message bus delays.
    """
    def __init__(
        self, 
        env: gym.Env, 
        mean_delay_ms: float = 20.0, 
        std_delay_ms: float = 5.0, 
        dt_ms: float = 10.0
    ):
        super().__init__(env)
        self.dt = dt_ms
        self.mean = mean_delay_ms
        self.std = std_delay_ms
        self.buffer_size = int((mean_delay_ms + 4 * std_delay_ms) / dt_ms) + 1
        self.obs_buffer = deque(maxlen=self.buffer_size)
        self.rng = np.random.default_rng()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.obs_buffer.clear()
        for _ in range(self.buffer_size):
            self.obs_buffer.append(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs_buffer.append(obs)
        
        # Sample current latency
        latency_ms = max(0, self.rng.normal(self.mean, self.std))
        delay_steps = int(round(latency_ms / self.dt))
        delay_steps = min(delay_steps, len(self.obs_buffer) - 1)
        
        # Retrieve the stale observation from history
        delayed_obs = self.obs_buffer[-(delay_steps + 1)]
        
        info["latency_ms"] = latency_ms
        info["delay_steps"] = delay_steps
        return delayed_obs, reward, terminated, truncated, info

class ActuatorLagWrapper(gym.Wrapper):
    """
    Models the first-order lag of real physical actuators.
    a_real = (1 - alpha) * a_real_prev + alpha * a_target
    """
    def __init__(self, env: gym.Env, alpha: float = 0.2):
        super().__init__(env)
        self.alpha = alpha
        self.prev_action = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        return obs, info

    def step(self, action):
        if self.prev_action is None:
            self.prev_action = np.zeros_like(action, dtype=np.float32)
            
        # First-order filter
        effective_action = (1.0 - self.alpha) * self.prev_action + self.alpha * action
        self.prev_action = effective_action
        
        # If discrete, we must map back to an integer
        if isinstance(self.action_space, gym.spaces.Discrete):
            # Probability-based or rounding? Rounding is simpler for lag.
            final_action = int(np.round(effective_action))
            # Ensure it's in range
            final_action = np.clip(final_action, 0, self.action_space.n - 1)
        else:
            final_action = effective_action
            
        return self.env.step(final_action)

class SensorQuantizationWrapper(gym.ObservationWrapper):
    """
    Simulates limited sensor resolution (e.g. 12-bit ADC, encoder ticks).
    """
    def __init__(self, env: gym.Env, levels: int = 1024):
        super().__init__(env)
        self.levels = levels

    def observation(self, obs):
        if isinstance(obs, np.ndarray) and np.issubdtype(obs.dtype, np.floating):
            # Map to [0, 1] based on space bounds if available, else just quantize
            low = self.observation_space.low if hasattr(self.observation_space, 'low') else -1.0
            high = self.observation_space.high if hasattr(self.observation_space, 'high') else 1.0
            
            # Clip and normalize
            normalized = np.clip((obs - low) / (high - low + 1e-8), 0, 1)
            # Quantize
            quantized = np.round(normalized * self.levels) / self.levels
            # Map back
            return quantized * (high - low) + low
        return obs
