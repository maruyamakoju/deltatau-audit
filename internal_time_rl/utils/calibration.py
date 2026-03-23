"""
Meta-Time Calibrator (DeepMind-style Online Adaptation).

Dynamically adjusts the agent's internal time scale (delta_tau bias)
to minimize TD-error bias in novel environments.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List

class MetaTimeCalibrator:
    """
    Online adapter that tunes the agent's time-perception bias.
    
    Threat Model: The agent is trained at speed 1.0 but deployed at speed 2.5.
    Observation: Value predictions will consistently overshoot/undershoot.
    Action: Shift the bias of the TimeModule until TD-error bias is zero.
    """
    def __init__(
        self, 
        agent: nn.Module, 
        lr: float = 0.05, 
        ema_alpha: float = 0.95,
        target_bias_param_name: str = "time_module.net.2.bias"
    ):
        self.agent = agent
        self.lr = lr
        self.ema_alpha = ema_alpha
        self.td_error_ema = 0.0
        
        # Access the bias of the output layer of TimeModule
        # (InternalTimeAgent has self.time_module.net[2] usually)
        self.bias_param = None
        for name, param in agent.named_parameters():
            if "time_module" in name and "bias" in name:
                # Target the last bias layer
                self.bias_param = param
        
        if self.bias_param is None:
            print("Warning: MetaTimeCalibrator could not find time_module bias.")

    def step_adaptation(self, reward: float, value: float, next_value: float, gamma: float, dt: float):
        """
        Update bias based on a single transition.
        Standard TD-error: delta = r + gamma^dt * V(s') - V(s)
        (Assuming internal time dt is used for discounting)
        """
        if self.bias_param is None: return
        
        # Discounting depends on subjective time
        discount = gamma ** dt
        td_target = reward + discount * next_value
        td_error = td_target - value
        
        # Update EMA of TD error
        self.td_error_ema = self.ema_alpha * self.td_error_ema + (1 - self.ema_alpha) * td_error
        
        # If TD error is positive (underestimating), we might need to think slower (smaller dt)?
        # Or faster? Depends on the sign of d(TD)/d(dt).
        # Heuristic: if we are underestimating, we are expecting reward TOO LATE. 
        # Making dt larger makes gamma^dt smaller, decreasing next_value contribution.
        
        # Simple gradient-free adaptation:
        # If EMA > 0, decrease bias (think slower). If EMA < 0, increase bias (think faster).
        # (Magnitude relative to EMA)
        with torch.no_grad():
            adjustment = -self.lr * np.sign(self.td_error_ema) * abs(self.td_error_ema)
            self.bias_param.add_(adjustment)
            
        return self.td_error_ema
