"""
Temporal Interpretability Engine.

Analyzes the relationship between agent observations and internal time (delta_tau)
to generate human-readable insights about the agent's 'subjective experience'.
"""

import torch
import numpy as np
from typing import List, Dict, Any
import scipy.stats as stats

class TemporalInterpreter:
    """
    Infers the 'reasoning' behind delta_tau fluctuations.
    Matches observation features to subjective time shifts.
    """
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    def analyze_episode(self, observations: np.ndarray, delta_taus: np.ndarray) -> Dict[str, Any]:
        """
        Calculates correlation between specific features and internal time.
        """
        # Ensure observations is (T, D) and dts is (T,)
        T, D = observations.shape
        dts = delta_taus.flatten()
        
        correlations = {}
        for i, name in enumerate(self.feature_names):
            if i >= D: break
            feat = observations[:, i]
            # Spearman correlation for non-linear monotonic relationship
            corr, p_val = stats.spearmanr(feat, dts)
            correlations[name] = {"correlation": corr, "p_value": p_val}
            
        # Detect 'Events' - large shifts in dt
        dt_mean = np.mean(dts)
        dt_std = np.std(dts)
        events = []
        for t in range(1, T):
            diff = dts[t] - dts[t-1]
            if abs(diff) > 2 * dt_std:
                events.append({
                    "step": t,
                    "type": "acceleration" if diff > 0 else "deceleration",
                    "magnitude": float(diff),
                    "trigger_features": self._get_leading_features(observations[t-1:t+1], i=t)
                })
                
        return {
            "feature_correlations": correlations,
            "events": events,
            "summary": self._generate_summary(correlations, events)
        }

    def _get_leading_features(self, obs_pair, i):
        # Find which feature changed most at the event step
        diffs = np.abs(obs_pair[1] - obs_pair[0])
        top_idx = np.argmax(diffs)
        return self.feature_names[top_idx] if top_idx < len(self.feature_names) else "unknown"

    def _generate_summary(self, correlations, events) -> str:
        # Heuristic insight generation
        insights = []
        
        # 1. Identify primary driver
        sorted_feats = sorted(correlations.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)
        if sorted_feats and abs(sorted_feats[0][1]['correlation']) > 0.3:
            top_feat, data = sorted_feats[0]
            direction = "increases" if data['correlation'] > 0 else "decreases"
            insights.append(f"Internal clock {direction} as '{top_feat}' increases (corr={data['correlation']:.2f}).")
            
        # 2. Event summary
        if events:
            accels = sum(1 for e in events if e['type'] == 'acceleration')
            decels = len(events) - accels
            insights.append(f"Detected {len(events)} major temporal shifts (accel={accels}, decel={decels}).")
            
        if not insights:
            return "No strong temporal patterns detected. The agent's clock is stable."
            
        return " ".join(insights)
