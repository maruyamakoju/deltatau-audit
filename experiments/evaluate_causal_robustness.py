"""Axis 10: Robustness Audit (Causal vs Subjective).

Evaluates the new CausalResolutionAgent against the previous 
SubjectiveResolutionAgent under severe timing jitter.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from deltatau_audit.auditors import RobustnessAuditor
from deltatau_audit._constants import ROBUSTNESS_SCENARIO_LABELS
from internal_time_rl.models.subjective_resolution import SubjectiveResolutionAgent
from internal_time_rl.models.causal_reasoning import CausalResolutionAgent
from deltatau_audit.adapters.subjective_resolution import SubjectiveResolutionAdapter
from deltatau_audit.adapters.causal_reasoning import CausalResolutionAdapter


import gymnasium as gym

# Register the custom environment
gym.envs.registration.register(
    id='VarFreqChain-v0',
    entry_point='internal_time_rl.envs.variable_frequency:VariableFrequencyChainEnv',
    kwargs={'chain_length': 20, 'train_speeds': (1, 2, 3), 'speed_in_obs': False}
)

def load_agent(agent_class, checkpoint_path, obs_dim=23, act_dim=2, device="cpu", **kwargs):
    agent = agent_class(obs_dim, act_dim, **kwargs).to(device)
    if os.path.exists(checkpoint_path):
        agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Using random weights.")
    agent.eval()
    return agent

def main():
    device = torch.device("cpu")
    print(f"Running Ultimate Robustness Audit on {device}...")
    
    # 1. Load Agents
    # Gen 2: Subjective Resolution
    agent_sub = load_agent(
        SubjectiveResolutionAgent, 
        "checkpoints/agent_dynamic_debug.pt", 
        max_ponder_base=4, tau_scale=2.0, device=device
    )
    adapter_sub = SubjectiveResolutionAdapter(agent_sub, device=device)
    
    # Gen 3: Causal Resolution
    agent_cau = load_agent(
        CausalResolutionAgent, 
        "checkpoints/agent_causal_axis10.pt", 
        max_ponder_base=4, tau_scale=2.0, causal_depth=3, device=device
    )
    adapter_cau = CausalResolutionAdapter(agent_cau, device=device)
    
    # 2. Run Auditor
    auditor = RobustnessAuditor(n_episodes=20, verbose=True, seed=42)
    env_id = "VarFreqChain-v0"  # Use the custom registered env
    scenarios = ["nominal", "jitter"] # Jitter introduces timing uncertainty
    
    print("\n--- Auditing Gen 2: Subjective Resolution Agent ---")
    report_sub = auditor.run(adapter_sub, env_id, scenarios=scenarios)
    
    print("\n--- Auditing Gen 3: Causal Resolution Agent ---")
    report_cau = auditor.run(adapter_cau, env_id, scenarios=scenarios)
    
    # 3. Compare Results
    nom_label = ROBUSTNESS_SCENARIO_LABELS.get("nominal", "nominal")
    jit_label = ROBUSTNESS_SCENARIO_LABELS.get("jitter", "jitter")
    
    def get_stage_metrics(report, stage_name):
        for stage in report.stages:
            if stage.stage_name == stage_name:
                return stage.metrics
        return None
        
    sub_nom = get_stage_metrics(report_sub, nom_label)
    sub_jit = get_stage_metrics(report_sub, jit_label)
    
    cau_nom = get_stage_metrics(report_cau, nom_label)
    cau_jit = get_stage_metrics(report_cau, jit_label)
    
    print("\n================ FINAL RESULTS ================")
    print(f"{'Metric':<25} | {'Gen 2 (Subjective)':<20} | {'Gen 3 (Causal)':<20}")
    print("-" * 70)
    
    r_sub_nom = sub_nom["mean_reward"].value if sub_nom else 0.0
    r_cau_nom = cau_nom["mean_reward"].value if cau_nom else 0.0
    print(f"{'Nominal Reward':<25} | {r_sub_nom:<20.2f} | {r_cau_nom:<20.2f}")
    
    r_sub_jit = sub_jit["mean_reward"].value if sub_jit else 0.0
    r_cau_jit = cau_jit["mean_reward"].value if cau_jit else 0.0
    print(f"{'Jitter Reward (Stress)':<25} | {r_sub_jit:<20.2f} | {r_cau_jit:<20.2f}")
    
    deg_sub = sub_jit["degradation"].value if sub_jit else 100.0
    deg_cau = cau_jit["degradation"].value if cau_jit else 100.0
    print(f"{'Degradation % (Lower is better)':<25} | {deg_sub:<20.2f}% | {deg_cau:<20.2f}%")
    
    print(f"{'Overall Reliability Score':<25} | {report_sub.reliability_score:<20.2f} | {report_cau.reliability_score:<20.2f}")
    print("===============================================")
    
    # 4. Generate Plot
    labels = ['Nominal', 'Jitter (Stress)']
    sub_scores = [r_sub_nom, r_sub_jit]
    cau_scores = [r_cau_nom, r_cau_jit]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, sub_scores, width, label='Gen 2 (Subjective Resolution)')
    rects2 = ax.bar(x + width/2, cau_scores, width, label='Gen 3 (Causal Resolution)')
    
    ax.set_ylabel('Mean Reward')
    ax.set_title('Robustness under Timing Uncertainty (Jitter)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    ax.bar_label(rects1, padding=3, fmt='%.1f')
    ax.bar_label(rects2, padding=3, fmt='%.1f')
    
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/causal_robustness_audit.png")
    print("\nPlot saved to results/causal_robustness_audit.png")
    
if __name__ == "__main__":
    main()
