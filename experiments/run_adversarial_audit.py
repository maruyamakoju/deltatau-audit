"""
Experiment: Temporal Adversarial Audit.

Finds the 'worst-case' timing jitter that minimizes agent performance.
Compares a baseline agent with a Time-Aware agent under adversarial pressure.
"""

import gymnasium as gym
import torch
from internal_time_rl.models.policy import InternalTimeAgent
from deltatau_audit.auditor import run_full_audit
from deltatau_audit.report import generate_report

def run_adversarial_experiment():
    print("Initializing Temporal Adversarial Audit...")
    
    env_name = "CartPole-v1"
    obs_dim = 4
    act_dim = 2
    
    # 1. Baseline Agent
    baseline = InternalTimeAgent(obs_dim, act_dim, use_internal_time=False)
    
    # 2. Time-Aware Agent
    robust = InternalTimeAgent(obs_dim, act_dim, use_internal_time=True)
    
    from deltatau_audit.adapters.internal_time import InternalTimeAdapter
    
    # We audit only the 'adversarial_jitter' scenario
    print("\n[1/2] Auditing Baseline Agent (Time-Blind)...")
    res_base = run_full_audit(
        InternalTimeAdapter(baseline),
        lambda: gym.make(env_name),
        robustness_scenarios=["adversarial_jitter"],
        n_episodes=10,
        verbose=False
    )
    
    print("\n[2/2] Auditing Robust Agent (Time-Aware)...")
    res_robust = run_full_audit(
        InternalTimeAdapter(robust),
        lambda: gym.make(env_name),
        robustness_scenarios=["adversarial_jitter"],
        n_episodes=10,
        verbose=False
    )
    
    print("\n--- Adversarial Impact Analysis ---")
    base_score = res_base['robustness']['per_scenario_scores']['adversarial_jitter']['return_ratio']
    robust_score = res_robust['robustness']['per_scenario_scores']['adversarial_jitter']['return_ratio']
    
    print(f"Baseline Score under Adversarial Jitter: {base_score:.2f}")
    print(f"Robust Agent Score under Adversarial Jitter: {robust_score:.2f}")
    
    improvement = (robust_score / base_score - 1) * 100 if base_score > 0 else 0
    print(f"Adversarial Resilience Improvement: {improvement:.1f}%")

    generate_report(res_robust, "adversarial_report", title="Adversarial Timing Audit")
    print("\nAudit complete. See adversarial_report/index.html")

if __name__ == "__main__":
    run_adversarial_experiment()
