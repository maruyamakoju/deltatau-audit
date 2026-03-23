"""
Experiment: Auditing a JAX/Equinox Internal Time Agent.

Demonstrates the 'DeepMind Tech Stack' level up:
- JIT-compiled agents for high-speed simulation.
- Pure-functional state management.
- Seamless integration with the deltatau_audit framework via JaxAdapter.
"""

import jax
import jax.numpy as jnp
import gymnasium as gym
import numpy as np
from internal_time_rl.jax_models import JaxInternalTimeAgent, JaxAdapter
from deltatau_audit.auditor import run_full_audit
from deltatau_audit.report import generate_report

def run_jax_experiment():
    print("Initializing JAX/Equinox Agent (DeepMind Stack)...")
    
    key = jax.random.PRNGKey(42)
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    hidden_dim = 64
    
    # Create the JAX agent
    agent = JaxInternalTimeAgent(obs_dim, act_dim, hidden_dim, key)
    
    # Wrap in our new JAX adapter
    adapter = JaxAdapter(agent)
    
    print(f"Running Full 2-Axis Audit on JAX Agent...")
    # JIT the agent call inside the adapter for performance (optional here but good for real runs)
    
    result = run_full_audit(
        adapter,
        lambda: gym.make(env_name),
        speeds=[1, 2, 3, 5],
        n_episodes=10,
        verbose=True,
        seed=42
    )
    
    report_dir = "jax_audit_report"
    generate_report(result, report_dir, title="JAX/Equinox Agent Audit")
    
    print(f"\nAudit complete. JAX-based results saved to {report_dir}/")
    print(f"Deployment Rating: {result['summary']['deployment_rating']}")
    print(f"Quadrant: {result['summary']['quadrant']}")

if __name__ == "__main__":
    run_jax_experiment()
