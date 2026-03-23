"""
Ray-based Distributed Auditor.

Scales the deltatau-audit framework to clusters of thousands of nodes.
Used for massive architecture searches or foundation model evaluation.
"""

import ray
from typing import List, Dict, Any, Callable
import gymnasium as gym

@ray.remote
def run_remote_audit(
    adapter_factory: Callable[[], Any],
    env_factory: Callable[[], Any],
    n_episodes: int = 10,
    seed: int = 42
):
    """Ray task to run an audit on a remote worker."""
    from .auditor import run_full_audit
    
    # Re-create adapter on the worker to avoid serialization issues with large models
    adapter = adapter_factory()
    
    result = run_full_audit(
        adapter,
        env_factory,
        n_episodes=n_episodes,
        verbose=False,
        seed=seed
    )
    return result

class RayClusterAuditor:
    """Orchestrates distributed audits across a Ray cluster."""
    def __init__(self):
        if not ray.is_initialized():
            ray.init()

    def mass_audit(
        self, 
        jobs: List[Dict[str, Any]], 
        episodes_per_job: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Runs multiple audit jobs in parallel.
        jobs: list of {adapter_factory, env_factory, name}
        """
        print(f"📡 Ray Cluster Auditor: Dispatching {len(jobs)} jobs...")
        
        futures = []
        for i, job in enumerate(jobs):
            futures.append(
                run_remote_audit.remote(
                    job['adapter_factory'],
                    job['env_factory'],
                    n_episodes=episodes_per_job,
                    seed=42 + i
                )
            )
            
        results = ray.get(futures)
        print(f"   Success: {len(results)} jobs completed on the cluster.")
        return results
