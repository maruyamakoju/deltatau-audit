"""
Experiment: Demonstrating the Apex Tier (Verification & Export).

1. Formally proves the agent's safety margin.
2. Exports the agent to ONNX for edge deployment.
"""

import os
from deltatau_audit.api import atlas

def run_apex_demo():
    print("🚀 Initiating Apex Tier (Formal Verification & Export)...")
    
    # We use our dummy agent for verification
    if not os.path.exists("dummy.pt"):
        import torch
        from internal_time_rl.models.policy import InternalTimeAgent
        agent = InternalTimeAgent(4, 2)
        torch.save({'agent': agent.state_dict()}, 'dummy.pt')
        
    env_id = "CartPole-v1"
    
    # 1. Formal Verification
    # (Calculates the Lipschitz-bound safety range)
    verification_result = atlas.verify("dummy.pt", env_id=env_id, n_steps=20)
    
    # 2. Edge Export
    # (Converts to high-performance ONNX format)
    atlas.export("dummy.pt", output_path="agent_v1.onnx", env_id=env_id)
    
    print("\n--- Apex Tier Results ---")
    print(f"Safety Proof: {verification_result['description']}")
    if os.path.exists("agent_v1.onnx"):
        print(f"Deployment Artifact: agent_v1.onnx (Ready for Jetson/Edge)")
    else:
        print("Export failed.")

if __name__ == "__main__":
    run_apex_demo()
