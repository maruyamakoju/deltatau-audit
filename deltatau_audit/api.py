"""
The Atlas API: The Unified High-Level Interface for deltatau-audit.

Provides a 3-line entry point for the entire ecosystem.
Designed for maximum ease-of-use without sacrificing research depth.
"""

import os
from typing import Any, Optional, Dict, Union
import gymnasium as gym

from .auditor import run_full_audit
from .report import generate_report
from .report.certification import generate_safety_certificate
from .adapters.internal_time import InternalTimeAdapter
from .adapters.sb3 import SB3Adapter

class Atlas:
    """The central coordinator for the deltatau-audit ecosystem."""
    
    @staticmethod
    def load_agent(
        path: str, 
        agent_type: str = "sb3", 
        env_id: Optional[str] = None,
        algo: str = "ppo",
        device: str = "cpu"
    ) -> Any:
        """
        Loads any supported agent and returns its adapter.
        agent_type: 'sb3', 'internal_time', 'ltc', 'cleanrl'
        """
        if agent_type == "sb3":
            return SB3Adapter.from_path(path, algo=algo, device=device)
        elif agent_type in ["internal_time", "ltc", "baseline"]:
            # Need obs/act dim for these
            temp_env = gym.make(env_id) if env_id else None
            obs_dim = temp_env.observation_space.shape[0] if temp_env else 4
            if temp_env and isinstance(temp_env.action_space, gym.spaces.Discrete):
                act_dim = temp_env.action_space.n
            else:
                act_dim = temp_env.action_space.shape[0] if temp_env else 2
            if temp_env: temp_env.close()
            
            return InternalTimeAdapter.from_checkpoint(
                path, obs_dim, act_dim, agent_type=agent_type, device=device
            )
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")

    @staticmethod
    def certify(
        path_or_adapter: Any,
        env_id: str,
        out_dir: str = "certified_result",
        episodes: int = 50,
        agent_type: str = "sb3"
    ) -> Dict[str, Any]:
        """
        The 'Do-Everything' Command:
        1. Loads/Wraps the agent.
        2. Runs Full 2-Axis Audit.
        3. Generates HTML Report.
        4. Generates Formal Safety Certificate.
        5. Returns structured results.
        """
        if isinstance(path_or_adapter, str):
            adapter = Atlas.load_agent(path_or_adapter, agent_type=agent_type, env_id=env_id)
        else:
            adapter = path_or_adapter
            
        print(f"🚀 ATLAS: Initiating Full Certification Pipeline for {env_id}...")
        
        result = run_full_audit(
            adapter,
            lambda: gym.make(env_id),
            n_episodes=episodes,
            verbose=True
        )
        
        # Add manifest info for certificate
        result["manifest"] = {
            "title": f"Atlas Certification: {env_id}",
            "env": env_id,
            "agent_class": type(adapter).__name__
        }
        
        os.makedirs(out_dir, exist_ok=True)
        generate_report(result, out_dir)
        
        cert_path = os.path.join(out_dir, "certificate.html")
        status, reg_id = generate_safety_certificate(result, cert_path)
        
        print(f"\n✅ ATLAS: Certification Complete.")
        print(f"   Status:      {status}")
        print(f"   Registry ID: DT-{reg_id}")
        print(f"   Artifacts:   {out_dir}/")
        
        return result

    @staticmethod
    def fix(
        path: str,
        env_id: str,
        algo: str = "ppo",
        out_dir: str = "fixed_agent",
        agent_type: str = "sb3"
    ) -> Dict[str, Any]:
        """
        The 'Self-Healing' Command:
        1. Audits the agent to identify failure modes.
        2. Configures optimal Speed-Randomization based on diagnosis.
        3. Retrains the agent to patch the temporal vulnerability.
        4. Re-audits the fixed agent and certifies it.
        """
        from .fixer import fix_sb3_model
        
        print(f"🔧 ATLAS: Initiating Auto-Fix Pipeline for {env_id}...")
        
        # fix_sb3_model handles the audit -> retrain -> re-audit loop
        result = fix_sb3_model(
            model_path=path,
            algo=algo,
            env_id=env_id,
            output_dir=out_dir
        )
        
        print(f"\n✨ ATLAS: Self-Healing Complete. Certified model saved to {out_dir}/after/")
        return result

    @staticmethod
    def audit_hub(
        repo_id: str,
        env_id: str,
        algo: str = "ppo",
        out_dir: str = "hub_audit"
    ) -> Dict[str, Any]:
        """
        The 'Universal Benchmarking' Command:
        1. Downloads a model from HuggingFace Hub.
        2. Audits its temporal robustness.
        3. Prepares it for the Global Leaderboard.
        """
        print(f"🌍 ATLAS: Auditing Universal Model from Hub: {repo_id}...")
        
        from .adapters.sb3 import SB3Adapter
        adapter = SB3Adapter.from_hub(repo_id=repo_id, algo=algo)
        
        result = Atlas.certify(adapter, env_id=env_id, out_dir=out_dir)
        
        print(f"📊 ATLAS: Hub Audit Complete. Ranking data ready in {out_dir}/summary.json")
        return result

    @staticmethod
    def verify(
        path_or_adapter: Any,
        env_id: str,
        agent_type: str = "internal_time",
        n_steps: int = 100
    ) -> Dict[str, Any]:
        """
        The 'Formal Verification' Command:
        Calculates mathematical stability boundaries for timing variations.
        """
        if isinstance(path_or_adapter, str):
            adapter = Atlas.load_agent(path_or_adapter, agent_type=agent_type, env_id=env_id)
        else:
            adapter = path_or_adapter
            
        from .verification.formal import FormalTemporalVerifier
        verifier = FormalTemporalVerifier(adapter)
        
        print(f"🧐 ATLAS: Formally Verifying Agent for {env_id}...")
        result = verifier.verify_agent(lambda: gym.make(env_id), n_steps=n_steps)
        
        print(f"   Formal Safety Range: {result['average_safety_range']:.2f}x")
        print(f"   Status: {'PROVEN ROBUST' if result['is_formally_robust'] else 'UNPROVEN'}")
        
        return result

    @staticmethod
    def export(
        path_or_adapter: Any,
        output_path: str = "agent.onnx",
        agent_type: str = "internal_time",
        env_id: Optional[str] = None
    ):
        """
        The 'Edge Production' Command:
        Exports the agent to ONNX format for deployment on Jetson/TPU.
        """
        import torch
        if isinstance(path_or_adapter, str):
            adapter = Atlas.load_agent(path_or_adapter, agent_type=agent_type, env_id=env_id)
        else:
            adapter = path_or_adapter
            
        print(f"📦 ATLAS: Exporting Agent to ONNX ({output_path})...")
        
        # Determine dummy input shapes
        dummy_obs = torch.randn(1, 4) # Default for CartPole, should be env-specific
        dummy_hidden = adapter.reset_hidden(1)
        
        # We export the internal agent model
        model = adapter.agent
        model.eval()
        
        # ONNX doesn't like torch.distributions (Categorical)
        # We wrap the model to return logits instead
        class ONNXWrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
            def forward(self, obs, hidden):
                # We assume forward() returns (dist, value, hidden, dt)
                dist, value, hidden_new, dt = self.inner(obs, hidden)
                # Return logits instead of dist
                return dist.logits, value, hidden_new, dt
        
        export_model = ONNXWrapper(model)
        
        # Map forward pass for ONNX
        torch.onnx.export(
            export_model,
            (dummy_obs, dummy_hidden),
            output_path,
            input_names=['observation', 'hidden_in'],
            output_names=['logits', 'value', 'hidden_out', 'dt'],
            opset_version=12
        )
        
        print(f"   Export Complete. Use with TensorRT or ONNX Runtime for microsecond latency.")

# Convenience instance
atlas = Atlas()
