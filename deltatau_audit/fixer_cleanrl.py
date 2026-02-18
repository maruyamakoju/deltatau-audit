"""Fix timing-fragile CleanRL agents via speed-randomized retraining.

Uses a self-contained PPO loop (no SB3 dependency) that works with any
CleanRL agent implementing the standard interface:
    agent.get_action_and_value(obs) -> (action, logprob, entropy, value)

Complete pipeline: audit original -> retrain -> re-audit -> compare.

Usage (Python API):
    from deltatau_audit.fixer_cleanrl import fix_cleanrl_agent
    result = fix_cleanrl_agent(
        agent_class=Agent,
        agent_kwargs={"obs_dim": 4, "act_dim": 2},
        env_id="CartPole-v1",
        output_dir="fix_output/",
    )

Usage (CLI):
    deltatau-audit fix-cleanrl \\
        --checkpoint agent.pt \\
        --agent-module ppo_cartpole.py \\
        --agent-class Agent \\
        --agent-kwargs obs_dim=4,act_dim=2 \\
        --env CartPole-v1
"""

import os
import time
from typing import Any, Dict, Optional, Type

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def _ppo_train_cleanrl(
    agent: nn.Module,
    env_id: str,
    total_steps: int,
    speed_min: int = 1,
    speed_max: int = 5,
    device: str = "cpu",
    verbose: bool = True,
    lr: float = 2.5e-4,
    num_steps: int = 128,
    num_minibatches: int = 4,
    update_epochs: int = 4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
) -> nn.Module:
    """Train a CleanRL agent with speed-randomized PPO.

    Works with any agent implementing:
        agent.get_action_and_value(obs) -> (action, logprob, entropy, value)

    Uses JitterWrapper for speed randomization during training.

    Returns the trained agent (same object, mutated in-place).
    """
    from .wrappers.speed import JitterWrapper

    base_speed = (speed_min + speed_max) // 2
    jitter = max(base_speed - speed_min, 1)

    # Create env
    base_env = gym.make(env_id)
    env = JitterWrapper(base_env, base_speed=base_speed, jitter=jitter)

    obs_dim = env.observation_space.shape[0]
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    # Detect action space (discrete vs continuous) from a test forward pass
    agent.eval()
    with torch.no_grad():
        _test_obs = torch.zeros(1, obs_dim, device=device)
        _test_action, _, _, _ = agent.get_action_and_value(_test_obs)
    _is_discrete = _test_action.dtype in (torch.int32, torch.int64, torch.bool)
    _act_dim = 1 if _is_discrete else int(_test_action.numel())

    # Rollout buffers
    obs_buf = torch.zeros(num_steps, obs_dim, device=device)
    if _is_discrete:
        act_buf = torch.zeros(num_steps, dtype=torch.long, device=device)
    else:
        act_buf = torch.zeros(num_steps, _act_dim, dtype=torch.float32,
                              device=device)
    logp_buf = torch.zeros(num_steps, device=device)
    rew_buf = torch.zeros(num_steps, device=device)
    done_buf = torch.zeros(num_steps, device=device)
    val_buf = torch.zeros(num_steps, device=device)

    obs, _ = env.reset()
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    done = False
    global_step = 0
    ep_returns = []
    ep_return = 0.0
    n_updates = 0

    agent.train()

    while global_step < total_steps:
        # Collect rollout
        for step in range(num_steps):
            obs_buf[step] = obs_t
            done_buf[step] = float(done)

            with torch.no_grad():
                action, logp, _, value = agent.get_action_and_value(
                    obs_t.unsqueeze(0))
            if _is_discrete:
                act_buf[step] = action.squeeze()
                action_for_env = int(action.squeeze().item())
            else:
                act_buf[step] = action.reshape(_act_dim)
                action_for_env = action.squeeze(0).cpu().numpy()
            logp_buf[step] = logp.squeeze()
            val_buf[step] = value.squeeze()

            obs, reward, terminated, truncated, _ = env.step(action_for_env)
            done = terminated or truncated
            rew_buf[step] = reward
            ep_return += reward
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            global_step += 1

            if done:
                ep_returns.append(ep_return)
                ep_return = 0.0
                obs, _ = env.reset()
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
                done = False

        # GAE advantages
        with torch.no_grad():
            next_val = agent.get_action_and_value(
                obs_t.unsqueeze(0))[3].squeeze()

        advantages = torch.zeros(num_steps, device=device)
        last_gae = 0.0
        for t in reversed(range(num_steps)):
            nv = next_val if t == num_steps - 1 else val_buf[t + 1]
            nd = 0.0 if t == num_steps - 1 else done_buf[t + 1]
            delta = rew_buf[t] + gamma * nv * (1 - nd) - val_buf[t]
            last_gae = delta + gamma * gae_lambda * (1 - nd) * last_gae
            advantages[t] = last_gae
        returns = advantages + val_buf

        # PPO update
        b_inds = np.arange(num_steps)
        mb_size = num_steps // num_minibatches
        for _ in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, num_steps, mb_size):
                idx = b_inds[start: start + mb_size]
                _, new_logp, entropy, new_val = agent.get_action_and_value(
                    obs_buf[idx], act_buf[idx])
                ratio = (new_logp - logp_buf[idx]).exp()
                adv = advantages[idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg_loss = torch.max(
                    -adv * ratio,
                    -adv * ratio.clamp(1 - clip_coef, 1 + clip_coef),
                ).mean()
                vf_loss = 0.5 * (new_val.squeeze() - returns[idx]).pow(2).mean()
                loss = pg_loss - ent_coef * entropy.mean() + vf_coef * vf_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

        n_updates += 1
        if verbose and n_updates % 10 == 0:
            if ep_returns:
                recent = ep_returns[-20:]
                print(f"    step={global_step:7d}  "
                      f"mean_return={np.mean(recent):.1f}  "
                      f"(last {len(recent)} eps)")

    env.close()
    agent.eval()
    return agent


def fix_cleanrl_agent(
    agent_class: Type[nn.Module],
    agent_kwargs: Optional[Dict] = None,
    env_id: str = "CartPole-v1",
    output_dir: str = "fix_output",
    checkpoint_path: Optional[str] = None,
    timesteps: Optional[int] = None,
    speed_min: int = 1,
    speed_max: int = 5,
    n_audit_episodes: int = 30,
    audit_speeds: Optional[list] = None,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict:
    """Fix a timing-fragile CleanRL agent via speed-randomized PPO retraining.

    Pipeline:
        1. Audit the original agent (Before) — if checkpoint_path provided
        2. Train a fresh agent with speed randomization (JitterWrapper)
        3. Audit the fixed agent (After)
        4. Generate Before/After comparison report

    Args:
        agent_class: CleanRL Agent class (must implement get_action_and_value).
        agent_kwargs: Kwargs for agent_class(**agent_kwargs).
        env_id: Gymnasium environment ID.
        output_dir: Where to write reports and fixed checkpoint.
        checkpoint_path: Path to original .pt checkpoint for Before audit.
                         If None, no Before audit is run.
        timesteps: PPO training timesteps (None = auto-estimate).
        speed_min: Min speed during training (default: 1).
        speed_max: Max speed during training (default: 5).
        n_audit_episodes: Episodes per audit condition.
        audit_speeds: Speed multipliers for audit (default: [1, 2, 3, 5, 8]).
        device: Device string.
        verbose: Print progress.

    Returns:
        Dict with before/after results, fixed checkpoint path, skipped flag.
    """
    from .auditor import run_full_audit
    from .report import generate_report
    from .diff import generate_comparison
    from .ci import write_ci_summary
    from .adapters.cleanrl import CleanRLAdapter
    from . import __version__

    if audit_speeds is None:
        audit_speeds = [1, 2, 3, 5, 8]
    if agent_kwargs is None:
        agent_kwargs = {}

    if timesteps is None:
        timesteps = _estimate_timesteps_cleanrl(env_id)

    base_speed = (speed_min + speed_max) // 2
    jitter = max(base_speed - speed_min, 1)
    effective_min = max(1, base_speed - jitter)
    effective_max = base_speed + jitter

    os.makedirs(output_dir, exist_ok=True)

    env_factory = lambda: gym.make(env_id)

    if verbose:
        print(f"deltatau-audit v{__version__} -- fix-cleanrl")
        print(f"  Agent:     {agent_class.__name__}")
        print(f"  Env:       {env_id}")
        print(f"  Timesteps: {timesteps:,}")
        print(f"  Speed range: {effective_min}-{effective_max}")
        print(f"  Device:    {device}")
        print(f"  Output:    {output_dir}/")
        print()

    before_result = None
    before_dir = os.path.join(output_dir, "before")

    # ═══════════════════════════════════════════════════════════════
    # STEP 1: Audit original agent (Before) — optional
    # ═══════════════════════════════════════════════════════════════
    if checkpoint_path is not None:
        if verbose:
            print("=" * 60)
            print("STEP 1/3: Auditing original agent")
            print("=" * 60)
            print()

        original_agent = agent_class(**agent_kwargs).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        original_agent.load_state_dict(ckpt)

        adapter_before = CleanRLAdapter(original_agent, device=device)

        t0 = time.time()
        before_result = run_full_audit(
            adapter_before, env_factory,
            speeds=audit_speeds,
            n_episodes=n_audit_episodes,
            sensitivity_episodes=0,
            device=device,
        )
        audit_time = time.time() - t0

        generate_report(before_result, before_dir,
                        title=f"{agent_class.__name__} on {env_id} — Before Fix")
        write_ci_summary(before_result["summary"],
                         before_result["robustness"], before_dir)

        before_dep = before_result["summary"]["deployment_score"]
        before_rating = before_result["summary"]["deployment_rating"]

        if verbose:
            print(f"\n  Audit time: {audit_time:.0f}s")
            print(f"  Result: {before_rating} (deployment={before_dep:.2f})")

        # Skip if already robust
        if before_dep >= 0.95:
            if verbose:
                print()
                print("  Model already PASSES deployment robustness (>=0.95)!")
                print("  No fix needed.")
                print(f"\n  Report: {before_dir}/index.html")
            return {
                "before": before_result,
                "after": None,
                "fixed_checkpoint_path": None,
                "skipped": True,
                "reason": "Already passes (deployment >= 0.95)",
            }
    else:
        if verbose:
            print("  (No original checkpoint — skipping Before audit)")
            print()

    # ═══════════════════════════════════════════════════════════════
    # STEP 2: Retrain with speed randomization
    # ═══════════════════════════════════════════════════════════════
    step_num = 2 if checkpoint_path else 1
    total_steps = 3 if checkpoint_path else 2

    if verbose:
        print("=" * 60)
        print(f"STEP {step_num}/{total_steps}: Training with speed randomization")
        print(f"          {timesteps:,} steps, speed {effective_min}-{effective_max}")
        print("=" * 60)
        print()

    fresh_agent = agent_class(**agent_kwargs).to(device)

    t0 = time.time()
    trained_agent = _ppo_train_cleanrl(
        fresh_agent, env_id,
        total_steps=timesteps,
        speed_min=speed_min,
        speed_max=speed_max,
        device=device,
        verbose=verbose,
    )
    train_time = time.time() - t0

    # Save fixed checkpoint
    fixed_ckpt_path = os.path.join(output_dir, "agent_fixed.pt")
    torch.save(trained_agent.state_dict(), fixed_ckpt_path)

    if verbose:
        print(f"\n  Training completed in {train_time:.0f}s")
        print(f"  Saved: {fixed_ckpt_path}")

    # ═══════════════════════════════════════════════════════════════
    # STEP 3: Audit fixed agent (After)
    # ═══════════════════════════════════════════════════════════════
    if verbose:
        print()
        print("=" * 60)
        print(f"STEP {step_num + 1}/{total_steps}: Auditing fixed agent")
        print("=" * 60)
        print()

    after_adapter = CleanRLAdapter(trained_agent, device=device)

    t0 = time.time()
    after_result = run_full_audit(
        after_adapter, env_factory,
        speeds=audit_speeds,
        n_episodes=n_audit_episodes,
        sensitivity_episodes=0,
        device=device,
    )
    audit_time_after = time.time() - t0

    after_dir = os.path.join(output_dir, "after")
    generate_report(after_result, after_dir,
                    title=f"{agent_class.__name__} on {env_id} — After Fix")
    write_ci_summary(after_result["summary"],
                     after_result["robustness"], after_dir)

    after_dep = after_result["summary"]["deployment_score"]
    after_rating = after_result["summary"]["deployment_rating"]

    if verbose:
        print(f"\n  Audit time: {audit_time_after:.0f}s")
        print(f"  Result: {after_rating} (deployment={after_dep:.2f})")

    # ═══════════════════════════════════════════════════════════════
    # COMPARISON
    # ═══════════════════════════════════════════════════════════════
    comp_path = os.path.join(output_dir, "comparison.md")
    if before_result is not None:
        generate_comparison(
            os.path.join(before_dir, "summary.json"),
            os.path.join(after_dir, "summary.json"),
            output_path=comp_path,
        )

    if verbose:
        _print_summary(before_result, after_result,
                       fixed_ckpt_path, output_dir, train_time)

    return {
        "before": before_result,
        "after": after_result,
        "fixed_checkpoint_path": fixed_ckpt_path,
        "skipped": False,
        "training_time": train_time,
        "training_timesteps": timesteps,
        "speed_range": (effective_min, effective_max),
    }


def _estimate_timesteps_cleanrl(env_id: str) -> int:
    """Estimate PPO training budget based on environment."""
    env_lower = env_id.lower()
    if any(k in env_lower for k in ("cartpole", "mountaincar", "acrobot",
                                     "pendulum", "lunar")):
        return 100_000
    if any(k in env_lower for k in ("cheetah", "hopper", "walker", "ant",
                                     "humanoid", "swimmer", "mujoco")):
        return 1_000_000
    return 300_000


def _print_summary(before, after, fixed_path, output_dir, train_time):
    """Print Before vs After summary."""
    print()
    print("=" * 60)
    print("  BEFORE vs AFTER")
    print("=" * 60)
    print()

    if before is not None:
        b_rob = before["robustness"]["per_scenario_scores"]
        a_rob = after["robustness"]["per_scenario_scores"]

        print(f"  {'Scenario':12s}  {'Before':>10s}  {'After':>10s}  {'Change':>10s}")
        print(f"  {'-' * 12}  {'-' * 10}  {'-' * 10}  {'-' * 10}")

        for sc in b_rob:
            b_pct = b_rob[sc]["return_ratio"] * 100
            a_pct = a_rob.get(sc, {}).get("return_ratio", 0) * 100
            delta = a_pct - b_pct
            sign = "+" if delta >= 0 else ""
            print(f"  {sc:12s}  {b_pct:9.1f}%  {a_pct:9.1f}%  {sign}{delta:8.1f}pp")

        b_sum = before["summary"]
        a_sum = after["summary"]
        print()
        print(f"  Deployment: {b_sum['deployment_rating']} "
              f"({b_sum['deployment_score']:.2f}) -> "
              f"{a_sum['deployment_rating']} "
              f"({a_sum['deployment_score']:.2f})")
        print(f"  Stress:     {b_sum['stress_rating']} "
              f"({b_sum['stress_score']:.2f}) -> "
              f"{a_sum['stress_rating']} "
              f"({a_sum['stress_score']:.2f})")
    else:
        a_sum = after["summary"]
        a_rob = after["robustness"]["per_scenario_scores"]
        print("  After-only results:")
        for sc, info in a_rob.items():
            pct = info["return_ratio"] * 100
            print(f"  {sc:12s}: {pct:.1f}%")
        print()
        print(f"  Deployment: {a_sum['deployment_rating']} "
              f"({a_sum['deployment_score']:.2f})")

    print()
    print(f"  Training time:   {train_time:.0f}s")
    print(f"  Fixed checkpoint: {fixed_path}")
    print(f"  Report: {output_dir}/after/index.html")
