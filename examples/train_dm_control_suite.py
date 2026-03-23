#!/usr/bin/env python3
"""Train dm_control suite checkpoints expected by bench manifests.

Produces:
  checkpoints/dm_control/walker_walk_standard.zip
  checkpoints/dm_control/walker_walk_robust.zip
  checkpoints/dm_control/cheetah_run_standard.zip
  checkpoints/dm_control/cheetah_run_robust.zip
  checkpoints/dm_control/reacher_easy_standard.zip
  checkpoints/dm_control/reacher_easy_robust.zip
  checkpoints/dm_control/humanoid_stand_standard.zip
  checkpoints/dm_control/humanoid_stand_robust.zip
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _algo_cls(name: str):
    import stable_baselines3 as sb3

    mapping = {
        "ppo": sb3.PPO,
        "sac": sb3.SAC,
        "td3": sb3.TD3,
        "a2c": sb3.A2C,
    }
    key = str(name).lower()
    if key not in mapping:
        raise ValueError(f"Unsupported algo: {name}")
    return mapping[key]


def _make_env(env_id: str, *, robust: bool, seed: int, base_speed: int, jitter: int):
    import gymnasium as gym
    from gymnasium.wrappers import FlattenObservation

    from deltatau_audit.adapters.dm_control import make_dm_control_env
    from deltatau_audit.wrappers.speed import JitterWrapper

    env = make_dm_control_env(env_id, seed=seed)
    if isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)
    if robust:
        env = JitterWrapper(env, base_speed=base_speed, jitter=jitter, seed=seed)
    return env


def _train_one(
    *,
    env_id: str,
    algo: str,
    robust: bool,
    seed: int,
    timesteps: int,
    device: str,
    out_path: Path,
    base_speed: int,
    jitter: int,
) -> dict[str, Any]:
    from stable_baselines3.common.vec_env import DummyVecEnv

    model_cls = _algo_cls(algo)

    env = DummyVecEnv(
        [lambda: _make_env(env_id, robust=robust, seed=seed, base_speed=base_speed, jitter=jitter)]
    )
    t0 = datetime.now(timezone.utc).timestamp()
    try:
        model = model_cls("MlpPolicy", env, seed=seed, device=device, verbose=0)
        model.learn(total_timesteps=timesteps, progress_bar=False)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(out_path.with_suffix("")))
    finally:
        env.close()
    t1 = datetime.now(timezone.utc).timestamp()
    return {
        "status": "trained",
        "duration_s": round(float(t1 - t0), 3),
        "model_path": str(out_path),
        "timesteps": int(timesteps),
    }


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train dm_control suite checkpoints for bench manifests")
    parser.add_argument("--timesteps", type=int, default=30_000, help="Timesteps per model")
    parser.add_argument("--seed", type=int, default=0, help="Training seed")
    parser.add_argument("--device", type=str, default="cpu", help="SB3 device")
    parser.add_argument("--out-root", type=str, default="checkpoints/dm_control", help="Checkpoint output root")
    parser.add_argument("--base-speed", type=int, default=3, help="Base speed for robust jitter training")
    parser.add_argument("--jitter", type=int, default=2, help="Jitter range for robust training")
    parser.add_argument("--force", action="store_true", default=False, help="Retrain even if checkpoint exists")
    parser.add_argument(
        "--envs",
        type=str,
        nargs="+",
        default=["walker_walk", "cheetah_run", "reacher_easy", "humanoid_stand"],
        help="Subset of env aliases to train",
    )
    parser.add_argument(
        "--algo-map",
        type=str,
        default="walker_walk:ppo,cheetah_run:ppo,reacher_easy:ppo,humanoid_stand:ppo",
        help="Comma-separated env:algo mapping",
    )
    parser.add_argument(
        "--summary-out",
        type=str,
        default="_status_demo/dm_control_suite_training/summary.json",
        help="Summary JSON output path",
    )
    args = parser.parse_args()

    # Dependency preflight
    try:
        import dm_control  # noqa: F401
        import shimmy  # noqa: F401
        import stable_baselines3  # noqa: F401
    except Exception as exc:
        print("Dependency check failed. Install: pip install shimmy[dm-control] dm-control stable-baselines3")
        print(f"Error: {exc}")
        return 1

    canonical = {
        "walker_walk": "dm_control/walker-walk-v0",
        "cheetah_run": "dm_control/cheetah-run-v0",
        "reacher_easy": "dm_control/reacher-easy-v0",
        "humanoid_stand": "dm_control/humanoid-stand-v0",
    }
    selected = [e for e in args.envs if e in canonical]
    if not selected:
        print("No valid env aliases provided. Choose from: walker_walk cheetah_run reacher_easy humanoid_stand")
        return 1

    algo_map: dict[str, str] = {}
    for pair in str(args.algo_map).split(","):
        item = pair.strip()
        if not item or ":" not in item:
            continue
        env_name, algo_name = item.split(":", 1)
        algo_map[env_name.strip()] = algo_name.strip()

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, Any]] = []
    n_trained = 0
    n_skipped = 0
    n_failed = 0

    for env_alias in selected:
        env_id = canonical[env_alias]
        algo = algo_map.get(env_alias, "ppo")
        for mode in ("standard", "robust"):
            robust = mode == "robust"
            ckpt = out_root / f"{env_alias}_{mode}.zip"
            if ckpt.exists() and not args.force:
                jobs.append(
                    {
                        "env": env_alias,
                        "env_id": env_id,
                        "mode": mode,
                        "algo": algo,
                        "status": "skipped",
                        "reason": "exists",
                        "model_path": str(ckpt),
                    }
                )
                n_skipped += 1
                continue

            print(f"[train] env={env_alias} mode={mode} algo={algo} steps={args.timesteps}")
            try:
                trained = _train_one(
                    env_id=env_id,
                    algo=algo,
                    robust=robust,
                    seed=int(args.seed),
                    timesteps=int(args.timesteps),
                    device=str(args.device),
                    out_path=ckpt,
                    base_speed=int(args.base_speed),
                    jitter=int(args.jitter),
                )
                jobs.append(
                    {
                        "env": env_alias,
                        "env_id": env_id,
                        "mode": mode,
                        "algo": algo,
                        **trained,
                    }
                )
                n_trained += 1
            except Exception as exc:
                jobs.append(
                    {
                        "env": env_alias,
                        "env_id": env_id,
                        "mode": mode,
                        "algo": algo,
                        "status": "failed",
                        "error": str(exc),
                        "model_path": str(ckpt),
                    }
                )
                n_failed += 1
                print(f"[failed] env={env_alias} mode={mode}: {exc}")

    payload = {
        "generated_at_utc": _utc_now(),
        "config": {
            "timesteps": int(args.timesteps),
            "seed": int(args.seed),
            "device": str(args.device),
            "out_root": str(out_root),
            "base_speed": int(args.base_speed),
            "jitter": int(args.jitter),
            "envs": selected,
            "algo_map": algo_map,
            "force": bool(args.force),
        },
        "counts": {"trained": n_trained, "skipped": n_skipped, "failed": n_failed},
        "jobs": jobs,
    }
    _write_summary(Path(args.summary_out), payload)

    print("dm_control suite training complete")
    print(f"  trained: {n_trained}")
    print(f"  skipped: {n_skipped}")
    print(f"  failed:  {n_failed}")
    print(f"  summary: {Path(args.summary_out).resolve()}")
    return 1 if n_failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
