"""Research-suite orchestration with staged execution and resume support."""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np


class StageSkippedError(RuntimeError):
    """Raised when a stage is intentionally skipped due to unmet prerequisites."""


@dataclass
class ResearchSuiteConfig:
    env: str
    out: str
    episodes: int
    seed: int | None
    speeds: list[int]
    deliberative_max_thinking_steps: int
    bridge_delay_ms: float
    bridge_delay_std_ms: float
    bridge_dt_ms: float
    bridge_actuator_alpha: float
    resume: bool
    fail_fast: bool


@dataclass
class StageOutcome:
    name: str
    status: str
    reason: str | None
    deployment_score: float | None
    stress_score: float | None
    output_dir: str
    duration_sec: float
    traceback_text: str | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _extract_scores(result: dict[str, Any]) -> tuple[float | None, float | None]:
    summary = result.get("summary")
    if not isinstance(summary, dict):
        return None, None
    return _to_float(summary.get("deployment_score")), _to_float(summary.get("stress_score"))


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(loaded, dict):
        return loaded
    return None


def _cached_stage_outcome(stage_name: str, stage_dir: Path) -> StageOutcome | None:
    summary = _load_json(stage_dir / "summary.json")
    if summary is None:
        return None
    dep, stress = _extract_scores(summary)
    return StageOutcome(
        name=stage_name,
        status="cached",
        reason="existing summary.json reused (--resume)",
        deployment_score=dep,
        stress_score=stress,
        output_dir=str(stage_dir),
        duration_sec=0.0,
    )


def _write_suite_markdown(path: Path, cfg: ResearchSuiteConfig, outcomes: list[StageOutcome], recommendations: list[str]) -> None:
    lines: list[str] = []
    lines.append("# Research Suite Summary")
    lines.append("")
    lines.append(f"- Generated (UTC): `{_now_iso()}`")
    lines.append(f"- Env: `{cfg.env}`")
    lines.append(f"- Episodes: `{cfg.episodes}`")
    lines.append(f"- Seed: `{cfg.seed}`")
    lines.append(f"- Speeds: `{cfg.speeds}`")
    lines.append(f"- Resume: `{cfg.resume}`")
    lines.append("")
    lines.append("| Stage | Status | Deployment | Stress | Reason |")
    lines.append("| --- | --- | --- | --- | --- |")
    for out in outcomes:
        dep = "n/a" if out.deployment_score is None else f"{out.deployment_score:.3f}"
        stress = "n/a" if out.stress_score is None else f"{out.stress_score:.3f}"
        reason = out.reason or ""
        lines.append(f"| {out.name} | {out.status} | {dep} | {stress} | {reason} |")
    lines.append("")
    lines.append("## Recommendations")
    lines.append("")
    if recommendations:
        for idx, rec in enumerate(recommendations, start=1):
            lines.append(f"{idx}. {rec}")
    else:
        lines.append("1. No additional recommendations.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def derive_recommendations(
    outcomes: list[StageOutcome],
    *,
    deployment_threshold: float = 0.80,
    stress_threshold: float = 0.50,
) -> list[str]:
    recs: list[str] = []
    failed = [o for o in outcomes if o.status == "failed"]
    skipped = [o for o in outcomes if o.status == "skipped"]
    succeeded = [o for o in outcomes if o.status in {"success", "cached"}]

    if failed:
        names = ", ".join(o.name for o in failed)
        recs.append(f"Fix failed stages first ({names}); rerun with --no-resume after resolving errors.")

    low_deploy = [o.name for o in succeeded if o.deployment_score is not None and o.deployment_score < deployment_threshold]
    if low_deploy:
        recs.append(
            "Deployment robustness below threshold for: "
            + ", ".join(low_deploy)
            + f". Apply speed-randomized retraining and rerun paper protocol ({deployment_threshold:.2f}+)."
        )

    low_stress = [o.name for o in succeeded if o.stress_score is not None and o.stress_score < stress_threshold]
    if low_stress:
        recs.append(
            "Stress robustness below threshold for: "
            + ", ".join(low_stress)
            + f". Run stress analyze/ablate and retrain interventions ({stress_threshold:.2f}+)."
        )

    if skipped:
        names = ", ".join(o.name for o in skipped)
        recs.append(f"Resolve skipped stages ({names}) by adding prerequisites or choosing a compatible env/action space.")

    if not failed and not low_deploy and not low_stress and not skipped and succeeded:
        recs.append("All stages passed thresholds; proceed to paper artifact freeze and internal review.")

    if not recs:
        recs.append("Collect at least one successful stage outcome before making a go/no-go decision.")
    return recs


def _run_stage(
    *,
    name: str,
    title: str,
    out_root: Path,
    resume: bool,
    runner: Callable[[], dict[str, Any]],
) -> StageOutcome:
    stage_dir = out_root / name
    stage_dir.mkdir(parents=True, exist_ok=True)

    if resume:
        cached = _cached_stage_outcome(name, stage_dir)
        if cached is not None:
            return cached

    start = time.perf_counter()
    try:
        result = runner()
        from .report import generate_report

        generate_report(result, str(stage_dir), title=title)
        dep, stress = _extract_scores(result)
        return StageOutcome(
            name=name,
            status="success",
            reason=None,
            deployment_score=dep,
            stress_score=stress,
            output_dir=str(stage_dir),
            duration_sec=time.perf_counter() - start,
        )
    except StageSkippedError as exc:
        return StageOutcome(
            name=name,
            status="skipped",
            reason=str(exc),
            deployment_score=None,
            stress_score=None,
            output_dir=str(stage_dir),
            duration_sec=time.perf_counter() - start,
        )
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        return StageOutcome(
            name=name,
            status="failed",
            reason=f"{exc.__class__.__name__}: {exc}",
            deployment_score=None,
            stress_score=None,
            output_dir=str(stage_dir),
            duration_sec=time.perf_counter() - start,
            traceback_text=traceback.format_exc(),
        )


def _probe_env(env_id: str) -> dict[str, Any]:
    import gymnasium as gym

    env = gym.make(env_id)
    try:
        obs_space = env.observation_space
        action_space = env.action_space
        if not isinstance(obs_space, gym.spaces.Box):
            raise ValueError(f"research-full requires Box observation space, got: {type(obs_space).__name__}")
        obs_dim = int(np.prod(obs_space.shape))

        if isinstance(action_space, gym.spaces.Discrete):
            return {
                "obs_dim": obs_dim,
                "action_kind": "discrete",
                "action_dim": int(action_space.n),
                "action_shape": (),
                "action_low": None,
                "action_high": None,
            }
        if isinstance(action_space, gym.spaces.Box):
            return {
                "obs_dim": obs_dim,
                "action_kind": "box",
                "action_dim": int(np.prod(action_space.shape)),
                "action_shape": tuple(int(x) for x in action_space.shape),
                "action_low": np.asarray(action_space.low, dtype=np.float32),
                "action_high": np.asarray(action_space.high, dtype=np.float32),
            }
        raise ValueError(f"Unsupported action space: {type(action_space).__name__}")
    finally:
        env.close()


def _build_deliberative_adapter(env_info: dict[str, Any], max_thinking_steps: int):
    if env_info["action_kind"] != "discrete":
        raise StageSkippedError("Deliberative stage currently supports discrete action spaces only.")

    import torch

    from internal_time_rl.models.deliberative import DeliberativeInternalTimeAgent

    from .adapters.base import AgentAdapter

    agent = DeliberativeInternalTimeAgent(
        env_info["obs_dim"],
        env_info["action_dim"],
        max_thinking_steps=max_thinking_steps,
    )

    class DelibAdapter(AgentAdapter):
        def __init__(self, model):
            self.model = model

        def reset_hidden(self, batch: int = 1, device: str = "cpu"):
            return self.model.get_initial_hidden(batch, torch.device(device))

        def act(self, obs, hidden):
            obs_t = obs if torch.is_tensor(obs) else torch.tensor(obs, dtype=torch.float32)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)
            with torch.no_grad():
                action, _, value, _, hidden_new, dt = self.model.get_action_and_value(obs_t, hidden)
            return int(action[0].item()), float(value[0].item()), hidden_new, float(dt[0].item())

    return DelibAdapter(agent)


def _build_ltc_adapter(env_info: dict[str, Any]):
    import torch

    from internal_time_rl.models.advanced import LiquidTimeCell
    from internal_time_rl.models.encoder import ObservationEncoder
    from internal_time_rl.models.time_module import TimeModule

    from .adapters.base import AgentAdapter

    class LTCAgent(torch.nn.Module):
        def __init__(self, obs_dim: int, action_dim: int):
            super().__init__()
            self.encoder = ObservationEncoder(obs_dim, 64)
            self.time_module = TimeModule(128, 64)
            self.rnn = LiquidTimeCell(64, 128)
            self.policy = torch.nn.Linear(128, action_dim)
            self.value_head = torch.nn.Linear(128, 1)

        def forward(self, obs, hidden):
            enc = self.encoder(obs)
            dt = self.time_module(hidden, enc)
            hidden_new = self.rnn(enc, hidden, dt)
            logits = self.policy(hidden_new)
            value = self.value_head(hidden_new).squeeze(-1)
            return logits, value, hidden_new, dt

    model = LTCAgent(env_info["obs_dim"], env_info["action_dim"])

    class LTCAdapter(AgentAdapter):
        def __init__(self, model_ref):
            self.model_ref = model_ref
            self.action_kind = str(env_info["action_kind"])
            self.action_shape = tuple(env_info["action_shape"])
            self.action_low = env_info["action_low"]
            self.action_high = env_info["action_high"]

        def reset_hidden(self, batch: int = 1, device: str = "cpu"):
            return torch.zeros(batch, 128, device=torch.device(device), dtype=torch.float32)

        def act(self, obs, hidden):
            obs_t = obs if torch.is_tensor(obs) else torch.tensor(obs, dtype=torch.float32)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)
            with torch.no_grad():
                logits, value, hidden_new, dt = self.model_ref(obs_t, hidden)
            if self.action_kind == "discrete":
                action = int(torch.argmax(logits[0]).item())
            else:
                raw = torch.tanh(logits[0]).detach().cpu().numpy().reshape(self.action_shape)
                action = np.clip(raw, self.action_low, self.action_high)
            return action, float(value[0].item()), hidden_new, float(dt[0].item())

    return LTCAdapter(model)


def run_research_suite(config: ResearchSuiteConfig) -> dict[str, Any]:
    """Run staged research suite and return summary payload."""
    import gymnasium as gym

    from .auditor import run_full_audit

    out_root = Path(config.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    env_info = _probe_env(config.env)

    outcomes: list[StageOutcome] = []
    adapter_cache: dict[str, Any] = {}

    def _delib_runner() -> dict[str, Any]:
        adapter = _build_deliberative_adapter(env_info, config.deliberative_max_thinking_steps)
        adapter_cache["deliberative"] = adapter
        return run_full_audit(
            adapter,
            lambda: gym.make(config.env),
            speeds=config.speeds,
            n_episodes=config.episodes,
            verbose=False,
            seed=config.seed,
        )

    out = _run_stage(
        name="deliberative",
        title="Deliberative Audit",
        out_root=out_root,
        resume=config.resume,
        runner=_delib_runner,
    )
    outcomes.append(out)
    if out.status == "failed" and config.fail_fast:
        return _finalize_suite(config, outcomes, out_root, dashboard=False)

    def _ltc_runner() -> dict[str, Any]:
        adapter = _build_ltc_adapter(env_info)
        adapter_cache["ltc"] = adapter
        return run_full_audit(
            adapter,
            lambda: gym.make(config.env),
            speeds=config.speeds,
            n_episodes=config.episodes,
            verbose=False,
            seed=config.seed,
        )

    out = _run_stage(
        name="ltc",
        title="LTC Audit",
        out_root=out_root,
        resume=config.resume,
        runner=_ltc_runner,
    )
    outcomes.append(out)
    if out.status == "failed" and config.fail_fast:
        return _finalize_suite(config, outcomes, out_root, dashboard=False)

    def _bridge_runner() -> dict[str, Any]:
        chosen = adapter_cache.get("deliberative") or adapter_cache.get("ltc")
        if chosen is None:
            raise StageSkippedError("Bridge stage needs a successful deliberative or ltc stage.")

        from .bridge.real_world import ActuatorLagWrapper, TransportDelayWrapper

        def bridge_env_factory():
            base = gym.make(config.env)
            delayed = TransportDelayWrapper(
                base,
                mean_delay_ms=config.bridge_delay_ms,
                std_delay_ms=config.bridge_delay_std_ms,
                dt_ms=config.bridge_dt_ms,
            )
            return ActuatorLagWrapper(delayed, alpha=config.bridge_actuator_alpha)

        return run_full_audit(
            chosen,
            bridge_env_factory,
            speeds=[1],
            n_episodes=config.episodes,
            verbose=False,
            seed=config.seed,
        )

    out = _run_stage(
        name="bridge",
        title="Sim-to-Real Bridge Audit",
        out_root=out_root,
        resume=config.resume,
        runner=_bridge_runner,
    )
    outcomes.append(out)

    return _finalize_suite(config, outcomes, out_root, dashboard=True)


def _finalize_suite(
    config: ResearchSuiteConfig,
    outcomes: list[StageOutcome],
    out_root: Path,
    *,
    dashboard: bool,
) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    recommendations = derive_recommendations(outcomes)
    payload = {
        "generated_at_utc": _now_iso(),
        "config": asdict(config),
        "stages": [asdict(o) for o in outcomes],
        "recommendations": recommendations,
    }
    summary_path = out_root / "suite_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_path = out_root / "suite_summary.md"
    _write_suite_markdown(md_path, config, outcomes, recommendations)

    dashboard_path: str | None = None
    if dashboard:
        try:
            from .report.meta_dashboard import generate_meta_dashboard

            dash = out_root / "dashboard.html"
            generate_meta_dashboard(str(out_root), str(dash))
            dashboard_path = str(dash)
        except Exception:
            dashboard_path = None

    return {
        "summary_path": str(summary_path),
        "summary_md_path": str(md_path),
        "dashboard_path": dashboard_path,
        "outcomes": outcomes,
    }
