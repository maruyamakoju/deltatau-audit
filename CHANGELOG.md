# Changelog

All notable changes to `deltatau-audit` are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

---

## [0.4.5] — 2026-02-19

### Added
- **`--workers auto`**: `--workers` now accepts the string `"auto"` (maps to `os.cpu_count()`) in addition to integers. Supported on all `audit-*` and `demo` subcommands.
- **Workers hint in CLI output**: `audit-sb3` and `demo` print a one-line tip (`— tip: --workers auto for faster auditing`) when running serially, so new users discover the feature naturally.
- **Performance section in README**: Documents `--workers` speedup table, auto-detect usage, and interaction with `--seed`.
- **README updates**: `obs_noise` scenario documented in "What It Measures"; `ObsNoiseWrapper` added to wrappers list; Python API example shows `n_workers=4` and `seed=42`; `audit-sb3` example updated.

### Changed
- Demo default episodes reduced from 30 to 20 (faster first run; use `--episodes 30` to restore).

---

## [0.4.4] — 2026-02-19

### Added
- **`ObsNoiseWrapper`** (`wrappers/latency.py`): Gaussian observation noise (σ=0.1) simulating noisy sensors. Uses a seeded, thread-local `numpy.random.Generator` for reproducibility.
- **`obs_noise` robustness scenario**: Added to `ROBUSTNESS_SCENARIOS` and `DEPLOYMENT_SCENARIOS`. Now 4 deployment scenarios: `jitter`, `delay`, `spike`, `obs_noise`.
- **Parallel episode execution**: `run_reliance_audit`, `run_robustness_audit`, `run_full_audit` accept `n_workers: int = 1`. When > 1, episodes are dispatched via `ThreadPoolExecutor` for 2-8× speedup on multi-core machines. Serial path unchanged when `n_workers=1`.
- **`--workers` CLI flag** on `audit`, `audit-sb3`, `audit-cleanrl`, `demo` subcommands.
- **Visual score card in HTML reports**: Each badge now shows a colored meter bar and numeric score. A verdict pill (green/orange/red) displays the quadrant classification prominently.
- **`examples/validate_mujoco.py`**: End-to-end validation script proving fix-sb3 on HalfCheetah-v5. Trains initial PPO → before audit → fix → after audit → prints Before/After comparison table.
- 8 new tests (170 total, all passing): `TestObsNoiseWrapper` (5 tests), `TestParallelExecution` (3 tests).

---

## [0.4.3] — 2026-02-19

### Fixed
- **Episode timeout guard** (`P0`): `_run_single_episode` now accepts `max_steps=10_000` to prevent infinite loops on envs without episode termination. Truncated episodes emit a `RuntimeWarning`.
- **Continuous action space** (`P0`): `fixer_cleanrl._ppo_train_cleanrl` now detects action space type via a test forward pass (dtype check) and allocates the correct buffer — `(num_steps,) long` for discrete, `(num_steps, act_dim) float32` for continuous. Previously only discrete was supported.
- **Negative nominal return ratio** (`P1`): `compute_return_ratio` and `bootstrap_return_ratio` now use the sign-aware formula `1 + (perturbed − nominal) / |nominal|` when `nominal < 0`, so that reduced penalty correctly maps to ratio > 1.0 (improvement). Previously the sign was inverted for penalty-heavy environments.

### Added
- `--seed` flag on all audit CLI subcommands (`audit`, `audit-sb3`, `audit-cleanrl`, `demo`) for reproducible results. Seed is threaded through `run_full_audit` → `run_reliance_audit` / `run_robustness_audit` → `_run_single_episode` with per-episode offsets.
- `tqdm` progress bars in `run_reliance_audit` and `run_robustness_audit` when tqdm is installed. Falls back to a plain print statement.
- `tqdm>=4.60` added to package dependencies.
- 36 new unit tests in `tests/test_quality_fixes.py` covering timeout behavior, seed reproducibility, negative return ratio semantics, and continuous action buffer shape.

---

## [0.4.2] — 2026-02-19

### Added
- `fix-cleanrl` CLI command: audit → retrain → re-audit pipeline for CleanRL agents (no SB3 dependency)
- `deltatau_audit/fixer_cleanrl.py`: self-contained PPO training loop with JitterWrapper, works with any agent implementing `get_action_and_value(obs)`
- `notebooks/quickstart.ipynb`: Google Colab notebook — install, run demo, view Before/After table
- Open in Colab badge on README
- `examples/audit_before_after.py`: auto-downloads HalfCheetah pre-trained models from GitHub Releases if not found locally

---

## [0.4.1] — 2026-02-19

### Added
- `CleanRLAdapter`: wraps any CleanRL MLP or LSTM agent; `from_checkpoint()` and `from_module_path()` (dynamic class loading for CLI)
- `TorchPolicyAdapter`: generic callable adapter for IsaacLab/RSL-RL and any custom PyTorch actor-critic
  - Auto-detects RSL-RL checkpoint format (`{"model_state_dict": {"actor.*": ...}}`)
  - `from_actor_critic()`, `from_checkpoint()` class methods
- `audit-cleanrl` CLI subcommand: one-command CleanRL agent auditing with `--agent-module`
- `examples/audit_cleanrl.py`: train minimal CleanRL PPO, audit end-to-end
- `examples/isaaclab_skeleton.py`: IsaacLab/RSL-RL integration skeleton
- README: "Audit CleanRL Agents" section, "Sim-to-Real Transfer" section, "IsaacLab / RSL-RL" section
- 38 new tests (131 total)

---

## [0.4.0] — 2026-02-18

### Added
- `fix-sb3` CLI command: diagnose + fix in one command
  - Audits original model → retrains with speed randomization → re-audits → Before/After report
  - Skips retraining if deployment score ≥ 0.95
- `deltatau_audit/fixer.py`: `fix_sb3_model()` Python API
- `action.yml`: GitHub Action composite action (`uses: maruyamakoju/deltatau-audit@main`)
  - Inputs: `command`, `model`, `algo`, `env`, `extras`, `episodes`, `deploy-threshold`
  - Outputs: `status`, `deployment-score`, `stress-score`
- `examples/fix_cartpole.py`: train CartPole PPO, fix in one script
- `JitterWrapper`, `FixedSpeedWrapper`, `PiecewiseSwitchWrapper`, `ObservationDelayWrapper` documented in README
- `tests/test_fixer.py`: 6 tests for fix-sb3 pipeline

### Changed
- README hero section updated: "Find and fix timing failures in RL agents"

---

## [0.3.9] — 2026-02-17

### Added
- `audit-sb3` CLI: zero-friction SB3 model auditing with smart error hints
  - Auto-detects MuJoCo / Box2D / Atari dependencies and prints install hints
  - `--ci` flag for pipeline gate mode
- SB3 sample model download snippet in README

---

## [0.3.7] — 2026-02-16

### Added
- PyPI metadata polish: keywords, classifiers, long description
- CI snippet in README
- Stable `assets` release tag for download links

---

## [0.3.5] — 2026-02-15

### Added
- Before/After audit story in README: speed-randomized PPO fixes deployment failures
- CartPole Before/After demo as the hero experience
- Sample HTML reports on GitHub Pages

---

## [0.3.2] — 2026-02-14

### Added
- MuJoCo showcase: HalfCheetah PPO timing audit results
- Bootstrap 95% confidence intervals on all return ratios
- Statistical significance testing per scenario
- `SB3Adapter`: wraps PPO/SAC/TD3/A2C from stable-baselines3
- `SB3RecurrentAdapter`: wraps RecurrentPPO from sb3-contrib
- `examples/audit_halfcheetah.py`, `examples/train_robust_halfcheetah.py`
- `diff` subcommand: compare two `summary.json` files → `comparison.md`
- 75 unit tests

---

## [0.3.0] — 2026-02-13

### Initial release

- 3-badge evaluation: **Reliance** (intervention ablation), **Deployment** (jitter/delay/spike), **Stress** (5x speed)
- `InternalTimeAdapter`: wraps Δτ-GRU agents with internal time module
- `GenericRecurrentAdapter`: wraps standard GRU/LSTM policies
- `VariableFrequencyChainEnv` integration
- HTML report generation with charts
- CI mode: `--ci` flag → `ci_summary.json` + `ci_summary.md` + exit codes (0/1/2)
- Bundled CartPole checkpoints for `demo` subcommand
- `deltatau_audit/wrappers/`: `JitterWrapper`, `FixedSpeedWrapper`, `PiecewiseSwitchWrapper`, `ObservationDelayWrapper`
