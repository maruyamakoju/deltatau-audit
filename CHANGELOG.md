# Changelog

All notable changes to `deltatau-audit` are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

---

## [0.5.5] — 2026-02-19

### Added
- **Experiment tracker integration** (`deltatau_audit/tracker.py`): Push audit metrics to Weights & Biases or MLflow after any audit command.
  ```bash
  # W&B
  deltatau-audit audit-sb3 --model m.zip --algo ppo --env CartPole-v1 \
      --wandb --wandb-project my-project --wandb-run baseline

  # MLflow
  deltatau-audit audit-sb3 --model m.zip --algo ppo --env CartPole-v1 \
      --mlflow --mlflow-experiment my-experiment
  ```
  - New flags on all four audit subcommands (`audit`, `audit-sb3`, `audit-cleanrl`, `audit-hf`):
    `--wandb`, `--wandb-project PROJECT`, `--wandb-run RUN`,
    `--mlflow`, `--mlflow-experiment EXPERIMENT`
  - Python API: `log_to_wandb(result)`, `log_to_mlflow(result)` in `deltatau_audit.tracker`
  - Logged scalars: `deployment_score`, `stress_score`, `robustness_score`, `reliance_score`, `sensitivity_mean`, per-scenario `scenario/<name>/return_ratio`
  - Logged params: `deployment_rating`, `stress_rating`, `reliance_rating`, `quadrant`, `_deltatau_version`
  - Graceful degradation: missing `wandb`/`mlflow` package prints a `WARNING` rather than crashing
- **Optional extras** for tracker dependencies:
  ```bash
  pip install "deltatau-audit[wandb]"    # installs wandb>=0.12
  pip install "deltatau-audit[mlflow]"   # installs mlflow>=2.0
  ```

### Tests
- 33 new tests in `tests/test_v055.py` (307 total): cover `_build_metrics`, `_build_params`, `log_to_wandb`/`log_to_mlflow` (mocked), `maybe_log` dispatch, ImportError graceful handling, and all CLI parser flags.

---

## [0.5.4] — 2026-02-19

### Added
- **`py.typed` marker** (PEP 561): Package now exports type information for downstream users. Static type checkers (mypy, pyright, pylance) will use the annotations directly.
- **mypy CI step**: `unit-test` job in `audit-smoke.yml` now runs `mypy` on `auditor.py`, `diagnose.py`, and `adapters/base.py` with `--ignore-missing-imports --follow-imports=skip`. Catches annotation regressions on every push.
- **mypy in dev dependencies**: `pip install ".[dev]"` now installs `mypy>=1.0`.
- **`[tool.mypy]` config in `pyproject.toml`**: Centralises mypy settings; overrides suppress errors from `rdkit.*`, `stable_baselines3.*`, `gymnasium.*`, `sb3_contrib.*` stubs.

### Fixed
- **Type annotations in `auditor.py`**:
  - `callable` → `Callable[[], Any]` on all 5 `env_factory` parameters
  - `List[int] = None` → `Optional[List[int]] = None` on `speeds`/`interventions`/`scenarios`/`robustness_scenarios`
  - `_print_summary(summary, diagnosis: Dict = None)` → `Optional[Dict] = None`
  - `_run_episodes_parallel` parallel-path `results` list annotated correctly with `# type: ignore[list-item]`
- **Flaky test `test_run_full_audit_strict_threshold_changes_quadrant`**: Changed `deploy_threshold=0.99` → `1.01` (above maximum possible return ratio), making the test deterministically pass.

### Tests
- 11 new tests in `tests/test_v054.py` (274 total): verify `py.typed` exists, return annotations on public functions, `AgentAdapter` method annotations, and `generate_diagnosis` annotations.

---

## [0.5.3] — 2026-02-20

### Added
- **Adaptive episode sampling** (`--adaptive` flag on `audit`, `audit-sb3`, `audit-cleanrl`, `audit-hf`): Instead of a fixed `n_episodes`, run episode batches and keep sampling until every scenario's 95% bootstrap CI width on the return ratio drops below `--target-ci-width` (default: `0.10`), or until `--max-episodes` per scenario is reached (default: `500`).
  ```bash
  deltatau-audit audit-sb3 --model m.zip --algo ppo --env CartPole-v1 \
      --adaptive --target-ci-width 0.05 --max-episodes 300
  ```
  - `--adaptive` / `--target-ci-width WIDTH` / `--max-episodes N` added to all four audit subcommands.
  - `run_robustness_audit()` and `run_full_audit()` accept `adaptive`, `target_ci_width`, `max_episodes`.
  - When adaptive, result includes `n_episodes_used` dict (per-scenario count) and `adaptive: True`.
  - Non-adaptive default path unchanged.
- **Flaky test fix**: `test_run_full_audit_strict_threshold_changes_quadrant` now uses `seed=42` and `n_episodes=10` for deterministic results.
- 11 new tests in `tests/test_v053.py` (263 total).

---

## [0.5.2] — 2026-02-19

### Added
- **Failure diagnostics (`diagnose.py`)**: Every audit now includes a structured failure analysis that maps each failing or degraded scenario to a named failure pattern, root cause, and actionable fix recommendation.
  - 5 named patterns: *Speed Jitter Sensitivity*, *Observation Recency Dependency*, *Frequency Spike Fragility*, *Observation Noise Sensitivity*, *Extreme Frequency Fragility*
  - Unknown/custom scenarios get a generic pattern automatically
  - Issues sorted by severity (FAIL first, then DEGRADED)
- **CLI output**: `_print_summary()` now prints a `Failure Analysis` block after the prescription when issues exist, showing: Pattern, Cause, Fix, and any secondary issues.
- **Markdown output**: `--format markdown` now includes a `> Failure Analysis` blockquote section with the primary pattern, cause, and fix.
- **HTML report**: The Prescription section is followed by a styled `Failure Analysis` card when failures are detected, showing the pattern, cause, fix, and secondary issue badges.
- **`diagnosis` key in audit result**: `run_full_audit()` now returns `diagnosis` dict with `status`, `failing_scenarios`, `issues`, `primary_pattern`, `root_cause`, `fix_recommendation`, `summary_line`.
- 17 new tests in `tests/test_v052.py` (252 total).

---

## [0.5.1] — 2026-02-19

### Added
- **`--deploy-threshold` and `--stress-threshold` flags** on `audit-sb3`, `audit-cleanrl`, `audit-hf`, and `audit`: Override the default PASS/FAIL thresholds for quadrant classification.
  ```bash
  # Stricter standard: require 85% retention to be "deployment_ready"
  deltatau-audit audit-sb3 --model m.zip --algo ppo --env CartPole-v1 \
      --deploy-threshold 0.85 --stress-threshold 0.60
  ```
  - `--deploy-threshold` (default: 0.80): affects quadrant classification (`deployment_ready` vs `deployment_fragile`, `time_aware_robust` vs `time_aware_fragile`)
  - `--stress-threshold` (default: 0.50): stored in `summary.json` for downstream use
  - Both thresholds saved in `summary["deploy_threshold"]` and `summary["stress_threshold"]`
- 9 new tests in `tests/test_v051.py` (235 total).

---

## [0.5.0] — 2026-02-19

### Added
- **`audit-hf` command — HuggingFace Hub integration**: Audit any SB3 model directly from the HuggingFace Model Hub without downloading manually.
  ```bash
  deltatau-audit audit-hf --repo sb3/ppo-CartPole-v1 --algo ppo --env CartPole-v1
  ```
  - Auto-detects model filename (`{repo-name}.zip` → `model.zip` fallback)
  - Supports `--filename` for explicit override, `--hf-token` for private repos
  - All `audit-sb3` flags available: `--quiet`, `--format markdown`, `--compare`, `--ci`, `--workers`
- **`SB3Adapter.from_hub()`**: New classmethod for programmatic Hub downloads.
- **`[hf]` optional extra**: `pip install "deltatau-audit[hf]"` installs `huggingface_hub` + `stable-baselines3`.
- 10 new tests in `tests/test_v050.py` (226 total).

---

## [0.4.9] — 2026-02-19

### Added
- **`--quiet` / `-q` flag on `audit-sb3`, `audit-cleanrl`, `audit`**: Suppresses episode-level progress bars and verbose mid-audit output. Final PASS/FAIL summary is always shown. Useful for clean CI log output and piped commands.
- 9 new tests in `tests/test_v049.py` (216 total).

---

## [0.4.8] — 2026-02-19

### Added
- **Colored terminal output** (`color.py`): New `deltatau_audit.color` module with ANSI color helpers. Ratings are color-coded (`PASS`=bright green, `MILD`=green, `DEGRADED`=yellow, `FAIL`=bold red, `N/A`=gray). Auto-disabled on `NO_COLOR` / `TERM=dumb`; force-enabled with `FORCE_COLOR`. Works in GitHub Actions, standard terminals, and CI pipelines.
- **`_print_summary()` colored output**: `auditor._print_summary()` now uses colored ratings and dim secondary text for improved readability.
- **`--format markdown` flag on `audit-sb3` and `audit-cleanrl`**: Prints a PR-ready markdown table instead of the default text summary. When running in GitHub Actions, automatically appends to `$GITHUB_STEP_SUMMARY` for step-level audit cards.
- 16 new tests in `tests/test_v048.py` (207 total).

---

## [0.4.7] — 2026-02-19

### Added
- **`--compare` flag on `audit-sb3` and `audit-cleanrl`**: After any audit, pass `--compare before/summary.json` to automatically generate `comparison.html` comparing the new audit against a previous one. No need to run `fix-sb3` to get a Before/After report — works with any two audits.
- **`_version` + `_timestamp` in `ci_summary.json`**: CI output now stamped with audit tool version and ISO 8601 UTC timestamp, matching `summary.json` behavior from v0.4.6.
- README: `--compare` usage, `--workers`/`--seed` added to fix-sb3 options, `comparison.html` references updated.
- 9 new tests in `tests/test_v047.py` (191 total).

---

## [0.4.6] — 2026-02-19

### Fixed
- **`obs_noise` category in diff** (`P1`): `_DEPLOY_SCENARIOS` in `diff.py` now includes `obs_noise`, so `generate_comparison()` and `generate_comparison_html()` correctly label it as a Deployment scenario (not Stress). Previously only `jitter`, `delay`, `spike` were listed.

### Added
- **HTML comparison report**: `generate_comparison_html()` in `diff.py` generates a rich HTML diff with side-by-side badge cards, color-coded per-scenario delta bars, and verdict pills. The `diff` CLI command now writes both `comparison.md` and `comparison.html`. The `fix-sb3` and `fix-cleanrl` pipelines also generate `comparison.html`.
- **`_version` + `_timestamp` in `summary.json`**: Every `generate_report()` call now stamps the output JSON with `_version` (e.g. `"0.4.6"`) and `_timestamp` (ISO 8601 UTC). Enables audit traceability and `generate_comparison_html()` shows version/time per audit.
- **`n_workers` + `seed` in fix pipelines**: `fix_sb3_model()` and `fix_cleanrl_agent()` now accept `n_workers` and `seed` parameters, threaded through to both Before and After `run_full_audit()` calls. `--workers` and `--seed` CLI flags added to `fix-sb3` and `fix-cleanrl` subcommands.
- 12 new tests in `tests/test_v046.py` (182 total).

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
