# arxiv / NeurIPS Submission Checklist

## Status: PREPARING (target: NeurIPS 2026 / ICLR 2026)

Stable artifact surfaces are now frozen in:

- `docs/core_output_contract.md`
- `docs/submission_artifact_contract.md`
- `docs/pipeline_artifact_contract.md`

Use `python scripts/check_contracts.py` as the fast gate before running the full stack.

---

## Phase 0: Artifact Discipline

- [x] Core output contract documented
- [x] Submission artifact contract documented
- [x] Pipeline artifact contract documented
- [x] Contract gate script (`python scripts/check_contracts.py`)
- [x] CI smoke workflow runs contract gate
- [x] Release workflow blocks publish if contract gate fails
- [x] README aligned to contract-tested surfaces

---

## Phase 1: Theory & Paper Skeleton ✅ (in progress)

- [x] Formal Δτ definition (docs/theory.md)
- [x] TimeAwareGRU exponential kernel derivation (Proposition 1)
- [x] Lipschitz bound on return degradation (Theorem 1)
- [x] Related work positioning vs ODE-RNN, CARE, Skip-RNN (docs/related_work.md)
- [x] LaTeX paper draft (docs/paper/paper.tex)
- [ ] Internal review by co-authors (Cambridge / DeepMind)
- [ ] Proof of Theorem 1 (need full derivation, not just sketch)

---

## Phase 2: Experiments (✅ Chain + HalfCheetah, 🚧 dm_control)

### Chain Environment (done)
- [x] 5-seed experiments: speed_gen_hidden_5seed/
- [x] Spearman ρ=1.000, Monotonicity=100%
- [x] Δτ slope: +0.0245 (IT) vs -0.0027 (Skip-RNN)
- [x] Value ablation: +105% RMSE at S=8 (reverse Δτ)
- [ ] **10-seed rerun** for paper (currently 5 seeds)
- [ ] Bootstrap 95% CI with 5000 samples (currently 2000)

### CartPole (done)
- [x] Before (baseline): FAIL deploy (0.23)
- [x] After (speed-randomized): DEGRADED deploy (0.62)
- [x] 2-grade improvement on deployment axis
- [ ] 10-seed version with full CI

### HalfCheetah (done)
- [x] Standard PPO: FAIL (0.02) — observation delay: -96%
- [x] Robust PPO: PASS (1.00) after speed-randomized training
- [x] HTML reports with interactive figures
- [ ] Statistical significance test (bootstrap CI on HalfCheetah scores)

### dm_control Suite (🚧 in progress)
- [ ] Walker-walk: standard vs robust training
- [ ] Cheetah-run: standard vs robust training
- [ ] Reacher-easy: standard vs robust training
- [ ] Humanoid-stand: (stretch goal)
- [ ] 10-seed evaluation with "paper" protocol

---

## Phase 3: Statistical Rigor

- [ ] All main results: 10+ seeds
- [ ] Bootstrap CI: 5000 samples (currently 2000)
- [ ] Effect sizes: Cohen's d + Cliff's δ for all comparisons
- [ ] Bonferroni correction for multiple comparisons
- [ ] Wilcoxon signed-rank test for paired comparisons (same seeds)

**Current gaps:**
- Chain env: 5 seeds → need 10 (run `python experiments/run_speed_generalization.py --seeds 10 --speed-hidden --output-dir runs/speed_gen_hidden_10seed`)
- CartPole: needs 10-seed ablation via `python -m deltatau_audit bench run --manifest bench/high_rigor_10seed_manifest.yaml --protocol paper`
- HalfCheetah: needs CI on robustness scores

---

## Phase 4: Ablation Studies

### PPOTimeV2 Variants (partially done)
- [x] Variant 0 (baseline): no internal time
- [x] Variant 1 (state only): Δτ modulates hidden state only
- [x] Variant 2 (full reparam): Δτ modulates hidden + discount + GAE
- [x] Variant 3 (discount only): Δτ modulates discount + GAE only
- [ ] **Full 5-seed × 4-variant table** for paper
- [ ] Statistical test across variants

### Time Module Components
- [ ] Ablate: TimeModule hidden dim (32 → 64 → 128)
- [ ] Ablate: Δτ range [0.3, 2.5] vs [0.5, 2.0] vs [0.1, 4.0]
- [ ] Ablate: Smoothness coef (0.02) vs variance penalty
- [ ] Ablate: Mean-centering coef (0.01) vs no centering

### Architecture
- [ ] Ablate: Encoder depth (1L vs 2L vs 3L)
- [ ] Ablate: Hidden dim (64 vs 128 vs 256)

---

## Phase 5: Comparison to Baselines

### Key baselines needed
- [ ] ODE-RNN with oracle dt (already in results: External dt ODE-RNN = FAILS on hidden speed)
- [ ] CARE-style (discrete speed input): requires implementing
- [ ] Frame-stacking baseline (intervention3_memory): in checkpoints/
- [ ] TimeFeatureWrapper baseline: in checkpoints/

### Currently available
- [x] Baseline GRU (no internal time): in all experiments
- [x] Skip-RNN (ACT): Spearman ρ=-1.000 (wrong direction!)
- [x] ODE-RNN with external dt: FAILS on hidden speed experiments
- [x] TimeFeature (explicit dt injection): in ablation checkpoints

---

## Phase 6: Writing

### Paper sections (target: 9 pages + references)
- [ ] Abstract: 150 words ✏️
- [ ] Introduction: 1 page ✏️
- [ ] Background: 0.5 page ✏️
- [ ] Δτ Framework: 1 page ✏️
- [ ] InternalTime Agent: 1 page ✏️
- [ ] deltatau-audit: 0.5 page ✏️
- [ ] Experiments: 2.5 pages ✏️
- [ ] Ablation: 0.5 page ✏️
- [ ] Related Work: 0.5 page ✏️
- [ ] Conclusion: 0.2 page ✏️

### Figures needed
- [x] fig_hero.png — main result figure
- [x] fig_main_result.png — Δτ tracking across speeds
- [x] fig_ablation.png — ablation studies
- [x] fig_dt_tracking_detail.png — Δτ dynamics
- [x] fig_killer.png — HalfCheetah failure
- [ ] fig_dm_control.png — dm_control before/after
- [ ] fig_quadrant.png — 2D reliance × robustness plot
- [ ] fig_theory.png — TimeAwareGRU diagram

### Tables needed
- [x] results_table.tex — main results (Δτ slope, Spearman ρ)
- [ ] ablation_table.tex — PPOTimeV2 variant comparison
- [ ] robustness_table.tex — deployment/stress scores
- [ ] comparison_table.tex — vs baselines

---

## Phase 7: Code Release

- [x] PyPI package: deltatau-audit v0.8.0
- [x] GitHub repo: maruyamakoju/deltatau-audit
- [x] Colab quickstart notebook
- [x] HuggingFace Hub integration
- [x] Contract docs for output / submission / pipeline artifacts
- [x] Fast contract gate (`python scripts/check_contracts.py`)
- [x] CI workflow runs contract gate before full pytest
- [x] Release workflow runs contract gate before publish
- [ ] Zenodo DOI for reproducibility
- [ ] Docker container for exact environment
- [ ] Pre-trained checkpoints for all paper experiments on HF Hub

---

## Phase 8: arxiv Submission

- [ ] Paper PDF from paper.tex
- [ ] Anonymized version (remove author names / institution for double-blind)
- [ ] Appendix with proofs and additional experiments
- [ ] Code supplement (anonymized URL or supplemental .zip)
- [ ] Camera-ready version after review

---

## Critical Path

1. **10-seed experiments** (Chain + CartPole) — 2-3 days compute
2. **dm_control experiments** (Walker + Cheetah) — 2-3 days compute
3. **Ablation table** — 1 day compute
4. **Paper writing** — in parallel with experiments
5. **Internal review** — Cambridge / DeepMind co-authors
6. **arxiv submission** — target 4 weeks

---

## Operations Commands

- Run contract suites only:
  `python scripts/check_contracts.py`
- Train missing CartPole seeds (5 variants x seeds):
  `python -m deltatau_audit stress train-sb3 --env CartPole-v1 --algo ppo --out-root checkpoints_cartpole_ppo --seeds 5 6 7 8 9 --variants baseline intervention1_curriculum intervention2_time_feature intervention1_plus_2 intervention3_memory --timesteps 30000 --fail-fast`
- Train dm_control suite checkpoints:
  `python examples/train_dm_control_suite.py --timesteps 20000 --force --summary-out _status_demo/dm_control_suite_training/full_20k_summary.json`
- Run CartPole paper-grade bench:
  `python -m deltatau_audit bench run --manifest bench/high_rigor_10seed_manifest.yaml --protocol paper`
- Run dm_control paper-grade bench:
  `python -m deltatau_audit bench run --manifest bench/dm_control_research_manifest.yaml --protocol paper --no-resume`
- Monitor bench progress:
  `python scripts/monitor_bench_progress.py --manifest bench/high_rigor_10seed_manifest.yaml --output-root bench_runs/cartpole_high_rigor_10seed`
- Launch + monitor + finalize orchestration:
  `python scripts/run_submission_pipeline.py --mode preflight`
  `python scripts/run_submission_pipeline.py --mode launch`
  `python scripts/run_submission_pipeline.py --mode status --watch --interval 120`
  `python scripts/run_submission_pipeline.py --mode diagnose --stall-seconds 1800`
  `python scripts/run_submission_pipeline.py --mode supervise --interval 120 --stall-seconds 1800 --max-cycles 0`
  `python scripts/run_submission_pipeline.py --mode supervise --interval 120 --stall-seconds 1800 --auto-recover --recover-after-consecutive 2 --max-restarts-per-job 2 --max-restarts-per-signature 2 --max-restarts-per-window 3 --restart-window-seconds 10800 --restart-cooldown-seconds 1800 --recovery-grace-seconds 900 --max-no-progress-cycles 60 --max-no-progress-seconds 7200`
  `python scripts/run_submission_pipeline.py --mode autopilot --preflight --auto-recover --recover-after-consecutive 2 --max-restarts-per-job 2 --restart-cooldown-seconds 1800 --max-no-progress-cycles 60 --max-no-progress-seconds 7200 --interval 120 --stall-seconds 1800`
  `python scripts/run_submission_pipeline.py --mode autopilot --preflight --auto-recover --recover-after-consecutive 2 --max-restarts-per-job 2 --restart-cooldown-seconds 1800 --max-no-progress-cycles 60 --max-no-progress-seconds 7200 --interval 120 --stall-seconds 1800 --auto-finalize`
  `python scripts/run_submission_pipeline.py --mode autopilot --preflight --auto-recover --recover-after-consecutive 2 --max-restarts-per-job 3 --max-restarts-per-reason 2 --max-restarts-per-signature 2 --max-total-restarts 6 --max-restarts-per-window 3 --restart-window-seconds 10800 --restart-cooldown-seconds 1800 --restart-backoff-factor 1.5 --max-restart-cooldown-seconds 21600 --recovery-grace-seconds 900 --max-no-progress-cycles 60 --max-no-progress-seconds 7200 --interval 120 --stall-seconds 1800`
  `python scripts/run_submission_pipeline.py --mode report --event-tail 500 --stall-seconds 1800`
  `python scripts/run_submission_pipeline.py --mode recommend --stall-seconds 1800`
  `python scripts/run_submission_pipeline.py --mode report --event-tail 500 --stall-seconds 1800 && python scripts/run_submission_pipeline.py --mode diagnose --stall-seconds 1800`
  `python scripts/run_submission_pipeline.py --mode finalize`
- Strict readiness gate (fails CI when paper bench evidence is incomplete):
  `python scripts/prepare_submission.py --check-only --strict-check`
- Supervisor/audit trail log:
  `_status_demo/long_runs/supervisor_events.jsonl`

---

## Quick Reference: Key Numbers

| Metric | Value | Significance |
|--------|-------|--------------|
| Δτ slope (IT, 5-seed) | +0.0245 | 95% CI: [+0.0034, +0.0275] |
| Δτ slope (Skip-RNN) | -0.0027 | NEGATIVE (wrong direction!) |
| Spearman ρ (IT) | 1.000 | p < 1.4e-24 |
| Monotonicity rate (IT) | 100% | All 10 speed pairs correct |
| Value ablation (S=8) | +105% RMSE | Reverse Δτ intervention |
| HalfCheetah delay | -96% | 1-step observation delay |
| HalfCheetah 5x speed | -109% | Goes negative! |
| HalfCheetah fix | FAIL→PASS | Speed-randomized training |
| CartPole fix | FAIL→DEGRADED | +0.39 deployment score |
