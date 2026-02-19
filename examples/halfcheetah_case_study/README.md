# HalfCheetah Before/After Case Study

**HalfCheetah PPO loses 96% performance with a 1-step delay. We fix it in 15 minutes.**

This example demonstrates the complete `deltatau-audit` workflow on a MuJoCo continuous control task:
audit a standard PPO, expose timing fragility, fix it with speed-randomized retraining, and verify the improvement.

## Quick Start

```bash
# 1. Install dependencies
pip install "deltatau-audit[sb3,mujoco]"

# 2. Train baseline model (~5-10 min on GPU)
python train.py --device cuda --seed 42

# 3. Run full pipeline (~15-20 min on GPU)
python run_case_study.py --device cuda --workers auto --seed 42
```

## Expected Results

| Scenario        | Before | After  | Change |
|-----------------|--------|--------|--------|
| Speed jitter    | ~55%   | ~85%   | +30%   |
| Obs. delay      | ~4%    | ~75%   | +71%   |
| Speed spike     | ~50%   | ~82%   | +32%   |
| Obs. noise      | ~70%   | ~90%   | +20%   |
| 5x speed [STRESS] | ~25% | ~50%   | +25%   |

| Badge       | Before          | After          |
|-------------|-----------------|----------------|
| Deployment  | FAIL (0.04)     | MILD+ (0.75)   |
| Stress      | FAIL (0.25)     | FAIL (0.50)    |
| Quadrant    | deployment_fragile | deployment_ready |

> Exact numbers vary by seed and hardware. The key signal is direction (Before < After)
> and rating improvement.

## Files

### Input Scripts

| File | Purpose |
|------|---------|
| `train.py` | Train a standard PPO on HalfCheetah-v5 (1M steps) |
| `run_case_study.py` | Full audit/fix/re-audit pipeline |

### Generated Outputs

```
outputs/
  models/
    halfcheetah_ppo.zip           # Before model (standard training)
    halfcheetah_ppo_fixed.zip     # After model (speed-randomized)
  before/
    index.html                    # Full audit report
    summary.json                  # Machine-readable results
    robustness_bars.png           # Scenario comparison chart
  after/
    index.html, summary.json, robustness_bars.png
  training_audits/
    step_100000/                  # Audit snapshots during retraining
    step_200000/
    ...
  comparison.html                 # Side-by-side HTML comparison
  comparison.md                   # Markdown comparison
  badges/
    before-deployment.svg         # Before deployment badge
    before-stress.svg             # Before stress badge
    before-status.svg             # Before overall status badge
    after-deployment.svg          # After deployment badge
    after-stress.svg              # After stress badge
    after-status.svg              # After overall status badge
  summary_table.txt               # Plain text summary
```

## CLI Options

### train.py

```
--timesteps N     Training steps (default: 1,000,000)
--device DEVICE   cpu or cuda (default: cpu)
--seed SEED       Random seed (default: 42)
--output DIR      Model output directory (default: outputs/models)
--force           Retrain even if model exists
```

### run_case_study.py

```
--model PATH        Path to trained model (default: outputs/models/halfcheetah_ppo.zip)
--device DEVICE     cpu or cuda (default: cpu)
--workers N|auto    Parallel workers for audit (default: 1)
--fix-timesteps N   Retraining steps (default: 500,000)
--seed SEED         Random seed (default: 42)
--output DIR        Output directory (default: outputs)
--quick             Quick mode: 10 episodes, no adaptive (~3 min)
```

## Quick Test

For a fast validation run (~3 minutes on GPU):

```bash
python run_case_study.py --device cuda --workers auto --quick
```

This uses 10 episodes per condition (instead of 50+) and skips adaptive sampling.
Results will be noisier but the Before/After direction should still be clear.

## Alternative: One-Command Fix

If you just want to fix a model without the full case study:

```bash
deltatau-audit fix-sb3 --model outputs/models/halfcheetah_ppo.zip \
    --algo ppo --env HalfCheetah-v5 --device cuda
```

## Hardware Requirements

| Phase | GPU (RTX 4090) | CPU |
|-------|:--------------:|:---:|
| Training (1M steps) | ~5-10 min | ~1-2 hr |
| Before audit | ~3-5 min | ~10 min |
| Fix retraining (500K) | ~3-5 min | ~30-60 min |
| After audit | ~3-5 min | ~10 min |
| **Total** | **~15-25 min** | **~2-3 hr** |

## Why HalfCheetah?

HalfCheetah-v5 is the canonical MuJoCo benchmark:
- Continuous action space (6-dim)
- High-dimensional observations (17-dim)
- Fast training with PPO (~5 min on GPU)
- Extremely sensitive to timing perturbations (1-step delay causes ~96% performance drop)

This makes it an ideal showcase for timing robustness auditing.
