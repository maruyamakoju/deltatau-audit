# deltatau-audit

[![PyPI version](https://img.shields.io/pypi/v/deltatau-audit)](https://pypi.org/project/deltatau-audit/)
[![Tests](https://github.com/maruyamakoju/deltatau-audit/actions/workflows/audit-smoke.yml/badge.svg)](https://github.com/maruyamakoju/deltatau-audit/actions/workflows/audit-smoke.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)

**Find timing bugs in your RL agents before deployment breaks them.**

`deltatau-audit` tests whether your trained RL agent breaks when timing changes — speed shifts, observation delays, jitter, and mid-episode spikes. One command, full report.

```
$ deltatau-audit audit-sb3 my_model.zip --env CartPole-v1

Robustness Test
  Nominal (speed=1):              reward = 487.2
  5x speed (unseen):              reward =  24.1  ↓ 95%
  Speed jitter (2 ± 1):           reward = 289.5  ↓ 41%
  Observation delay (1 step):     reward = 412.8  ↓ 15%
  Mid-episode spike (1→5→1):      reward =  87.6  ↓ 82%

  Deployment: FAIL (worst drop: 95%)
  Fix: retrain with speed augmentation
```

## Install

```bash
pip install deltatau-audit
```

Extras: `pip install deltatau-audit[sb3]` for Stable-Baselines3, `[demo]` for bundled demos.

## 30-Second Quick Start

```bash
# See it in action — no model needed
deltatau-audit demo

# Audit your own SB3 model
deltatau-audit audit-sb3 path/to/model.zip --env CartPole-v1

# Audit a CleanRL checkpoint
deltatau-audit audit-cleanrl path/to/agent.pt --env CartPole-v1

# Audit directly from HuggingFace Hub
deltatau-audit audit-hf sb3/ppo-CartPole-v1 --env CartPole-v1
```

Every command writes a `summary.json` + `index.html` report.

## What It Tests

| Scenario | What Breaks | Real-World Cause |
|----------|-------------|------------------|
| **Speed change** (2x, 5x, 8x) | Agent trained at one frequency fails at another | Servo loop running faster/slower than training |
| **Observation delay** (1-3 steps) | Agent acts on stale data | Network latency, sensor lag |
| **Speed jitter** (±random) | Inconsistent timing | OS scheduling, garbage collection |
| **Mid-episode spike** (1→5→1) | Sudden speed burst destroys state | CPU throttling, competing processes |
| **Observation noise** (σ=0.1) | Noisy sensor readings | Hardware degradation |
| **Adversarial jitter** | Worst-case timing | Deliberate attack on timing channel |

## Auto-Fix

Found a timing-fragile model? Fix it:

```bash
# Retrain SB3 model with timing augmentation
deltatau-audit fix-sb3 fragile_model.zip --env CartPole-v1 --out fixed_model/

# Retrain CleanRL agent
deltatau-audit fix-cleanrl agent.pt --env CartPole-v1 --out fixed_agent/
```

Before/after comparison is automatic.

## CI Integration

### GitHub Actions (1 line)

```yaml
- uses: maruyamakoju/deltatau-audit@v1
  with:
    model: models/agent.zip
    env: CartPole-v1
    threshold-deploy: 0.80
    threshold-stress: 0.50
```

Fails the build if your agent isn't timing-robust. Generates a badge:

![Timing Robust](https://img.shields.io/badge/timing-robust-brightgreen)

### Any CI

```bash
deltatau-audit audit-sb3 model.zip --env CartPole-v1 --ci
# Writes ci_summary.json with pass/fail for your pipeline
```

## Formal Verification

Go beyond empirical testing with mathematical guarantees:

```bash
# Generate a formal safety certificate
deltatau-audit certify model.zip --env CartPole-v1

# 6 verification levels:
#   L1: Empirical sampling
#   L2: Statistical bounds (Clopper-Pearson)
#   L3: Interval Bound Propagation (IBP)
#   L4: Spectral norm Lipschitz bound
#   L5: CROWN linear relaxation
```

## Supported Frameworks

| Framework | Command | Model Format |
|-----------|---------|--------------|
| **Stable-Baselines3** | `audit-sb3` | `.zip` |
| **CleanRL** | `audit-cleanrl` | `.pt` |
| **HuggingFace Hub** | `audit-hf` | Hub ID |
| **Any PyTorch** | `audit` | `.pt` checkpoint |
| **Custom** | Python API | Any `nn.Module` |

## Python API

```python
from deltatau_audit import run_full_audit
from deltatau_audit.adapters.sb3 import SB3Adapter

adapter = SB3Adapter.from_path("model.zip", env_id="CartPole-v1")
result = adapter.audit(episodes=50)

print(result["deployment_rating"])  # "PASS" or "FAIL"
print(result["robustness_score"])   # 0.0 - 1.0
```

## Advanced Features

- **Stress analysis**: `deltatau-audit stress analyze summary.json` — identifies failure mechanisms
- **Seed sweep**: Multi-seed statistical evaluation with bootstrap confidence intervals
- **Benchmarking**: `deltatau-audit bench run --manifest bench/manifest.yaml` — matrix evaluation
- **Diff**: `deltatau-audit diff before.json after.json` — compare two audits
- **Badges**: `deltatau-audit badge summary.json` — generate SVG status badges

## Who Needs This

- **Robotics teams** deploying learned controllers on hardware with variable loop rates
- **Autonomous driving** where sensor timing is never perfectly consistent
- **Trading systems** where execution speed determines profit/loss
- **Any RL deployment** where training and production timing differ

## Citation

```bibtex
@software{deltatau_audit2026,
  author = {maruyamakoju},
  title = {deltatau-audit: Timing Robustness Audit for RL Agents},
  version = {1.0.0},
  year = {2026},
  url = {https://github.com/maruyamakoju/deltatau-audit}
}
```

## License

MIT
