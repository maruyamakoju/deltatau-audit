# deltatau-audit

[![PyPI version](https://img.shields.io/pypi/v/deltatau-audit)](https://pypi.org/project/deltatau-audit/)
[![CI](https://github.com/maruyamakoju/deltatau-audit/actions/workflows/audit-smoke.yml/badge.svg)](https://github.com/maruyamakoju/deltatau-audit/actions/workflows/audit-smoke.yml)
[![Python 3.9+](https://img.shields.io/pypi/pyversions/deltatau-audit)](https://pypi.org/project/deltatau-audit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Time Robustness Audit for RL agents.**

Evaluates whether an RL agent breaks when the environment's timing changes — the kind of failure that silently appears in deployment but never shows up in training.

## Try it in 30 seconds

```bash
pip install "deltatau-audit[demo]"
python -m deltatau_audit demo cartpole
```

No GPU. No MuJoCo. Just `pip install` and run. You'll see a Before/After comparison:

| Scenario | Before (Baseline) | After (Speed-Randomized) | Change |
|----------|:-----------------:|:------------------------:|:------:|
| 5x speed | **12%** | **49%** | +37pp |
| Speed jitter | **66%** | **115%** | +49pp |
| Observation delay | **82%** | **95%** | +13pp |
| Mid-episode spike | **23%** | **62%** | +39pp |
| **Deployment** | **FAIL** (0.23) | **DEGRADED** (0.62) | +0.39 |

The standard agent collapses under timing perturbations. Speed-randomized training dramatically improves robustness. Full HTML reports with charts are generated in `demo_report/`.

## The same pattern at MuJoCo scale: HalfCheetah PPO

A PPO agent trained to reward ~990 on HalfCheetah-v5 shows even more catastrophic timing failures — **all 4 scenarios statistically significant (95% bootstrap CI)**:

| Scenario | Return (% of nominal) | 95% CI | Drop |
|----------|:--------------------:|:------:|:----:|
| Observation delay (1 step) | **3.8%** | [2.4%, 5.2%] | -96% |
| Speed jitter (2 +/- 1) | **25.4%** | [23.5%, 27.8%] | -75% |
| 5x speed (unseen) | **-9.3%** | [-10.6%, -8.4%] | -109% |
| Mid-episode spike (1->5->1) | **90.9%** | [86.3%, 97.8%] | -9% |

A single step of observation delay destroys 96% of performance. The agent goes *negative* at 5x speed.

![HalfCheetah robustness audit results](https://raw.githubusercontent.com/maruyamakoju/deltatau-audit/main/assets/halfcheetah_robustness.png)

[View interactive report](https://maruyamakoju.github.io/deltatau-audit/sample/halfcheetah/) | [Download report ZIP](https://github.com/maruyamakoju/deltatau-audit/releases/latest/download/halfcheetah_audit_report.zip)

### Speed-randomized training fixes the problem

| Scenario | Before (Standard) | After (Speed-Randomized) | Change |
|----------|:-----------------:|:------------------------:|:------:|
| Observation delay | **2%** | **148%** | +146pp |
| Speed jitter | **28%** | **121%** | +93pp |
| 5x speed (unseen) | **-12%** | **38%** | +50pp |
| Mid-episode spike | **100%** | **113%** | +13pp |
| **Deployment** | **FAIL** (0.02) | **PASS** (1.00) | |
| **Quadrant** | deployment_fragile | deployment_ready | |

![Robust agent audit results](https://raw.githubusercontent.com/maruyamakoju/deltatau-audit/main/assets/halfcheetah_robust_robustness.png)

[View Before report](https://maruyamakoju.github.io/deltatau-audit/sample/halfcheetah_before/) | [View After report](https://maruyamakoju.github.io/deltatau-audit/sample/halfcheetah_after/)

<details>
<summary>Reproduce HalfCheetah results</summary>

```bash
pip install "deltatau-audit[sb3,mujoco]"
git clone https://github.com/maruyamakoju/deltatau-audit.git
cd deltatau-audit
python examples/audit_halfcheetah.py              # standard PPO audit (~30 min)
python examples/train_robust_halfcheetah.py        # train robust PPO (~30 min)
python examples/audit_before_after.py              # Before/After comparison
```

Or download pre-trained models from [Releases](https://github.com/maruyamakoju/deltatau-audit/releases/latest).

</details>

## Install

```bash
pip install deltatau-audit            # core
pip install "deltatau-audit[demo]"    # + CartPole demo (recommended start)
pip install "deltatau-audit[sb3,mujoco]"  # + SB3 + MuJoCo environments
```

## Audit Your Own SB3 Model

```python
from stable_baselines3 import PPO
from deltatau_audit.adapters.sb3 import SB3Adapter
from deltatau_audit.auditor import run_full_audit
from deltatau_audit.report import generate_report
import gymnasium as gym

model = PPO.load("my_model.zip")
adapter = SB3Adapter(model)

result = run_full_audit(
    adapter,
    lambda: gym.make("HalfCheetah-v5"),
    speeds=[1, 2, 3, 5, 8],
    n_episodes=30,
)
generate_report(result, "my_audit/", title="My Agent Audit")
```

## What It Measures

| Badge | What it tests | How |
|-------|--------------|-----|
| **Reliance** | Does the agent *use* internal timing? | Tampers with internal Dt, measures value prediction error |
| **Deployment** | Does the agent *survive* realistic timing changes? | Jitter, observation delay, mid-episode speed spikes |
| **Stress** | Does the agent *survive* extreme timing changes? | 5x speed (unseen during training) |

Agents without internal timing (standard PPO, SAC, etc.) get **Reliance: N/A** — only Deployment and Stress are tested.

## Rating Scale

| Rating | Return Ratio | Meaning |
|--------|-------------|---------|
| PASS | > 95% | Production ready |
| MILD | > 80% | Minor degradation |
| DEGRADED | > 50% | Significant loss |
| FAIL | <= 50% | Agent breaks |

All return ratios include bootstrap 95% confidence intervals with significance testing.

## CI Mode

```bash
python -m deltatau_audit demo cartpole --ci --out ci_report/
# exit 0 = pass, exit 1 = warn (stress), exit 2 = fail (deployment)
```

Outputs `ci_summary.json` and `ci_summary.md` for pipeline gates and PR comments.

## Custom Adapters

Implement `AgentAdapter` (see `deltatau_audit/adapters/base.py`):

```python
from deltatau_audit.adapters.base import AgentAdapter

class MyAdapter(AgentAdapter):
    def reset_hidden(self, batch=1, device="cpu"):
        return torch.zeros(batch, hidden_dim)

    def act(self, obs, hidden):
        # Returns: (action, value, hidden_new, dt_or_None)
        ...
        return action, value, hidden_new, None
```

Built-in adapters: `SB3Adapter` (PPO/SAC/TD3/A2C), `SB3RecurrentAdapter` (RecurrentPPO), `InternalTimeAdapter` (Dt-GRU models).

## Comparing Results

```bash
python -m deltatau_audit diff before/summary.json after/summary.json --out comparison.md
```
