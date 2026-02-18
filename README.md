# deltatau-audit

[![PyPI version](https://img.shields.io/pypi/v/deltatau-audit)](https://pypi.org/project/deltatau-audit/)
[![CI](https://github.com/maruyamakoju/deltatau-audit/actions/workflows/audit-smoke.yml/badge.svg)](https://github.com/maruyamakoju/deltatau-audit/actions/workflows/audit-smoke.yml)
[![Python 3.9+](https://img.shields.io/pypi/pyversions/deltatau-audit)](https://pypi.org/project/deltatau-audit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Time Robustness Audit for RL agents.**

Evaluates whether an RL agent breaks when the environment's timing changes — the kind of failure that silently appears in deployment but never shows up in training.

## A standard PPO agent trained on HalfCheetah collapses under timing perturbations

A PPO agent trained to reward ~990 on HalfCheetah-v5 was audited with `deltatau-audit`. **All 4 timing scenarios cause statistically significant performance drops (95% bootstrap CI):**

| Scenario | Return (% of nominal) | 95% CI | Drop |
|----------|:--------------------:|:------:|:----:|
| Observation delay (1 step) | **3.8%** | [2.4%, 5.2%] | -96% |
| Speed jitter (2 +/- 1) | **25.4%** | [23.5%, 27.8%] | -75% |
| 5x speed (unseen) | **-9.3%** | [-10.6%, -8.4%] | -109% |
| Mid-episode spike (1->5->1) | **90.9%** | [86.3%, 97.8%] | -9% |

A single step of observation delay destroys 96% of performance. The agent goes *negative* at 5x speed. These are deployment-realistic conditions that never appear during standard training.

[Download the sample HTML report](https://github.com/maruyamakoju/deltatau-audit/releases/download/v0.3.2/halfcheetah_audit_report_v0.3.2.zip) (open `index.html`)

<details>
<summary>Reproduce this result</summary>

```bash
pip install "deltatau-audit[sb3,mujoco]"
git clone https://github.com/maruyamakoju/deltatau-audit.git
cd deltatau-audit
python examples/audit_halfcheetah.py  # trains PPO 500K steps + runs audit (~30 min)
```

To skip training, download the [pre-trained model](https://github.com/maruyamakoju/deltatau-audit/releases/download/v0.3.2/halfcheetah_ppo_500k.zip) to `runs/halfcheetah_ppo_500k.zip`.

</details>

## Install

```bash
pip install deltatau-audit            # core
pip install "deltatau-audit[demo]"    # + CartPole demo
pip install "deltatau-audit[sb3,mujoco]"  # + SB3 + MuJoCo environments
```

## Quick Start

### CartPole demo (no GPU, 30 seconds)

```bash
python -m deltatau_audit demo cartpole --out demo_report/
```

### HalfCheetah MuJoCo audit (CPU, ~30 min including training)

```bash
pip install "deltatau-audit[sb3,mujoco]"
git clone https://github.com/maruyamakoju/deltatau-audit.git
cd deltatau-audit && python examples/audit_halfcheetah.py
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
