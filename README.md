# deltatau-audit

**Time Robustness Audit for RL agents.**

Evaluates whether an RL agent breaks when the environment's timing changes — the kind of failure that silently appears in deployment but never shows up in training.

## What It Measures

| Badge | What it tests | How |
|-------|--------------|-----|
| **Reliance** | Does the agent *use* internal timing? | Tampers with internal Δτ, measures value prediction error |
| **Deployment** | Does the agent *survive* realistic timing changes? | Jitter, observation delay, mid-episode speed spikes |
| **Stress** | Does the agent *survive* extreme timing changes? | 5× speed (unseen during training) |

Agents without internal timing (standard GRU, SB3 models, etc.) get **Reliance: N/A** — only Deployment and Stress are tested.

## Quick Start

```bash
pip install .

# Run the bundled CartPole demo (Before/After comparison)
python -m deltatau_audit demo cartpole --out demo_report/

# Audit your own checkpoint
python -m deltatau_audit audit \
    --checkpoint path/to/model.pt \
    --agent-type internal_time \
    --env chain \
    --out audit_report/
```

Open `demo_report/baseline/index.html` and `demo_report/robust_wide/index.html` to see the Before/After reports.

## CI Mode

Use `--ci` to get machine-readable output and exit codes for CI pipelines:

```bash
python -m deltatau_audit demo cartpole --ci --out ci_report/
# exit 0 = pass, exit 1 = warn (stress), exit 2 = fail (deployment)
```

Outputs:
- `ci_summary.json` — scores and ratings
- `ci_summary.md` — one-line summary for PR comments

## Rating Scale

**Deployment / Stress Robustness** (return ratio vs nominal):

| Rating | Threshold | Meaning |
|--------|-----------|---------|
| PASS | > 95% | Production ready |
| MILD | > 80% | Minor degradation |
| DEGRADED | > 50% | Significant loss |
| FAIL | ≤ 50% | Agent breaks |

## Adapting to Your Model

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

Then audit:

```python
from deltatau_audit.auditor import run_full_audit
from deltatau_audit.report import generate_report

result = run_full_audit(adapter, env_factory, speeds=[1,2,3,5,8], n_episodes=30)
generate_report(result, "my_report/", title="My Agent Audit")
```
