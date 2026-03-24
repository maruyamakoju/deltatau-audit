---
title: deltatau-audit Demo
emoji: ⏱️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: mit
short_description: RL timing robustness audit — Before vs After comparison
---

# deltatau-audit: RL Timing Robustness Demo

Does your RL agent break when the clock changes? This demo audits two GRU
agents under 6 timing perturbation scenarios and shows the Before (fragile)
vs After (robust) comparison.

## How it works

1. **Before agent** — trained at nominal speed only (no timing augmentation)
2. **After agent** — trained with speed randomization (1x-3x jitter)
3. Both agents are evaluated under: 5x speed, jitter, observation delay,
   mid-episode speed spike, observation noise, and adversarial jitter
4. Results show per-scenario return ratios and an overall PASS/FAIL verdict

## Run locally

```bash
cd demo
pip install -r requirements.txt
python app.py
```

## Links

- [GitHub Repository](https://github.com/jmcoholich/deltatau-audit)
- [PyPI Package](https://pypi.org/project/deltatau-audit/)
