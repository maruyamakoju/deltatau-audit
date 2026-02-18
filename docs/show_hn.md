# Show HN Post

**Title:** Show HN: deltatau-audit – audit and fix timing failures in RL agents (HalfCheetah PPO: 990 → 38 with 1-step delay)

We built a tool that measures how badly an RL agent degrades under real-world timing conditions — sensor delays, variable control rates, speed jitter — and optionally fixes it by retraining with a speed-randomized wrapper.

The motivating finding: a standard PPO agent trained on HalfCheetah-v5 scores 990. Add a single observation step of delay (something common in real deployments) and the return drops to 38 — a 96% collapse. At 5x speed, the agent goes negative. Speed jitter causes a 75% drop. All four stress scenarios are statistically significant under 95% bootstrap CI. The model itself is unchanged; only the timing assumptions are violated.

The audit runs three evaluations — deployment robustness (jitter/delay/spike), stress robustness (5x speed), and where architecturally supported, a reliance test (intervention ablation). Results are rated PASS / MILD / DEGRADED / FAIL with numeric ratios. The fix command (`fix-sb3`) rewraps the environment with speed randomization and retrains; after retraining, the delay scenario recovers +146 percentage points and crosses from FAIL to PASS.

There is a working CartPole demo you can run in 30 seconds:

```
pip install "deltatau-audit[demo]"
python -m deltatau_audit demo cartpole
```

For your own SB3 model:

```
deltatau-audit fix-sb3 --algo ppo --model my_model.zip --env HalfCheetah-v5
```

Colab notebook: https://colab.research.google.com/github/maruyamakoju/deltatau-audit/blob/main/notebooks/quickstart.ipynb

Honest limitations: the fix (speed randomization) is a training-time intervention, not an architectural change. It helps but does not fully solve the problem in all cases — 5x speed recovery is partial (+50pp). The stress test is intentionally extreme. The tool currently wraps SB3 models; custom architectures need an adapter. The chain-env results in our internal experiments are less dramatic than HalfCheetah because the task is simpler.

Feedback we are looking for: are the audit thresholds (PASS >= 0.80 deployment ratio) calibrated sensibly for continuous control? Are there timing failure modes we are not testing? Would it be useful to support non-SB3 checkpoints directly?

GitHub: https://github.com/maruyamakoju/deltatau-audit
PyPI: pip install deltatau-audit
