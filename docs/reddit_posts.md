# Reddit Posts for deltatau-audit

## r/reinforcementlearning

**Title:** I measured how badly standard PPO breaks under timing changes — and built a tool to fix it (with data)

I ran some experiments on HalfCheetah-v5 to quantify something that rarely gets measured: what happens to a trained PPO agent when the timing conditions at deployment differ from training? The results were worse than I expected.

**The numbers:**

- Standard PPO, trained at fixed dt, then evaluated with a 1-step observation delay: return drops from 990 to 38. That's **-96%**.
- Same agent at **5x simulation speed**: return goes **negative**. -109% relative to baseline.
- Speed jitter (random step-to-step timing variation): **-75%**.

All four tested scenarios are statistically significant at 95% bootstrap CI. This isn't noise.

**Why does this happen?**

Standard recurrent policies (GRU, LSTM) treat every timestep as equally spaced. When you train at a fixed dt and then deploy into a world where dt is variable — latency spikes, faster sim, slower hardware — the hidden state dynamics break. The agent's "sense of time" is baked in as a constant, so even small perturbations to timing compound across an episode.

This is a well-known problem in principle, but I didn't find many tools that actually *measure* it systematically on your specific checkpoint, let alone fix it automatically.

**The fix:**

Speed-randomized training. During training, randomly sample the simulation speed each episode so the agent learns to be robust to dt variation. After doing this:

- Observation delay: **+146 percentage points**
- Speed jitter: **+93pp**
- 5x speed: **+50pp**
- Deployment rating: **FAIL → PASS**

**The tool:**

I packaged this into `deltatau-audit`. It audits your checkpoint across timing perturbation scenarios, then optionally retrains with speed randomization and re-audits, giving you a Before/After report.

```bash
pip install deltatau-audit

# Audit + fix in one command (SB3 models):
deltatau-audit fix-sb3 --algo ppo --model my_model.zip --env HalfCheetah-v5

# No GPU, no MuJoCo? Try the CartPole demo:
pip install "deltatau-audit[demo]"
python -m deltatau_audit demo cartpole
# CartPole demo: Deployment score 0.23 (FAIL) → 0.62 (DEGRADED) in ~30 seconds
```

You can also drop it into CI so timing regressions get caught before they ship:

```yaml
# .github/workflows/audit.yml
- uses: maruyamakoju/deltatau-audit@main
```

It's open source, free, pip installable, no proprietary dependencies. Works with any SB3 checkpoint out of the box.

GitHub: https://github.com/maruyamakoju/deltatau-audit

Happy to answer questions about methodology — the audit uses intervention ablation, jitter/delay/spike scenarios, and stress tests (5x speed), each with bootstrap CIs.

---

## r/robotics

**Title:** Sim-to-real timing gap: quantifying and fixing it automatically (with before/after numbers)

One of the more underappreciated failure modes in sim-to-real transfer is timing mismatch. Simulation runs at a fixed, clean dt. Real robots don't — there's sensor latency, compute jitter, actuator delays, and variable control loop timing. Most training pipelines don't account for this, and most evaluation pipelines don't measure it.

I built a tool to do both.

**The core problem:**

A standard PPO agent trained in simulation at fixed dt implicitly assumes constant, uniform time steps. Its recurrent hidden state evolves under that assumption. When you deploy to real hardware — or even just change sim speed or add observation latency — the hidden state dynamics are wrong from step one, and the error accumulates.

To put numbers on it, I evaluated a standard PPO agent on HalfCheetah-v5 under controlled timing perturbations:

- **1-step observation delay**: return drops from 990 to 38 (**-96%**)
- **5x simulation speed**: return goes negative (**-109%**)
- **Speed jitter**: **-75%**

All results carry 95% bootstrap confidence intervals. A single step of observation delay — something trivially common on real hardware — nearly zeros out the agent's performance.

**The fix:**

Speed-randomized training during the sim phase. The agent sees variable dt during training and learns policies that generalize across timing conditions. After retraining:

- Delay robustness: **+146 percentage points**
- Jitter robustness: **+93pp**
- 5x speed robustness: **+50pp**
- Overall deployment rating: **FAIL → PASS**

**The tool — `deltatau-audit`:**

```bash
pip install deltatau-audit

# Audit a checkpoint, then retrain with speed randomization, then re-audit:
deltatau-audit fix-sb3 --algo ppo --model my_model.zip --env HalfCheetah-v5

# CartPole demo (no GPU, no MuJoCo needed):
pip install "deltatau-audit[demo]"
python -m deltatau_audit demo cartpole
```

The tool works with Stable-Baselines3 checkpoints out of the box. IsaacLab / RSL-RL support is included for sim-to-real workflows where you're already using those frameworks. The audit runs three classes of test: deployment perturbations (jitter, delay, spike), stress tests (5x speed), and intervention ablation to measure time-reliance structure.

You get a structured Before/After report and a deployment readiness score. It also integrates into CI via GitHub Actions:

```yaml
- uses: maruyamakoju/deltatau-audit@main
```

Open source, no license cost, no proprietary sim dependency for the audit itself.

GitHub: https://github.com/maruyamakoju/deltatau-audit

If you're doing sim-to-real with recurrent policies and haven't audited timing robustness, the -96% delay number is probably worth taking seriously. The fix isn't complicated — it's mostly a training distribution problem — but you need to measure it first to know where you stand.
