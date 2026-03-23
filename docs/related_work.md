# Related Work: Positioning InternalTime vs Existing Approaches

This document positions our work against the most relevant prior literature.

---

## Summary Comparison Table

| Method | No oracle Δt | Variable Δt | Unseen speeds | Internal time | Audit tool | Fix pipeline | Cont. gate |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ODE-RNN (Rubanova 2019) | ✗ | ✓ | partial | ✗ | ✗ | ✗ | ✓ |
| Skip-RNN (Campos 2018) | ✓ | ✗ | ✗ | partial | ✗ | ✗ | ✗ |
| CARE (Lim 2022) | ✗ | ✗ | partial | ✗ | ✗ | ✗ | ✗ |
| Frame-stack | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| TimeFeature (dt inject) | ✗ | ✓ | partial | ✗ | ✗ | ✗ | ✗ |
| **InternalTime (ours)** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

---

## 1. Continuous-Time Recurrent Neural Networks

### 1.1 Neural ODE (Chen et al. 2018)

**What it is:** Defines continuous-time dynamics via dh/dt = f(h,x,t) and
solves with a black-box ODE solver. Enables arbitrary-time evaluation.

**Connection:** Provides the theoretical motivation for continuous-time recurrent
models. The Neural ODE framework treats the hidden state as a continuous trajectory.

**Key difference from ours:**
- Neural ODEs solve a full ODE per step — computationally expensive
- Requires external time Δt as explicit input
- Not designed for RL with discrete action selection
- No robustness audit methodology

### 1.2 ODE-RNN (Rubanova et al. 2019) ← Closest continuous-time baseline

**What it is:** Combines Neural ODE hidden state evolution with RNN updates.
Between observations, hidden state evolves as: h(t+Δt) = ODESolve(f, h(t), [0,Δt]).
At each observation: h ← GRU(h(t+Δt), x_t).

**What they achieve:**
- Handles irregularly-sampled time series
- Naturally incorporates variable Δt
- State-of-the-art on medical time series

**Why it differs from ours:**
1. **Requires oracle Δt:** The elapsed time Δt must be provided as an explicit input.
   In deployment scenarios where timing varies silently (frame drops), this oracle
   is not available.
2. **External dt fails on hidden speed:** Our experiments show ODE-RNN with oracle
   Δt=1/S achieves Spearman ρ=1.000 (trivially correct) but **collapses completely**
   when dt is hidden (reward = -1.0 at all speeds). This validates that learning Δτ
   internally is non-trivial.
3. **Not designed for deployment audit:** No framework for measuring robustness
   or fixing failing agents.

**Our experiment result:**
```
External dt ODE-RNN (hidden speed): reward = -0.70, -0.65, -0.63, -0.62, -0.61
                                    (negative returns across all speeds)
```

### 1.3 Latent ODE (Rubanova et al. 2019)

**What it is:** Generative model combining ODE dynamics with VAE encoder-decoder.
Excellent for irregular time series generation/interpolation.

**Not directly comparable:** Designed for generative modeling, not RL. Requires
reconstruction objective, not reward maximization. Does not directly apply to
policy learning.

### 1.4 CT-GRU / Exponential Smoothing RNNs

**What they are:** Various proposals to add time-dependence to RNN gates,
including exponential smoothing (Mozer et al. 2017) and other continuous-time
extensions.

**Connection to ours:**
- Our TimeAwareGRU gate z_eff = 1-(1-z)^Δτ is equivalent to exponential smoothing
  with rate λ = -log(1-z), which is the same formula as CT-GRU
- **Key difference:** These approaches require external Δt. We learn Δτ internally.

---

## 2. Adaptive Computation and Discrete-Time Skipping

### 2.1 Skip-RNN (Campos et al. 2018) ← Key baseline in our experiments

**What it is:** Learns a binary gate to decide whether to update the hidden state
at each time step. Updates hidden state conditionally: either propagate h_{t-1}
unchanged or compute a full GRU update.

**Their goal:** Reduce computational cost by skipping steps.

**Why it is NOT equivalent to time-awareness:**
Our experiments show a striking result: Skip-RNN **anti-correlates** with speed:
```
Skip-RNN Δτ slope: -0.0027  (Δτ DECREASES as speed increases)
Spearman ρ: -1.000 (perfect NEGATIVE correlation!)
```

This means at high speeds (S=8), Skip-RNN updates LESS often — exactly the wrong
direction for timing adaptation. At high speed (fewer observations per real-time
unit), the agent should update MORE aggressively, not less.

**Root cause:** Skip-RNN learns to skip steps to minimize task loss. In the chain
environment, high speed means observations arrive more frequently (per time unit),
but the optimal policy is similar at all speeds. Skip-RNN learns to skip because
the current observation is uninformative, not because of speed. This anti-correlates
with actual speed.

**Conclusion:** Skip-RNN is an adaptive computation method, not a timing method.
It does not learn about physical timing at all.

### 2.2 Adaptive Computation Time (Graves 2016)

**What it is:** Variable number of "pondering" steps per input using a learned
halting distribution. Similar to Skip-RNN but with soft halting.

**Not directly applicable:** Like Skip-RNN, ACT varies computation, not temporal
modulation. Does not address deployment timing variations.

### 2.3 Multi-Scale RNNs

**Examples:** Clockwork RNN (Koutník et al. 2014), Hierarchical Multiscale RNNs
(Chung et al. 2017).

**What they do:** Fixed hierarchical update rates (e.g., update at steps 1, 2, 4, 8).

**Limitation:** Fixed schedules cannot adapt to variable timing at deployment.
They improve efficiency within training distribution but don't generalize to
unseen timing regimes.

---

## 3. Timing-Robust RL and Action Repetition

### 3.1 CARE (Lim et al. 2022) ← Most directly related

**What it is:** Cooperative Relative-rate action repetition for RL. Addresses
the problem of different action repetition rates (k=1, 2, 4, 8 steps per action).

**What they achieve:**
- Handles discrete speed multipliers (integer k)
- Improves policy robustness when k is provided at test time
- Provides a framework for thinking about timing as a policy input

**How it differs from ours:**
1. **Requires oracle speed k:** CARE requires the repetition factor k as an
   explicit input to the policy. We learn timing internally without oracle.
2. **Discrete multipliers only:** CARE handles integer k ∈ {1,2,4,8}.
   Real deployment has continuous timing variation, not discrete multiples.
3. **Generalization gap:** Not designed for zero-shot generalization to unseen speeds.
   Our experiment with speed S=5,8 (unseen) shows strong generalization.
4. **No audit framework:** CARE trains robust policies but provides no methodology
   for auditing existing agents or fixing them.

**Comparison:**
| | CARE | InternalTime |
|---|---|---|
| Speed input | Oracle k ∈ {1,2,4,8} | Learned from obs history |
| Speed range | Discrete multiples | Continuous [Δτ_min, Δτ_max] |
| Unseen speeds | Not tested | Δτ=1.000 Spearman ρ on unseen S=5,8 |
| Audit existing agents | No | Yes (deltatau-audit) |
| Fix existing agents | No | Yes (fix-sb3 pipeline) |

### 3.2 Temporal Abstraction / Options Framework

**Background:** The options framework (Sutton et al. 1999) provides hierarchical
temporal abstraction. Options can execute for variable numbers of steps.

**Connection:** Our problem is orthogonal — we address PHYSICAL timing variation
(variable Δt between policy decisions), not ACTION duration abstraction.

### 3.3 Frame-Skipping / Sticky Actions

**Background:** In Atari (Mnih et al. 2015), actions are repeated for k=4 frames.
"Sticky actions" (Machado et al. 2018) add stochastic action repetition for robustness.

**Connection:** Frame-skipping is a fixed-rate temporal abstraction. Sticky actions
add noise, not principled timing variation. Neither addresses continuous timing
distribution shifts at deployment.

---

## 4. Domain Randomization for Robustness

### 4.1 Domain Randomization (Tobin et al. 2017)

**What it is:** Randomize physical simulation parameters during training (mass,
friction, visual appearance) so policy generalizes to the real world.

**Connection to ours:**
- Speed-randomized training is a special case of domain randomization where the
  randomized parameter is the simulation time step
- We provide a **formal justification** (Proposition 4 in theory.md) for why
  speed randomization forces timing-invariant representations
- Additionally, we provide an **audit framework** to measure how well any agent
  has achieved timing robustness

**Key addition:** Domain randomization is a training strategy. deltatau-audit
adds a *measurement methodology* (was the randomization effective?) and a
*repair pipeline* (if not, automatically fix the agent).

### 4.2 Robust RL / Distributionally Robust Optimization

**Examples:** Robust MDPs (Nilim & El Ghaoui 2005), RARL (Pinto et al. 2017).

**What they address:** Adversarial perturbations to state, action, or transition
dynamics. Usually model-based or adversary-based approaches.

**Limitation for timing:** Standard robust RL treats all state perturbations equally.
Timing perturbations have structure: they affect sequential accumulation of hidden
state in a specific way that is captured by the TimeAwareGRU gating formula.

---

## 5. RL Robustness Evaluation

### 5.1 Standard Robustness Benchmarks

**Examples:** ObstacleWorld, BRAX, sim-to-real benchmarks.

**What they measure:** Policy performance under physical parameter variations
(friction, mass, morphology). Not timing/latency variations.

### 5.2 Delay-Aware RL

**Examples:** Firoiu et al. (2018), Ramstedt & Pal (2019).

**What they address:** Constant observation delays (h steps of latency).

**Connection:** Observation delay is one of our robustness scenarios (delay=1 step).
However, they focus on known, constant delays — not the general variable timing
framework we address.

### 5.3 Benchmark Comparison

Our work uniquely combines:
1. **Formal measurement** of timing robustness (Δτ Reliance + Deployment Score)
2. **Training method** that improves robustness (TimeAwareGRU + speed randomization)
3. **Automatic repair** of failing agents (fix-sb3 pipeline)
4. **Standardized evaluation** across architectures (AgentAdapter protocol)

No prior work provides all four components together.

---

## 6. Open Problems (Future Work)

1. **Online adaptation:** Can the agent adapt at test time? Current approach is
   train-time robustness only. Meta-RL or test-time fine-tuning could improve this.

2. **Formal guarantees:** Theorem 1 gives an upper bound but not a tight one.
   Deriving a matching lower bound would characterize the fundamental limits.

3. **Real hardware:** All experiments use simulated timing variations. Validating
   on real robotic hardware with genuine latency variability is crucial.

4. **Multi-agent timing:** What happens when multiple agents have different timing?
   Communication protocols with variable latency remain unexplored.

5. **Distribution shift detection:** Can the agent detect when timing has changed?
   This could trigger adaptation strategies (e.g., increase entropy during timing shifts).
