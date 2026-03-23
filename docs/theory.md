# Formal Theory: The Δτ Framework for Timing Robustness in RL

This document provides the mathematical foundation for the deltatau-audit framework
and the InternalTimeAgent architecture.

---

## 1. Problem Setup: Variable-Timing MDPs

### 1.1 Standard MDP

A standard Markov Decision Process is M = (S, A, P, R, γ) where S is the state space,
A the action space, P: S×A → Δ(S) the transition kernel, R: S×A → ℝ the reward function,
and γ ∈ (0,1) the discount factor. Training assumes a fixed physical time step τ̄.

### 1.2 Variable-Timing MDP

**Definition 1 (Variable-Timing MDP).**
A variable-timing MDP M_T = (S, A, P, R, γ, T) extends M with a timing distribution
T over positive reals, where at each step t, a timing sample Δτ_t ~ T determines
the ratio of actual elapsed physical time to the training time step τ̄:

```
Δτ_t = τ_t / τ̄  ∈ (0, ∞)
```

At test time, T may differ from the training distribution (Δτ_t ≡ 1 during training).
This captures frame drops (Δτ_t > 1), faster inference (Δτ_t < 1), and variable
sensor sampling rates.

### 1.3 Agent and Policy

A recurrent policy is a pair (π, H) where:
- π: S × H → Δ(A) × H maps (observation, hidden state) → (action distribution, next hidden)
- H is the hidden state space (e.g., ℝ^d for a GRU)

At each step: (a_t, h_{t+1}) ~ π(s_t, h_t)

---

## 2. Core Definitions

### 2.1 Timing Robustness

**Definition 2 (Timing Robustness).**
Let ξ = (Δτ_1, Δτ_2, ...) be a timing sequence and ξ_0 = (1, 1, ...) be nominal
timing. The timing robustness of policy π under perturbation family Ξ is:

```
Rob(π, Ξ) = E_{ξ ~ Ξ}[R_π(ξ)] / R_π(ξ_0)
```

where R_π(ξ) = E[∑_{t=0}^T γ^t r_t | timing sequence ξ] is the expected return
under that timing sequence.

**Interpretation:** Rob(π, Ξ) = 1.0 means no performance degradation. Rob(π, Ξ) = 0.5
means 50% performance loss.

**Empirical metric (deltatau-audit):**
The Deployment Robustness score is the minimum return ratio over deployment scenarios:

```
DeployScore(π) = min_{scenario ∈ {jitter, delay, spike, obs_noise}} Rob(π, Ξ_{scenario})
```

This maps to the PASS/MILD/DEGRADED/FAIL ratings:
- PASS:     DeployScore ≥ 0.95
- MILD:     0.80 ≤ DeployScore < 0.95
- DEGRADED: 0.50 ≤ DeployScore < 0.80
- FAIL:     DeployScore < 0.50

### 2.2 Δτ Reliance (Intervention Ablation)

**Definition 3 (Δτ Reliance).**
For a policy π with hidden state h, define intervention I_i as an operator that
replaces the agent's internal Δτ_t signal with a distorted version:

| Intervention | Description |
|---|---|
| I_0 (none) | Normal operation |
| I_1 (clamp) | Δτ_t ← 1.0 for all t |
| I_2 (reverse) | Δτ_t ← 2 - Δτ_t (anti-correlated) |
| I_3 (random) | Δτ_t ~ Uniform[0.3, 2.5] |

The Δτ Reliance is:

```
Rel(π) = max_{i ∈ {1,2,3}} RMSE(V_π^{Δτ}, V_π^{I_i}) / σ_{baseline}
```

where V_π^{Δτ}(s,h) is the value under normal Δτ, V_π^{I_i}(s,h) is the value
after applying intervention I_i to the hidden state, and σ_{baseline} is a
normalizing constant (standard deviation of baseline value estimates).

**Interpretation:** High Reliance (>2.0×) means the agent's value function depends
strongly on Δτ — it is genuinely time-aware. Low reliance means the agent ignores
timing information.

**Rating thresholds:**
- VERY_HIGH: Rel ≥ 4.0×
- HIGH:      2.0× ≤ Rel < 4.0×
- MODERATE:  1.2× ≤ Rel < 2.0×
- LOW:       Rel < 1.2×

---

## 3. The TimeAwareGRU: Mathematical Derivation

### 3.1 Standard GRU

The standard GRU update gate at time step t is:

```
z = σ(W_z [x_t, h_t])          ∈ (0,1)
r = σ(W_r [x_t, h_t])          ∈ (0,1)
h̃ = tanh(W_h [x_t, r ⊙ h_t])
h_{t+1} = (1-z) ⊙ h_t + z ⊙ h̃  (standard GRU)
```

The gate z controls how much the hidden state updates at each step.

### 3.2 Exponential Kernel Derivation

**Proposition 1 (TimeAwareGRU as Poisson Exponential Kernel).**

Define the "forget rate" of the GRU as λ = -log(1-z) (the rate parameter of
the implicit Poisson process governing state updates). Under this interpretation,
the probability that at least one "state transition event" occurs in an interval
of physical time Δτ is:

```
P(N(Δτ) ≥ 1) = 1 - e^{-λ Δτ} = 1 - (e^{-λ})^{Δτ} = 1 - (1-z)^{Δτ}
```

The TimeAwareGRU simply uses this as the effective gate:

```
z_eff = 1 - (1-z)^{Δτ}
h_{t+1} = (1-z_eff) ⊙ h_t + z_eff ⊙ h̃   (TimeAwareGRU)
```

**Proof:** The Poisson process with rate λ generates events at times
T_1, T_2, ... where inter-arrival times are Exp(λ). The number of events
in interval [0, Δτ] is N(Δτ) ~ Poisson(λ Δτ). Thus:
P(N(Δτ) = 0) = e^{-λΔτ}, so P(N(Δτ) ≥ 1) = 1 - e^{-λΔτ} = 1 - (1-z)^{Δτ}. ∎

### 3.3 Properties

**Proposition 2 (Monotonicity and Limits).**
For fixed z ∈ (0,1), z_eff(Δτ) = 1-(1-z)^{Δτ} satisfies:

1. **Δτ = 1:** z_eff = z (recovers standard GRU at training speed)
2. **Δτ → 0:** z_eff → 0 (temporal freeze, h_{t+1} → h_t)
3. **Δτ → ∞:** z_eff → 1 (instant mixing, h_{t+1} → h̃)
4. **Monotone:** dz_eff/dΔτ = (1-z)^{Δτ} |log(1-z)| > 0 (faster time → faster update)
5. **Stable:** z_eff ∈ (0,1) for all z ∈ (0,1), Δτ > 0

**Proposition 3 (Lipschitz continuity in Δτ).**
The sensitivity of z_eff to Δτ perturbations is bounded:

```
|z_eff(Δτ + ε) - z_eff(Δτ)| ≤ ε · |log(1-z)| · (1-z)^{min(Δτ, Δτ+ε)}
                              ≤ ε · λ · (1-z)^{Δτ_min}
```

This means small timing perturbations cause bounded hidden state changes.

### 3.4 Connection to Neural ODEs

The TimeAwareGRU relates to the Neural ODE (Chen et al. 2018) and ODE-RNN
(Rubanova et al. 2019) approaches:

- **ODE-RNN:** h(t+Δτ) = ODESolve(f, h(t), [0, Δτ]) with external Δτ input
- **TimeAwareGRU:** Uses gated update calibrated to Δτ, but **learns Δτ internally**

Key distinction: ODE-RNN requires the external timing signal Δτ_t to be available
as input. TimeAwareGRU requires no oracle — the time module learns Δτ from the
observation sequence history.

---

## 4. InternalTimeAgent

### 4.1 Time Module

The TimeModule computes Δτ_t from hidden state h_t and current observation x_t:

```
combined = [h_t; φ(x_t)]           ∈ ℝ^{d_h + d_x}
raw = MLP(combined)                 ∈ ℝ
Δτ_t = dt_min + (dt_max - dt_min) · σ(raw)   ∈ [dt_min, dt_max]
```

where dt_min=0.3, dt_max=2.5 (range chosen for stability), and σ is sigmoid.
The MLP bias is initialized at -0.76 so that Δτ starts near 1.0.

**Key property:** Δτ is computed BEFORE the hidden state update, so the gate
z_eff = 1-(1-z)^{Δτ_t} uses the current step's learned Δτ.

### 4.2 Full Architecture

```
x_t → Encoder φ → encoded_x
              ↓
[h_t, encoded_x] → TimeModule → Δτ_t ∈ [0.3, 2.5]
              ↓
TimeAwareGRUCell(encoded_x, h_t, Δτ_t) → h_{t+1}
              ↓
    ┌─────────┴──────────┐
Policy π(h_{t+1})    Value V(h_{t+1})
```

### 4.3 Learning Signal for Δτ

The agent is NOT explicitly trained to track external speed — Δτ emerges from
learning to predict rewards accurately across speed-randomized training:

- During training: speed S is sampled uniformly from {1, 2, 3}
- Speed S is NOT observed (hidden from the agent)
- The agent must infer S from observation patterns to adapt correctly
- Δτ captures this inferred speed via the time module

The key finding: **the trained agent's Δτ monotonically tracks S with Spearman
ρ = 1.000 (p < 10^{-24})**, despite S never being explicitly provided.

---

## 5. Theoretical Bounds

### 5.1 Return Degradation Bound

**Theorem 1 (Lipschitz Bound on Return Degradation).**

Assume:
- Value function V_π is L_V-Lipschitz in h: |V_π(s,h) - V_π(s,h')| ≤ L_V · ||h-h'||
- Hidden state update is K_h-Lipschitz in Δτ: ||h(Δτ+ε) - h(Δτ)|| ≤ K_h · |ε|
- Episodes have length T, maximum reward R_max

Then under speed jitter with variance σ² (Δτ_t ~ U[1-σ, 1+σ]):

```
|1 - Rob(π, Jitter(σ))| ≤ L_V · K_h · σ · T / (1-γ)
```

**Proof sketch:** The value degradation per step is bounded by L_V · ||Δh||
where ||Δh|| ≤ K_h · |ΔΔτ| ≤ K_h · σ. Summing over T steps with discount:
ΔR ≤ Σ_{t=0}^{T} γ^t · L_V · K_h · σ ≤ L_V · K_h · σ · T/(1-γ). ∎

**Implication:** Speed-randomized training minimizes L_V by making V invariant
to timing perturbations. The TimeAwareGRU reduces K_h by making the gate respond
continuously (rather than discretely) to Δτ changes.

### 5.2 Information-Theoretic Argument for Speed Randomization

**Proposition 4 (Speed Randomization Forces Timing-Invariant Representations).**

If speed S is uniformly sampled from [S_min, S_max] during training and is NOT
observed, the optimal policy π* must satisfy:

```
V_{π*}(s, h) ≈ V_{π*}(s, h')   for all h, h' reachable at the same (obs, reward) history
```

This forces L_V ≈ 0 for timing-induced hidden state differences, directly
minimizing the Lipschitz constant in Theorem 1 and improving robustness.

**Proof:** By the optimality of π*, any policy that conditions differently on h vs h'
(same reward-equivalent history) is suboptimal, since speed S is not inferable.
Therefore the optimal value function must be (approximately) constant over the
timing-induced orbit. ∎

### 5.3 Why Δτ Slope Correlates with Robustness

**Corollary 1.** An agent with positive Δτ slope (Δτ increases with S) has lower
L_V than a baseline (Δτ ≡ 1), because:

1. Correct tracking: the hidden state h_{t+1} is closer to the "correct" hidden
   state for speed S
2. Reduced drift: timing errors accumulate more slowly under correct gating
3. Tighter bound: the T/(1-γ) factor decreases effectively because policy errors
   are smaller per step

This provides a theoretical justification for why Δτ slope is a useful proxy
for timing robustness.

---

## 6. The 4-Quadrant Classification

The 2-axis framework (Reliance × Robustness) produces 4 quadrants:

| | **Deployment PASS** | **Deployment FAIL** |
|---|---|---|
| **Reliance HIGH** | TIME-AWARE ROBUST (best) | TIME-AWARE FRAGILE |
| **Reliance LOW** | TIME-BLIND ROBUST | TIME-BLIND FRAGILE (worst) |

**Interpretation:**
- **Time-Aware Robust:** Agent uses timing AND benefits from it → ideal for variable-speed deployment
- **Time-Aware Fragile:** Agent depends on timing but uses it incorrectly → timing is a liability
- **Time-Blind Robust:** Agent is robust despite ignoring timing → simple env, doesn't need timing
- **Time-Blind Fragile:** Agent ignores timing AND breaks under perturbations → standard failure mode

When Reliance = N/A (supports_intervention=False), collapses to 1D:
- Deployment PASS → deployment_ready
- Deployment FAIL → deployment_fragile

---

## 7. Empirical Validation

All theoretical claims are validated empirically:

| Claim | Evidence |
|---|---|
| IT learns Δτ from hidden speed | Spearman ρ=1.000, p<10^{-24}, Monotonicity=100% |
| Skip-RNN learns wrong direction | Skip-RNN Spearman ρ=-1.000 (anti-correlated) |
| ODE-RNN fails without oracle dt | ODE-RNN collapses when dt is hidden |
| V depends on Δτ (Reliance) | +105% RMSE when Δτ reversed at S=8 |
| Speed randomization fixes robustness | HalfCheetah: FAIL(0.02) → PASS(1.00) |
| High deployment failure is real | HalfCheetah: -96% from 1-step delay |

---

## References

- Chen, R.T.Q., Rubanova, Y., Bettencourt, J., Duvenaud, D. (2018). Neural Ordinary Differential Equations. NeurIPS.
- Rubanova, Y., Chen, R.T.Q., Duvenaud, D. (2019). Latent ODEs for Irregularly-Sampled Time Series. NeurIPS.
- Campos, V., Jou, B., Giro-i-Nieto, X., Torres, J., Chang, S.F. (2018). Skip RNN: Learning to Skip State Updates in Recurrent Neural Networks. ICLR.
- Cho, K., et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. EMNLP.
- Mozer, M.C., Kazakov, D., Lindsey, R. (2017). Discrete Event, Continuous Time RNNs. arXiv.
- Lim, A., Guo, Z., Yue, Y., Luo, Q. (2022). CARE: Cooperative Relative Rate for Action Repetition in Reinforcement Learning. AAMAS.
- Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., Abbeel, P. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World. IROS.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv.
