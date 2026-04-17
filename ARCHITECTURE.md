# deltatau-audit: Architecture Overview

This document describes the unified, multi-axis auditing architecture implemented in `deltatau-audit` v1.0.0.

## Core Design Principles

1.  **Axis-based Evaluation**: Audits are organized into independent axes (e.g., Timing Reliance, Temporal Robustness, Deliberative Efficiency).
2.  **Protocol-First**: Agents and Auditors interact through strictly defined Protocols (`AgentAdapter`, `Auditor`).
3.  **Unified Execution**: All episodes are run through a central `EpisodeRunner` to ensure consistency in state management and metric collection.
4.  **Serialization Ready**: Results are stored in a strictly typed `AuditReport` schema that is JSON-serializable and validatable.

---

## Component Stack

### 1. Agent Adapters (`deltatau_audit.adapters`)
Adapters wrap agents from different frameworks (SB3, CleanRL, custom) into a standard interface.
- **`AgentAdapter`**: The unified protocol.
- **`act()`**: Returns `(action, info_dict)`. The `info_dict` contains `value`, `dt`, and optional `reasoning_trace`.
- **`reset_internal_state()`**: Resets RNN hidden states or MCTS trees.
- **`rerun_with_dt()`**: Enables causal intervention by forcing a specific $\Delta\tau$.

### 2. Episode Execution Engine (`deltatau_audit.core.runner`)
- **`EpisodeRunner`**: Manages the agent-environment loop.
- Supports parallel execution via `ThreadPoolExecutor`.
- Captures reasoning traces and applies interventions if requested by the auditor.
- Computes discounted returns and value-error metrics (RMSE, MAE, Bias).

### 3. Auditors (`deltatau_audit.auditors`)
Class-based evaluating logic.
- **`RelianceAuditor` (Axis 1)**: Measures how much the agent's value function depends on internal timing via ablation.
- **`RobustnessAuditor` (Axis 2)**: Tests performance under realistic timing perturbations (Jitter, Delay, etc.).
- **`ReasoningAuditor` (Axis 3)**: Evaluates deliberative efficiency (pondering depth vs performance/uncertainty).
- **`TemporalHorizonAuditor`**: Tests cascading scenarios over long horizons.

### 4. Orchestration (`deltatau_audit.core.session`)
- **`AuditSession`**: The top-level manager.
- Coordinates multiple auditors and aggregates results into a single `AuditReport`.
- Manages output directories and artifact persistence.

---

## Data Flow

1.  **Setup**: CLI/API creates an `AgentAdapter` and an `AuditSession`.
2.  **Configuration**: Auditors are initialized with specific parameters (episodes, sampling mode).
3.  **Execution**: `session.run_full_audit()` calls each auditor's `run()` method.
4.  **Rollouts**: Auditors use `EpisodeRunner` to perform parallel rollouts.
5.  **Aggregation**: `EpisodeResult` objects are converted to `AuditStageResult` metrics.
6.  **Reporting**: A final `AuditReport` is generated, saved as JSON, and used to produce HTML/Markdown reports.

---

## Roadmap & Evolution (v1.0.0 Status)

The 10-axis roadmap is now **fully functional** as of Cycle 11516:

1.  **World Models**: Stable foundation for planning.
2.  **Consistency Distillation**: Distilling slow planning into fast reflexive policies.
3.  **Multi-Scale Deliberation**: Hierarchical temporal abstraction.
4.  **Certified MCTS**: Lipschitz-bounded safety.
5.  **ACT (Adaptive Computation Time)**: Subjective time-aware pondering.
6.  **Adversarial Audit Synthesis**: Co-evolving auditors and agents.
7.  **Meta-Policy Distillation**: **BREAKTHROUGH.** Universal Adversaries with 100% zero-shot kill rate on MountainCar-v0 (Composite: 200.61).
8.  **Temporal Subjectification**: Neural ODE-based internal state dynamics.
9.  **Recursive Self-Architecture**: **VALIDATED.** Dynamic scaling of ODE resolution based on state complexity.
10. **Causal Temporal Reasoning**: **FUNCTIONAL.** System 1/System 2 counterfactual unrolling with unblocked adapter initialization.

---

## Core Design Principles

- **Axis-based Evaluation**: Audits are organized into independent axes (e.g., Timing Reliance, Temporal Robustness, Deliberative Efficiency).
- **Protocol-First**: Agents and Auditors interact through strictly defined Protocols (`AgentAdapter`, `Auditor`).
- **Unified Execution**: All episodes are run through a central `EpisodeRunner` to ensure consistency in state management and metric collection.
- **Serialization Ready**: Results are stored in a strictly typed `AuditReport` schema defined in `deltatau_audit/schema.py`.
- **Level**: `ReliabilityLevel` (Certified, Robust, Degraded, Unreliable).
- **Metrics**: `MetricValue` with confidence intervals.
- **Capabilities**: `TemporalCapability` detected from the agent.
