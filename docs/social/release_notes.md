# v1.0.0 Release Notes

## deltatau-audit v1.0.0 — Timing Robustness Audit for RL Agents

**Find timing bugs in your RL agents before deployment breaks them.**

### Highlights

- **One-command audit**: `deltatau-audit audit-sb3 model.zip --env CartPole-v1`
- **6 timing scenarios**: speed change, jitter, delay, spikes, noise, adversarial
- **Auto-fix**: retrain fragile models with timing augmentation
- **CI integration**: GitHub Actions — fail the build if your agent isn't robust
- **Formal verification**: spectral norms, IBP, CROWN with mathematical guarantees
- **612 tests passing**, Python 3.10-3.12

### Quick Start

```bash
pip install deltatau-audit
deltatau-audit demo  # see it in action, no model needed
```

### Supported Frameworks

| Framework | Command |
|-----------|---------|
| Stable-Baselines3 | `audit-sb3` |
| CleanRL | `audit-cleanrl` |
| HuggingFace Hub | `audit-hf` |
| Any PyTorch | `audit` |

### What's New in v1.0.0

**Core Algorithms (Publication-Grade)**
- Adaptive Computation Time (ACT) with geometric halting prior, multi-head deliberation
- Dreamer v3 RSSM with categorical latents, symlog, KL balancing
- AlphaZero/MuZero-grade MCTS with PUCT, progressive widening, Gumbel search

**Formal Verification (6-Level Hierarchy)**
- L1: Empirical Jacobian sampling
- L2: Clopper-Pearson statistical bounds
- L3: Interval Bound Propagation (IBP)
- L4: Spectral norm Lipschitz bounds
- L5: CROWN/alpha-CROWN linear relaxation
- Hölder continuity analysis

**Audit Framework**
- Real temporal stress testing (not stubs)
- Multi-scale horizon analysis with CUSUM phase detection
- Adversarial timing attacks (value-based + gradient PGD)
- Foundation model adapters (Octo VLA, Decision Transformer)

### Install

```bash
pip install deltatau-audit        # core
pip install deltatau-audit[sb3]   # + Stable-Baselines3
pip install deltatau-audit[demo]  # + bundled demos
pip install deltatau-audit[dev]   # + development tools
```

### Links

- [GitHub](https://github.com/maruyamakoju/deltatau-audit)
- [PyPI](https://pypi.org/project/deltatau-audit/)
- [Documentation](https://github.com/maruyamakoju/deltatau-audit#readme)
