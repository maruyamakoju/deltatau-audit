"""deltatau-audit: 2-axis Time Robustness Audit for RL agents.

Evaluates RL agents along two orthogonal axes:

1. Timing Reliance — Does the agent USE internal timing? (intervention ablation)
2. Timing Robustness — Does the agent SURVIVE timing perturbations? (env wrappers)

This separation prevents confusing "the agent depends on timing" (good for
time-aware agents) with "the agent breaks under timing changes" (bad for
deployment).
"""

__version__ = "0.6.1"
