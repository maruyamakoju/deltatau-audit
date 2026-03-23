"""deltatau-audit: 2-axis Time Robustness Audit for RL agents.

Evaluates RL agents along two orthogonal axes:

1. Timing Reliance — Does the agent USE internal timing? (intervention ablation)
2. Timing Robustness — Does the agent SURVIVE timing perturbations? (env wrappers)

This separation prevents confusing "the agent depends on timing" (good for
time-aware agents) with "the agent breaks under timing changes" (bad for
deployment).
"""

from .schema import SCHEMA_VERSION as __schema_version__
from .seed_sweep import run_seed_sweep, summarize_seed_sweep

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "__schema_version__",
    "run_seed_sweep",
    "summarize_seed_sweep",
]

# Optional dm_control adapter (requires shimmy + dm-control)
try:
    from .adapters.dm_control import DMControlSB3Adapter, make_dm_control_env
    __all__ += ["DMControlSB3Adapter", "make_dm_control_env"]
except ImportError:
    pass  # dm_control/shimmy not installed
