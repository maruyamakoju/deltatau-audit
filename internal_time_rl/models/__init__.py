"""Internal Time RL — Model architectures.

All model classes are importable directly from ``internal_time_rl.models``:

    >>> from internal_time_rl.models import InternalTimeAgent, TemporalRSSM
    >>> from internal_time_rl.models import NeuralODEAgent, TemporalDiffusionModel

The ``__model_registry__`` dict provides name-to-class mapping for
dynamic loading and configuration-driven instantiation:

    >>> from internal_time_rl.models import __model_registry__
    >>> ModelClass = __model_registry__["NeuralODEAgent"]
    >>> agent = ModelClass(obs_dim=4, act_dim=2)
"""

# --- Core modules (always available) ---
from .encoder import ObservationEncoder
from .time_module import TimeModule, TimeAwareGRUCell, NeuralODETransition
from .policy import InternalTimeAgent

# --- Advanced architectures ---
from .advanced import (
    LiquidTimeCell,
    ContinuousRoPE,
    ContinuousPositionalEmbedding,
    TimeAwareAttention,
    TimeAwareTransformerBlock,
    DecisionTransformerInternalTime,
    TemporalDiffusionModel,
    SinusoidalTimestepEmbedding,
    TemporalUNetBlock,
    ODESolver,
)

# --- Continuous-time models ---
from .continuous import (
    LTCAgent,
    NeuralODEAgent,
    ContinuousNormalizingFlowTiming,
    ODEFunc,
    ConditionedODEFunc,
    CNFTimingDynamics,
    odeint,
    odeint_adjoint,
)

# --- World model ---
from .world_model import TemporalWorldModel, TemporalRSSM


# ── Model registry for dynamic loading ─────────────────────────────────────

__model_registry__ = {
    # Core agents
    "InternalTimeAgent": InternalTimeAgent,
    "LTCAgent": LTCAgent,
    "NeuralODEAgent": NeuralODEAgent,

    # Transformer architectures
    "DecisionTransformerInternalTime": DecisionTransformerInternalTime,

    # World models
    "TemporalRSSM": TemporalRSSM,
    "TemporalWorldModel": TemporalWorldModel,

    # Diffusion models
    "TemporalDiffusionModel": TemporalDiffusionModel,

    # Components (for advanced users)
    "LiquidTimeCell": LiquidTimeCell,
    "TimeAwareAttention": TimeAwareAttention,
    "ContinuousNormalizingFlowTiming": ContinuousNormalizingFlowTiming,
    "ODEFunc": ODEFunc,
    "ConditionedODEFunc": ConditionedODEFunc,
}

__all__ = list(__model_registry__.keys()) + [
    "ObservationEncoder",
    "TimeModule",
    "TimeAwareGRUCell",
    "NeuralODETransition",
    "ContinuousRoPE",
    "ContinuousPositionalEmbedding",
    "TimeAwareTransformerBlock",
    "SinusoidalTimestepEmbedding",
    "TemporalUNetBlock",
    "ODESolver",
    "CNFTimingDynamics",
    "odeint",
    "odeint_adjoint",
    "__model_registry__",
]
