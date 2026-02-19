from .base import AgentAdapter
from .generic import GenericRecurrentAdapter
from .internal_time import InternalTimeAdapter

__all__ = [
    "AgentAdapter",
    "InternalTimeAdapter",
    "GenericRecurrentAdapter",
]

# SB3 adapters are optional (require stable-baselines3 / sb3-contrib)
try:
    from .sb3 import SB3Adapter  # noqa: F401
    __all__.append("SB3Adapter")
except ImportError:
    pass

try:
    from .sb3_recurrent import SB3RecurrentAdapter  # noqa: F401
    __all__.append("SB3RecurrentAdapter")
except ImportError:
    pass

# CleanRL adapter (requires torch, which is a core dep)
try:
    from .cleanrl import CleanRLAdapter  # noqa: F401
    __all__.append("CleanRLAdapter")
except ImportError:
    pass

# Generic PyTorch adapter (requires torch)
try:
    from .torch_policy import TorchPolicyAdapter  # noqa: F401
    __all__.append("TorchPolicyAdapter")
except ImportError:
    pass
