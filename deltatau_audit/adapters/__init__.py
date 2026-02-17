from .base import AgentAdapter
from .internal_time import InternalTimeAdapter
from .generic import GenericRecurrentAdapter

__all__ = [
    "AgentAdapter",
    "InternalTimeAdapter",
    "GenericRecurrentAdapter",
]

# SB3 adapters are optional (require stable-baselines3 / sb3-contrib)
try:
    from .sb3 import SB3Adapter
    __all__.append("SB3Adapter")
except ImportError:
    pass

try:
    from .sb3_recurrent import SB3RecurrentAdapter
    __all__.append("SB3RecurrentAdapter")
except ImportError:
    pass
