"""Internal Time RL — Training infrastructure.

Exports:
    WorldModelTrainer: Publication-grade trainer for TemporalRSSM.
    TrainerConfig: Full configuration dataclass for the trainer.
    GradStats: Gradient statistics per parameter group.
"""

from .world_model_trainer import WorldModelTrainer, TrainerConfig, GradStats

__all__ = ["WorldModelTrainer", "TrainerConfig", "GradStats"]
