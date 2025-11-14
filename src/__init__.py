"""Active Learning package."""

from .config import Config, ModelConfig, TrainingConfig, DataConfig, ActiveLearningConfig
from .models import get_model, get_model_info, list_available_models
from .dataloader import get_dataloaders, get_dataset_info, get_class_names
from .trainer import Trainer
from .strategies import (
    UncertaintySampling,
    MarginSampling,
    DiversitySampling,
    RandomSampling,
    get_strategy,
    list_available_strategies,
)

__version__ = "0.1.0"
__all__ = [
    "Config",
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "ActiveLearningConfig",
    "get_model",
    "get_model_info",
    "list_available_models",
    "get_dataloaders",
    "get_dataset_info",
    "get_class_names",
    "Trainer",
    "UncertaintySampling",
    "MarginSampling",
    "DiversitySampling",
    "RandomSampling",
    "get_strategy",
    "list_available_strategies",
]