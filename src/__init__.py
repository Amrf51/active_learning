"""Active Learning package for vehicle classification."""

from .config import Config, ModelConfig, TrainingConfig, DataConfig, ActiveLearningConfig
from .models import get_model, get_model_info, list_available_models
from .dataloader import get_datasets, get_dataloaders, get_dataset_info, get_class_names
from .trainer import Trainer
from .data_manager import ALDataManager
from .active_loop import ActiveLearningLoop
from .strategies import (
    uncertainty_least_confidence,
    uncertainty_entropy,
    margin_sampling,
    random_sampling,
    get_strategy,
    list_available_strategies,
)

__version__ = "0.2.0"
__all__ = [
    # Config
    "Config",
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "ActiveLearningConfig",
    # Models
    "get_model",
    "get_model_info",
    "list_available_models",
    # Data
    "get_datasets",
    "get_dataloaders",
    "get_dataset_info",
    "get_class_names",
    # Training
    "Trainer",
    # Active Learning
    "ALDataManager",
    "ActiveLearningLoop",
    # Strategies
    "uncertainty_least_confidence",
    "uncertainty_entropy",
    "margin_sampling",
    "random_sampling",
    "get_strategy",
    "list_available_strategies",
]