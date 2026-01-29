"""Backend components for Active Learning Dashboard."""

from .trainer import Trainer
from .active_loop import ActiveLearningLoop
from .data_manager import ALDataManager
from .strategies import (
    uncertainty_least_confidence,
    uncertainty_entropy,
    margin_sampling,
    random_sampling,
    get_strategy,
    list_available_strategies,
)
from .models import get_model, get_model_info, list_available_models
from .dataloader import get_datasets, get_dataloaders, get_class_names, get_dataset_info

__version__ = "1.0.0"
__all__ = [
    # Training
    "Trainer",
    # Active Learning
    "ActiveLearningLoop",
    "ALDataManager",
    # Strategies
    "uncertainty_least_confidence",
    "uncertainty_entropy",
    "margin_sampling",
    "random_sampling",
    "get_strategy",
    "list_available_strategies",
    # Models
    "get_model",
    "get_model_info",
    "list_available_models",
    # Data
    "get_datasets",
    "get_dataloaders",
    "get_class_names",
    "get_dataset_info",
]