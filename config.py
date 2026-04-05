"""
config.py — YAML-based configuration with validation and merging.

Usage:
    config = load_config()                          # loads default.yaml
    config = load_config("configs/quick_test.yaml") # merges over default
    config = load_config(overrides={"training.epochs": 10})  # CLI/UI override
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
import torch
import logging

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"


@dataclass
class ExperimentConfig:
    name: str = "al_experiment"
    seed: int = 42
    device: str = "auto"
    exp_dir: str = "experiments/"


@dataclass
class DataConfig:
    data_dir: str = "data/raw"
    test_dir: Optional[str] = None
    val_split: float = 0.15
    test_split: float = 0.15
    augmentation: bool = True
    num_workers: int = 4
    image_size: int = 224


@dataclass
class ModelConfig:
    name: str = "resnet50"
    pretrained: bool = True
    num_classes: Optional[int] = None


@dataclass
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    early_stopping_patience: int = 3
    scheduler: str = "cosine"       # options: "cosine", "plateau", "none"
    warmup_epochs: int = 2
    grad_clip_norm: float = 1.0     # set to 0 to disable
    label_smoothing: float = 0.1
    backbone_lr_factor: float = 0.1  # backbone LR = learning_rate × this factor
    freeze_backbone_epochs: int = 2  # freeze backbone for first N epochs per cycle
    loss_fn: str = "cross_entropy"   # options: "cross_entropy", "supcon", "combined"
    supcon_temperature: float = 0.07
    supcon_weight: float = 0.5       # α in: loss = (1-α)·CE + α·SupCon


@dataclass
class ALConfig:
    num_cycles: int = 10
    initial_pool_size: int = 100
    batch_size_al: int = 50
    sampling_strategy: str = "entropy"
    uncertainty_method: str = "entropy"
    reset_mode: str = "continue"  # options: "continue", "pretrained", "head_only", "none"
    step_mode: bool = False  # Require explicit UI "Next Step" between cycles
    auto_annotate: bool = True  # Auto-annotate with ground truth for simulation
    stratified_init: bool = True  # Guarantee ≥1 sample per class in initial pool


@dataclass
class CheckpointConfig:
    save_best_model: bool = True
    save_best_per_cycle: bool = True
    save_every_n_epochs: int = 5


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_to_file: bool = True


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    active_learning: ALConfig = field(default_factory=ALConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def resolve_device(self):
        """Resolve 'auto' device to actual device."""
        if self.experiment.device == "auto":
            self.experiment.device = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self) -> dict:
        """Serialize entire config for queue transport (no dataclass objects)."""
        return asdict(self)

    def save_to_file(self, path: Path) -> None:
        """Save resolved config as YAML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def validate(self) -> list[str]:
        """
        Validate configuration values.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate data config
        if not (0.0 < self.data.val_split < 1.0):
            errors.append(f"data.val_split must be between 0 and 1, got {self.data.val_split}")
        if not (0.0 < self.data.test_split < 1.0):
            errors.append(f"data.test_split must be between 0 and 1, got {self.data.test_split}")
        if self.data.val_split + self.data.test_split >= 1.0:
            errors.append(f"data.val_split + data.test_split must be < 1.0")
        if self.data.num_workers < 0:
            errors.append(f"data.num_workers must be >= 0, got {self.data.num_workers}")
        if self.data.image_size <= 0:
            errors.append(f"data.image_size must be > 0, got {self.data.image_size}")
        
        # Check data directory exists
        data_path = Path(self.data.data_dir)
        if not data_path.exists():
            errors.append(f"data.data_dir does not exist: {self.data.data_dir}")
        
        # Validate training config
        if self.training.epochs <= 0:
            errors.append(f"training.epochs must be > 0, got {self.training.epochs}")
        if self.training.batch_size <= 0:
            errors.append(f"training.batch_size must be > 0, got {self.training.batch_size}")
        if self.training.learning_rate <= 0:
            errors.append(f"training.learning_rate must be > 0, got {self.training.learning_rate}")
        if self.training.weight_decay < 0:
            errors.append(f"training.weight_decay must be >= 0, got {self.training.weight_decay}")
        
        valid_optimizers = ["adam", "adamw", "sgd"]
        if self.training.optimizer not in valid_optimizers:
            errors.append(f"training.optimizer must be one of {valid_optimizers}, got {self.training.optimizer}")

        valid_schedulers = ["cosine", "plateau", "none"]
        if self.training.scheduler not in valid_schedulers:
            errors.append(f"training.scheduler must be one of {valid_schedulers}, got {self.training.scheduler}")
        if self.training.warmup_epochs < 0:
            errors.append(f"training.warmup_epochs must be >= 0, got {self.training.warmup_epochs}")
        if self.training.grad_clip_norm < 0:
            errors.append(f"training.grad_clip_norm must be >= 0, got {self.training.grad_clip_norm}")
        if not (0.0 <= self.training.label_smoothing < 1.0):
            errors.append(f"training.label_smoothing must be in [0, 1), got {self.training.label_smoothing}")
        if not (0.0 < self.training.backbone_lr_factor <= 1.0):
            errors.append(f"training.backbone_lr_factor must be in (0, 1], got {self.training.backbone_lr_factor}")
        if self.training.freeze_backbone_epochs < 0:
            errors.append(f"training.freeze_backbone_epochs must be >= 0, got {self.training.freeze_backbone_epochs}")
        valid_loss_fns = ["cross_entropy", "supcon", "combined"]
        if self.training.loss_fn not in valid_loss_fns:
            errors.append(f"training.loss_fn must be one of {valid_loss_fns}, got {self.training.loss_fn}")
        if not (0.0 < self.training.supcon_temperature <= 1.0):
            errors.append(f"training.supcon_temperature must be in (0, 1], got {self.training.supcon_temperature}")
        if not (0.0 <= self.training.supcon_weight <= 1.0):
            errors.append(f"training.supcon_weight must be in [0, 1], got {self.training.supcon_weight}")
        
        # Validate active learning config
        if self.active_learning.num_cycles <= 0:
            errors.append(f"active_learning.num_cycles must be > 0, got {self.active_learning.num_cycles}")
        if self.active_learning.initial_pool_size <= 0:
            errors.append(f"active_learning.initial_pool_size must be > 0, got {self.active_learning.initial_pool_size}")
        if self.active_learning.batch_size_al <= 0:
            errors.append(f"active_learning.batch_size_al must be > 0, got {self.active_learning.batch_size_al}")
        
        valid_strategies = ["entropy", "margin", "least_confidence", "random"]
        if self.active_learning.sampling_strategy not in valid_strategies:
            errors.append(f"active_learning.sampling_strategy must be one of {valid_strategies}, got {self.active_learning.sampling_strategy}")
        
        valid_uncertainty = ["entropy", "margin", "least_confidence"]
        if self.active_learning.uncertainty_method not in valid_uncertainty:
            errors.append(f"active_learning.uncertainty_method must be one of {valid_uncertainty}, got {self.active_learning.uncertainty_method}")
        
        valid_reset_modes = ["continue", "pretrained", "head_only", "none"]
        if self.active_learning.reset_mode not in valid_reset_modes:
            errors.append(f"active_learning.reset_mode must be one of {valid_reset_modes}, got {self.active_learning.reset_mode}")
        if not isinstance(self.active_learning.step_mode, bool):
            errors.append(
                f"active_learning.step_mode must be boolean, got {type(self.active_learning.step_mode).__name__}"
            )
        
        # Validate experiment config
        if self.experiment.seed < 0:
            errors.append(f"experiment.seed must be >= 0, got {self.experiment.seed}")
        
        valid_devices = ["auto", "cuda", "cpu"]
        if self.experiment.device not in valid_devices:
            errors.append(f"experiment.device must be one of {valid_devices}, got {self.experiment.device}")
        
        # Validate logging config
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.logging.level not in valid_log_levels:
            errors.append(f"logging.level must be one of {valid_log_levels}, got {self.logging.level}")
        
        return errors


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _dict_to_config(d: dict) -> Config:
    """Convert nested dict to Config dataclass."""
    return Config(
        experiment=ExperimentConfig(**d.get("experiment", {})),
        data=DataConfig(**d.get("data", {})),
        model=ModelConfig(**d.get("model", {})),
        training=TrainingConfig(**d.get("training", {})),
        active_learning=ALConfig(**d.get("active_learning", {})),
        checkpoint=CheckpointConfig(**d.get("checkpoint", {})),
        logging=LoggingConfig(**d.get("logging", {})),
    )


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[dict] = None
) -> Config:
    """
    Load config with layered merging:
      1. default.yaml (base)
      2. config_path yaml (experiment-specific overrides)
      3. overrides dict (runtime/UI overrides)
    
    Args:
        config_path: Path to experiment-specific YAML config
        overrides: Runtime overrides (supports dotted keys like "training.epochs")
    
    Returns:
        Config object with all settings merged
    """
    # Layer 1: defaults
    with open(DEFAULT_CONFIG) as f:
        base = yaml.safe_load(f)

    # Layer 2: experiment config
    if config_path:
        with open(config_path) as f:
            experiment = yaml.safe_load(f) or {}
        base = _deep_merge(base, experiment)

    # Layer 3: runtime overrides (from Streamlit UI, CLI args, etc.)
    if overrides:
        # Support dotted keys: {"training.epochs": 10}
        expanded = {}
        for key, value in overrides.items():
            parts = key.split(".")
            d = expanded
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = value
        base = _deep_merge(base, expanded)

    config = _dict_to_config(base)
    config.resolve_device()
    
    # Validate configuration
    validation_errors = config.validate()
    if validation_errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {err}" for err in validation_errors)
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"Config loaded: {config.experiment.name} | "
                f"Model: {config.model.name} | "
                f"Strategy: {config.active_learning.sampling_strategy}")

    return config
