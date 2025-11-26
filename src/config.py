"""Configuration management using dataclasses and YAML."""

from dataclasses import dataclass, asdict, field
from typing import Any, Dict
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "resnet18"
    pretrained: bool = True
    num_classes: int = 4


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "adam"
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    seed: int = 42


@dataclass
class DataConfig:
    """Data loading configuration."""
    data_dir: str = "./data/raw/kaggle-vehicle/"
    val_split: float = 0.15
    test_split: float = 0.15
    augmentation: bool = True
    num_workers: int = 4


@dataclass
class ActiveLearningConfig:
    """Active Learning parameters."""
    enabled: bool = False
    num_cycles: int = 5
    sampling_strategy: str = "uncertainty"
    initial_pool_size: int = 50
    batch_size_al: int = 20
    uncertainty_method: str = "least_confidence"
    reset_weights_each_cycle: bool = True


@dataclass
class CheckpointConfig:
    """Checkpoint and logging configuration."""
    save_every_n_epochs: int = 5
    save_best_model: bool = True
    save_best_per_cycle: bool = True
    log_every_n_batches: int = 50


@dataclass
class Config:
    """Master configuration object."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    active_learning: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        if data is None:
            data = {}
        
        model_config = ModelConfig(**data.get("model", {}))
        training_config = TrainingConfig(**data.get("training", {}))
        data_config = DataConfig(**data.get("data", {}))
        al_config = ActiveLearningConfig(**data.get("active_learning", {}))
        checkpoint_config = CheckpointConfig(**data.get("checkpoint", {}))
        
        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            active_learning=al_config,
            checkpoint=checkpoint_config,
        )

    def save_to(self, path: str) -> None:
        """Save configuration to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = {
            "model": asdict(self.model),
            "training": asdict(self.training),
            "data": asdict(self.data),
            "active_learning": asdict(self.active_learning),
            "checkpoint": asdict(self.checkpoint),
        }
        
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model": asdict(self.model),
            "training": asdict(self.training),
            "data": asdict(self.data),
            "active_learning": asdict(self.active_learning),
            "checkpoint": asdict(self.checkpoint),
        }