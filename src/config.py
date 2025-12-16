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
    reset_mode: str = "pretrained"


@dataclass
class CheckpointConfig:
    """Checkpoint and logging configuration."""
    save_every_n_epochs: int = 5
    save_best_model: bool = True
    save_best_per_cycle: bool = True
    log_every_n_batches: int = 50


@dataclass
class DashboardConfig:
    """Dashboard and worker process configuration."""
    heartbeat_interval: int = 5
    heartbeat_timeout: int = 30
    auto_refresh_interval: int = 3000
    num_probe_images_per_class: int = 2
    experiments_base_dir: str = "./experiments"


@dataclass
class Config:
    """Master configuration object."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    active_learning: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)

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
        dashboard_config = DashboardConfig(**data.get("dashboard", {}))
        
        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            active_learning=al_config,
            checkpoint=checkpoint_config,
            dashboard=dashboard_config,
        )

    def save_to(self, path: str, include_metadata: bool = True) -> None:
        """Save configuration to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = {
            "model": asdict(self.model),
            "training": asdict(self.training),
            "data": asdict(self.data),
            "active_learning": asdict(self.active_learning),
            "checkpoint": asdict(self.checkpoint),
            "dashboard": asdict(self.dashboard),
        }
        
        # Add metadata if requested
        if include_metadata:
            from datetime import datetime
            config_dict["_metadata"] = {
                "created_at": datetime.now().isoformat(),
                "config_version": "1.0",
                "description": f"Configuration for {self.model.name} model"
            }
        
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def save_as_template(self, template_name: str, description: str = "") -> str:
        """Save configuration as a reusable template."""
        templates_dir = Path("config/templates")
        templates_dir.mkdir(parents=True, exist_ok=True)
        
        template_path = templates_dir / f"{template_name}.yaml"
        
        config_dict = {
            "model": asdict(self.model),
            "training": asdict(self.training),
            "data": asdict(self.data),
            "active_learning": asdict(self.active_learning),
            "checkpoint": asdict(self.checkpoint),
            "dashboard": asdict(self.dashboard),
        }
        
        # Add template metadata
        from datetime import datetime
        config_dict["_template_info"] = {
            "name": template_name,
            "description": description or f"Template for {self.model.name} model",
            "created_at": datetime.now().isoformat(),
            "tags": [self.model.name, "active_learning" if self.active_learning.enabled else "standard"]
        }
        
        with open(template_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        return str(template_path)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model": asdict(self.model),
            "training": asdict(self.training),
            "data": asdict(self.data),
            "active_learning": asdict(self.active_learning),
            "checkpoint": asdict(self.checkpoint),
            "dashboard": asdict(self.dashboard),
        }

    @classmethod
    def from_template(cls, template_name: str) -> "Config":
        """Load configuration from a template."""
        templates_dir = Path("config/templates")
        template_path = templates_dir / f"{template_name}.yaml"
        
        if not template_path.exists():
            # Fallback to main config directory
            fallback_path = Path("config") / f"{template_name}.yaml"
            if fallback_path.exists():
                template_path = fallback_path
            else:
                raise FileNotFoundError(f"Template not found: {template_name}")
        
        return cls.from_yaml(str(template_path))


class ConfigManager:
    """Manages experiment-specific configurations and templates."""
    
    def __init__(self, experiments_base_dir: str = "./experiments"):
        self.experiments_base_dir = Path(experiments_base_dir)
        self.templates_dir = Path("config/templates")
        self.templates_dir.mkdir(parents=True, exist_ok=True)
    
    def create_experiment_config(
        self, 
        experiment_name: str, 
        base_template: str = "base_config",
        **overrides
    ) -> tuple[Path, Config]:
        """
        Create a new experiment with its own config file.
        
        Args:
            experiment_name: Name for the experiment
            base_template: Template to start from
            **overrides: Config parameters to override
            
        Returns:
            Tuple of (experiment_dir, config)
        """
        from datetime import datetime
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = self.experiments_base_dir / f"{experiment_name}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Load base config
        try:
            config = Config.from_template(base_template)
        except FileNotFoundError:
            # Use default config if template not found
            config = Config()
        
        # Apply overrides
        for key, value in overrides.items():
            self._set_nested_attr(config, key, value)
        
        # Save config to experiment directory
        config_path = exp_dir / "config.yaml"
        config.save_to(str(config_path))
        
        return exp_dir, config
    
    def load_experiment_config(self, experiment_dir: Path) -> Config:
        """Load config from an experiment directory."""
        config_path = experiment_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found in experiment: {experiment_dir}")
        
        return Config.from_yaml(str(config_path))
    
    def update_experiment_config(
        self, 
        experiment_dir: Path, 
        **updates
    ) -> Config:
        """Update an experiment's config file."""
        config = self.load_experiment_config(experiment_dir)
        
        # Apply updates
        for key, value in updates.items():
            self._set_nested_attr(config, key, value)
        
        # Save updated config
        config_path = experiment_dir / "config.yaml"
        config.save_to(str(config_path))
        
        return config
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """List all available config templates."""
        templates = []
        
        # Check templates directory
        if self.templates_dir.exists():
            for template_file in self.templates_dir.glob("*.yaml"):
                templates.append(self._get_template_info(template_file))
        
        # Check main config directory for legacy templates
        config_dir = Path("config")
        if config_dir.exists():
            for config_file in config_dir.glob("*.yaml"):
                if config_file.name not in [t["filename"] for t in templates]:
                    templates.append(self._get_template_info(config_file, is_legacy=True))
        
        return sorted(templates, key=lambda x: x["name"])
    
    def copy_config_as_template(
        self, 
        experiment_dir: Path, 
        template_name: str, 
        description: str = ""
    ) -> str:
        """Copy an experiment's config as a new template."""
        config = self.load_experiment_config(experiment_dir)
        return config.save_as_template(template_name, description)
    
    def _set_nested_attr(self, obj, key: str, value):
        """Set nested attribute using dot notation (e.g., 'model.name')."""
        parts = key.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    
    def _get_template_info(self, template_path: Path, is_legacy: bool = False) -> Dict[str, Any]:
        """Extract template information from YAML file."""
        try:
            with open(template_path, "r") as f:
                data = yaml.safe_load(f) or {}
            
            template_info = data.get("_template_info", {})
            
            return {
                "name": template_info.get("name", template_path.stem),
                "filename": template_path.name,
                "path": str(template_path),
                "description": template_info.get("description", "No description"),
                "created_at": template_info.get("created_at", "Unknown"),
                "tags": template_info.get("tags", []),
                "is_legacy": is_legacy,
                "has_active_learning": data.get("active_learning", {}).get("enabled", False)
            }
        except Exception as e:
            return {
                "name": template_path.stem,
                "filename": template_path.name,
                "path": str(template_path),
                "description": f"Error reading template: {e}",
                "created_at": "Unknown",
                "tags": [],
                "is_legacy": is_legacy,
                "has_active_learning": False
            }