"""
Data Schemas for Active Learning Dashboard.

This module defines the data structures used throughout the application.
These are extracted from the original state.py and adapted for the MVC architecture.

Key Differences from state.py:
- Lighter weight (no Pydantic validation overhead for hot paths)
- Clear separation between transient (WorldState) and persistent (SQLite) data
- Explicit to_dict/from_dict for serialization
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ExperimentPhase(str, Enum):
    """
    Possible phases of an experiment lifecycle.
    
    State machine:
        IDLE → INITIALIZING → PREPARING → TRAINING → EVALUATING → QUERYING → AWAITING_ANNOTATION
                                                                                    ↓
                                                              IDLE ← (annotations submitted)
                                                                ↓
                                                           COMPLETED (after final cycle)
        
        Any state can transition to ERROR
    """
    IDLE = "IDLE"
    INITIALIZING = "INITIALIZING"
    PREPARING = "PREPARING"
    TRAINING = "TRAINING"
    VALIDATING = "VALIDATING"
    EVALUATING = "EVALUATING"
    QUERYING = "QUERYING"
    AWAITING_ANNOTATION = "AWAITING_ANNOTATION"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    PAUSED = "PAUSED"


@dataclass
class EpochMetrics:
    """
    Metrics for a single training epoch.
    
    Stored in SQLite for historical analysis.
    Current epoch metrics also kept in WorldState for fast access.
    """
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON/SQLite storage."""
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "train_accuracy": self.train_accuracy,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_accuracy,
            "learning_rate": self.learning_rate,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpochMetrics":
        """Deserialize from JSON/SQLite storage."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        return cls(
            epoch=data["epoch"],
            train_loss=data["train_loss"],
            train_accuracy=data["train_accuracy"],
            val_loss=data.get("val_loss"),
            val_accuracy=data.get("val_accuracy"),
            learning_rate=data.get("learning_rate"),
            timestamp=timestamp or datetime.now()
        )


@dataclass
class CycleSummary:
    """
    Summary metrics for a completed AL cycle.
    
    Stored in SQLite for historical comparison across cycles.
    """
    cycle: int
    labeled_count: int
    unlabeled_count: int
    epochs_trained: int
    best_val_accuracy: float
    best_epoch: int
    test_accuracy: float
    test_f1: float
    test_precision: float = 0.0
    test_recall: float = 0.0
    per_class_metrics: Optional[Dict[str, Dict[str, float]]] = None
    annotation_accuracy: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON/SQLite storage."""
        return {
            "cycle": self.cycle,
            "labeled_count": self.labeled_count,
            "unlabeled_count": self.unlabeled_count,
            "epochs_trained": self.epochs_trained,
            "best_val_accuracy": self.best_val_accuracy,
            "best_epoch": self.best_epoch,
            "test_accuracy": self.test_accuracy,
            "test_f1": self.test_f1,
            "test_precision": self.test_precision,
            "test_recall": self.test_recall,
            "per_class_metrics": self.per_class_metrics,
            "annotation_accuracy": self.annotation_accuracy,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CycleSummary":
        """Deserialize from JSON/SQLite storage."""
        started_at = data.get("started_at")
        completed_at = data.get("completed_at")
        if isinstance(started_at, str):
            started_at = datetime.fromisoformat(started_at)
        if isinstance(completed_at, str):
            completed_at = datetime.fromisoformat(completed_at)
        
        return cls(
            cycle=data["cycle"],
            labeled_count=data["labeled_count"],
            unlabeled_count=data["unlabeled_count"],
            epochs_trained=data["epochs_trained"],
            best_val_accuracy=data["best_val_accuracy"],
            best_epoch=data["best_epoch"],
            test_accuracy=data["test_accuracy"],
            test_f1=data["test_f1"],
            test_precision=data.get("test_precision", 0.0),
            test_recall=data.get("test_recall", 0.0),
            per_class_metrics=data.get("per_class_metrics"),
            annotation_accuracy=data.get("annotation_accuracy"),
            started_at=started_at,
            completed_at=completed_at
        )


@dataclass
class QueriedImage:
    """
    Information about an image selected for annotation.
    
    Kept in WorldState during annotation phase.
    """
    image_id: int
    image_path: str
    display_path: str
    ground_truth: int
    ground_truth_name: str
    predicted_class: str
    predicted_confidence: float
    uncertainty_score: float
    selection_reason: str
    model_probabilities: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for transmission."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueriedImage":
        """Deserialize from transmission."""
        return cls(**data)


@dataclass
class ExperimentConfig:
    """
    Experiment configuration - immutable after experiment starts.
    
    Stored in SQLite as JSON blob.
    """
    # Model settings
    model_name: str
    pretrained: bool = True
    num_classes: int = 4
    
    # Training settings
    epochs_per_cycle: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "adam"
    weight_decay: float = 0.0001
    early_stopping_patience: int = 5
    
    # Active Learning settings
    num_cycles: int = 5
    sampling_strategy: str = "uncertainty"
    uncertainty_method: str = "entropy"
    initial_pool_size: int = 40
    batch_size_al: int = 20
    reset_mode: str = "reset_to_pretrained"
    
    # Data settings
    data_dir: str = ""
    val_split: float = 0.15
    test_split: float = 0.15
    augmentation: bool = True
    
    # Misc
    seed: int = 42
    class_names: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for SQLite storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Deserialize from SQLite storage."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class DatasetInfo:
    """Information about the loaded dataset."""
    total_images: int
    num_classes: int
    class_names: List[str]
    class_counts: Dict[str, int]
    train_samples: int
    val_samples: int
    test_samples: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetInfo":
        return cls(**data)


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)


@dataclass 
class TrainingProgress:
    """
    Current training progress - for UI display.
    
    This is a view-ready structure created by ModelHandler.
    """
    phase: ExperimentPhase
    current_cycle: int
    total_cycles: int
    current_epoch: int
    epochs_per_cycle: int
    train_loss: Optional[float] = None
    train_accuracy: Optional[float] = None
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    progress_percentage: float = 0.0
    epoch_history: List[EpochMetrics] = field(default_factory=list)
    
    @property
    def is_training(self) -> bool:
        return self.phase == ExperimentPhase.TRAINING
    
    @property
    def loss_history_df(self) -> Dict[str, List[float]]:
        """Return loss history in a format suitable for st.line_chart."""
        return {
            "epoch": [m.epoch for m in self.epoch_history],
            "train_loss": [m.train_loss for m in self.epoch_history],
            "val_loss": [m.val_loss for m in self.epoch_history if m.val_loss]
        }


@dataclass
class ExperimentStatus:
    """
    Current experiment status - for UI display.
    
    This is a view-ready structure created by ModelHandler.
    """
    experiment_id: str
    experiment_name: str
    phase: ExperimentPhase
    current_cycle: int
    total_cycles: int
    labeled_count: int
    unlabeled_count: int
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    
    @property
    def is_running(self) -> bool:
        """Check if experiment is in an active phase."""
        active_phases = {
            ExperimentPhase.INITIALIZING,
            ExperimentPhase.PREPARING,
            ExperimentPhase.TRAINING,
            ExperimentPhase.VALIDATING,
            ExperimentPhase.EVALUATING,
            ExperimentPhase.QUERYING
        }
        return self.phase in active_phases
    
    @property
    def is_waiting_for_user(self) -> bool:
        """Check if experiment is waiting for user action."""
        return self.phase == ExperimentPhase.AWAITING_ANNOTATION
    
    @property
    def has_error(self) -> bool:
        return self.phase == ExperimentPhase.ERROR
