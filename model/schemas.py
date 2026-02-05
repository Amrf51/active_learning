"""Data schemas for the Active Learning Dashboard.

This module defines dataclasses for configuration, dataset info, and validation.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


@dataclass
class DatasetInfo:
    """Information about a scanned dataset."""
    total_images: int
    num_classes: int
    class_names: List[str]
    class_counts: Dict[str, int]
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetInfo':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ExperimentConfig:
    """Configuration for an active learning experiment."""
    # Model settings
    model_name: str
    pretrained: bool
    num_classes: int
    class_names: List[str] = field(default_factory=list)
    
    # Training settings
    epochs_per_cycle: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "adam"
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5
    
    # Active Learning settings
    num_cycles: int = 5
    sampling_strategy: str = "uncertainty"
    uncertainty_method: str = "entropy"
    initial_pool_size: int = 50
    batch_size_al: int = 10
    reset_mode: str = "pretrained"
    
    # Data settings
    data_dir: str = "./data"
    val_split: float = 0.15
    test_split: float = 0.15
    augmentation: bool = True
    num_workers: int = 4
    
    # Misc
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, message: str):
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)


class ExperimentPhase(Enum):
    """Current phase of an experiment."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    PREPARING = "preparing"
    TRAINING = "training"
    VALIDATING = "validating"
    EVALUATING = "evaluating"
    QUERYING = "querying"
    AWAITING_ANNOTATION = "awaiting_annotation"
    ANNOTATIONS_SUBMITTED = "annotations_submitted"
    COMPLETED = "completed"
    ERROR = "error"
    ABORT = "abort"


@dataclass
class EpochMetrics:
    """Metrics for a single training epoch."""
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    timestamp: Optional[datetime] = None


@dataclass
class QueriedImage:
    """An image selected by the AL strategy for annotation."""
    image_id: str
    image_path: str
    display_path: Optional[str] = None
    predicted_class: str = ""
    predicted_confidence: float = 0.0
    uncertainty_score: float = 0.0
    selection_reason: str = ""
    ground_truth: int = -1
    ground_truth_name: str = ""
    model_probabilities: Dict[str, float] = field(default_factory=dict)


@dataclass
class ProbeImage:
    """A probe image for tracking predictions across cycles."""
    image_id: str
    image_path: str
    true_class: str
    predictions_by_cycle: Dict[int, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class UserAnnotation:
    """A user annotation for a queried image."""
    image_id: str
    user_label: int
    user_label_name: str
    timestamp: datetime
    was_correct: bool


@dataclass
class AnnotationSubmission:
    """A batch of user annotations submitted for a cycle."""
    experiment_id: str
    cycle: int
    annotations: List[UserAnnotation]
    submitted_at: datetime


@dataclass
class CycleSummary:
    """Summary of a completed cycle."""
    cycle: int
    labeled_count: int
    unlabeled_count: int
    test_accuracy: float
    test_f1: float
    timestamp: datetime


@dataclass
class ExperimentStatus:
    """Status information for an experiment."""
    experiment_id: str
    experiment_name: str
    phase: str
    current_cycle: int
    total_cycles: int
    created_at: datetime
    last_updated: datetime
