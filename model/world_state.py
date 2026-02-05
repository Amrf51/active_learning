"""WorldState and Phase definitions for the Active Learning Dashboard.

WorldState is the single source of truth for UI state - fast reads, no I/O.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Phase(Enum):
    """Current phase of an experiment."""
    IDLE = "idle"                           # Ready to start
    INITIALIZING = "initializing"           # Setting up experiment
    TRAINING = "training"                   # Training in progress
    QUERYING = "querying"                   # Running AL strategy
    AWAITING_ANNOTATION = "awaiting_annotation"  # Waiting for user
    COMPLETED = "completed"                 # All cycles done
    ERROR = "error"                         # Something went wrong


@dataclass
class EpochMetrics:
    """Metrics for a single training epoch."""
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None


@dataclass
class QueriedImage:
    """An image selected by the AL strategy for annotation."""
    index: int
    path: str
    predicted_label: Optional[str] = None
    confidence: Optional[float] = None
    annotation: Optional[str] = None


@dataclass
class ProbeImage:
    """A probe image for tracking predictions across cycles."""
    index: int
    path: str
    true_label: str
    predictions: Dict[int, str] = field(default_factory=dict)  # cycle -> predicted_label


@dataclass
class WorldState:
    """In-memory state for fast UI reads.
    
    This is the single source of truth for the UI layer. All fields are
    updated by the ModelHandler and read by the views via the controller.
    """
    # Identity
    experiment_id: Optional[str] = None
    experiment_name: Optional[str] = None
    
    # Phase
    phase: Phase = Phase.IDLE
    
    # Progress
    current_cycle: int = 0
    total_cycles: int = 0
    current_epoch: int = 0
    epochs_per_cycle: int = 0
    
    # Pool sizes
    labeled_count: int = 0
    unlabeled_count: int = 0
    
    # Live metrics (updated during training)
    epoch_metrics: List[EpochMetrics] = field(default_factory=list)
    
    # Queried images for annotation
    queried_images: List[QueriedImage] = field(default_factory=list)
    
    # Probe images for prediction tracking
    probe_images: List[ProbeImage] = field(default_factory=list)
    
    # Error
    error_message: Optional[str] = None
    
    def reset(self) -> None:
        """Reset state to initial values."""
        self.experiment_id = None
        self.experiment_name = None
        self.phase = Phase.IDLE
        self.current_cycle = 0
        self.total_cycles = 0
        self.current_epoch = 0
        self.epochs_per_cycle = 0
        self.labeled_count = 0
        self.unlabeled_count = 0
        self.epoch_metrics = []
        self.queried_images = []
        self.probe_images = []
        self.error_message = None
    
    def clear_error(self) -> None:
        """Clear any error message."""
        self.error_message = None
        if self.phase == Phase.ERROR:
            self.phase = Phase.IDLE
