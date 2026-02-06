"""WorldState and Phase definitions for the Active Learning Dashboard.

WorldState is the single source of truth for UI state - fast reads, no I/O.
All data in WorldState must be picklable for multiprocessing pipe transport.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import time

# Import from backend - single source of truth for these types
# These are all picklable dataclasses
from backend.state import (
    EpochMetrics,
    CycleMetrics,
    QueriedImage,
    ProbeImage,
)

# Re-export for convenience
__all__ = [
    'Phase',
    'WorldState',
    'EpochMetrics',
    'CycleMetrics', 
    'QueriedImage',
    'ProbeImage',
]


class Phase(Enum):
    """Current phase of an experiment (MVC-specific)."""
    IDLE = "idle"                           # Ready to start
    INITIALIZING = "initializing"           # Setting up experiment
    TRAINING = "training"                   # Training in progress
    QUERYING = "querying"                   # Running AL strategy
    AWAITING_ANNOTATION = "awaiting_annotation"  # Waiting for user
    COMPLETED = "completed"                 # All cycles done
    ERROR = "error"                         # Something went wrong


@dataclass
class WorldState:
    """In-memory state for fast UI reads.
    
    This is the single source of truth for the UI layer. All fields are
    updated by the ModelHandler (in service process) and read by the views 
    via the controller.
    
    IMPORTANT: All fields must be picklable for multiprocessing pipe transport.
    - Use primitives (int, float, str, bool)
    - Use dataclasses with primitive fields
    - Use lists/dicts with picklable contents
    - Do NOT store PyTorch models, DataLoaders, or open file handles
    """
    # Identity
    experiment_id: Optional[str] = None
    experiment_name: Optional[str] = None
    
    # Phase
    phase: Phase = Phase.IDLE
    
    # Progress
    current_cycle: int = 1  # Next cycle to execute (1-indexed)
    total_cycles: int = 0
    current_epoch: int = 0
    epochs_per_cycle: int = 0
    
    # Pool sizes
    labeled_count: int = 0
    unlabeled_count: int = 0
    
    # Class distribution (for Dataset Explorer)
    class_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Live metrics (updated during training) - uses backend.state.EpochMetrics
    epoch_metrics: List[EpochMetrics] = field(default_factory=list)
    
    # Queried images for annotation - uses backend.state.QueriedImage
    queried_images: List[QueriedImage] = field(default_factory=list)
    
    # Probe images for prediction tracking - uses backend.state.ProbeImage
    probe_images: List[ProbeImage] = field(default_factory=list)
    
    # Error
    error_message: Optional[str] = None
    
    # Timestamp for state versioning (multiprocessing)
    updated_at: float = field(default_factory=time.time)
    
    def reset(self) -> None:
        """Reset state to initial values."""
        self.experiment_id = None
        self.experiment_name = None
        self.phase = Phase.IDLE
        self.current_cycle = 1  # Ready to execute cycle 1
        self.total_cycles = 0
        self.current_epoch = 0
        self.epochs_per_cycle = 0
        self.labeled_count = 0
        self.unlabeled_count = 0
        self.class_distribution = {}
        self.epoch_metrics = []
        self.queried_images = []
        self.probe_images = []
        self.error_message = None
        self.updated_at = time.time()
    
    def clear_error(self) -> None:
        """Clear any error message."""
        self.error_message = None
        if self.phase == Phase.ERROR:
            self.phase = Phase.IDLE
        self.updated_at = time.time()
    
    def touch(self) -> None:
        """Update timestamp to mark state as modified."""
        self.updated_at = time.time()
