"""
Event System for Active Learning Dashboard.

This module defines the event-driven communication protocol between
View, Controller, and Service layers.

Event Flow:
    VIEW → CONTROLLER: User actions (INITIALIZE, START_CYCLE, etc.)
    CONTROLLER → SERVICE: Commands (CMD_START_CYCLE, CMD_PAUSE, etc.)
    SERVICE → CONTROLLER: Progress/completion (EPOCH_COMPLETE, QUERY_READY, etc.)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, Optional


class EventType(Enum):
    """
    All event types in the system.
    
    Naming convention:
    - No prefix: View → Controller events (user actions)
    - CMD_*: Controller → Service commands
    - SERVICE_*: Service → Controller notifications
    """
    
    # ═══════════════════════════════════════════════════════════════════
    # VIEW → CONTROLLER EVENTS (User Actions)
    # ═══════════════════════════════════════════════════════════════════
    
    # Experiment lifecycle
    INITIALIZE_EXPERIMENT = auto()  # User configures and starts new experiment
    LOAD_EXPERIMENT = auto()        # User loads existing experiment
    
    # Training control
    START_CYCLE = auto()            # User clicks "Start Cycle"
    PAUSE_TRAINING = auto()         # User clicks "Pause"
    RESUME_TRAINING = auto()        # User clicks "Resume"
    STOP_EXPERIMENT = auto()        # User clicks "Stop"
    
    # Annotation
    SUBMIT_ANNOTATIONS = auto()     # User submits annotation batch
    
    # Data requests (synchronous)
    REQUEST_STATUS = auto()         # Request current status
    REQUEST_METRICS = auto()        # Request training metrics
    REQUEST_HISTORY = auto()        # Request historical data
    
    # ═══════════════════════════════════════════════════════════════════
    # CONTROLLER → SERVICE COMMANDS
    # ═══════════════════════════════════════════════════════════════════
    
    CMD_START_CYCLE = auto()        # Begin training cycle
    CMD_PAUSE = auto()              # Pause training
    CMD_RESUME = auto()             # Resume training
    CMD_STOP = auto()               # Stop experiment
    CMD_ANNOTATIONS = auto()        # Process submitted annotations
    CMD_SHUTDOWN = auto()           # Graceful service shutdown
    
    # ═══════════════════════════════════════════════════════════════════
    # SERVICE → CONTROLLER EVENTS (Notifications)
    # ═══════════════════════════════════════════════════════════════════
    
    SERVICE_READY = auto()          # Service initialized and ready
    SERVICE_ERROR = auto()          # Service encountered error
    
    # Training progress
    EPOCH_COMPLETE = auto()         # Single epoch finished
    TRAINING_COMPLETE = auto()      # All epochs in cycle finished
    
    # Evaluation
    EVALUATION_COMPLETE = auto()    # Test evaluation finished
    
    # Active Learning
    QUERY_READY = auto()            # Queried samples ready for annotation
    CYCLE_COMPLETE = auto()         # Full AL cycle completed
    EXPERIMENT_COMPLETE = auto()    # All cycles completed


@dataclass
class Event:
    """
    Immutable event object for inter-layer communication.
    
    Attributes:
        type: The event type
        payload: Event-specific data (optional)
        timestamp: When the event was created
        source: Origin of the event (for debugging)
    
    Usage:
        # Create event
        event = Event(EventType.START_CYCLE)
        
        # Create event with data
        event = Event(
            EventType.EPOCH_COMPLETE,
            payload={"epoch": 5, "train_loss": 0.123, "val_acc": 0.95}
        )
    """
    
    type: EventType
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "unknown"
    
    def __post_init__(self):
        """Validate event after creation."""
        if not isinstance(self.type, EventType):
            raise ValueError(f"Invalid event type: {self.type}")
        if self.payload is None:
            self.payload = {}
    
    def __repr__(self) -> str:
        payload_preview = str(self.payload)[:50] + "..." if len(str(self.payload)) > 50 else str(self.payload)
        return f"Event({self.type.name}, payload={payload_preview}, source={self.source})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize event for Pipe transmission."""
        return {
            "type": self.type.value,
            "type_name": self.type.name,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Deserialize event from Pipe transmission."""
        return cls(
            type=EventType(data["type"]),
            payload=data.get("payload", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source=data.get("source", "unknown")
        )


# ═══════════════════════════════════════════════════════════════════════════
# Event Payload Schemas (for documentation and validation)
# ═══════════════════════════════════════════════════════════════════════════

"""
PAYLOAD SCHEMAS BY EVENT TYPE:

INITIALIZE_EXPERIMENT:
    {
        "experiment_name": str,
        "config": {
            "model_name": str,
            "sampling_strategy": str,
            "num_cycles": int,
            "epochs_per_cycle": int,
            "batch_size_al": int,
            "initial_pool_size": int,
            "data_dir": str,
            ...
        }
    }

SUBMIT_ANNOTATIONS:
    {
        "annotations": [
            {"image_id": int, "label": int},
            ...
        ]
    }

EPOCH_COMPLETE:
    {
        "epoch": int,
        "train_loss": float,
        "train_accuracy": float,
        "val_loss": float,
        "val_accuracy": float,
        "learning_rate": float
    }

QUERY_READY:
    {
        "queried_images": [
            {
                "image_id": int,
                "image_path": str,
                "predicted_class": str,
                "confidence": float,
                "uncertainty_score": float
            },
            ...
        ]
    }

CYCLE_COMPLETE:
    {
        "cycle": int,
        "labeled_count": int,
        "unlabeled_count": int,
        "test_accuracy": float,
        "test_f1": float,
        "best_val_accuracy": float
    }

SERVICE_ERROR:
    {
        "message": str,
        "traceback": str (optional),
        "recoverable": bool
    }
"""


def create_event(
    event_type: EventType,
    payload: Optional[Dict[str, Any]] = None,
    source: str = "unknown"
) -> Event:
    """
    Factory function for creating events.
    
    Args:
        event_type: The type of event
        payload: Event-specific data
        source: Origin identifier
    
    Returns:
        New Event instance
    """
    return Event(
        type=event_type,
        payload=payload or {},
        source=source
    )


# Convenience functions for common events
def start_cycle_event(source: str = "view") -> Event:
    """Create START_CYCLE event."""
    return create_event(EventType.START_CYCLE, source=source)


def pause_event(source: str = "view") -> Event:
    """Create PAUSE_TRAINING event."""
    return create_event(EventType.PAUSE_TRAINING, source=source)


def epoch_complete_event(
    epoch: int,
    train_loss: float,
    train_accuracy: float,
    val_loss: float,
    val_accuracy: float,
    learning_rate: float,
    source: str = "service"
) -> Event:
    """Create EPOCH_COMPLETE event with metrics."""
    return create_event(
        EventType.EPOCH_COMPLETE,
        payload={
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": learning_rate
        },
        source=source
    )


def error_event(
    message: str,
    traceback: Optional[str] = None,
    recoverable: bool = False,
    source: str = "service"
) -> Event:
    """Create SERVICE_ERROR event."""
    return create_event(
        EventType.SERVICE_ERROR,
        payload={
            "message": message,
            "traceback": traceback,
            "recoverable": recoverable
        },
        source=source
    )
