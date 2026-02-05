"""Event system for the Active Learning Dashboard.

All user actions go through events. Views never call backend directly.
This implements the Command Pattern for decoupling UI from backend.

All events must be picklable for multiprocessing pipe transport.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict
import time


class EventType(Enum):
    """Types of events that can be dispatched to the controller."""
    # Experiment lifecycle
    CREATE_EXPERIMENT = "create_experiment"
    LOAD_EXPERIMENT = "load_experiment"
    DELETE_EXPERIMENT = "delete_experiment"
    
    # Training control
    START_CYCLE = "start_cycle"
    PAUSE = "pause"
    STOP = "stop"
    CONTINUE = "continue"
    
    # Annotation
    SUBMIT_ANNOTATIONS = "submit_annotations"
    
    # Data queries
    GET_POOL_DATA = "get_pool_data"
    
    # Service control (multiprocessing)
    SHUTDOWN = "shutdown"  # Graceful service termination


@dataclass
class Event:
    """A command object representing a user action.
    
    Events are dispatched by views to the controller, which sends them
    to the service process via pipe for processing.
    
    IMPORTANT: All fields must be picklable for pipe transport.
    
    Attributes:
        type: The type of event (from EventType enum)
        payload: Optional dictionary of event-specific data
        timestamp: When the event was created (for ordering/debugging)
    """
    type: EventType
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
