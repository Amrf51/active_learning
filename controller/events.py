"""Event system for the Active Learning Dashboard.

All user actions go through events. Views never call backend directly.
This implements the Command Pattern for decoupling UI from backend.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class EventType(Enum):
    """Types of events that can be dispatched to the controller."""
    CREATE_EXPERIMENT = "create_experiment"
    LOAD_EXPERIMENT = "load_experiment"
    START_CYCLE = "start_cycle"
    PAUSE = "pause"
    STOP = "stop"
    CONTINUE = "continue"
    SUBMIT_ANNOTATIONS = "submit_annotations"


@dataclass
class Event:
    """A command object representing a user action.
    
    Events are dispatched by views to the controller, which routes them
    to the ModelHandler for processing.
    
    Attributes:
        type: The type of event (from EventType enum)
        payload: Optional dictionary of event-specific data
    """
    type: EventType
    payload: Dict[str, Any] = field(default_factory=dict)
