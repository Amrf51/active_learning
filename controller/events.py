"""Event system for Active Learning Dashboard MVC architecture.

This module defines the event-driven communication system between View, Controller, and Service layers.
Events flow in three directions:
- VIEW → CONTROLLER: User actions (button clicks, form submissions)
- CONTROLLER → SERVICE: Commands to background processes
- SERVICE → CONTROLLER: Status updates and results from background processes
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class EventType(Enum):
    """Event types for the Active Learning Dashboard.
    
    Event flow directions:
    - VIEW → CONTROLLER: User interface events
    - CONTROLLER → SERVICE: Commands to service processes
    - SERVICE → CONTROLLER: Status updates from service processes
    """
    
    # VIEW → CONTROLLER Events (User Actions)
    INITIALIZE_EXPERIMENT = "initialize_experiment"
    START_CYCLE = "start_cycle"
    PAUSE_TRAINING = "pause_training"
    RESUME_TRAINING = "resume_training"
    STOP_EXPERIMENT = "stop_experiment"
    SUBMIT_ANNOTATIONS = "submit_annotations"
    
    # CONTROLLER → SERVICE Commands
    CMD_START_CYCLE = "cmd_start_cycle"
    CMD_PAUSE = "cmd_pause"
    CMD_RESUME = "cmd_resume"
    CMD_STOP = "cmd_stop"
    CMD_ANNOTATIONS = "cmd_annotations"
    CMD_SHUTDOWN = "cmd_shutdown"
    
    # SERVICE → CONTROLLER Events (Status Updates)
    SERVICE_READY = "service_ready"
    EPOCH_COMPLETE = "epoch_complete"
    CYCLE_COMPLETE = "cycle_complete"
    QUERY_READY = "query_ready"
    SERVICE_ERROR = "service_error"


@dataclass
class Event:
    """Event data structure for inter-component communication.
    
    Attributes:
        type: The type of event (from EventType enum)
        payload: Event-specific data (dict, can be empty)
        timestamp: When the event was created
        source: Component that created the event (for debugging)
    """
    
    type: EventType
    payload: Dict[str, Any]
    timestamp: datetime
    source: Optional[str] = None
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if not hasattr(self, 'timestamp') or self.timestamp is None:
            self.timestamp = datetime.now()
    
    @classmethod
    def create(cls, event_type: EventType, payload: Optional[Dict[str, Any]] = None, source: Optional[str] = None) -> 'Event':
        """Convenience method to create an event.
        
        Args:
            event_type: Type of event
            payload: Event data (optional)
            source: Source component (optional)
            
        Returns:
            New Event instance
        """
        return cls(
            type=event_type,
            payload=payload or {},
            timestamp=datetime.now(),
            source=source
        )
    
    def __str__(self) -> str:
        """String representation for logging."""
        source_str = f" from {self.source}" if self.source else ""
        return f"Event({self.type.value}{source_str} at {self.timestamp.strftime('%H:%M:%S.%f')[:-3]})"


# Event payload schemas for documentation and validation

class EventPayloads:
    """Documentation of expected payload structures for each event type."""
    
    # VIEW → CONTROLLER Event Payloads
    INITIALIZE_EXPERIMENT = {
        "config": "Dict[str, Any]",  # Experiment configuration
        "description": "User-submitted experiment configuration"
    }
    
    START_CYCLE = {
        "description": "Start a new active learning cycle"
    }
    
    PAUSE_TRAINING = {
        "description": "Pause current training"
    }
    
    RESUME_TRAINING = {
        "description": "Resume paused training"
    }
    
    STOP_EXPERIMENT = {
        "description": "Stop current experiment"
    }
    
    SUBMIT_ANNOTATIONS = {
        "annotations": "List[Dict[str, Any]]",  # User annotations for queried images
        "description": "User-provided annotations for queried samples"
    }
    
    # CONTROLLER → SERVICE Command Payloads
    CMD_START_CYCLE = {
        "cycle_number": "int",  # Current cycle number
        "description": "Command to start training cycle"
    }
    
    CMD_PAUSE = {
        "description": "Command to pause training"
    }
    
    CMD_RESUME = {
        "description": "Command to resume training"
    }
    
    CMD_STOP = {
        "description": "Command to stop training"
    }
    
    CMD_ANNOTATIONS = {
        "annotations": "List[Dict[str, Any]]",  # Processed annotations
        "description": "Annotations to incorporate into training"
    }
    
    CMD_SHUTDOWN = {
        "description": "Command to gracefully shutdown service"
    }
    
    # SERVICE → CONTROLLER Event Payloads
    SERVICE_READY = {
        "service_id": "str",  # Service process identifier
        "description": "Service is ready to receive commands"
    }
    
    EPOCH_COMPLETE = {
        "epoch": "int",  # Completed epoch number
        "cycle": "int",  # Current cycle number
        "metrics": "Dict[str, float]",  # Training metrics (loss, accuracy, etc.)
        "description": "Training epoch completed with metrics"
    }
    
    CYCLE_COMPLETE = {
        "cycle": "int",  # Completed cycle number
        "results": "Dict[str, Any]",  # Cycle results and statistics
        "description": "Active learning cycle completed"
    }
    
    QUERY_READY = {
        "cycle": "int",  # Current cycle number
        "queried_images": "List[Dict[str, Any]]",  # Images selected for annotation
        "description": "Query selection completed, awaiting user annotations"
    }
    
    SERVICE_ERROR = {
        "error_type": "str",  # Type of error
        "message": "str",  # Error message
        "traceback": "Optional[str]",  # Stack trace if available
        "description": "Service encountered an error"
    }


def validate_event_payload(event: Event) -> bool:
    """Validate that an event has the expected payload structure.
    
    Args:
        event: Event to validate
        
    Returns:
        True if payload is valid, False otherwise
        
    Note:
        This is a basic validation. In production, you might want
        to use a schema validation library like Pydantic or Cerberus.
    """
    expected_payload = getattr(EventPayloads, event.type.name, None)
    if expected_payload is None:
        return True  # No validation defined
    
    # Basic validation - check if required keys exist
    # This is simplified; real validation would check types too
    if not isinstance(event.payload, dict):
        return False
    
    return True  # Simplified validation for now


# Convenience functions for creating common events

def create_view_event(event_type: EventType, payload: Optional[Dict[str, Any]] = None) -> Event:
    """Create an event from the View layer."""
    return Event.create(event_type, payload, source="View")


def create_controller_event(event_type: EventType, payload: Optional[Dict[str, Any]] = None) -> Event:
    """Create an event from the Controller layer."""
    return Event.create(event_type, payload, source="Controller")


def create_service_event(event_type: EventType, payload: Optional[Dict[str, Any]] = None) -> Event:
    """Create an event from the Service layer."""
    return Event.create(event_type, payload, source="Service")