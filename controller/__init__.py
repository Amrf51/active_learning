"""
Controller Layer - Event Handling and Orchestration.

This package provides:
- EventDispatcher: Routes events between View, Model, and Service
- ModelHandler: Transforms WorldState into view-ready data
- ServiceManager: Manages the Service process lifecycle
- SessionManager: Multi-tab detection

Usage:
    from controller import get_controller
    from controller.events import Event, EventType
    
    # Get controller singleton
    ctrl = get_controller()
    
    # Dispatch event
    ctrl.dispatch(Event(EventType.START_CYCLE))
    
    # Get data for view
    status = ctrl.get_status()
    progress = ctrl.get_training_progress()
"""

from .events import Event, EventType, create_event

__all__ = [
    "Event",
    "EventType", 
    "create_event"
]

# Note: get_controller() and other components will be added in Phase 4
# when the full Controller layer is implemented.