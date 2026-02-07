"""
Protocol definitions for message passing and event signaling in the MVC architecture.

This module defines:
- Message type constants for queue communication
- Event name constants for multiprocessing.Event signaling
- Message builder helpers for type-safe queue messages
- Helper to initialize multiprocessing.Event dictionary
"""

from typing import Dict, Any, Optional
import multiprocessing as mp


# ============================================================================
# MESSAGE TYPE CONSTANTS
# ============================================================================

# Worker initialization and lifecycle
INIT_MODEL = "init_model"
SHUTDOWN = "shutdown"

# Active learning cycle operations
RUN_CYCLE = "run_cycle"
QUERY = "query"
ANNOTATE = "annotate"

# Progress and status updates
PROGRESS_UPDATE = "progress_update"
TRAIN_COMPLETE = "train_complete"
QUERY_COMPLETE = "query_complete"
ANNOTATE_COMPLETE = "annotate_complete"
CYCLE_COMPLETE = "cycle_complete"

# Error handling
ERROR = "error"

# Model and state operations
SAVE_STATE = "save_state"
LOAD_STATE = "load_state"


# ============================================================================
# EVENT NAME CONSTANTS
# ============================================================================

# Lifecycle events
MODEL_READY = "model_ready"
WORKER_INITIALIZED = "worker_initialized"

# Training events
TRAINING_STARTED = "training_started"
TRAINING_DONE = "training_done"

# Query events
QUERY_STARTED = "query_started"
QUERY_DONE = "query_done"

# Annotation events
ANNOTATION_STARTED = "annotation_started"
ANNOTATION_DONE = "annotation_done"

# Control events
STOP_REQUESTED = "stop_requested"
WORKER_ERROR = "worker_error"
SHUTDOWN_COMPLETE = "shutdown_complete"


# ============================================================================
# MESSAGE BUILDER HELPERS
# ============================================================================

def build_message(msg_type: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Build a type-safe message for queue communication.
    
    Args:
        msg_type: Message type constant (e.g., INIT_MODEL, RUN_CYCLE)
        payload: Optional dictionary containing message-specific data
        
    Returns:
        Dictionary with 'type' and 'payload' keys
    """
    return {
        "type": msg_type,
        "payload": payload or {}
    }


def build_init_model_message(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build INIT_MODEL message.
    
    Args:
        config_dict: Configuration dictionary for model initialization
        
    Returns:
        Message dictionary
    """
    return build_message(INIT_MODEL, {"config": config_dict})


def build_run_cycle_message(cycle_num: int) -> Dict[str, Any]:
    """
    Build RUN_CYCLE message.
    
    Args:
        cycle_num: Active learning cycle number
        
    Returns:
        Message dictionary
    """
    return build_message(RUN_CYCLE, {"cycle_num": cycle_num})


def build_query_message(query_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Build QUERY message.
    
    Args:
        query_size: Optional number of samples to query (overrides config)
        
    Returns:
        Message dictionary
    """
    payload = {}
    if query_size is not None:
        payload["query_size"] = query_size
    return build_message(QUERY, payload)


def build_annotate_message(annotations: Dict[int, int]) -> Dict[str, Any]:
    """
    Build ANNOTATE message.
    
    Args:
        annotations: Dictionary mapping image indices to class labels
        
    Returns:
        Message dictionary
    """
    return build_message(ANNOTATE, {"annotations": annotations})


def build_progress_update_message(
    stage: str,
    current: int,
    total: int,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build PROGRESS_UPDATE message.
    
    Args:
        stage: Current stage (e.g., "training", "querying")
        current: Current progress value
        total: Total progress value
        details: Optional additional details
        
    Returns:
        Message dictionary
    """
    payload = {
        "stage": stage,
        "current": current,
        "total": total
    }
    if details:
        payload["details"] = details
    return build_message(PROGRESS_UPDATE, payload)


def build_train_complete_message(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build TRAIN_COMPLETE message.
    
    Args:
        metrics: Training metrics dictionary
        
    Returns:
        Message dictionary
    """
    return build_message(TRAIN_COMPLETE, {"metrics": metrics})


def build_query_complete_message(queried_images: list) -> Dict[str, Any]:
    """
    Build QUERY_COMPLETE message.
    
    Args:
        queried_images: List of queried image dictionaries
        
    Returns:
        Message dictionary
    """
    return build_message(QUERY_COMPLETE, {"queried_images": queried_images})


def build_annotate_complete_message(num_annotated: int) -> Dict[str, Any]:
    """
    Build ANNOTATE_COMPLETE message.
    
    Args:
        num_annotated: Number of images annotated
        
    Returns:
        Message dictionary
    """
    return build_message(ANNOTATE_COMPLETE, {"num_annotated": num_annotated})


def build_cycle_complete_message(cycle_num: int, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build CYCLE_COMPLETE message.
    
    Args:
        cycle_num: Completed cycle number
        metrics: Cycle metrics dictionary
        
    Returns:
        Message dictionary
    """
    return build_message(CYCLE_COMPLETE, {
        "cycle_num": cycle_num,
        "metrics": metrics
    })


def build_error_message(error_type: str, error_msg: str, traceback: Optional[str] = None) -> Dict[str, Any]:
    """
    Build ERROR message.
    
    Args:
        error_type: Type of error (e.g., "ModelInitError", "TrainingError")
        error_msg: Error message string
        traceback: Optional traceback string
        
    Returns:
        Message dictionary
    """
    payload = {
        "error_type": error_type,
        "error_msg": error_msg
    }
    if traceback:
        payload["traceback"] = traceback
    return build_message(ERROR, payload)


def build_shutdown_message() -> Dict[str, Any]:
    """
    Build SHUTDOWN message.
    
    Returns:
        Message dictionary
    """
    return build_message(SHUTDOWN)


# ============================================================================
# EVENT INITIALIZATION HELPER
# ============================================================================

def create_event_dict(mp_context: Optional[mp.context.BaseContext] = None) -> Dict[str, mp.Event]:
    """
    Create a dictionary of multiprocessing.Event objects for all defined events.
    
    Args:
        mp_context: Optional multiprocessing context (e.g., mp.get_context('spawn'))
                   If None, uses default multiprocessing context
        
    Returns:
        Dictionary mapping event name constants to mp.Event objects
    """
    if mp_context is None:
        mp_context = mp
    
    event_names = [
        # Lifecycle events
        MODEL_READY,
        WORKER_INITIALIZED,
        
        # Training events
        TRAINING_STARTED,
        TRAINING_DONE,
        
        # Query events
        QUERY_STARTED,
        QUERY_DONE,
        
        # Annotation events
        ANNOTATION_STARTED,
        ANNOTATION_DONE,
        
        # Control events
        STOP_REQUESTED,
        WORKER_ERROR,
        SHUTDOWN_COMPLETE,
    ]
    
    return {name: mp_context.Event() for name in event_names}
