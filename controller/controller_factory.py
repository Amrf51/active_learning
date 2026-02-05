"""Controller factory for managing ExperimentController in Streamlit session state.

This module provides utilities for:
1. Creating and caching the ExperimentController in st.session_state
2. Managing session heartbeat for cleanup
3. Providing a consistent interface for all pages to access the controller
4. Service lifecycle management with atexit cleanup
5. Automatic service restart if it dies
"""

import streamlit as st
from pathlib import Path
from datetime import datetime
import logging
import atexit
import weakref

from controller.experiment_controller import ExperimentController

logger = logging.getLogger(__name__)

# Default experiments directory
DEFAULT_EXPERIMENTS_DIR = Path("./experiments")

# Global reference for atexit cleanup (weak reference to avoid memory leaks)
_controller_refs: list = []


def _cleanup_controller(controller_ref: weakref.ref) -> None:
    """Cleanup function called by atexit to shutdown the controller.
    
    Args:
        controller_ref: Weak reference to the controller
    """
    controller = controller_ref()
    if controller is not None:
        try:
            logger.info("atexit: Shutting down ExperimentController")
            controller.shutdown()
            logger.info("atexit: ExperimentController shutdown complete")
        except Exception as e:
            logger.error(f"atexit: Error shutting down controller: {e}")


def _register_cleanup(controller: ExperimentController) -> None:
    """Register the controller for cleanup on app shutdown.
    
    Uses a weak reference to avoid preventing garbage collection.
    
    Args:
        controller: The controller to register for cleanup
    """
    # Create weak reference
    controller_ref = weakref.ref(controller)
    _controller_refs.append(controller_ref)
    
    # Register atexit handler
    atexit.register(_cleanup_controller, controller_ref)
    logger.info("Registered atexit cleanup handler for ExperimentController")


def get_controller() -> ExperimentController:
    """Get or create the ExperimentController from session state.
    
    This function ensures that only one controller instance exists per
    Streamlit session, stored in st.session_state for persistence across
    reruns.
    
    Also handles:
    - Registering atexit cleanup handler on first creation
    - Automatic service restart if the service process died
    
    Returns:
        ExperimentController instance
    """
    # Initialize controller if not present
    if "experiment_controller" not in st.session_state:
        logger.info("Creating new ExperimentController instance")
        
        # Ensure experiments directory exists
        experiments_dir = DEFAULT_EXPERIMENTS_DIR
        experiments_dir.mkdir(parents=True, exist_ok=True)
        
        # Create controller
        controller = ExperimentController(experiments_dir)
        
        # Register cleanup handler for graceful shutdown
        _register_cleanup(controller)
        
        # Store in session state
        st.session_state.experiment_controller = controller
        st.session_state.controller_created_at = datetime.now()
        
        logger.info("ExperimentController created and stored in session state")
    
    # Get the controller
    controller = st.session_state.experiment_controller
    
    # Check if service needs restart
    _check_and_restart_service(controller)
    
    return controller


def _check_and_restart_service(controller: ExperimentController) -> bool:
    """Check if the service process died and restart if needed.
    
    Args:
        controller: The ExperimentController to check
        
    Returns:
        True if service was restarted, False otherwise
    """
    if not controller.is_service_alive():
        # Check if this is an unexpected death (not during IDLE/COMPLETED)
        state = controller.get_state()
        
        # Only auto-restart if we're not in a terminal state
        from model.world_state import Phase
        if state.phase not in (Phase.COMPLETED, Phase.ERROR):
            logger.warning("Service process died unexpectedly, attempting restart")
            if controller.restart_service_if_dead():
                logger.info("Service process restarted successfully")
                return True
            else:
                logger.error("Failed to restart service process")
    
    return False


def ensure_service_alive() -> bool:
    """Ensure the service process is alive, restarting if necessary.
    
    This is a convenience function for pages that need to ensure
    the service is running before performing operations.
    
    Returns:
        True if service is alive (or was successfully restarted), False otherwise
    """
    controller = get_controller()
    
    if controller.is_service_alive():
        return True
    
    # Try to restart
    if controller.restart_service_if_dead():
        return True
    
    return False


def shutdown_controller() -> bool:
    """Explicitly shutdown the controller and service.
    
    This can be called when the user wants to explicitly stop
    the service, or during cleanup.
    
    Returns:
        True if shutdown was successful, False otherwise
    """
    if "experiment_controller" not in st.session_state:
        return True
    
    controller = st.session_state.experiment_controller
    try:
        result = controller.shutdown()
        logger.info(f"Controller shutdown: {'graceful' if result else 'forced'}")
        return result
    except Exception as e:
        logger.error(f"Error during controller shutdown: {e}")
        return False


def update_session_heartbeat():
    """Update the session heartbeat timestamp.
    
    This can be used to track when the session was last active,
    useful for cleanup or timeout logic.
    """
    st.session_state.last_heartbeat = datetime.now()


def get_session_age() -> float:
    """Get the age of the current session in seconds.
    
    Returns:
        Session age in seconds, or 0 if controller not yet created
    """
    if "controller_created_at" not in st.session_state:
        return 0.0
    
    created_at = st.session_state.controller_created_at
    return (datetime.now() - created_at).total_seconds()
