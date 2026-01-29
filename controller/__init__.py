"""
Controller Layer Initialization and Singleton Management.

This module provides the main entry points for accessing Controller layer
components. It implements singleton patterns using Streamlit's session_state
to ensure components survive script reruns while maintaining proper lifecycle.
"""

import logging
from pathlib import Path
from typing import Optional

# Import Streamlit with error handling for non-Streamlit environments
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    # Create a mock session state for testing
    class MockSessionState:
        def __init__(self):
            self._state = {}
        
        def __contains__(self, key):
            return key in self._state
        
        def __getitem__(self, key):
            return self._state[key]
        
        def __setitem__(self, key, value):
            self._state[key] = value
        
        def __delitem__(self, key):
            del self._state[key]
    
    st = type('MockStreamlit', (), {'session_state': MockSessionState()})()

from controller.dispatcher import EventDispatcher
from controller.model_handler import ModelHandler
from controller.service_manager import ServiceManager
from controller.session_manager import SessionManager
from model.world_state import WorldState
from model.database import DatabaseManager

logger = logging.getLogger(__name__)

# Global instances for non-Streamlit environments
_global_controller: Optional[EventDispatcher] = None
_global_session_manager: Optional[SessionManager] = None


def get_controller() -> EventDispatcher:
    """
    Get or create the Controller singleton.
    
    The Controller is stored in st.session_state to survive Streamlit reruns.
    This is the primary way to access the Controller from view components.
    
    Returns:
        EventDispatcher instance (Controller)
        
    Raises:
        RuntimeError: If controller creation fails
    """
    global _global_controller
    
    if STREAMLIT_AVAILABLE:
        # Use Streamlit session state
        if 'controller' not in st.session_state:
            logger.info("Creating new controller instance")
            st.session_state.controller = _create_controller()
        
        return st.session_state.controller
    else:
        # Use global instance for non-Streamlit environments (testing)
        if _global_controller is None:
            logger.info("Creating global controller instance")
            _global_controller = _create_controller()
        
        return _global_controller


def _create_controller() -> EventDispatcher:
    """
    Create and initialize the Controller with all dependencies.
    
    This function is called once per session to create the complete
    Controller layer with all its dependencies properly initialized.
    
    Returns:
        EventDispatcher instance
        
    Raises:
        RuntimeError: If initialization fails
    """
    try:
        logger.info("Initializing Controller layer...")
        
        # Get database path
        db_path = _get_database_path()
        logger.debug(f"Using database path: {db_path}")
        
        # Initialize Model layer components
        world_state = WorldState()
        db_manager = DatabaseManager(db_path)
        
        # Check for existing active experiment to restore
        active_experiment = db_manager.get_active_experiment()
        if active_experiment:
            logger.info(f"Restoring active experiment: {active_experiment['id']}")
            world_state.restore_from_db(active_experiment)
        else:
            logger.info("No active experiment found")
        
        # Create Controller layer components
        model_handler = ModelHandler(world_state, db_manager)
        service_manager = ServiceManager()
        
        # Create and return the main dispatcher
        dispatcher = EventDispatcher(
            model_handler=model_handler,
            service_manager=service_manager
        )
        
        logger.info("Controller layer initialized successfully")
        return dispatcher
        
    except Exception as e:
        logger.error(f"Failed to create controller: {e}")
        raise RuntimeError(f"Controller initialization failed: {str(e)}")


def get_session_manager() -> SessionManager:
    """
    Get or create the SessionManager singleton.
    
    Similar to get_controller(), but for session management.
    Uses the same singleton pattern with st.session_state.
    
    Returns:
        SessionManager instance
    """
    global _global_session_manager
    
    if STREAMLIT_AVAILABLE:
        # Use Streamlit session state
        if 'session_manager' not in st.session_state:
            logger.info("Creating new session manager instance")
            st.session_state.session_manager = SessionManager()
        
        return st.session_state.session_manager
    else:
        # Use global instance for non-Streamlit environments (testing)
        if _global_session_manager is None:
            logger.info("Creating global session manager instance")
            _global_session_manager = SessionManager()
        
        return _global_session_manager


def reset_controller() -> None:
    """
    Reset the Controller (for testing or error recovery).
    
    This function cleans up the existing controller and removes it
    from session state. The next call to get_controller() will
    create a fresh instance.
    
    Use this for:
    - Error recovery when controller is in bad state
    - Testing scenarios that need clean state
    - Manual reset operations
    """
    global _global_controller
    
    logger.info("Resetting controller...")
    
    try:
        if STREAMLIT_AVAILABLE:
            # Cleanup existing controller in Streamlit
            if 'controller' in st.session_state:
                ctrl = st.session_state.controller
                if hasattr(ctrl, 'shutdown'):
                    ctrl.shutdown()
                del st.session_state.controller
                logger.info("Controller removed from session state")
        else:
            # Cleanup global controller
            if _global_controller is not None:
                if hasattr(_global_controller, 'shutdown'):
                    _global_controller.shutdown()
                _global_controller = None
                logger.info("Global controller reset")
                
    except Exception as e:
        logger.error(f"Error during controller reset: {e}")
        # Force cleanup even if shutdown fails
        if STREAMLIT_AVAILABLE and 'controller' in st.session_state:
            del st.session_state.controller
        else:
            _global_controller = None


def reset_session_manager() -> None:
    """
    Reset the SessionManager (for testing or cleanup).
    
    Similar to reset_controller() but for session management.
    """
    global _global_session_manager
    
    logger.info("Resetting session manager...")
    
    try:
        if STREAMLIT_AVAILABLE:
            # Cleanup existing session manager in Streamlit
            if 'session_manager' in st.session_state:
                sm = st.session_state.session_manager
                if hasattr(sm, 'release_session'):
                    sm.release_session()
                del st.session_state.session_manager
                logger.info("Session manager removed from session state")
        else:
            # Cleanup global session manager
            if _global_session_manager is not None:
                if hasattr(_global_session_manager, 'release_session'):
                    _global_session_manager.release_session()
                _global_session_manager = None
                logger.info("Global session manager reset")
                
    except Exception as e:
        logger.error(f"Error during session manager reset: {e}")
        # Force cleanup even if release fails
        if STREAMLIT_AVAILABLE and 'session_manager' in st.session_state:
            del st.session_state.session_manager
        else:
            _global_session_manager = None


def _get_database_path() -> Path:
    """
    Get path to SQLite database.
    
    Creates the data directory if it doesn't exist and returns
    the path to the experiments database file.
    
    Returns:
        Path to SQLite database file
    """
    # Use user's home directory for data storage
    data_dir = Path.home() / ".al_dashboard" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    db_path = data_dir / "experiments.db"
    return db_path


def get_controller_info() -> dict:
    """
    Get information about the current controller state.
    
    Useful for debugging and monitoring. Returns information about
    whether components are initialized and their current state.
    
    Returns:
        Dictionary with controller information
    """
    info = {
        'streamlit_available': STREAMLIT_AVAILABLE,
        'controller_exists': False,
        'session_manager_exists': False,
        'database_path': str(_get_database_path()),
        'controller_info': None,
        'session_info': None
    }
    
    try:
        # Check controller existence
        if STREAMLIT_AVAILABLE:
            info['controller_exists'] = 'controller' in st.session_state
            info['session_manager_exists'] = 'session_manager' in st.session_state
        else:
            info['controller_exists'] = _global_controller is not None
            info['session_manager_exists'] = _global_session_manager is not None
        
        # Get controller info if it exists
        if info['controller_exists']:
            try:
                ctrl = get_controller()
                info['controller_info'] = {
                    'service_alive': ctrl.is_service_alive(),
                    'service_info': ctrl.get_service_info(),
                    'experiment_summary': ctrl.get_experiment_summary()
                }
            except Exception as e:
                info['controller_info'] = {'error': str(e)}
        
        # Get session manager info if it exists
        if info['session_manager_exists']:
            try:
                sm = get_session_manager()
                info['session_info'] = sm.get_session_info()
            except Exception as e:
                info['session_info'] = {'error': str(e)}
                
    except Exception as e:
        info['error'] = str(e)
    
    return info


def cleanup_all() -> None:
    """
    Cleanup all controller components.
    
    This function performs a complete cleanup of all controller
    components. Should be called when the application is shutting down.
    """
    logger.info("Cleaning up all controller components...")
    
    try:
        # Reset both controller and session manager
        reset_controller()
        reset_session_manager()
        
        logger.info("Controller cleanup complete")
        
    except Exception as e:
        logger.error(f"Error during controller cleanup: {e}")


# Convenience functions for common operations

def dispatch_event(event) -> bool:
    """
    Convenience function to dispatch an event.
    
    Args:
        event: Event to dispatch
        
    Returns:
        True if event was handled successfully
    """
    try:
        controller = get_controller()
        return controller.dispatch(event)
    except Exception as e:
        logger.error(f"Error dispatching event: {e}")
        return False


def get_experiment_status() -> dict:
    """
    Convenience function to get current experiment status.
    
    Returns:
        Dictionary with experiment status
    """
    try:
        controller = get_controller()
        return controller.get_status()
    except Exception as e:
        logger.error(f"Error getting experiment status: {e}")
        return {'error': str(e), 'phase': 'ERROR'}


def is_controller_ready() -> bool:
    """
    Check if the controller is ready for use.
    
    Returns:
        True if controller is initialized and ready
    """
    try:
        controller = get_controller()
        # Basic readiness check
        return controller is not None
    except Exception:
        return False


# Module-level cleanup registration
import atexit
atexit.register(cleanup_all)