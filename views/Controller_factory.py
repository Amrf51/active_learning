"""
Controller Factory - Singleton access to Controller for Streamlit pages.

This provides a clean interface for views to access the Controller layer
without importing complex dependencies. The factory ensures only one
Controller instance exists per Streamlit session.

Usage in Streamlit pages:
    from views.controller_factory import get_controller
    
    def my_page():
        ctrl = get_controller()
        status = ctrl.get_status()
        # ... use controller ...
"""

import streamlit as st
import logging
from pathlib import Path
from typing import Optional

# Import from your actual structure
from controller.dispatcher import EventDispatcher
from controller.model_handler import ModelHandler
from controller.service_manager import ServiceManager
from controller.session_manager import SessionManager

logger = logging.getLogger(__name__)


def get_controller() -> EventDispatcher:
    """
    Get or create the Controller singleton.
    
    This function ensures only one Controller instance exists per Streamlit session.
    The Controller is stored in st.session_state and reused across page interactions.
    
    The Controller is automatically initialized with:
    - ModelHandler: For WorldState and Database access
    - ServiceManager: For spawning and managing the Service process
    
    Returns:
        EventDispatcher: The singleton Controller instance
    
    Example:
        >>> ctrl = get_controller()
        >>> status = ctrl.get_status()
        >>> print(status.phase)
    """
    if "controller" not in st.session_state:
        logger.info("Initializing Controller for new session")
        
        try:
            # Create dependencies
            from model.world_state import WorldState
            from model.database import DatabaseManager
            
            # Initialize model layer components
            world_state = WorldState()
            db_manager = DatabaseManager()
            
            # Create model handler with dependencies
            model_handler = ModelHandler(world_state, db_manager)
            service_manager = ServiceManager()
            
            # Create Controller
            controller = EventDispatcher(model_handler, service_manager)
            
            # Store in session state
            st.session_state.controller = controller
            
            logger.info("Controller initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Controller: {e}")
            st.error(f"❌ Failed to initialize system: {str(e)}")
            st.stop()
    
    return st.session_state.controller


def get_session_manager() -> SessionManager:
    """
    Get or create the SessionManager singleton.
    
    SessionManager prevents multiple browser tabs from running simultaneously
    by using a file-based heartbeat mechanism.
    
    Returns:
        SessionManager: The singleton SessionManager instance
    
    Example:
        >>> session_mgr = get_session_manager()
        >>> if session_mgr.acquire_session():
        ...     print("Session acquired!")
    """
    if "session_manager" not in st.session_state:
        logger.info("Initializing SessionManager for new session")
        st.session_state.session_manager = SessionManager()
    
    return st.session_state.session_manager


def initialize_controller_session() -> bool:
    """
    Initialize controller and check for active sessions.
    
    This function should be called ONCE at the start of dashboard.py before
    any pages load. It ensures:
    1. Only one browser tab can run the dashboard at a time
    2. The Controller is properly initialized
    3. Heartbeat mechanism is started
    
    Returns:
        bool: True if session acquired successfully, False if another session is active
    
    Example in dashboard.py:
        >>> from views.controller_factory import initialize_controller_session
        >>> 
        >>> # Page configuration
        >>> st.set_page_config(...)
        >>> 
        >>> # Initialize session (FIRST THING)
        >>> if not initialize_controller_session():
        ...     st.stop()
        >>> 
        >>> # ... rest of dashboard ...
    """
    session_mgr = get_session_manager()
    
    # Try to acquire session
    if not session_mgr.acquire_session():
        st.error("⚠️ **Another Dashboard Instance is Running**")
        st.markdown("""
        Another browser tab or window is already running the Active Learning Dashboard.
        
        **To proceed:**
        - Close the other dashboard tab/window
        - Wait 30 seconds for the session to timeout
        - Refresh this page
        
        **Why this check exists:**
        The dashboard manages background training processes and database connections.
        Running multiple instances simultaneously could cause data corruption or
        conflicting commands to the training service.
        """)
        st.stop()
        return False
    
    # Initialize controller
    try:
        get_controller()
        logger.info("Controller session initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize controller session: {e}")
        st.error(f"❌ Failed to initialize dashboard: {str(e)}")
        st.stop()
        return False


def update_session_heartbeat() -> None:
    """
    Update the session heartbeat to indicate this session is still active.
    
    Should be called periodically in the main dashboard page to prevent
    session timeout. Recommended to call this at the start of main page
    rendering or in a periodic callback.
    
    Example in dashboard.py:
        >>> from views.controller_factory import update_session_heartbeat
        >>> 
        >>> def main():
        ...     update_session_heartbeat()  # Update at start of page
        ...     # ... render page content ...
    """
    try:
        session_mgr = get_session_manager()
        session_mgr.update_heartbeat()
    except Exception as e:
        logger.warning(f"Failed to update session heartbeat: {e}")


def cleanup_controller_session() -> None:
    """
    Clean up controller session and release resources.
    
    Should be called when the dashboard is closing or when explicitly
    stopping the session. This releases the session lock and terminates
    the service process.
    
    Note: Streamlit doesn't have reliable shutdown hooks, so this may
    need to be called manually or via a "Shutdown" button.
    
    Example:
        >>> from views.controller_factory import cleanup_controller_session
        >>> 
        >>> if st.button("🛑 Shutdown Dashboard"):
        ...     cleanup_controller_session()
        ...     st.success("Dashboard shut down successfully")
        ...     st.stop()
    """
    try:
        # Release session
        if "session_manager" in st.session_state:
            session_mgr = st.session_state.session_manager
            session_mgr.release_session()
            logger.info("Session released")
        
        # Stop service if running
        if "controller" in st.session_state:
            ctrl = st.session_state.controller
            # Dispatcher can access service_manager to terminate process
            if hasattr(ctrl, '_service_manager'):
                ctrl._service_manager.terminate()
                logger.info("Service process terminated")
        
        # Clear session state
        if "controller" in st.session_state:
            del st.session_state.controller
        if "session_manager" in st.session_state:
            del st.session_state.session_manager
        
        logger.info("Controller session cleaned up successfully")
        
    except Exception as e:
        logger.error(f"Error during session cleanup: {e}")


def with_controller(func):
    """
    Decorator to ensure controller is available before executing view function.
    
    Usage:
        >>> from views.controller_factory import with_controller
        >>> 
        >>> @with_controller
        >>> def my_page():
        ...     ctrl = get_controller()
        ...     # ... use controller ...
    """
    def wrapper(*args, **kwargs):
        try:
            # Ensure controller is initialized
            get_controller()
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            st.error(f"❌ Error: {str(e)}")
            return None
    
    return wrapper


# Export main functions
__all__ = [
    'get_controller',
    'get_session_manager',
    'initialize_controller_session',
    'update_session_heartbeat',
    'cleanup_controller_session',
    'with_controller'
]