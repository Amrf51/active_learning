"""Controller factory for managing ExperimentController in Streamlit session state.

This module provides utilities for:
1. Creating and caching the ExperimentController in st.session_state
2. Managing session heartbeat for cleanup
3. Providing a consistent interface for all pages to access the controller
"""

import streamlit as st
from pathlib import Path
from datetime import datetime
import logging

from controller.experiment_controller import ExperimentController

logger = logging.getLogger(__name__)

# Default experiments directory
DEFAULT_EXPERIMENTS_DIR = Path("./experiments")


def get_controller() -> ExperimentController:
    """Get or create the ExperimentController from session state.
    
    This function ensures that only one controller instance exists per
    Streamlit session, stored in st.session_state for persistence across
    reruns.
    
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
        
        # Store in session state
        st.session_state.experiment_controller = controller
        st.session_state.controller_created_at = datetime.now()
        
        logger.info(f"ExperimentController created and stored in session state")
    
    return st.session_state.experiment_controller


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
