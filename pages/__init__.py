"""
Views Package - View layer components for Active Learning Dashboard.

This package contains view-related utilities for the Streamlit dashboard,
primarily the controller factory for accessing the MVC Controller layer.
"""

from .controller_factory import (
    get_controller,
    get_session_manager,
    initialize_controller_session,
    update_session_heartbeat,
    cleanup_controller_session,
    with_controller
)

__all__ = [
    'get_controller',
    'get_session_manager',
    'initialize_controller_session',
    'update_session_heartbeat',
    'cleanup_controller_session',
    'with_controller'
]