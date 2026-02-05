"""Controller layer for the Active Learning Dashboard MVC architecture.

This module provides event routing and thread management for the dashboard.
"""

from .events import Event, EventType
from .experiment_controller import ExperimentController

__all__ = ['Event', 'EventType', 'ExperimentController']
