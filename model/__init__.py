"""Model layer for the Active Learning Dashboard MVC architecture.

This module provides state management, persistence, and backend orchestration.
"""

from .world_state import Phase, WorldState

__all__ = ['Phase', 'WorldState']
