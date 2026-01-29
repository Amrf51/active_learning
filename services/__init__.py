"""
Services package for Active Learning Dashboard.

This package contains service layer components that run in separate
processes and handle background operations like ML training.
"""

from .al_service import run_active_learning_service, ActiveLearningService

__all__ = ['run_active_learning_service', 'ActiveLearningService']