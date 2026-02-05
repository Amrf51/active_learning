"""Services package for the Active Learning Dashboard.

This package contains service processes that run in separate processes
for isolation and scalability.
"""

from .active_learning_service import ActiveLearningService, run_service_loop

__all__ = ['ActiveLearningService', 'run_service_loop']
