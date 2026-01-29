"""
Service Layer - Background Computation.

This package provides the ActiveLearningService which runs in a
separate process and handles all heavy computation.

The Service replaces:
- run_worker.py (manual worker start)
- worker_command_loop.py (command polling)

Usage:
    # This is called by ServiceManager, not directly
    from services.al_service import run_active_learning_service
    
    # ServiceManager does:
    parent_conn, child_conn = Pipe()
    process = Process(target=run_active_learning_service, args=(child_conn, config))
    process.start()
"""

from .al_service import run_active_learning_service, ActiveLearningService

__all__ = [
    "run_active_learning_service",
    "ActiveLearningService"
]
