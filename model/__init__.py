"""
Model Layer - Data Storage and State Management.

This package provides:
- WorldState: In-memory current experiment state (~1μs access)
- DatabaseManager: SQLite persistence for historical data (~50ms access)
- Schemas: Data structures used throughout the application

Usage:
    from model import WorldState, DatabaseManager
    from model.schemas import ExperimentPhase, EpochMetrics
    
    # Create instances
    world_state = WorldState()
    db = DatabaseManager("experiments.db")
"""

from .world_state import WorldState
from .database import DatabaseManager
from .schemas import (
    ExperimentPhase,
    EpochMetrics,
    CycleSummary,
    QueriedImage,
    ExperimentConfig,
    DatasetInfo,
    ValidationResult,
    TrainingProgress,
    ExperimentStatus
)

__all__ = [
    # Core classes
    "WorldState",
    "DatabaseManager",
    
    # Schemas
    "ExperimentPhase",
    "EpochMetrics",
    "CycleSummary",
    "QueriedImage",
    "ExperimentConfig",
    "DatasetInfo",
    "ValidationResult",
    "TrainingProgress",
    "ExperimentStatus"
]
