"""
Model layer for the Active Learning Dashboard.

This package provides the data structures and persistence layer for the
hybrid WorldState/SQLite architecture.
"""

from .world_state import WorldState
from .database import DatabaseManager
from .schemas import (
    EpochMetrics,
    CycleSummary,
    ExperimentConfig,
    ValidationResult,
    serialize_to_json,
    deserialize_from_json,
    validate_schema_dict
)

__version__ = "1.0.0"

__all__ = [
    'WorldState',
    'DatabaseManager',
    'EpochMetrics',
    'CycleSummary',
    'ExperimentConfig',
    'ValidationResult',
    'serialize_to_json',
    'deserialize_from_json',
    'validate_schema_dict'
]