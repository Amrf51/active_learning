"""
Backend package for Active Learning MVC architecture.

This package contains the core backend modules for the active learning framework:
- models: Model creation and management
- state: Data structures for tracking AL state
- data_manager: Dataset and pool management
- trainer: Model training logic
- strategies: Active learning query strategies
- active_loop: Main active learning loop orchestration
- dataloader: Data loading utilities
"""

# Import key components for easier access
# Note: These imports will work once the modules are moved into the backend/ directory
# For now, they remain at the root level, so this package serves as a namespace marker

__all__ = [
    # Models
    "get_model",
    
    # State
    "EpochMetrics",
    "CycleMetrics",
    "QueriedImage",
    "ProbeImage",
    
    # Core components
    "DataManager",
    "Trainer",
    
    # Strategies
    "random_sampling",
    "uncertainty_entropy",
    "margin_sampling",
    "uncertainty_least_confidence",
    
    # Main loop
    "ActiveLearningLoop",
]
