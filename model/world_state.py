"""
WorldState - In-Memory Experiment State.

This module provides the WorldState class which holds the current experiment
state in memory for microsecond-level access. This replaces the JSON file
approach that required ~500ms reads.

Key Design Decisions:
1. Pure data container - no business logic
2. Thread-safe with simple Lock (for listener thread updates)
3. Only holds CURRENT state - historical data goes to SQLite
4. Lightweight - no Pydantic validation overhead

Usage:
    world_state = WorldState()
    world_state.phase = ExperimentPhase.TRAINING
    world_state.current_epoch = 5
    
    # Thread-safe update from listener
    with world_state.lock:
        world_state.current_metrics = new_metrics
        world_state.pending_updates = True
"""

from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional
import logging

from .schemas import (
    ExperimentPhase,
    ExperimentConfig,
    EpochMetrics,
    CycleSummary,
    QueriedImage,
    DatasetInfo
)

logger = logging.getLogger(__name__)


@dataclass
class WorldState:
    """
    In-memory representation of current experiment state.
    
    This is the SINGLE SOURCE OF TRUTH for current state.
    Historical data is stored in SQLite.
    
    Thread Safety:
        The listener thread (receiving events from Service) and the main
        Streamlit thread both access WorldState. Use the lock for updates:
        
        with world_state.lock:
            world_state.current_metrics = new_metrics
    
    Attributes:
        # Identity
        experiment_id: Unique identifier
        experiment_name: Human-readable name
        created_at: When experiment was created
        
        # Phase & Status
        phase: Current experiment phase
        error_message: Error details if phase is ERROR
        
        # Progress
        current_cycle: Current AL cycle (1-indexed)
        total_cycles: Total cycles to run
        current_epoch: Current epoch within cycle (1-indexed)
        epochs_per_cycle: Epochs per cycle from config
        
        # Pool Sizes
        labeled_count: Current labeled pool size
        unlabeled_count: Current unlabeled pool size
        
        # Current Cycle Data (reset each cycle)
        current_metrics: Latest epoch metrics
        epoch_history: All epochs in current cycle
        
        # Query Data
        queried_images: Images awaiting annotation
        
        # Configuration
        config: Experiment configuration (immutable)
        dataset_info: Dataset statistics
        
        # UI Synchronization
        pending_updates: Flag for UI to check for new data
    """
    
    # ═══════════════════════════════════════════════════════════════════
    # Identity
    # ═══════════════════════════════════════════════════════════════════
    experiment_id: Optional[str] = None
    experiment_name: Optional[str] = None
    created_at: Optional[datetime] = None
    
    # ═══════════════════════════════════════════════════════════════════
    # Phase & Status
    # ═══════════════════════════════════════════════════════════════════
    phase: ExperimentPhase = ExperimentPhase.IDLE
    error_message: Optional[str] = None
    
    # ═══════════════════════════════════════════════════════════════════
    # Progress
    # ═══════════════════════════════════════════════════════════════════
    current_cycle: int = 0
    total_cycles: int = 0
    current_epoch: int = 0
    epochs_per_cycle: int = 0
    
    # ═══════════════════════════════════════════════════════════════════
    # Pool Sizes
    # ═══════════════════════════════════════════════════════════════════
    labeled_count: int = 0
    unlabeled_count: int = 0
    
    # ═══════════════════════════════════════════════════════════════════
    # Current Cycle Data
    # ═══════════════════════════════════════════════════════════════════
    current_metrics: Optional[EpochMetrics] = None
    epoch_history: List[EpochMetrics] = field(default_factory=list)
    
    # ═══════════════════════════════════════════════════════════════════
    # Query Data
    # ═══════════════════════════════════════════════════════════════════
    queried_images: List[QueriedImage] = field(default_factory=list)
    
    # ═══════════════════════════════════════════════════════════════════
    # Configuration
    # ═══════════════════════════════════════════════════════════════════
    config: Optional[ExperimentConfig] = None
    dataset_info: Optional[DatasetInfo] = None
    
    # ═══════════════════════════════════════════════════════════════════
    # UI Synchronization
    # ═══════════════════════════════════════════════════════════════════
    pending_updates: bool = False
    
    # ═══════════════════════════════════════════════════════════════════
    # Thread Safety
    # ═══════════════════════════════════════════════════════════════════
    _lock: Lock = field(default_factory=Lock, repr=False)
    
    @property
    def lock(self) -> Lock:
        """Get the thread lock for safe updates."""
        return self._lock
    
    # ═══════════════════════════════════════════════════════════════════
    # Initialization Methods
    # ═══════════════════════════════════════════════════════════════════
    
    def initialize(
        self,
        experiment_id: str,
        experiment_name: str,
        config: ExperimentConfig,
        dataset_info: Optional[DatasetInfo] = None
    ) -> None:
        """
        Initialize WorldState for a new experiment.
        
        Args:
            experiment_id: Unique identifier
            experiment_name: Human-readable name
            config: Experiment configuration
            dataset_info: Optional dataset statistics
        """
        with self._lock:
            self.experiment_id = experiment_id
            self.experiment_name = experiment_name
            self.created_at = datetime.now()
            self.config = config
            self.dataset_info = dataset_info
            
            # Set from config
            self.total_cycles = config.num_cycles
            self.epochs_per_cycle = config.epochs_per_cycle
            
            # Reset progress
            self.phase = ExperimentPhase.IDLE
            self.error_message = None
            self.current_cycle = 0
            self.current_epoch = 0
            self.labeled_count = config.initial_pool_size
            self.unlabeled_count = 0  # Will be set when data loads
            
            # Clear cycle data
            self.current_metrics = None
            self.epoch_history = []
            self.queried_images = []
            
            self.pending_updates = True
            
        logger.info(f"WorldState initialized: {experiment_id}")
    
    def reset(self) -> None:
        """
        Reset WorldState to initial empty state.
        
        Called when starting a new experiment or on error recovery.
        """
        with self._lock:
            self.experiment_id = None
            self.experiment_name = None
            self.created_at = None
            self.config = None
            self.dataset_info = None
            
            self.phase = ExperimentPhase.IDLE
            self.error_message = None
            
            self.current_cycle = 0
            self.total_cycles = 0
            self.current_epoch = 0
            self.epochs_per_cycle = 0
            
            self.labeled_count = 0
            self.unlabeled_count = 0
            
            self.current_metrics = None
            self.epoch_history = []
            self.queried_images = []
            
            self.pending_updates = False
            
        logger.info("WorldState reset to empty")
    
    def restore_from_db(self, experiment_data: Dict[str, Any]) -> None:
        """
        Restore WorldState from database record.
        
        Called on session recovery when loading an existing experiment.
        
        Args:
            experiment_data: Dict from DatabaseManager.get_active_experiment()
        """
        with self._lock:
            self.experiment_id = experiment_data["experiment_id"]
            self.experiment_name = experiment_data["experiment_name"]
            self.created_at = experiment_data.get("created_at")
            
            if experiment_data.get("config"):
                self.config = ExperimentConfig.from_dict(experiment_data["config"])
                self.total_cycles = self.config.num_cycles
                self.epochs_per_cycle = self.config.epochs_per_cycle
            
            self.phase = ExperimentPhase(experiment_data.get("phase", "IDLE"))
            self.current_cycle = experiment_data.get("current_cycle", 0)
            self.labeled_count = experiment_data.get("labeled_count", 0)
            self.unlabeled_count = experiment_data.get("unlabeled_count", 0)
            
            # Clear transient data - will be populated if needed
            self.current_epoch = 0
            self.current_metrics = None
            self.epoch_history = []
            self.queried_images = []
            
            self.pending_updates = True
            
        logger.info(f"WorldState restored from DB: {self.experiment_id}")
    
    # ═══════════════════════════════════════════════════════════════════
    # State Update Methods
    # ═══════════════════════════════════════════════════════════════════
    
    def set_phase(self, phase: ExperimentPhase, error_message: Optional[str] = None) -> None:
        """
        Update the experiment phase.
        
        Args:
            phase: New phase
            error_message: Error details if phase is ERROR
        """
        with self._lock:
            old_phase = self.phase
            self.phase = phase
            if error_message:
                self.error_message = error_message
            elif phase != ExperimentPhase.ERROR:
                self.error_message = None
            self.pending_updates = True
            
        logger.info(f"Phase changed: {old_phase.value} → {phase.value}")
    
    def set_error(self, message: str) -> None:
        """
        Set error state with message.
        
        Args:
            message: Error description
        """
        self.set_phase(ExperimentPhase.ERROR, error_message=message)
    
    def start_cycle(self, cycle: int, labeled_count: int, unlabeled_count: int) -> None:
        """
        Start a new AL cycle.
        
        Args:
            cycle: Cycle number (1-indexed)
            labeled_count: Current labeled pool size
            unlabeled_count: Current unlabeled pool size
        """
        with self._lock:
            self.current_cycle = cycle
            self.current_epoch = 0
            self.labeled_count = labeled_count
            self.unlabeled_count = unlabeled_count
            self.current_metrics = None
            self.epoch_history = []
            self.queried_images = []
            self.pending_updates = True
            
        logger.info(f"Started cycle {cycle}: labeled={labeled_count}, unlabeled={unlabeled_count}")
    
    def update_epoch(self, metrics: EpochMetrics) -> None:
        """
        Update with new epoch metrics.
        
        Called by listener thread when EPOCH_COMPLETE event arrives.
        
        Args:
            metrics: Metrics for completed epoch
        """
        with self._lock:
            self.current_epoch = metrics.epoch
            self.current_metrics = metrics
            self.epoch_history.append(metrics)
            self.pending_updates = True
            
        logger.debug(f"Epoch {metrics.epoch} complete: loss={metrics.train_loss:.4f}, acc={metrics.val_accuracy:.4f}")
    
    def set_queried_images(self, images: List[QueriedImage]) -> None:
        """
        Set images awaiting annotation.
        
        Args:
            images: List of queried images
        """
        with self._lock:
            self.queried_images = images
            self.pending_updates = True
            
        logger.info(f"Set {len(images)} queried images")
    
    def finalize_cycle(self, labeled_count: int, unlabeled_count: int) -> None:
        """
        Finalize current cycle after annotations.
        
        Args:
            labeled_count: Updated labeled pool size
            unlabeled_count: Updated unlabeled pool size
        """
        with self._lock:
            self.labeled_count = labeled_count
            self.unlabeled_count = unlabeled_count
            self.queried_images = []
            self.pending_updates = True
            
        logger.info(f"Cycle {self.current_cycle} finalized: labeled={labeled_count}")
    
    # ═══════════════════════════════════════════════════════════════════
    # UI Synchronization
    # ═══════════════════════════════════════════════════════════════════
    
    def has_pending_updates(self) -> bool:
        """Check if there are pending updates for UI."""
        return self.pending_updates
    
    def clear_pending_updates(self) -> None:
        """Clear the pending updates flag after UI refresh."""
        with self._lock:
            self.pending_updates = False
    
    def mark_updated(self) -> None:
        """Mark that state has been updated (triggers UI refresh)."""
        with self._lock:
            self.pending_updates = True
    
    # ═══════════════════════════════════════════════════════════════════
    # Computed Properties
    # ═══════════════════════════════════════════════════════════════════
    
    @property
    def is_initialized(self) -> bool:
        """Check if an experiment is loaded."""
        return self.experiment_id is not None
    
    @property
    def is_running(self) -> bool:
        """Check if experiment is actively running."""
        active_phases = {
            ExperimentPhase.INITIALIZING,
            ExperimentPhase.PREPARING,
            ExperimentPhase.TRAINING,
            ExperimentPhase.VALIDATING,
            ExperimentPhase.EVALUATING,
            ExperimentPhase.QUERYING
        }
        return self.phase in active_phases
    
    @property
    def is_complete(self) -> bool:
        """Check if experiment is complete."""
        return self.phase == ExperimentPhase.COMPLETED
    
    @property
    def has_error(self) -> bool:
        """Check if experiment is in error state."""
        return self.phase == ExperimentPhase.ERROR
    
    @property
    def progress_percentage(self) -> float:
        """
        Calculate overall progress percentage.
        
        Returns:
            Progress from 0.0 to 100.0
        """
        if self.total_cycles == 0:
            return 0.0
        
        # Base progress from completed cycles
        cycle_progress = (self.current_cycle - 1) / self.total_cycles if self.current_cycle > 0 else 0
        
        # Add progress within current cycle
        if self.epochs_per_cycle > 0 and self.current_cycle > 0:
            epoch_progress = self.current_epoch / self.epochs_per_cycle
            cycle_contribution = epoch_progress / self.total_cycles
            return (cycle_progress + cycle_contribution) * 100
        
        return cycle_progress * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize WorldState to dictionary.
        
        Used for debugging and logging.
        """
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "phase": self.phase.value,
            "current_cycle": self.current_cycle,
            "total_cycles": self.total_cycles,
            "current_epoch": self.current_epoch,
            "epochs_per_cycle": self.epochs_per_cycle,
            "labeled_count": self.labeled_count,
            "unlabeled_count": self.unlabeled_count,
            "pending_updates": self.pending_updates,
            "error_message": self.error_message
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current experiment status for view layer.
        
        Returns:
            Dictionary with current status information
        """
        return {
            'experiment_id': self.experiment_id,
            'experiment_name': self.experiment_name,
            'created_at': self.created_at,
            'phase': self.phase.value,
            'error_message': self.error_message,
            'current_cycle': self.current_cycle,
            'total_cycles': self.total_cycles,
            'current_epoch': self.current_epoch,
            'epochs_per_cycle': self.epochs_per_cycle,
            'labeled_count': self.labeled_count,
            'unlabeled_count': self.unlabeled_count,
            'progress_percentage': self.progress_percentage,
            'is_initialized': self.is_initialized,
            'is_running': self.is_running,
            'is_complete': self.is_complete,
            'has_error': self.has_error
        }
    
    def get_training_progress(self) -> Dict[str, Any]:
        """
        Get current training progress for view layer.
        
        Returns:
            Dictionary with training progress information
        """
        return {
            'current_epoch': self.current_epoch,
            'epochs_per_cycle': self.epochs_per_cycle,
            'current_metrics': self.current_metrics.to_dict() if self.current_metrics else None,
            'epoch_history': [metrics.to_dict() for metrics in self.epoch_history],
            'has_metrics': self.current_metrics is not None,
            'has_history': len(self.epoch_history) > 0
        }
    
    def get_queried_images(self) -> List[Dict[str, Any]]:
        """
        Get current queried images for view layer.
        
        Returns:
            List of queried image data
        """
        return [img.to_dict() if hasattr(img, 'to_dict') else img for img in self.queried_images]
    
    def set_pending_updates(self, value: bool = True) -> None:
        """
        Set the pending updates flag.
        
        Args:
            value: Whether there are pending updates
        """
        with self._lock:
            self.pending_updates = value
    
    def update_training_progress(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """
        Update training progress with new epoch metrics.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary with training metrics
        """
        # Convert dict to EpochMetrics if needed
        if isinstance(metrics, dict):
            epoch_metrics = EpochMetrics(
                epoch=epoch,
                train_loss=metrics.get('train_loss', 0.0),
                train_accuracy=metrics.get('train_accuracy', 0.0),
                val_loss=metrics.get('val_loss', 0.0),
                val_accuracy=metrics.get('val_accuracy', 0.0),
                learning_rate=metrics.get('learning_rate', 0.0)
            )
        else:
            epoch_metrics = metrics
        
        self.update_epoch(epoch_metrics)
    
    def complete_cycle(self, results: Dict[str, Any]) -> None:
        """
        Complete the current cycle with results.
        
        Args:
            results: Cycle completion results
        """
        with self._lock:
            # Update pool counts from results
            self.labeled_count = results.get('labeled_count', self.labeled_count)
            self.unlabeled_count = results.get('unlabeled_count', self.unlabeled_count)
            
            # Clear cycle-specific data
            self.queried_images = []
            self.current_epoch = 0
            self.current_metrics = None
            self.epoch_history = []
            
            # Set to idle phase
            self.phase = ExperimentPhase.IDLE
            self.pending_updates = True
            
        logger.info(f"Cycle {self.current_cycle} completed")
    
    def initialize_experiment(self, experiment_name: str, config: Dict[str, Any]) -> str:
        """
        Initialize a new experiment and return generated ID.
        
        Args:
            experiment_name: Human-readable name
            config: Experiment configuration
            
        Returns:
            Generated experiment ID
        """
        # Generate unique experiment ID
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Convert config dict to ExperimentConfig if needed
        if isinstance(config, dict):
            exp_config = ExperimentConfig.from_dict(config)
        else:
            exp_config = config
        
        # Initialize the experiment
        self.initialize(experiment_id, experiment_name, exp_config)
        
        return experiment_id

    def __repr__(self) -> str:
        return (
            f"WorldState(id={self.experiment_id}, phase={self.phase.value}, "
            f"cycle={self.current_cycle}/{self.total_cycles}, "
            f"epoch={self.current_epoch}/{self.epochs_per_cycle})"
        )
