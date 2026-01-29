"""
WorldState: In-memory experiment state for fast UI access.

This module provides the WorldState dataclass that maintains current experiment
state in memory for fast access (~1ms). It's designed to be thread-safe for
updates from the service listener thread while providing immediate access for
the UI components.
"""

from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid


@dataclass
class WorldState:
    """
    In-memory experiment state for fast UI access.
    
    This class maintains the current experiment state in memory to provide
    fast access for UI components. It's synchronized with SQLite for persistence
    but optimized for immediate read access.
    
    Thread Safety:
    - Uses a Lock to ensure thread-safe updates from service listener thread
    - All state modifications should acquire the lock
    - Read operations are atomic for basic types
    
    State Categories:
    - Identity: experiment_id, experiment_name
    - Configuration: config dict
    - Phase: phase, error_message  
    - Progress: current_cycle, total_cycles, current_epoch, epochs_per_cycle
    - Metrics: current_metrics, epoch_history (current cycle only)
    - Query: queried_images (current batch only)
    - UI: pending_updates flag
    """
    
    # Identity
    experiment_id: Optional[str] = None
    experiment_name: Optional[str] = None
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Phase and Status
    phase: str = "IDLE"  # IDLE, INITIALIZING, TRAINING, AWAITING_ANNOTATION, ERROR
    error_message: Optional[str] = None
    
    # Progress Tracking
    current_cycle: int = 0
    total_cycles: int = 0
    current_epoch: int = 0
    epochs_per_cycle: int = 0
    
    # Current Metrics (for current cycle only)
    current_metrics: Dict[str, Any] = field(default_factory=dict)
    epoch_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Query State (current batch only)
    queried_images: List[Dict[str, Any]] = field(default_factory=list)
    
    # UI State
    pending_updates: bool = False
    
    # Thread safety
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    
    def __post_init__(self):
        """Initialize the lock after dataclass creation."""
        if not hasattr(self, '_lock'):
            self._lock = Lock()
    
    def reset(self) -> None:
        """
        Reset WorldState for a new experiment.
        
        Clears all experiment-specific state while preserving the lock.
        Should be called when starting a new experiment.
        """
        with self._lock:
            self.experiment_id = None
            self.experiment_name = None
            self.config = {}
            self.phase = "IDLE"
            self.error_message = None
            self.current_cycle = 0
            self.total_cycles = 0
            self.current_epoch = 0
            self.epochs_per_cycle = 0
            self.current_metrics = {}
            self.epoch_history = []
            self.queried_images = []
            self.pending_updates = False
    
    def restore_from_db(self, experiment_data: Dict[str, Any]) -> None:
        """
        Restore WorldState from database experiment data.
        
        Used during application startup to restore the state of an active
        experiment from the SQLite database.
        
        Args:
            experiment_data: Dictionary containing experiment data from database
                Expected keys: id, name, config_json, phase, current_cycle, etc.
        """
        with self._lock:
            self.experiment_id = experiment_data.get('id')
            self.experiment_name = experiment_data.get('name')
            self.config = experiment_data.get('config', {})
            self.phase = experiment_data.get('phase', 'IDLE')
            self.error_message = experiment_data.get('error_message')
            self.current_cycle = experiment_data.get('current_cycle', 0)
            self.total_cycles = experiment_data.get('total_cycles', 0)
            self.current_epoch = experiment_data.get('current_epoch', 0)
            self.epochs_per_cycle = experiment_data.get('epochs_per_cycle', 0)
            
            # Reset transient state (not persisted)
            self.current_metrics = {}
            self.epoch_history = []
            self.queried_images = []
            self.pending_updates = False
    
    def initialize_experiment(self, experiment_name: str, config: Dict[str, Any]) -> str:
        """
        Initialize a new experiment.
        
        Args:
            experiment_name: Name of the experiment
            config: Experiment configuration dictionary
            
        Returns:
            experiment_id: Generated unique experiment ID
        """
        with self._lock:
            self.experiment_id = str(uuid.uuid4())
            self.experiment_name = experiment_name
            self.config = config.copy()
            self.phase = "INITIALIZING"
            self.error_message = None
            self.current_cycle = 0
            self.total_cycles = config.get('total_cycles', 10)
            self.current_epoch = 0
            self.epochs_per_cycle = config.get('epochs_per_cycle', 5)
            self.current_metrics = {}
            self.epoch_history = []
            self.queried_images = []
            self.pending_updates = True
            
            return self.experiment_id
    
    def set_phase(self, phase: str, error_message: Optional[str] = None) -> None:
        """
        Update the experiment phase.
        
        Args:
            phase: New phase (IDLE, INITIALIZING, TRAINING, AWAITING_ANNOTATION, ERROR)
            error_message: Optional error message if phase is ERROR
        """
        with self._lock:
            self.phase = phase
            self.error_message = error_message
            self.pending_updates = True
    
    def update_training_progress(self, current_epoch: int, metrics: Dict[str, Any]) -> None:
        """
        Update training progress and metrics.
        
        Args:
            current_epoch: Current epoch number
            metrics: Training metrics for this epoch
        """
        with self._lock:
            self.current_epoch = current_epoch
            self.current_metrics = metrics.copy()
            
            # Add to epoch history for current cycle
            epoch_data = {
                'epoch': current_epoch,
                'timestamp': datetime.now().isoformat(),
                **metrics
            }
            self.epoch_history.append(epoch_data)
            
            self.pending_updates = True
    
    def start_new_cycle(self, cycle_number: int) -> None:
        """
        Start a new active learning cycle.
        
        Args:
            cycle_number: The cycle number being started
        """
        with self._lock:
            self.current_cycle = cycle_number
            self.current_epoch = 0
            self.current_metrics = {}
            self.epoch_history = []  # Reset for new cycle
            self.queried_images = []
            self.phase = "TRAINING"
            self.pending_updates = True
    
    def set_queried_images(self, images: List[Dict[str, Any]]) -> None:
        """
        Set the queried images for annotation.
        
        Args:
            images: List of queried image data
        """
        with self._lock:
            self.queried_images = images.copy()
            self.phase = "AWAITING_ANNOTATION"
            self.pending_updates = True
    
    def complete_cycle(self, cycle_results: Dict[str, Any]) -> None:
        """
        Complete the current cycle.
        
        Args:
            cycle_results: Results summary for the completed cycle
        """
        with self._lock:
            # Update any final metrics from cycle results
            if 'final_metrics' in cycle_results:
                self.current_metrics.update(cycle_results['final_metrics'])
            
            # Clear transient state
            self.queried_images = []
            self.phase = "IDLE"
            self.pending_updates = True
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current experiment status.
        
        Returns:
            Dictionary with phase, cycle info, and error message
        """
        # No lock needed for read-only access to atomic fields
        return {
            'experiment_id': self.experiment_id,
            'experiment_name': self.experiment_name,
            'phase': self.phase,
            'error_message': self.error_message,
            'current_cycle': self.current_cycle,
            'total_cycles': self.total_cycles,
            'progress_percentage': (self.current_cycle / max(self.total_cycles, 1)) * 100
        }
    
    def get_training_progress(self) -> Dict[str, Any]:
        """
        Get current training progress.
        
        Returns:
            Dictionary with epoch info, metrics, and history
        """
        # Create a snapshot under lock to ensure consistency
        with self._lock:
            return {
                'current_epoch': self.current_epoch,
                'epochs_per_cycle': self.epochs_per_cycle,
                'current_metrics': self.current_metrics.copy(),
                'epoch_history': self.epoch_history.copy(),
                'progress_percentage': (self.current_epoch / max(self.epochs_per_cycle, 1)) * 100
            }
    
    def get_queried_images(self) -> List[Dict[str, Any]]:
        """
        Get current queried images for annotation.
        
        Returns:
            List of queried image data
        """
        with self._lock:
            return self.queried_images.copy()
    
    def has_pending_updates(self) -> bool:
        """
        Check if there are pending UI updates.
        
        Returns:
            True if UI should refresh
        """
        return self.pending_updates
    
    def clear_pending_updates(self) -> None:
        """Clear the pending updates flag."""
        with self._lock:
            self.pending_updates = False
    
    def set_pending_updates(self, value: bool = True) -> None:
        """
        Set the pending updates flag.
        
        Args:
            value: Whether there are pending updates
        """
        with self._lock:
            self.pending_updates = value