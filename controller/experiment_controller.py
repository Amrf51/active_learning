"""ExperimentController - Routes events and manages background training threads.

The ExperimentController is the main entry point for the view layer. It:
1. Routes events to the ModelHandler
2. Manages background training threads
3. Provides thread-safe state access
4. Handles pause/stop control flow
"""

from pathlib import Path
from threading import Thread, Lock
from typing import Optional
import logging

from .events import Event, EventType
from model.model_handler import ModelHandler
from model.world_state import WorldState, Phase

logger = logging.getLogger(__name__)


class ExperimentController:
    """Controller for routing events and managing background threads."""
    
    def __init__(self, experiments_dir: Path):
        """Initialize ExperimentController.
        
        Args:
            experiments_dir: Root directory for experiment storage
        """
        self.experiments_dir = Path(experiments_dir)
        self._model_handler = ModelHandler(experiments_dir)
        
        # Threading
        self._training_thread: Optional[Thread] = None
        self._state_lock = Lock()  # Protects WorldState access
        
        logger.info(f"ExperimentController initialized with experiments_dir: {experiments_dir}")
    
    def dispatch(self, event: Event) -> None:
        """Route event to model handler.
        
        This is the main entry point for all user actions from the view layer.
        
        Args:
            event: Event to process
        """
        logger.info(f"Dispatching event: {event.type}")
        
        # Thread-safe event handling
        with self._state_lock:
            self._model_handler.handle_event(event)
    
    def get_state(self) -> WorldState:
        """Fast in-memory state read.
        
        Returns a consistent snapshot of WorldState, safe to call from
        any thread (including the UI thread during training).
        
        Returns:
            Current WorldState
        """
        with self._state_lock:
            return self._model_handler.world_state
    
    def get_status(self) -> dict:
        """Get experiment status as a dictionary.
        
        This is a convenience method for the view layer that returns
        the WorldState as a dictionary with additional computed fields.
        
        Returns:
            Dictionary with experiment status
        """
        with self._state_lock:
            state = self._model_handler.world_state
            return {
                'experiment_id': state.experiment_id,
                'experiment_name': state.experiment_name,
                'phase': state.phase.value if state.phase else 'idle',
                'current_cycle': state.current_cycle,
                'total_cycles': state.total_cycles,
                'current_epoch': state.current_epoch,
                'epochs_per_cycle': state.epochs_per_cycle,
                'labeled_count': state.labeled_count,
                'unlabeled_count': state.unlabeled_count,
                'error_message': state.error_message,
                'config': None,  # TODO: Add config to WorldState
                'current_cycle_epochs': state.epoch_metrics,
                'queried_images': state.queried_images,
                'probe_images': state.probe_images
            }
    
    def is_service_alive(self) -> bool:
        """Check if the service is alive.
        
        For the MVC implementation, the service is always alive since
        it runs in the same process. This method is provided for
        compatibility with the worker-based architecture.
        
        Returns:
            Always True for MVC implementation
        """
        return True
    
    def start_training_async(self) -> None:
        """Start training in background thread.
        
        Spawns a background thread that repeatedly calls train_epoch() until:
        - Training phase completes (all epochs done)
        - Pause is requested
        - Stop is requested
        - An error occurs
        
        The UI can poll get_state() to show live progress.
        """
        if self._training_thread is not None and self._training_thread.is_alive():
            logger.warning("Training thread already running")
            return
        
        logger.info("Starting background training thread")
        
        # Clear any previous control flags
        self._model_handler.clear_control_flags()
        
        # Create and start training thread
        self._training_thread = Thread(target=self._run_training, daemon=True)
        self._training_thread.start()
    
    def _run_training(self) -> None:
        """Training loop that runs in background thread.
        
        This method runs in a separate thread and repeatedly trains epochs
        until the training phase is complete or a control action is requested.
        """
        logger.info("Training thread started")
        
        try:
            while True:
                # Check control flags
                if self._model_handler.is_stopped():
                    logger.info("Training stopped by user")
                    break
                
                if self._model_handler.is_paused():
                    logger.info("Training paused by user")
                    break
                
                # Check if still in training phase
                with self._state_lock:
                    current_phase = self._model_handler.world_state.phase
                
                if current_phase != Phase.TRAINING:
                    logger.info(f"Training complete, phase transitioned to {current_phase}")
                    break
                
                # Train one epoch (thread-safe)
                with self._state_lock:
                    try:
                        metrics = self._model_handler.train_epoch()
                        logger.debug(f"Epoch {metrics.epoch} complete: "
                                   f"loss={metrics.train_loss:.4f}, "
                                   f"acc={metrics.train_accuracy:.4f}")
                    except Exception as e:
                        logger.error(f"Error during training epoch: {e}", exc_info=True)
                        self._model_handler.world_state.phase = Phase.ERROR
                        self._model_handler.world_state.error_message = str(e)
                        break
        
        except Exception as e:
            logger.error(f"Unexpected error in training thread: {e}", exc_info=True)
            with self._state_lock:
                self._model_handler.world_state.phase = Phase.ERROR
                self._model_handler.world_state.error_message = f"Training thread error: {str(e)}"
        
        finally:
            logger.info("Training thread finished")
    
    def is_training_active(self) -> bool:
        """Check if training thread is currently running.
        
        Returns:
            True if training thread is alive, False otherwise
        """
        return self._training_thread is not None and self._training_thread.is_alive()
    
    def wait_for_training(self, timeout: Optional[float] = None) -> bool:
        """Wait for training thread to complete.
        
        Args:
            timeout: Maximum time to wait in seconds (None = wait forever)
        
        Returns:
            True if thread completed, False if timeout occurred
        """
        if self._training_thread is None:
            return True
        
        self._training_thread.join(timeout=timeout)
        return not self._training_thread.is_alive()
