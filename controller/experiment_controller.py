"""ExperimentController - Routes events via BackgroundWorker for multiprocessing.

The ExperimentController is the main entry point for the view layer. It:
1. Routes events to the service process via BackgroundWorker
2. Caches WorldState locally for fast UI reads
3. Polls for state updates from the service
4. Handles graceful service shutdown
"""

from pathlib import Path
from typing import Optional
import logging

from .events import Event, EventType
from .background_worker import BackgroundWorker
from model.world_state import WorldState, Phase

logger = logging.getLogger(__name__)


class ExperimentController:
    """Controller that communicates with service process via pipes.
    
    The ExperimentController uses multiprocessing for process isolation.
    Training runs in a separate service process, communicating via pipes.
    The controller caches WorldState locally for fast UI reads.
    
    Usage:
        controller = ExperimentController(experiments_dir)
        controller.dispatch(Event(EventType.START_CYCLE))
        controller.poll_updates()  # Get latest state from service
        state = controller.get_state()  # Fast local read
        controller.shutdown()  # Graceful cleanup
    """
    
    def __init__(self, experiments_dir: Path):
        """Initialize ExperimentController.
        
        Args:
            experiments_dir: Root directory for experiment storage
        """
        self.experiments_dir = Path(experiments_dir)
        self._worker = BackgroundWorker()
        self._cached_state = WorldState()  # Local cache for fast UI reads
        
        # Start the service process
        self._worker.start(experiments_dir)
        
        logger.info(f"ExperimentController initialized with experiments_dir: {experiments_dir}")
    
    def dispatch(self, event: Event) -> bool:
        """Send event to service process via pipe.
        
        This is the main entry point for all user actions from the view layer.
        Events are sent to the service process for processing.
        
        Args:
            event: Event to send to the service
            
        Returns:
            True if event was sent successfully, False otherwise
        """
        logger.info(f"Dispatching event: {event.type}")
        return self._worker.send_event(event)
    
    def get_state(self) -> WorldState:
        """Return cached state for fast UI reads.
        
        Call poll_updates() first to get fresh data from the service.
        This method returns the locally cached state without any I/O.
        
        If the service has died unexpectedly, the state will be updated
        to reflect an error condition.
        
        Returns:
            Current cached WorldState
        """
        # Check service health before returning state
        if not self._worker.is_alive() and self._cached_state.phase != Phase.IDLE:
            # Service died unexpectedly during operation
            if self._cached_state.phase not in (Phase.ERROR, Phase.COMPLETED):
                self._cached_state.phase = Phase.ERROR
                self._cached_state.error_message = "Service process died unexpectedly"
                self._cached_state.touch()
        
        return self._cached_state
    
    def poll_updates(self) -> bool:
        """Poll for state updates from service.
        
        Non-blocking check for WorldState updates from the service process.
        If an update is available, the cached state is replaced.
        
        Returns:
            True if state was updated, False otherwise
        """
        new_state = self._worker.poll_state()
        if new_state is not None:
            self._cached_state = new_state
            logger.debug(f"State updated (phase: {new_state.phase.value})")
            return True
        return False
    
    def drain_updates(self) -> WorldState:
        """Drain all pending updates and return latest state.
        
        Reads all pending WorldState updates from the pipe and returns
        the most recent one. Useful when you want to skip intermediate
        states and get the current state.
        
        Returns:
            The latest WorldState (cached state if no updates pending)
        """
        latest = self._worker.drain_all_states()
        if latest is not None:
            self._cached_state = latest
            logger.debug(f"Drained to latest state (phase: {latest.phase.value})")
        return self._cached_state
    
    def is_service_alive(self) -> bool:
        """Check if service process is running.
        
        Returns:
            True if service process is alive, False otherwise
        """
        return self._worker.is_alive()
    
    def check_service_health(self) -> bool:
        """Check service health and update state if service died.
        
        This method checks if the service is alive and updates the
        cached state to ERROR if the service died unexpectedly during
        an active operation.
        
        Returns:
            True if service is healthy, False if service died
        """
        if self._worker.is_alive():
            return True
        
        # Service is not alive - check if this is expected
        if self._cached_state.phase in (Phase.IDLE, Phase.COMPLETED, Phase.ERROR):
            # Service may have been shut down intentionally or already in error
            return True
        
        # Service died unexpectedly during operation
        logger.error("Service process died unexpectedly")
        self._cached_state.phase = Phase.ERROR
        self._cached_state.error_message = "Service process died unexpectedly"
        self._cached_state.touch()
        return False
    
    def restart_service_if_dead(self) -> bool:
        """Restart the service if it has died.
        
        Attempts to restart the service process if it's not running.
        Clears any error state from the previous crash.
        
        Returns:
            True if service was restarted, False if already running
        """
        if self._worker.restart_if_dead():
            # Clear error state from previous crash
            if self._cached_state.phase == Phase.ERROR:
                self._cached_state.phase = Phase.IDLE
                self._cached_state.error_message = None
                self._cached_state.touch()
            logger.info("Service process restarted successfully")
            return True
        return False
    
    def shutdown(self) -> bool:
        """Graceful shutdown of service.
        
        Sends SHUTDOWN event to the service and waits for it to terminate.
        If the service doesn't terminate gracefully, it's forcefully killed.
        
        Returns:
            True if shutdown was graceful, False if forced termination
        """
        logger.info("Shutting down ExperimentController")
        return self._worker.shutdown()
    
    def get_status(self) -> dict:
        """Get experiment status as a dictionary.
        
        This is a convenience method for the view layer that returns
        the WorldState as a dictionary with additional computed fields.
        
        Returns:
            Dictionary with experiment status
        """
        state = self._cached_state
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
