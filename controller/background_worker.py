"""BackgroundWorker - Manages the ActiveLearning service process.

This module provides process management for the multiprocessing architecture.
The BackgroundWorker spawns a separate process for the ActiveLearningService
and handles bidirectional pipe communication.

Key responsibilities:
- Spawn and manage the service process lifecycle
- Send events to the service via pipe
- Receive WorldState updates from the service
- Handle graceful shutdown with timeout
"""

from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Optional, Callable
import logging
import time

from controller.events import Event, EventType
from model.world_state import WorldState

logger = logging.getLogger(__name__)


class BackgroundWorker:
    """Manages the ActiveLearning service process.
    
    The BackgroundWorker is responsible for:
    1. Spawning the service process with a bidirectional pipe
    2. Sending events to the service (controller -> service)
    3. Polling for WorldState updates (service -> controller)
    4. Monitoring process health
    5. Graceful shutdown with timeout
    
    Usage:
        worker = BackgroundWorker()
        worker.start(experiments_dir)
        worker.send_event(Event(EventType.START_CYCLE))
        state = worker.poll_state()
        worker.shutdown()
    """
    
    def __init__(self):
        """Initialize BackgroundWorker."""
        self._process: Optional[Process] = None
        self._pipe: Optional[Connection] = None  # Parent end of pipe
        self._is_started = False
        self._experiments_dir: Optional[Path] = None
    
    def start(self, experiments_dir: Path, service_entry_point: Optional[Callable] = None) -> None:
        """Spawn the service process with pipe connection.
        
        Args:
            experiments_dir: Root directory for experiment storage
            service_entry_point: Optional custom entry point function for testing.
                                 If None, uses the default run_service_loop.
        """
        if self._is_started:
            logger.warning("BackgroundWorker already started")
            return
        
        self._experiments_dir = Path(experiments_dir)
        
        # Create bidirectional pipe
        parent_conn, child_conn = Pipe(duplex=True)
        self._pipe = parent_conn
        
        # Import here to avoid circular imports
        if service_entry_point is None:
            from services.active_learning_service import run_service_loop
            service_entry_point = run_service_loop
        
        # Spawn service process
        self._process = Process(
            target=service_entry_point,
            args=(child_conn, self._experiments_dir),
            daemon=False  # Allow graceful shutdown
        )
        self._process.start()
        self._is_started = True
        
        logger.info(f"BackgroundWorker started service process (PID: {self._process.pid})")
    
    def send_event(self, event: Event) -> bool:
        """Send event to service via pipe.
        
        Args:
            event: Event to send to the service
            
        Returns:
            True if event was sent successfully, False otherwise
        """
        if not self._is_started or self._pipe is None:
            logger.warning("Cannot send event: worker not started")
            return False
        
        if not self.is_alive():
            logger.error("Cannot send event: service process is not alive")
            return False
        
        try:
            self._pipe.send(event)
            logger.debug(f"Sent event: {event.type.value}")
            return True
        except Exception as e:
            logger.error(f"Failed to send event: {e}")
            return False
    
    def poll_state(self, timeout: float = 0.0) -> Optional[WorldState]:
        """Non-blocking check for state updates from service.
        
        Args:
            timeout: How long to wait for data (0.0 = non-blocking)
            
        Returns:
            WorldState if available, None otherwise
        """
        if not self._is_started or self._pipe is None:
            return None
        
        try:
            if self._pipe.poll(timeout):
                state = self._pipe.recv()
                if isinstance(state, WorldState):
                    logger.debug(f"Received state update (phase: {state.phase.value})")
                    return state
                else:
                    logger.warning(f"Received unexpected data type: {type(state)}")
                    return None
        except EOFError:
            logger.warning("Pipe closed by service")
            return None
        except Exception as e:
            logger.error(f"Error polling state: {e}")
            return None
        
        return None
    
    def drain_all_states(self, timeout: float = 0.0) -> Optional[WorldState]:
        """Drain all pending states and return the latest one.
        
        This is useful when you want to get the most recent state
        without processing intermediate states.
        
        Args:
            timeout: Initial timeout for first poll
            
        Returns:
            The latest WorldState if any available, None otherwise
        """
        latest_state = None
        
        # First poll with timeout
        state = self.poll_state(timeout)
        if state:
            latest_state = state
        
        # Drain remaining states (non-blocking)
        while True:
            state = self.poll_state(0.0)
            if state is None:
                break
            latest_state = state
        
        return latest_state
    
    def is_alive(self) -> bool:
        """Check if service process is running.
        
        Returns:
            True if process is alive, False otherwise
        """
        return self._process is not None and self._process.is_alive()
    
    def shutdown(self, timeout: float = 5.0) -> bool:
        """Graceful shutdown of service process.
        
        Sends SHUTDOWN event and waits for process to terminate.
        If process doesn't terminate within timeout, it's forcefully killed.
        
        Args:
            timeout: Maximum time to wait for graceful shutdown
            
        Returns:
            True if shutdown was graceful, False if forced termination
        """
        if not self._is_started:
            logger.debug("BackgroundWorker not started, nothing to shutdown")
            return True
        
        graceful = True
        
        # Send shutdown event
        if self._pipe is not None and self.is_alive():
            try:
                self._pipe.send(Event(EventType.SHUTDOWN))
                logger.info("Sent SHUTDOWN event to service")
            except Exception as e:
                logger.warning(f"Failed to send SHUTDOWN event: {e}")
        
        # Wait for process to terminate
        if self._process is not None:
            self._process.join(timeout=timeout)
            
            # Force terminate if still alive
            if self._process.is_alive():
                logger.warning("Service process did not terminate gracefully, forcing termination")
                self._process.terminate()
                self._process.join(timeout=1.0)
                graceful = False
                
                # Last resort: kill
                if self._process.is_alive():
                    logger.error("Service process did not respond to terminate, killing")
                    self._process.kill()
                    self._process.join(timeout=1.0)
        
        # Close pipe
        if self._pipe is not None:
            try:
                self._pipe.close()
            except Exception as e:
                logger.warning(f"Error closing pipe: {e}")
        
        self._is_started = False
        self._process = None
        self._pipe = None
        
        if graceful:
            logger.info("BackgroundWorker shutdown complete (graceful)")
        else:
            logger.warning("BackgroundWorker shutdown complete (forced)")
        
        return graceful
    
    def restart_if_dead(self) -> bool:
        """Restart service if it crashed.
        
        Returns:
            True if service was restarted, False if it was already running
        """
        if self.is_alive():
            return False
        
        if not self._is_started:
            logger.warning("Cannot restart: worker was never started")
            return False
        
        if self._experiments_dir is None:
            logger.error("Cannot restart: experiments_dir not set")
            return False
        
        logger.info("Service process died, restarting...")
        
        # Clean up old resources
        if self._pipe is not None:
            try:
                self._pipe.close()
            except Exception:
                pass
        
        self._is_started = False
        self._process = None
        self._pipe = None
        
        # Restart
        self.start(self._experiments_dir)
        return True
    
    @property
    def process_id(self) -> Optional[int]:
        """Get the PID of the service process.
        
        Returns:
            Process ID if running, None otherwise
        """
        if self._process is not None and self._process.is_alive():
            return self._process.pid
        return None
