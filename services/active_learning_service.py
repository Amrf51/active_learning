"""ActiveLearningService - Service process for active learning operations.

This module implements the service that runs in a separate process,
handling all backend operations (training, querying, etc.) while
communicating with the controller via pipes.

Key responsibilities:
- Run the main event loop in a separate process
- Process events from the controller
- Execute training epochs and push state updates
- Handle graceful shutdown
"""

from multiprocessing.connection import Connection
from pathlib import Path
from typing import Optional
import logging
import time
import sys

from controller.events import Event, EventType
from model.world_state import WorldState, Phase
from model.model_handler import ModelHandler

# Configure logging for the service process
logging.basicConfig(
    level=logging.INFO,
    format='[SERVICE %(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def run_service_loop(pipe: Connection, experiments_dir: Path) -> None:
    """Entry point for the service process.
    
    This function is called by multiprocessing.Process to start
    the service in a separate process.
    
    Args:
        pipe: Child end of the bidirectional pipe for communication
        experiments_dir: Root directory for experiment storage
    """
    logger.info(f"Service process starting (PID: {__import__('os').getpid()})")
    logger.info(f"Experiments directory: {experiments_dir}")
    
    try:
        service = ActiveLearningService(pipe, experiments_dir)
        service.run()
    except Exception as e:
        logger.error(f"Service process crashed: {e}", exc_info=True)
        # Try to send error state before dying
        try:
            error_state = WorldState(
                phase=Phase.ERROR,
                error_message=f"Service crashed: {str(e)}"
            )
            pipe.send(error_state)
        except Exception:
            pass
        raise
    finally:
        logger.info("Service process exiting")
        try:
            pipe.close()
        except Exception:
            pass


class ActiveLearningService:
    """Service that runs in a separate process.
    
    The ActiveLearningService handles all backend operations in isolation
    from the UI process. It:
    1. Receives events from the controller via pipe
    2. Delegates event handling to ModelHandler
    3. Executes training epochs when in TRAINING phase
    4. Pushes WorldState updates back to the controller
    
    Communication Pattern:
        Controller -> Service: Events (commands)
        Service -> Controller: WorldState (state updates)
    """
    
    def __init__(self, pipe: Connection, experiments_dir: Path):
        """Initialize the service.
        
        Args:
            pipe: Child end of the bidirectional pipe
            experiments_dir: Root directory for experiment storage
        """
        self._pipe = pipe
        self._model_handler = ModelHandler(experiments_dir)
        self._running = True
        
        logger.info("ActiveLearningService initialized")
    
    def run(self) -> None:
        """Main event loop.
        
        This loop:
        1. Checks for incoming events (non-blocking with short timeout)
        2. If in TRAINING phase, trains one epoch
        3. Continues until SHUTDOWN event or error
        """
        logger.info("Service event loop starting")
        
        while self._running:
            try:
                # Check for incoming events (INBOX)
                # Use short timeout to allow training to proceed
                if self._pipe.poll(timeout=0.1):
                    event = self._pipe.recv()
                    self._handle_event(event)
                
                # If training, run one epoch and push state
                if self._model_handler.world_state.phase == Phase.TRAINING:
                    self._train_one_epoch()
                    
            except EOFError:
                logger.warning("Pipe closed by controller, shutting down")
                self._running = False
            except Exception as e:
                logger.error(f"Error in event loop: {e}", exc_info=True)
                self._model_handler.world_state.phase = Phase.ERROR
                self._model_handler.world_state.error_message = str(e)
                self._push_state()
                # Continue running to allow recovery
        
        logger.info("Service event loop ended")
    
    def _handle_event(self, event: Event) -> None:
        """Process event from controller.
        
        Args:
            event: Event received from the controller
        """
        logger.info(f"Handling event: {event.type.value}")
        
        # Handle SHUTDOWN specially
        if event.type == EventType.SHUTDOWN:
            logger.info("Received SHUTDOWN event")
            self._running = False
            return
        
        try:
            # Delegate to ModelHandler
            self._model_handler.handle_event(event)
            
            # Push updated state to controller
            self._push_state()
            
        except Exception as e:
            logger.error(f"Error handling event {event.type}: {e}", exc_info=True)
            self._model_handler.world_state.phase = Phase.ERROR
            self._model_handler.world_state.error_message = str(e)
            self._push_state()
    
    def _train_one_epoch(self) -> None:
        """Train one epoch and push state update.
        
        This method is called repeatedly while in TRAINING phase.
        It trains a single epoch and sends the updated state to the controller.
        """
        # Check for pause/stop requests
        if self._model_handler.is_paused():
            logger.info("Training paused")
            self._model_handler.world_state.phase = Phase.IDLE
            self._push_state()
            self._model_handler.clear_control_flags()
            return
        
        if self._model_handler.is_stopped():
            logger.info("Training stopped")
            self._model_handler.world_state.phase = Phase.IDLE
            self._push_state()
            self._model_handler.clear_control_flags()
            return
        
        try:
            # Train one epoch
            metrics = self._model_handler.train_epoch()
            
            # Push state after each epoch for live updates
            self._push_state()
            
            logger.debug(f"Epoch {metrics.epoch} complete: "
                        f"loss={metrics.train_loss:.4f}, acc={metrics.train_accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"Error during training: {e}", exc_info=True)
            self._model_handler.world_state.phase = Phase.ERROR
            self._model_handler.world_state.error_message = f"Training error: {str(e)}"
            self._push_state()
    
    def _push_state(self) -> None:
        """Send current WorldState to controller.
        
        Updates the timestamp before sending to mark the state as fresh.
        """
        try:
            # Update timestamp for state versioning
            self._model_handler.world_state.touch()
            
            # Send state via pipe
            self._pipe.send(self._model_handler.world_state)
            
            logger.debug(f"Pushed state (phase: {self._model_handler.world_state.phase.value})")
            
        except Exception as e:
            logger.error(f"Failed to push state: {e}")
            # Don't raise - we want to continue running even if push fails
    
    @property
    def world_state(self) -> WorldState:
        """Get current WorldState (for testing)."""
        return self._model_handler.world_state
    
    @property
    def is_running(self) -> bool:
        """Check if service is still running."""
        return self._running
    
    def stop(self) -> None:
        """Stop the service loop (for testing)."""
        self._running = False
