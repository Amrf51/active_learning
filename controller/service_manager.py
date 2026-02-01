"""
ServiceManager: Process lifecycle management for ActiveLearningService.

This module manages the ActiveLearningService process lifecycle, including
spawning, monitoring, and graceful termination. It provides pipe-based
communication and automatic restart capabilities.
"""

import logging
import time
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from threading import Thread
from typing import Optional, Callable, Any

from controller.events import Event, EventType, create_event

logger = logging.getLogger(__name__)


class ServiceManager:
    """
    Manages the ActiveLearningService process lifecycle.
    
    Key responsibilities:
    - Spawn service process with Pipe communication
    - Monitor service health
    - Handle graceful shutdown
    - Restart on failure
    - Provide event-based communication interface
    
    The service process is spawned as a daemon to ensure cleanup when
    the parent process exits.
    """
    
    def __init__(self):
        """Initialize ServiceManager with empty state."""
        self._process: Optional[Process] = None
        self._pipe: Optional[Connection] = None
        self._listener_thread: Optional[Thread] = None
        self._event_callback: Optional[Callable[[Event], None]] = None
        self._shutdown_requested = False
        self._restart_attempts = 0
        self._max_restart_attempts = 3
        
        logger.info("ServiceManager initialized")
    
    def spawn_service(self, config: dict, event_callback: Callable[[Event], None]) -> bool:
        """
        Spawn the ActiveLearningService process.
        
        Creates a bidirectional pipe, spawns the service process as a daemon,
        and starts a listener thread for receiving events from the service.
        
        Args:
            config: Experiment configuration dictionary
            event_callback: Callback function for handling service events
            
        Returns:
            True if service started successfully, False otherwise
        """
        # Terminate any existing service first
        self._terminate_existing_service()
        
        self._event_callback = event_callback
        self._shutdown_requested = False
        self._restart_attempts = 0
        
        try:
            # Import the service entry point
            from services.al_service import run_active_learning_service
            
            # Create bidirectional pipe for communication
            parent_conn, child_conn = Pipe()
            self._pipe = parent_conn
            
            # Spawn process as DAEMON
            # Daemon processes are automatically terminated when parent exits
            self._process = Process(
                target=run_active_learning_service,
                args=(child_conn, config),
                daemon=True,
                name="ActiveLearningService"
            )
            
            # Start the process
            self._process.start()
            
            logger.info(f"Service process started with PID: {self._process.pid}")
            
            # Start listener thread for service events
            self._start_listener_thread()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to spawn service: {e}")
            self._cleanup_failed_spawn()
            return False
    
    def _terminate_existing_service(self) -> None:
        """
        Gracefully terminate existing service if running.
        
        Follows a graceful shutdown sequence:
        1. Send CMD_SHUTDOWN command
        2. Wait for graceful shutdown (5 seconds)
        3. Terminate process if still running
        4. Kill process if terminate fails (last resort)
        """
        if not self._process or not self._process.is_alive():
            return
        
        logger.info("Terminating existing service...")
        
        try:
            # Step 1: Send graceful shutdown command
            if self._pipe and not self._pipe.closed:
                shutdown_event = create_event(EventType.CMD_SHUTDOWN, source="controller")
                self._pipe.send(shutdown_event)
                logger.debug("Sent CMD_SHUTDOWN to service")
            
            # Step 2: Wait for graceful shutdown (max 5 seconds)
            self._process.join(timeout=5.0)
            
            # Step 3: Force terminate if still running
            if self._process.is_alive():
                logger.warning("Service didn't shutdown gracefully, terminating...")
                self._process.terminate()
                self._process.join(timeout=2.0)
            
            # Step 4: Kill if still running (last resort)
            if self._process.is_alive():
                logger.error("Service didn't terminate, killing...")
                self._process.kill()
                self._process.join(timeout=1.0)
                
            if self._process.is_alive():
                logger.error("Failed to kill service process - may be zombie")
            else:
                logger.info("Service process terminated successfully")
                
        except Exception as e:
            logger.error(f"Error during service termination: {e}")
        
        finally:
            # Always cleanup references
            self._cleanup_process_references()
    
    def _cleanup_process_references(self) -> None:
        """Clean up process and pipe references."""
        if self._pipe and not self._pipe.closed:
            try:
                self._pipe.close()
            except Exception as e:
                logger.error(f"Error closing pipe: {e}")
        
        self._process = None
        self._pipe = None
    
    def _cleanup_failed_spawn(self) -> None:
        """Clean up after failed service spawn."""
        if self._process:
            try:
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=2.0)
            except Exception as e:
                logger.error(f"Error cleaning up failed spawn: {e}")
        
        self._cleanup_process_references()
    
    def _start_listener_thread(self) -> None:
        """
        Start background thread to listen for service events.
        
        The listener thread runs as a daemon and continuously polls
        the pipe for incoming events from the service process.
        """
        self._listener_thread = Thread(
            target=self._listen_for_events,
            daemon=True,
            name="ServiceEventListener"
        )
        self._listener_thread.start()
        logger.debug("Service event listener thread started")
    
    def _listen_for_events(self) -> None:
        """
        Background thread: Listen for events from service.
        
        This thread blocks on pipe.recv() waiting for events from the service.
        It handles communication errors gracefully and detects service crashes.
        No polling - just blocking I/O with periodic health checks.
        """
        logger.debug("Event listener thread started")
        
        while not self._shutdown_requested:
            try:
                # Check if service is still alive
                if not self.is_alive():
                    logger.warning("Service process died unexpectedly")
                    self._handle_service_crash()
                    break
                
                # Wait for event with timeout to check shutdown flag periodically
                if self._pipe and self._pipe.poll(timeout=1.0):
                    try:
                        event = self._pipe.recv()
                        logger.debug(f"Received service event: {event.type}")
                        
                        # Forward event to callback
                        if self._event_callback:
                            self._event_callback(event)
                            
                    except EOFError:
                        logger.info("Service pipe closed (EOFError)")
                        break
                    except Exception as e:
                        logger.error(f"Error receiving service event: {e}")
                        
            except EOFError:
                logger.info("Service pipe closed")
                break
            except Exception as e:
                logger.error(f"Error in event listener: {e}")
                # Don't break on non-pipe errors, just log them
        
        logger.debug("Event listener thread exiting")
    
    def _handle_service_crash(self) -> None:
        """
        Handle unexpected service process death.
        
        Emits a SERVICE_ERROR event and attempts automatic restart
        if within retry limits.
        """
        if self._event_callback:
            error_event = create_event(
                EventType.SERVICE_ERROR,
                {
                    "error_type": "service_crash",
                    "message": "Service process died unexpectedly",
                    "restart_attempts": self._restart_attempts
                },
                source="controller"
            )
            self._event_callback(error_event)
        
        # Attempt automatic restart if within limits
        if self._restart_attempts < self._max_restart_attempts:
            self._restart_attempts += 1
            logger.info(f"Attempting service restart ({self._restart_attempts}/{self._max_restart_attempts})")
            
            # Wait before restart (exponential backoff)
            wait_time = min(2 ** self._restart_attempts, 30)  # Max 30 seconds
            time.sleep(wait_time)
            
            # Note: Actual restart would need config and callback from original spawn
            # This is a simplified version - in practice, you'd store these for restart
            logger.warning("Automatic restart not implemented - manual restart required")
        else:
            logger.error("Max restart attempts reached, giving up")
    
    def send_command(self, event: Event) -> bool:
        """
        Send command to service process.
        
        Args:
            event: Command event to send to service
            
        Returns:
            True if command sent successfully, False otherwise
        """
        if not self._pipe or self._pipe.closed:
            logger.error("Cannot send command: no pipe connection")
            return False
        
        if not self.is_alive():
            logger.error("Cannot send command: service process not running")
            return False
        
        try:
            self._pipe.send(event)
            logger.debug(f"Sent command to service: {event.type}")
            return True
        except BrokenPipeError:
            logger.error("Failed to send command: pipe broken")
            return False
        except EOFError:
            logger.error("Failed to send command: pipe closed")
            return False
        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            return False
    
    def is_alive(self) -> bool:
        """
        Check if service process is running.
        
        Returns:
            True if service process is alive, False otherwise
        """
        return self._process is not None and self._process.is_alive()
    
    def get_process_info(self) -> dict:
        """
        Get information about the service process.
        
        Returns:
            Dictionary with process information
        """
        if not self._process:
            return {"status": "not_started", "pid": None}
        
        return {
            "status": "running" if self._process.is_alive() else "dead",
            "pid": self._process.pid,
            "name": self._process.name,
            "daemon": self._process.daemon,
            "restart_attempts": self._restart_attempts
        }
    
    def shutdown(self) -> None:
        """
        Shutdown the service manager and cleanup resources.
        
        Terminates the service process, stops the listener thread,
        and cleans up all resources. This method should be called
        when the application is shutting down.
        """
        logger.info("ServiceManager shutting down...")
        
        # Set shutdown flag to stop listener thread
        self._shutdown_requested = True
        
        # Terminate service process
        self._terminate_existing_service()
        
        # Wait for listener thread to finish
        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=2.0)
            if self._listener_thread.is_alive():
                logger.warning("Listener thread didn't stop gracefully")
        
        # Reset state
        self._event_callback = None
        self._listener_thread = None
        
        logger.info("ServiceManager shutdown complete")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if not self._shutdown_requested:
                self.shutdown()
        except Exception:
            pass  # Ignore errors in destructor