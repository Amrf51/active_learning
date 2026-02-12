"""
controller.py - State Machine and Dispatch Logic for MVC Architecture.

The Controller manages:
- Application state transitions (IDLE -> TRAINING -> QUERYING -> ANNOTATING)
- Dispatching commands to the worker process via task_queue
- Polling results from the worker process via result_queue
- State persistence (save/load state.json)
- State transition validation

Usage:
    controller = Controller(task_queue, result_queue, events, config)
    controller.dispatch_run_cycle(cycle_num)
    result = controller.poll_results()
"""

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List
from queue import Empty
import multiprocessing as mp

from protocol import (
    build_run_cycle_message,
    build_query_message,
    build_annotate_message,
    build_shutdown_message,
    STOP_REQUESTED,
    RUN_CYCLE,
    QUERY,
    ANNOTATE,
    PROGRESS_UPDATE,
    TRAIN_COMPLETE,
    QUERY_COMPLETE,
    ANNOTATE_COMPLETE,
    CYCLE_COMPLETE,
    ERROR,
)

logger = logging.getLogger(__name__)


# ============================================================================
# SUBTASK 7.1: Define AppState enum
# ============================================================================

class AppState(Enum):
    """
    Application state machine states.
    
    State transitions:
    - IDLE -> TRAINING (when run_cycle is dispatched)
    - TRAINING -> QUERYING (when training completes)
    - QUERYING -> ANNOTATING (when query completes)
    - ANNOTATING -> TRAINING (when annotations submitted)
    - ANY -> ERROR (when error occurs)
    - ERROR -> IDLE (when reset)
    """
    IDLE = "idle"
    TRAINING = "training"
    QUERYING = "querying"
    ANNOTATING = "annotating"
    ERROR = "error"


# ============================================================================
# SUBTASK 7.2: Create Controller class with state machine logic
# ============================================================================

class Controller:
    """
    Controller manages application state and dispatches commands to worker.
    
    Responsibilities:
    - Maintain current application state
    - Validate state transitions
    - Send commands to worker via task_queue
    - Poll results from worker via result_queue
    - Persist and restore application state
    """
    
    def __init__(
        self,
        task_queue: mp.Queue,
        result_queue: mp.Queue,
        events: Dict[str, mp.Event],
        config,
        state_file: str = "state.json"
    ):
        """
        Initialize Controller.
        
        Args:
            task_queue: Queue for sending commands to worker
            result_queue: Queue for receiving results from worker
            events: Dictionary of multiprocessing events
            config: Config object
            state_file: Path to state persistence file
        """
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.events = events
        self.state_file = Path(state_file)
        
        # Store config
        self.config = config
        self.experiment_config: Dict[str, Any] = config.to_dict() if config else {}
        self.total_cycles = config.active_learning.num_cycles if config else 0
        
        # Application state
        self.current_state = AppState.IDLE
        self.current_cycle = 0
        self.current_epoch = 0
        
        # Metrics and history
        self.metrics_history: List[Dict[str, Any]] = []
        self.queried_images: List[Dict[str, Any]] = []
        self.epoch_metrics: List[Dict[str, Any]] = []  # Store epoch-level metrics for current cycle
        
        # Error tracking
        self.last_error: Optional[Dict[str, Any]] = None
        
        logger.info("Controller initialized")
    
    def get_state(self) -> AppState:
        """Get current application state."""
        return self.current_state
    
    def _set_state(self, new_state: AppState):
        """
        Internal method to set state.
        
        Args:
            new_state: New application state
        """
        old_state = self.current_state
        self.current_state = new_state
        logger.info(f"State transition: {old_state.value} -> {new_state.value}")

    # ========================================================================
    # SUBTASK 7.10: Implement state transition validation
    # ========================================================================
    
    def _can_transition_to(self, target_state: AppState) -> bool:
        """
        Validate if transition to target state is allowed.
        
        Args:
            target_state: Desired target state
            
        Returns:
            True if transition is valid, False otherwise
        """
        # Define valid transitions
        valid_transitions = {
            AppState.IDLE: [AppState.TRAINING],  # Direct to TRAINING (worker already ready)
            AppState.TRAINING: [AppState.QUERYING, AppState.ERROR, AppState.IDLE],
            AppState.QUERYING: [AppState.ANNOTATING, AppState.ERROR, AppState.IDLE],
            AppState.ANNOTATING: [AppState.TRAINING, AppState.ERROR, AppState.IDLE],
            AppState.ERROR: [AppState.IDLE],
        }
        
        allowed = valid_transitions.get(self.current_state, [])
        return target_state in allowed
    
    def _validate_transition(self, target_state: AppState) -> None:
        """
        Validate state transition and raise error if invalid.
        
        Args:
            target_state: Desired target state
            
        Raises:
            ValueError: If transition is not allowed
        """
        if not self._can_transition_to(target_state):
            raise ValueError(
                f"Invalid state transition: {self.current_state.value} -> {target_state.value}"
            )
    
    # ========================================================================
    # SUBTASK 7.4: Implement dispatch_run_cycle
    # ========================================================================
    
    def dispatch_run_cycle(self, cycle_num: int) -> None:
        """
        Dispatch RUN_CYCLE command to worker.
        
        This triggers a full training cycle in the worker process.
        
        Args:
            cycle_num: Active learning cycle number
            
        Raises:
            ValueError: If not in appropriate state
        """
        # Can run cycle from IDLE (first cycle) or ANNOTATING (subsequent cycles)
        if self.current_state not in [AppState.IDLE, AppState.ANNOTATING]:
            raise ValueError(
                f"Cannot run cycle from state {self.current_state.value}. "
                f"Must be in IDLE or ANNOTATING state."
            )
        
        # Clear epoch metrics for new cycle
        self.epoch_metrics = []
        
        # Build and send message
        message = build_run_cycle_message(cycle_num)
        self.task_queue.put(message)
        
        # Update state
        self._set_state(AppState.TRAINING)
        self.current_cycle = cycle_num
        
        logger.info(f"Dispatched RUN_CYCLE for cycle {cycle_num}")
    
    # ========================================================================
    # SUBTASK 7.5: Implement dispatch_query
    # ========================================================================
    
    def dispatch_query(self, query_size: Optional[int] = None) -> None:
        """
        Dispatch QUERY command to worker.
        
        This triggers sample selection for annotation.
        
        Args:
            query_size: Optional number of samples to query (overrides config)
            
        Raises:
            ValueError: If not in TRAINING state
        """
        # Validate state transition
        self._validate_transition(AppState.QUERYING)
        
        # Build and send message
        message = build_query_message(query_size)
        self.task_queue.put(message)
        
        # Update state
        self._set_state(AppState.QUERYING)
        
        logger.info(f"Dispatched QUERY (size={query_size or 'default'})")
    
    # ========================================================================
    # SUBTASK 7.6: Implement dispatch_annotate
    # ========================================================================
    
    def dispatch_annotate(self, annotations: List[Dict[str, int]]) -> None:
        """
        Dispatch ANNOTATE command to worker.
        
        This sends user annotations to the worker to update the labeled pool.
        
        Args:
            annotations: List of dicts with 'image_id' and 'user_label' keys
                         e.g., [{"image_id": 5, "user_label": 3}, ...]
            
        Raises:
            ValueError: If not in ANNOTATING state
        """
        # Validate state
        if self.current_state != AppState.ANNOTATING:
            raise ValueError(
                f"Cannot annotate from state {self.current_state.value}. "
                f"Must be in ANNOTATING state."
            )
        
        # Build and send message
        message = build_annotate_message(annotations)
        self.task_queue.put(message)
        
        logger.info(f"Dispatched ANNOTATE with {len(annotations)} annotations")
    
    # ========================================================================
    # SUBTASK 7.7: Implement dispatch_stop
    # ========================================================================
    
    def dispatch_stop(self) -> None:
        """
        Dispatch stop request to worker.
        
        This sets the stop_requested event to signal the worker to halt
        the current operation gracefully.
        """
        self.events[STOP_REQUESTED].set()
        logger.info("Dispatched STOP request")
    
    def clear_stop(self) -> None:
        """Clear the stop_requested event."""
        self.events[STOP_REQUESTED].clear()
        logger.info("Cleared STOP request")
    
    # ========================================================================
    # SUBTASK 7.8: Implement poll_results
    # ========================================================================
    
    def poll_results(self, timeout: float = 0.01) -> Optional[Dict[str, Any]]:
        """
        Non-blocking check of result_queue.
        
        This method polls the result queue for messages from the worker.
        It should be called regularly (e.g., in Streamlit's rerun loop).
        
        Args:
            timeout: Timeout in seconds for queue.get (default: 0.01 for non-blocking)
            
        Returns:
            Message dictionary if available, None otherwise
        """
        try:
            message = self.result_queue.get(timeout=timeout)
            self._handle_result_message(message)
            return message
        except Empty:
            return None
    
    def _handle_result_message(self, message: Dict[str, Any]) -> None:
        """
        Handle result message from worker and update state accordingly.
        
        Args:
            message: Message dictionary from worker
        """
        msg_type = message.get("type")
        payload = message.get("payload", {})
        
        if msg_type == PROGRESS_UPDATE:
            # Progress updates don't change state
            stage = payload.get("stage")
            current = payload.get("current")
            total = payload.get("total")
            logger.debug(f"Progress: {stage} {current}/{total}")
            
            # Update current epoch if in training
            if stage == "training" and "details" in payload:
                details = payload["details"]
                if "epoch" in details:
                    self.current_epoch = details["epoch"]
                    # Store epoch metrics for live visualization
                    self.epoch_metrics.append(details)
        
        elif msg_type == TRAIN_COMPLETE:
            # Training complete, transition to QUERYING
            metrics = payload.get("metrics", {})
            self.metrics_history.append(metrics)
            logger.info(f"Training complete for cycle {self.current_cycle}")
            # State will transition to QUERYING when dispatch_query is called
        
        elif msg_type == QUERY_COMPLETE:
            # Query complete, transition to ANNOTATING
            queried_images = payload.get("queried_images", [])
            self.queried_images = queried_images
            self._set_state(AppState.ANNOTATING)
            logger.info(f"Query complete: {len(queried_images)} images selected")
        
        elif msg_type == ANNOTATE_COMPLETE:
            # Annotation complete, ready for next cycle
            num_annotated = payload.get("num_annotated", 0)
            logger.info(f"Annotation complete: {num_annotated} images labeled")
            # State will transition to TRAINING when dispatch_run_cycle is called
        
        elif msg_type == CYCLE_COMPLETE:
            # Full cycle complete
            cycle_num = payload.get("cycle_num")
            metrics = payload.get("metrics", {})
            logger.info(f"Cycle {cycle_num} complete")
        
        elif msg_type == ERROR:
            # Error occurred, transition to ERROR state
            error_type = payload.get("error_type")
            error_msg = payload.get("error_msg")
            traceback = payload.get("traceback")
            
            self.last_error = {
                "type": error_type,
                "message": error_msg,
                "traceback": traceback
            }
            
            self._set_state(AppState.ERROR)
            logger.error(f"Worker error: {error_type} - {error_msg}")
        
        else:
            logger.warning(f"Unknown message type: {msg_type}")
    
    # ========================================================================
    # SUBTASK 7.9: Implement save_state / load_state
    # ========================================================================
    
    def save_state(self) -> None:
        """
        Save current application state to state.json.
        
        This persists:
        - Current state and cycle
        - Metrics history
        - Queried images
        - Epoch metrics
        - Experiment configuration
        """
        state_data = {
            "current_state": self.current_state.value,
            "current_cycle": self.current_cycle,
            "current_epoch": self.current_epoch,
            "total_cycles": self.total_cycles,
            "metrics_history": self.metrics_history,
            "queried_images": self.queried_images,
            "epoch_metrics": self.epoch_metrics,
            "experiment_config": self.experiment_config,
            "last_error": self.last_error,
        }
        
        with open(self.state_file, "w") as f:
            json.dump(state_data, f, indent=2)
        
        logger.info(f"State saved to {self.state_file}")
    
    def load_state(self) -> bool:
        """
        Load application state from state.json.
        
        Returns:
            True if state was loaded successfully, False otherwise
        """
        if not self.state_file.exists():
            logger.info(f"No state file found at {self.state_file}")
            return False
        
        try:
            with open(self.state_file, "r") as f:
                state_data = json.load(f)
            
            # Restore state
            self.current_state = AppState(state_data["current_state"])
            self.current_cycle = state_data["current_cycle"]
            self.current_epoch = state_data["current_epoch"]
            self.total_cycles = state_data["total_cycles"]
            self.metrics_history = state_data["metrics_history"]
            self.queried_images = state_data["queried_images"]
            self.epoch_metrics = state_data.get("epoch_metrics", [])
            self.experiment_config = state_data["experiment_config"]
            self.last_error = state_data.get("last_error")
            
            logger.info(f"State loaded from {self.state_file}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False
    
    def reset_state(self) -> None:
        """
        Reset controller to initial state.
        
        This clears all state and returns to IDLE.
        """
        self.current_state = AppState.IDLE
        self.current_cycle = 0
        self.current_epoch = 0
        self.total_cycles = 0
        self.metrics_history = []
        self.queried_images = []
        self.epoch_metrics = []
        self.experiment_config = {}
        self.last_error = None
        
        # Clear stop event
        self.clear_stop()
        
        logger.info("Controller state reset")
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_progress(self) -> Dict[str, Any]:
        """
        Get current progress information.
        
        Returns:
            Dictionary with progress information
        """
        return {
            "state": self.current_state.value,
            "current_cycle": self.current_cycle,
            "total_cycles": self.total_cycles,
            "current_epoch": self.current_epoch,
            "cycles_completed": len(self.metrics_history),
        }
    
    def get_last_error(self) -> Optional[Dict[str, Any]]:
        """
        Get last error information.
        
        Returns:
            Error dictionary if error occurred, None otherwise
        """
        return self.last_error
    
    def is_busy(self) -> bool:
        """
        Check if controller is in a busy state.
        
        Returns:
            True if in TRAINING or QUERYING state
        """
        return self.current_state in [
            AppState.TRAINING,
            AppState.QUERYING,
        ]
