"""
Thread-safe shared experiment state for Streamlit UI + backend thread.
"""

from __future__ import annotations

import copy
import threading
import time
import traceback
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from events import Inbox


class AppState(Enum):
    """High-level lifecycle states for the active learning UI."""

    IDLE = "idle"
    INITIALIZING = "initializing"
    TRAINING = "training"
    QUERYING = "querying"
    ANNOTATING = "annotating"
    STOPPING = "stopping"
    ERROR = "error"
    FINISHED = "finished"


class ExperimentState:
    """Single source of truth shared between UI thread and backend thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # Core progress
        self.app_state: AppState = AppState.IDLE
        self.current_cycle: int = 0
        self.total_cycles: int = 0
        self.current_epoch: int = 0
        self.epoch_metrics: List[Dict[str, Any]] = []
        self.metrics_history: List[Dict[str, Any]] = []
        self.queried_images: List[Dict[str, Any]] = []
        self.class_names: List[str] = []
        self.unlabeled_pool_size: int = 0
        self.last_error: Optional[Dict[str, str]] = None
        self.progress_detail: str = "Idle"

        # Thread control
        self.stop_event = threading.Event()
        self.annotations_ready = threading.Event()
        self.annotations_data: List[Dict[str, Any]] = []
        self.thread: Optional[threading.Thread] = None
        self.thread_status: str = "stopped"
        self.run_id: str = ""
        self.heartbeat_ts: float = time.time()
        self.query_token: str = ""
        self.run_dir: str = ""

        # Event inbox for worker → controller communication
        self.inbox = Inbox()

    def reset(self, config: Any) -> str:
        """Reset run-scoped state, create a new run id, and clear events."""
        new_run_id = str(uuid.uuid4())
        with self._lock:
            self.app_state = AppState.IDLE
            self.current_cycle = 0
            self.total_cycles = int(getattr(config.active_learning, "num_cycles", 0))
            self.current_epoch = 0
            self.epoch_metrics = []
            self.metrics_history = []
            self.queried_images = []
            self.class_names = []
            self.unlabeled_pool_size = 0
            self.last_error = None
            self.progress_detail = "Ready"
            self.annotations_data = []
            self.run_id = new_run_id
            self.thread_status = "starting"
            self.heartbeat_ts = time.time()
            self.query_token = ""
            self.run_dir = ""

        self.stop_event.clear()
        self.annotations_ready.clear()
        self.inbox.reset()
        return new_run_id

    def snapshot(self) -> Dict[str, Any]:
        """Return an atomic snapshot for UI rendering."""
        with self._lock:
            thread_alive = self.thread.is_alive() if self.thread else False
            return {
                "app_state": self.app_state,
                "current_cycle": self.current_cycle,
                "total_cycles": self.total_cycles,
                "current_epoch": self.current_epoch,
                "epoch_metrics": copy.deepcopy(self.epoch_metrics),
                "metrics_history": copy.deepcopy(self.metrics_history),
                "queried_images": copy.deepcopy(self.queried_images),
                "class_names": list(self.class_names),
                "unlabeled_pool_size": self.unlabeled_pool_size,
                "last_error": copy.deepcopy(self.last_error),
                "progress_detail": self.progress_detail,
                "thread_status": self.thread_status,
                "run_id": self.run_id,
                "heartbeat_ts": self.heartbeat_ts,
                "thread_alive": thread_alive,
                "event_version": self.inbox.version,
                "query_token": self.query_token,
                "run_dir": self.run_dir,
            }

    def touch_heartbeat(self, run_id: Optional[str] = None) -> bool:
        """
        Update heartbeat timestamp.

        If run_id is provided, heartbeat is only updated when it matches
        the active run. Returns whether the heartbeat was updated.
        """
        with self._lock:
            if run_id is not None and self.run_id != run_id:
                return False
            self.heartbeat_ts = time.time()
            return True

    def set_error(self, exc: Exception, tb: Optional[str] = None) -> None:
        """Set terminal error state and record traceback."""
        if tb is None:
            tb = traceback.format_exc()
        with self._lock:
            self.app_state = AppState.ERROR
            self.thread_status = "failed"
            self.progress_detail = f"Error: {exc}"
            self.last_error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": tb,
            }
            self.heartbeat_ts = time.time()

    def set_annotations(self, run_id: str, cycle: int, annotations: List[Dict[str, Any]]) -> bool:
        """
        Accept user annotations only for the active run/cycle while annotating.
        """
        with self._lock:
            if self.run_id != run_id:
                return False
            if self.current_cycle != cycle:
                return False
            if self.app_state != AppState.ANNOTATING:
                return False
            self.annotations_data = copy.deepcopy(annotations)
            self.heartbeat_ts = time.time()
        self.annotations_ready.set()
        return True

    def clear_annotations(self) -> None:
        """Clear annotation payload + signal event."""
        with self._lock:
            self.annotations_data = []
        self.annotations_ready.clear()

    def consume_annotations(self, run_id: str, cycle: int) -> Optional[List[Dict[str, Any]]]:
        """Consume annotations if they belong to the current run/cycle."""
        with self._lock:
            if self.run_id != run_id or self.current_cycle != cycle:
                return None
            data = copy.deepcopy(self.annotations_data)
            self.annotations_data = []
            self.heartbeat_ts = time.time()
        self.annotations_ready.clear()
        return data

    def is_run_active(self, run_id: str) -> bool:
        """Check whether run_id still matches active run."""
        with self._lock:
            return self.run_id == run_id

    def update_for_run(self, run_id: str, **kwargs: Any) -> bool:
        """Update fields only if run_id still matches."""
        with self._lock:
            if self.run_id != run_id:
                return False
            for key, value in kwargs.items():
                setattr(self, key, value)
            self.heartbeat_ts = time.time()
            return True
