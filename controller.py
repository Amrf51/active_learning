"""
controller.py - Threaded controller for Streamlit Active Learning UI.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from al_thread import run_experiment
from experiment_state import AppState, ExperimentState

logger = logging.getLogger(__name__)


class Controller:
    """Thin controller over a shared ExperimentState + background thread."""

    def __init__(self, config: Any, state_file: str = "state.json") -> None:
        self.config = config
        self.state_file = Path(state_file)
        self.state = ExperimentState()

    def _enforce_ui_safety(self, config: Any) -> None:
        # Streamlit mode should not spawn dataloader workers.
        config.data.num_workers = 0

    def start_experiment(self, config: Any) -> str:
        """Start a new backend run (stopping any previous run first)."""
        self.stop_experiment(join_timeout=1.0)
        self._enforce_ui_safety(config)
        self.config = config

        run_id = self.state.reset(config)
        self.state.update_for_run(
            run_id,
            app_state=AppState.INITIALIZING,
            thread_status="starting",
            progress_detail="Starting backend thread...",
        )

        thread = threading.Thread(
            target=run_experiment,
            args=(self.state, config),
            daemon=True,
            name=f"ALThread-{run_id[:8]}",
        )
        self.state.update_for_run(run_id, thread=thread)
        thread.start()
        logger.info("Started run %s", run_id)
        return run_id

    def stop_experiment(self, join_timeout: float = 5.0) -> None:
        """Signal backend thread to stop and best-effort join."""
        snap = self.state.snapshot()
        run_id = snap["run_id"]

        if run_id:
            self.state.update_for_run(
                run_id,
                app_state=AppState.STOPPING,
                thread_status="stopping",
                progress_detail="Stop requested...",
            )
        self.state.stop_event.set()
        # Unblock annotation wait loops immediately.
        self.state.annotations_ready.set()

        thread_obj = self.state.thread
        if thread_obj is not None and thread_obj.is_alive():
            thread_obj.join(timeout=join_timeout)

        if run_id and self.state.is_run_active(run_id):
            post = self.state.snapshot()
            if not post["thread_alive"] and post["app_state"] == AppState.STOPPING:
                self.state.update_for_run(
                    run_id,
                    app_state=AppState.IDLE,
                    thread_status="finished",
                    progress_detail="Stopped",
                )

    def submit_annotations(self, annotations: List[Dict[str, Any]], run_id: str, cycle: int) -> bool:
        """Submit human annotations only if run/cycle is still current."""
        return self.state.set_annotations(run_id=run_id, cycle=cycle, annotations=annotations)

    def get_snapshot(self) -> Dict[str, Any]:
        """Return atomic snapshot used by UI."""
        return self.state.snapshot()

    def reset_to_idle(self, clear_history: bool = False) -> None:
        """Return UI to IDLE and optionally clear accumulated history."""
        self.stop_experiment(join_timeout=1.0)
        with self.state._lock:  # pylint: disable=protected-access
            self.state.last_error = None
            self.state.progress_detail = "Idle"
            self.state.app_state = AppState.IDLE
            self.state.current_epoch = 0
            self.state.queried_images = []
            self.state.annotations_data = []
            self.state.thread_status = "stopped"
            self.state.heartbeat_ts = time.time()
            if clear_history:
                self.state.current_cycle = 0
                self.state.metrics_history = []
                self.state.epoch_metrics = []
                self.state.class_names = []
                self.state.unlabeled_pool_size = 0
        self.state.stop_event.clear()
        self.state.annotations_ready.clear()

    def save_state(self) -> None:
        """Persist UI-visible state as JSON."""
        snap = self.get_snapshot()
        state_data = {
            "app_state": snap["app_state"].value,
            "current_cycle": snap["current_cycle"],
            "total_cycles": snap["total_cycles"],
            "current_epoch": snap["current_epoch"],
            "epoch_metrics": snap["epoch_metrics"],
            "metrics_history": snap["metrics_history"],
            "queried_images": snap["queried_images"],
            "class_names": snap["class_names"],
            "unlabeled_pool_size": snap["unlabeled_pool_size"],
            "last_error": snap["last_error"],
            "progress_detail": snap["progress_detail"],
            "run_id": snap["run_id"],
            "thread_status": snap["thread_status"],
            "heartbeat_ts": snap["heartbeat_ts"],
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state_data, f, indent=2)

    def load_state(self) -> bool:
        """Load previously persisted UI-visible state."""
        if not self.state_file.exists():
            return False
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                state_data = json.load(f)

            app_state_raw = state_data.get("app_state", AppState.IDLE.value)
            app_state = AppState(app_state_raw)

            with self.state._lock:  # pylint: disable=protected-access
                self.state.app_state = app_state
                self.state.current_cycle = int(state_data.get("current_cycle", 0))
                self.state.total_cycles = int(state_data.get("total_cycles", 0))
                self.state.current_epoch = int(state_data.get("current_epoch", 0))
                self.state.epoch_metrics = state_data.get("epoch_metrics", [])
                self.state.metrics_history = state_data.get("metrics_history", [])
                self.state.queried_images = state_data.get("queried_images", [])
                self.state.class_names = state_data.get("class_names", [])
                self.state.unlabeled_pool_size = int(state_data.get("unlabeled_pool_size", 0))
                self.state.last_error = state_data.get("last_error")
                self.state.progress_detail = state_data.get("progress_detail", "Idle")
                self.state.run_id = state_data.get("run_id", "")
                self.state.thread_status = state_data.get("thread_status", "stopped")
                self.state.heartbeat_ts = float(state_data.get("heartbeat_ts", time.time()))
                self.state.thread = None
            return True
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Failed to load controller state")
            return False

    def get_progress(self) -> Dict[str, Any]:
        """Compatibility helper used by some views."""
        snap = self.get_snapshot()
        return {
            "state": snap["app_state"].value,
            "current_cycle": snap["current_cycle"],
            "total_cycles": snap["total_cycles"],
            "current_epoch": snap["current_epoch"],
            "cycles_completed": len(snap["metrics_history"]),
        }

    def get_last_error(self) -> Optional[Dict[str, Any]]:
        return self.get_snapshot()["last_error"]

    def is_busy(self) -> bool:
        state = self.get_snapshot()["app_state"]
        return state in {
            AppState.INITIALIZING,
            AppState.TRAINING,
            AppState.QUERYING,
            AppState.ANNOTATING,
            AppState.STOPPING,
        }
