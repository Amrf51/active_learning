"""
controller.py - Threaded controller for Streamlit Active Learning UI.
"""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .worker import run_experiment
from .events import Event, EventType
from .experiment_state import AppState, ExperimentState

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

    def _sanitize_experiment_name(self, name: Any) -> str:
        """Return a filesystem-safe experiment folder name."""
        raw = str(name or "").strip()
        if not raw:
            return "al_experiment"

        # Windows-reserved filename characters + control chars.
        safe = re.sub(r'[<>:"/\\\\|?*\x00-\x1F]', "_", raw)
        safe = safe.strip().strip(".")
        return safe or "al_experiment"

    def dispatch(self, event: Event) -> Any:
        """Central event router for UI and worker lifecycle events."""
        match event.type:
            # UI -> Controller (immediate)
            case EventType.START_EXPERIMENT:
                return self._handle_start(event.data["config"])
            case EventType.STOP_EXPERIMENT:
                join_timeout = float(event.data.get("join_timeout", 5.0))
                return self._handle_stop(join_timeout=join_timeout)
            case EventType.NEXT_STEP:
                return self._handle_next_step(event)
            case EventType.SUBMIT_ANNOTATIONS:
                return self._handle_submit(event)

            # Worker -> Controller (from inbox)
            case EventType.CYCLE_STARTED:
                cycle = int(event.data.get("cycle", event.cycle))
                total_cycles = int(event.data.get("total_cycles", 0))
                cycle_suffix = f"/{total_cycles}" if total_cycles > 0 else ""
                self.state.update_for_run(
                    event.run_id,
                    app_state=AppState.TRAINING,
                    thread_status="running",
                    current_cycle=cycle,
                    current_epoch=0,
                    epoch_metrics=[],
                    query_token="",
                    class_names=list(event.data.get("class_names", [])),
                    labeled_pool_size=int(event.data.get("labeled_pool_size", 0)),
                    unlabeled_pool_size=int(event.data.get("unlabeled_pool_size", 0)),
                    labeled_class_distribution=dict(event.data.get("labeled_class_distribution", {})),
                    unlabeled_class_distribution=dict(event.data.get("unlabeled_class_distribution", {})),
                    progress_detail=f"Training cycle {cycle}{cycle_suffix}...",
                )
            case EventType.EPOCH_DONE:
                snap = self.state.snapshot()
                metrics = dict(event.data.get("metrics", {}))
                epoch_metrics = list(snap["epoch_metrics"])
                epoch_metrics.append(metrics)
                epoch = int(event.data.get("epoch", metrics.get("epoch", 0)))
                total_epochs = int(event.data.get("total_epochs", 0))
                self.state.update_for_run(
                    event.run_id,
                    app_state=AppState.TRAINING,
                    current_epoch=epoch,
                    epoch_metrics=epoch_metrics,
                    progress_detail=f"Cycle {event.cycle} - Epoch {epoch}/{total_epochs}",
                )
            case EventType.EVAL_COMPLETE:
                snap = self.state.snapshot()
                cycle_metrics = dict(event.data.get("cycle_metrics", {}))
                probe_images = list(event.data.get("probe_images", snap.get("probe_images", [])))
                history = list(snap["metrics_history"])
                history.append(cycle_metrics)
                self.state.update_for_run(
                    event.run_id,
                    metrics_history=history,
                    probe_images=probe_images,
                    queried_images=[],
                    labeled_pool_size=int(
                        cycle_metrics.get("labeled_pool_size", snap["labeled_pool_size"])
                    ),
                    unlabeled_pool_size=int(
                        cycle_metrics.get("unlabeled_pool_size", snap["unlabeled_pool_size"])
                    ),
                    labeled_class_distribution=dict(
                        event.data.get("labeled_class_distribution", snap["labeled_class_distribution"])
                    ),
                    unlabeled_class_distribution=dict(
                        event.data.get("unlabeled_class_distribution", snap["unlabeled_class_distribution"])
                    ),
                    progress_detail=f"Cycle {event.cycle} complete",
                )
            case EventType.QUERYING_STARTED:
                cycle = int(event.data.get("cycle", event.cycle))
                self.state.update_for_run(
                    event.run_id,
                    app_state=AppState.QUERYING,
                    progress_detail=f"Querying samples for cycle {cycle}...",
                )
            case EventType.WAITING_FOR_STEP:
                next_cycle = int(event.data.get("next_cycle", event.cycle))
                total_cycles = int(event.data.get("total_cycles", 0))
                cycle_suffix = f"/{total_cycles}" if total_cycles > 0 else ""
                self.state.update_for_run(
                    event.run_id,
                    app_state=AppState.WAITING_STEP,
                    progress_detail=f"Paused before cycle {next_cycle}{cycle_suffix}. Click Next Step.",
                )
            case EventType.NEW_IMAGES:
                self.state.update_for_run(
                    event.run_id,
                    app_state=AppState.ANNOTATING,
                    queried_images=list(event.data.get("queried_images", [])),
                    query_token=str(event.data.get("query_token", "")),
                    progress_detail="Waiting for annotations...",
                )
            case EventType.ANNOTATIONS_APPLIED:
                snap = self.state.snapshot()
                count = int(event.data.get("count", 0))
                self.state.update_for_run(
                    event.run_id,
                    queried_images=[],
                    labeled_pool_size=int(event.data.get("labeled_pool_size", snap["labeled_pool_size"])),
                    unlabeled_pool_size=int(event.data.get("unlabeled_pool_size", snap["unlabeled_pool_size"])),
                    labeled_class_distribution=dict(
                        event.data.get("labeled_class_distribution", snap["labeled_class_distribution"])
                    ),
                    unlabeled_class_distribution=dict(
                        event.data.get("unlabeled_class_distribution", snap["unlabeled_class_distribution"])
                    ),
                    progress_detail=f"Annotations applied ({count})",
                )
            case EventType.RUN_FINISHED:
                self.state.update_for_run(
                    event.run_id,
                    app_state=AppState.FINISHED,
                    thread_status="finished",
                    queried_images=[],
                    query_token="",
                    progress_detail="Experiment finished",
                )
            case EventType.RUN_ERROR:
                self.state.set_error(
                    Exception(str(event.data.get("error", "Unknown error"))),
                    event.data.get("traceback"),
                )
            case EventType.RUN_STOPPED:
                self.state.update_for_run(
                    event.run_id,
                    app_state=AppState.IDLE,
                    thread_status="finished",
                    queried_images=[],
                    query_token="",
                    progress_detail="Stopped",
                )
            case _:
                logger.warning("Unhandled event type: %s", event.type)
        return None

    def _handle_start(self, config: Any) -> str:
        """Immediate start: stop existing run, create run folder, save config, spawn thread."""
        self.stop_experiment(join_timeout=1.0)
        self._enforce_ui_safety(config)
        safe_exp_name = self._sanitize_experiment_name(config.experiment.name)
        if safe_exp_name != config.experiment.name:
            logger.warning(
                "Experiment name '%s' normalized to '%s' for folder creation",
                config.experiment.name,
                safe_exp_name,
            )
        config.experiment.name = safe_exp_name
        self.config = config

        run_id = self.state.reset(config)
        self.state.update_for_run(
            run_id,
            app_state=AppState.INITIALIZING,
            thread_status="starting",
            progress_detail="Starting backend thread...",
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        run_dir = (
            Path(config.experiment.exp_dir)
            / safe_exp_name
            / f"{timestamp}_{run_id[:8]}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        config.save_to_file(run_dir / "config.yaml")
        self.state.update_for_run(run_id, run_dir=str(run_dir))

        command_queue: "queue.Queue[Any]" = queue.Queue()
        self.state.command_queue = command_queue

        thread = threading.Thread(
            target=run_experiment,
            args=(command_queue, self.state.inbox, config, run_dir, run_id),
            daemon=True,
            name=f"ALThread-{run_id[:8]}",
        )
        self.state.update_for_run(run_id, thread=thread)
        thread.start()
        logger.info("Started run %s", run_id)
        return run_id

    def _handle_stop(self, join_timeout: float = 5.0) -> None:
        """Immediate stop: mark STOPPING, signal worker via command queue, and best-effort join."""
        snap = self.state.snapshot()
        run_id = snap["run_id"]

        if run_id:
            self.state.update_for_run(
                run_id,
                app_state=AppState.STOPPING,
                thread_status="stopping",
                query_token="",
                progress_detail="Stop requested...",
            )

        q = self.state.command_queue
        if q is not None:
            q.put({"command": "STOP"})

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
                    query_token="",
                    progress_detail="Stopped",
                )

    def _handle_submit(self, event: Event) -> bool:
        """Validate query token and forward annotations to the worker via command queue."""
        snap = self.state.snapshot()
        token = str(event.data.get("query_token", ""))
        if not token or token != snap["query_token"]:
            return False

        # Validate run/cycle/state before consuming the token
        if snap["app_state"] != AppState.ANNOTATING:
            return False
        if snap["current_cycle"] != event.cycle:
            return False

        if not self.state.update_for_run(event.run_id, query_token=""):
            return False

        q = self.state.command_queue
        if q is None:
            return False

        q.put({
            "command": "SUBMIT_ANNOTATIONS",
            "run_id": event.run_id,
            "cycle": event.cycle,
            "annotations": list(event.data.get("annotations", [])),
        })
        return True

    def _handle_next_step(self, event: Event) -> bool:
        """Release worker pause when step mode is waiting for user action."""
        snap = self.state.snapshot()
        target_run_id = event.run_id or snap["run_id"]
        if not target_run_id or target_run_id != snap["run_id"]:
            return False
        if snap["app_state"] != AppState.WAITING_STEP:
            return False
        if not self.state.update_for_run(
            target_run_id,
            app_state=AppState.TRAINING,
            progress_detail="Advancing to next cycle...",
        ):
            return False

        q = self.state.command_queue
        if q is None:
            return False
        q.put({"command": "NEXT_STEP"})
        return True

    def start_experiment(self, config: Any) -> str:
        """Backward-compatible wrapper for UI code paths."""
        return self.dispatch(
            Event(
                type=EventType.START_EXPERIMENT,
                data={"config": config},
            )
        )

    def stop_experiment(self, join_timeout: float = 5.0) -> None:
        """Backward-compatible wrapper for UI code paths."""
        self.dispatch(
            Event(
                type=EventType.STOP_EXPERIMENT,
                data={"join_timeout": join_timeout},
            )
        )

    def next_step(self, run_id: Optional[str] = None) -> bool:
        """Backward-compatible wrapper for step-mode progression."""
        rid = run_id if run_id is not None else self.state.snapshot().get("run_id", "")
        return bool(
            self.dispatch(
                Event(
                    type=EventType.NEXT_STEP,
                    run_id=rid,
                )
            )
        )

    def submit_annotations(
        self,
        annotations: List[Dict[str, Any]],
        run_id: str,
        cycle: int,
        query_token: Optional[str] = None,
    ) -> bool:
        """Backward-compatible wrapper for annotation submit."""
        token = query_token if query_token is not None else self.state.snapshot().get("query_token", "")
        return bool(
            self.dispatch(
                Event(
                    type=EventType.SUBMIT_ANNOTATIONS,
                    run_id=run_id,
                    cycle=cycle,
                    data={
                        "annotations": annotations,
                        "query_token": token,
                    },
                )
            )
        )

    def process_inbox(self, since_version: int) -> Tuple[List[Event], int]:
        """Drain worker events, filter stale items with a snapshot, and dispatch accepted events."""
        events, new_version = self.state.inbox.drain(since_version)

        snap = self.state.snapshot()
        current_run_id = snap["run_id"]
        current_cycle = snap["current_cycle"]

        accepted: List[Event] = []
        for event in events:
            if event.run_id != current_run_id:
                continue
            if event.type == EventType.ANNOTATIONS_APPLIED and event.cycle != current_cycle:
                continue
            self.dispatch(event)
            accepted.append(event)

        return accepted, new_version

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
            self.state.probe_images = []
            self.state.labeled_pool_size = 0
            self.state.unlabeled_pool_size = 0
            self.state.labeled_class_distribution = {}
            self.state.unlabeled_class_distribution = {}
            self.state.thread_status = "stopped"
            self.state.heartbeat_ts = time.time()
            if clear_history:
                self.state.current_cycle = 0
                self.state.metrics_history = []
                self.state.epoch_metrics = []
                self.state.class_names = []

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
            "probe_images": snap["probe_images"],
            "class_names": snap["class_names"],
            "labeled_pool_size": snap["labeled_pool_size"],
            "unlabeled_pool_size": snap["unlabeled_pool_size"],
            "labeled_class_distribution": snap["labeled_class_distribution"],
            "unlabeled_class_distribution": snap["unlabeled_class_distribution"],
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
                self.state.probe_images = state_data.get("probe_images", [])
                self.state.class_names = state_data.get("class_names", [])
                self.state.labeled_pool_size = int(state_data.get("labeled_pool_size", 0))
                self.state.unlabeled_pool_size = int(state_data.get("unlabeled_pool_size", 0))
                self.state.labeled_class_distribution = dict(state_data.get("labeled_class_distribution", {}))
                self.state.unlabeled_class_distribution = dict(state_data.get("unlabeled_class_distribution", {}))
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
            AppState.WAITING_STEP,
            AppState.STOPPING,
        }
