"""
events.py — Typed event system for worker-to-controller communication.

The worker thread emits Events to the Inbox. The Streamlit fragment drains
them on each poll tick. The controller dispatches each event via match/case.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from types import MappingProxyType
from typing import Any, List, Mapping, Tuple

logger = logging.getLogger(__name__)

HIGH_WATER_MARK = 500


class EventType(Enum):
    """All lifecycle events in the system."""

    # Worker → Controller (via inbox, processed on fragment tick)
    CYCLE_STARTED = auto()
    EPOCH_DONE = auto()
    EVAL_COMPLETE = auto()
    QUERYING_STARTED = auto()
    NEW_IMAGES = auto()
    ANNOTATIONS_APPLIED = auto()
    RUN_FINISHED = auto()
    RUN_ERROR = auto()
    RUN_STOPPED = auto()

    # UI → Controller (immediate, not through inbox)
    START_EXPERIMENT = auto()
    STOP_EXPERIMENT = auto()
    SUBMIT_ANNOTATIONS = auto()


@dataclass(frozen=True)
class Event:
    """Immutable event emitted by the worker or UI."""

    type: EventType
    run_id: str = ""
    cycle: int = 0
    timestamp: float = field(default_factory=time.time)
    data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Detach and freeze payload so producers cannot mutate it after emit."""
        frozen_data = MappingProxyType(copy.deepcopy(dict(self.data)))
        object.__setattr__(self, "data", frozen_data)


class Inbox:
    """
    Thread-safe event inbox with a version counter.

    The version counter increments on every put(), allowing the UI to
    cheaply detect "has anything changed?" without draining the queue.

    Events are stored in an unbounded list (not a bounded deque) to avoid
    silently dropping lifecycle events. A high-water-mark warning is logged
    if the list grows unexpectedly large.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: List[Event] = []
        self._version: int = 0

    @property
    def version(self) -> int:
        """Current version counter (monotonically increasing)."""
        with self._lock:
            return self._version

    def put(self, event: Event) -> None:
        """Append an event and bump the version. Called by the worker thread."""
        with self._lock:
            self._events.append(event)
            self._version += 1
            if len(self._events) > HIGH_WATER_MARK:
                logger.warning(
                    "Inbox high-water mark exceeded (%d events). "
                    "Fragment may not be draining fast enough.",
                    len(self._events),
                )

    def drain(self, since_version: int = 0) -> Tuple[List[Event], int]:
        """
        Return all events accumulated since *since_version* and the current version.

        Called by the Streamlit fragment on each poll tick.
        Returns ``(events_list, current_version)``.
        If *since_version* is already current, returns an empty list.
        """
        with self._lock:
            current = self._version
            if since_version >= current:
                return [], current
            events = list(self._events)
            self._events.clear()
            return events, current

    def reset(self) -> None:
        """Clear all events and reset the version counter. Called on new run."""
        with self._lock:
            self._events.clear()
            self._version = 0
