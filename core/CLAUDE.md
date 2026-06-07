# core/ — Threading, Events, and State

## Module Responsibilities

| File | Role |
|---|---|
| `events.py` | `EventType` enum, immutable `Event` dataclass, `Inbox` queue |
| `experiment_state.py` | `AppState` enum, `ExperimentState` (thread-safe shared state) |
| `controller.py` | Command/event router; owns `ExperimentState`; spawns worker |
| `worker.py` | Daemon thread; builds AL loop; emits events; handles commands |
| `active_loop.py` | Per-run training/query/eval orchestration (called by worker) |
| `state.py` | Backend dataclasses with no Streamlit dependency |

## Communication Architecture

Two strictly separated channels:

```
Worker → UI:   worker calls inbox.put(Event(...))
               UI fragment calls controller.process_inbox(last_version)
               controller.dispatch() applies state updates

UI → Worker:   UI calls controller.dispatch(Event(type=SUBMIT_ANNOTATIONS, ...))
               controller._handle_submit() pushes dict to command_queue
               worker pulls from command_queue in _wait_for_annotations()
```

The `Inbox` uses a version counter so the fragment can cheaply detect "has anything new arrived?" without draining the full queue. Never replace `Inbox` with a plain `queue.Queue` — the drain-by-version semantics are load-bearing.

## AppState Transitions

```
IDLE          → INITIALIZING   (START_EXPERIMENT command)
INITIALIZING  → TRAINING       (CYCLE_STARTED event)
TRAINING      → QUERYING       (QUERYING_STARTED event)
TRAINING      → WAITING_STEP   (WAITING_FOR_STEP event, step_mode=True)
WAITING_STEP  → TRAINING       (NEXT_STEP command)
QUERYING      → ANNOTATING     (NEW_IMAGES event, auto_annotate=False)
QUERYING      → TRAINING       (ANNOTATIONS_APPLIED event, auto_annotate=True)
ANNOTATING    → TRAINING       (ANNOTATIONS_APPLIED event)
Any active    → STOPPING       (STOP_EXPERIMENT command)
STOPPING      → IDLE           (worker thread exits cleanly)
Any active    → FINISHED       (RUN_FINISHED event)
Any active    → ERROR          (RUN_ERROR event)
```

`controller.dispatch()` is the **single authority** for all state transitions. Do not mutate `app_state` anywhere else.

## Adding a New EventType

1. Add the variant to `EventType` in `events.py`.
2. Emit it in `worker.py` via `_emit_event(inbox, run_id, cycle, EventType.YOUR_EVENT, data={...})`.
3. Add a `case EventType.YOUR_EVENT:` branch in `controller.dispatch()`.
4. If the event drives a new `AppState`, add that state to the `AppState` enum and handle it in `views/router.py` and the polling config in `app.py`.
5. Event `data` dict must be JSON-serializable — no dataclasses, no tensors, no numpy arrays.

## Thread Safety Rules

- All `ExperimentState` field writes go through `update_for_run(run_id, **kwargs)`. This validates the run_id under lock before writing. If the run has been superseded, the write is silently dropped.
- `snapshot()` returns a deep-copy under lock — the returned dict is safe to read in the UI thread without further locking.
- `set_error()` bypasses run_id validation (exceptions must always be recorded).
- The `command_queue` is a plain `queue.Queue`. The worker calls `get(timeout=...)` in a loop; the UI puts single command dicts.
- Never hold `ExperimentState._lock` while doing I/O or calling any Streamlit APIs.

## run_id and Token Guards

Every run gets a UUID (`run_id`). All worker events carry this `run_id`. `controller.process_inbox()` discards events whose `run_id` does not match the current active run — stale events from previous runs are dropped without dispatch.

The annotation handshake uses a one-time `query_token` (UUID set on `NEW_IMAGES`, cleared on `ANNOTATIONS_APPLIED`). The controller validates token + run_id + cycle + app_state before forwarding to the command queue. This prevents duplicate annotation submissions from stale gallery renders triggered by browser back-button or fragment re-runs.
