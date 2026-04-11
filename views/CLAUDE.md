# views/ — Streamlit UI Views

## Module Responsibilities

| File | Role |
|---|---|
| `router.py` | AppState → view dispatch; tab layout (Main / Results / Compare / Explorer) |
| `sidebar.py` | Config controls + Start/Stop/Next Step; dispatches controller events |
| `training.py` | Live training charts, epoch progress, pool size metrics |
| `gallery.py` | Manual annotation image grid + annotation submit flow |
| `results.py` | Disk-first run browser; reads from `experiments/` directory |
| `explorer.py` | Pool size and class distribution display (read-only) |

## How Rendering Works

All views receive a `snap` dict from `controller.get_snapshot()` — an atomic deep-copy taken once before the render. Never call `controller.get_snapshot()` again mid-render; state may change between calls.

The routing entry point is `views/router.py:render(snap)`. It builds the tab layout and delegates the Main tab to `_render_state_view()`, which matches `snap["app_state"]` to the appropriate view function. The Results, Compare, and Explorer tabs always render regardless of state.

## Streamlit Fragment Pattern

Auto-polling fragments (`@st.fragment(run_every=...)`) only appear in `app.py`. Do not add new polling fragments elsewhere. Route new live-update needs through the existing `fast_live_update_fragment` / `slow_live_update_fragment` by adjusting which `AppState` values map to fast vs. slow in `app.py`'s `FAST_POLL_STATES` / `SLOW_POLL_STATES` sets.

`@st.cache_data` is used in `gallery.py` for image loading (`_load_display_image`) since image bytes are stable for a given path. Do not use `@st.cache_data` on anything that depends on experiment state — state changes won't invalidate the cache.

## Session State Keys

Initialized in `app.py:init_session_state()`:

| Key | Type | Purpose |
|---|---|---|
| `controller` | `Controller` | Process-level singleton (from `st.cache_resource`) |
| `config` | `Config` | Current config (also from cache, re-pointed each session) |
| `annotations` | `dict` | `{image_id: label}` — cleared on `CYCLE_STARTED` and `NEW_IMAGES` |
| `poll_mode` | `str` | `"fast"` / `"slow"` / `"off"` — controls which fragment is active |
| `last_event_version` | `int` | Inbox version cursor for incremental event draining |
| `current_cycle_id` | `tuple` | `(run_id, cycle)` — set on `CYCLE_STARTED` |

Do not add new persistent session state keys without listing them here. Prefer deriving values from `snap` on each render rather than caching them in session state.

## Sidebar and Controller Dispatch

The sidebar never mutates `ExperimentState` directly. All actions go through `controller.dispatch(Event(...))`:

```python
# Start experiment
controller.dispatch(Event(type=EventType.START_EXPERIMENT, data={"config": config}))

# Stop experiment
controller.dispatch(Event(type=EventType.STOP_EXPERIMENT, data={"join_timeout": 5.0}))

# Next step (step mode)
controller.next_step(run_id=snap["run_id"])

# Submit annotations (called from gallery)
controller.submit_annotations(annotations, run_id, cycle, query_token)
```

Sidebar buttons are enabled/disabled based on `controller.is_busy()` and `snap["app_state"]`. Do not add buttons that bypass these guards.

## Manual Annotation Flow

1. State transitions to `ANNOTATING`; `snap["queried_images"]` holds the `QueriedImage` dicts.
2. `gallery.py` renders one card per image with a class selector. Selections accumulate in `st.session_state.annotations` keyed by `image_id`.
3. On submit, the gallery calls `controller.submit_annotations(...)` with the `query_token` from `snap`. The controller validates the token and forwards to the worker.
4. `st.session_state.annotations` is cleared by `app.py:_handle_ui_effects()` on `NEW_IMAGES` and `CYCLE_STARTED` events — do not clear it manually in view code.

## Results View (Disk-First)

`results.py` reads run directories from `experiments/` on disk, not from live controller state. It reads `al_cycle_results.json`, `config.yaml`, and `*.npy` confusion matrices. This means it shows all previous runs, including those from past sessions. Do not attempt to display `ExperimentState` history here — use on-disk artifacts only.

## Adding a New View State

1. Add the `AppState` variant (see `core/CLAUDE.md` for the full process).
2. Add a branch in `views/router.py:_render_state_view()`.
3. Write a `render_*_view(controller, snap)` function (or a standalone module if complex).
4. Decide fast/slow/off polling and update `FAST_POLL_STATES` / `SLOW_POLL_STATES` in `app.py`.
