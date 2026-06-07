# Active Learning Framework — CLAUDE.md

## What This Project Is

Bachelor thesis project implementing **Active Learning for vehicle image classification**. A PyTorch model iteratively selects its most uncertain training samples from an unlabeled pool; a human (or simulated oracle) annotates them. The experiment runs interactively through a Streamlit dashboard.

- **Entry point:** `streamlit run app.py`
- **Active development branch:** `events` — `main` is stable/thesis snapshots
- **No automated test suite.** Verification is done manually by running the UI.

## Running the Project

```bash
pip install -r requirements.txt       # Python 3.10+ required
streamlit run app.py                  # Start the UI
```

The app expects an ImageFolder-structured dataset at the path set in `configs/default.yaml` (`data.data_dir`). Use `configs/quick_test.yaml` for fast 4-class smoke-test runs.

## Architecture Overview

```
app.py                  Streamlit entry point. Polling fragments, session state init, view routing.
config.py               YAML config loading + dataclass validation + runtime override merging.
core/
  controller.py         Central router between UI actions and worker thread. Owns ExperimentState.
  worker.py             Background daemon thread. Builds AL loop, emits typed events.
  active_loop.py        Per-run AL cycle: train → evaluate → query → annotate. Called by worker.
  experiment_state.py   AppState enum + ExperimentState (thread-safe shared state object).
  events.py             EventType enum, immutable Event dataclass, Inbox queue.
  state.py              Backend dataclasses: EpochMetrics, CycleMetrics, QueriedImage, ProbeImage.
ml/
  trainer.py            Training/validation loops, checkpointing, early stopping.
  data_manager.py       Labeled/unlabeled pool index management (no data duplication).
  dataloader.py         ImageFolder loading, train/val/test splits, augmentation pipelines.
  strategies.py         Uncertainty sampling registry: entropy, margin, least_confidence, random.
  models.py             TIMM model loader (get_model, get_feature_dim).
  embeddings.py         UMAP 2D projection saved per cycle for the results dashboard.
  losses.py             SupConLoss (Khosla 2020) + ProjectionHead.
views/
  router.py             AppState → view dispatch; builds tab layout.
  sidebar.py            Config controls, Start/Stop/Next Step buttons.
  training.py           Live training charts and epoch metrics.
  gallery.py            Manual annotation image grid + submit flow.
  results.py            Disk-first run browser (reads experiments/ directory).
  explorer.py           Dataset pool size and class distribution viewer.
configs/
  default.yaml          Stanford Cars experiment (main thesis config).
  quick_test.yaml       Smoke-test config for fast end-to-end runs.
experiments/            Run outputs — NOT in git.
data/                   Raw datasets — NOT in git.
```

## Config System

Config is layered in order:

1. `configs/default.yaml` (base values)
2. Optional experiment YAML passed to `load_config(path)`
3. Runtime dict overrides (dotted keys e.g. `"training.epochs"`) — used by the sidebar

Sidebar changes **do not edit YAML files**. They build an overrides dict and call `load_config(overrides={...})` when Start is clicked. The resolved config is saved to `experiments/{name}/{timestamp}_{run_id8}/config.yaml` for reproducibility.

Top-level sections: `experiment`, `data`, `model`, `training`, `active_learning`, `checkpoint`, `logging`. `load_config()` raises `ValueError` on invalid values.

## Key Patterns

**Thread-safe communication only.** Two channels exist between the UI thread and the worker thread:
- `Inbox` (worker → UI): typed `Event` objects put by the worker, drained by the Streamlit fragment.
- `command_queue` (UI → worker): plain dicts (`STOP`, `NEXT_STEP`, `SUBMIT_ANNOTATIONS`).

Never share mutable state between threads except through these two channels.

**Always use snapshots.** The UI reads state via `controller.get_snapshot()`, which returns a deep-copy dict under lock. Never access `controller.state.*` fields directly from view code.

**AppState drives everything.** View routing, sidebar button states, and polling mode all key off `snap["app_state"]`. When adding a UI state: add it to `AppState`, handle it in `controller.dispatch()`, add a branch in `views/router.py`, and update `FAST_POLL_STATES`/`SLOW_POLL_STATES` in `app.py`.

**Event payloads are immutable.** `Event.data` is deep-copied and frozen (`MappingProxyType`) on construction.

**Adaptive polling.** Fast (0.5 s) for `QUERYING`/`ANNOTATING`. Slow (1.5 s) for `INITIALIZING`/`TRAINING`/`STOPPING`. Off (static render) for `IDLE`/`FINISHED`/`ERROR`/`WAITING_STEP`.

**Controller is a singleton.** `get_controller()` uses `@st.cache_resource` — same instance across browser refreshes. Do not re-create it per render.

**`num_workers` is platform-dependent.** On Windows, DataLoader workers inside a daemon thread deadlock — the controller forces `config.data.num_workers = 0` in that case. On Linux (e.g. the university JupyterHub cluster), workers function correctly and `num_workers=6` is confirmed working. Do not set a non-zero value when running on Windows.

## Output Directory Structure

```
experiments/
  {experiment_name}/
    {YYYYMMDD_HHMM}_{run_id8}/
      config.yaml
      al_cycle_results.json
      al_pool_state.json
      training_history.json
      training_log.txt
      checkpoints/
        best_model.pth
        best_model_cycle_{n}.pth
      confusion_matrices/cycle_{n}.npy
      queries/cycle_{n}/
      cycle_{n}_annotations.json
      cycle_{n}_embeddings.npz
```

## What NOT to Do

- **Do not add a test framework.** No pytest, unittest, `tests/` directory, or `conftest.py`. Testing is manual via the UI.
- **Do not call `st.rerun()` from inside a cached function or from a background thread.** Only fragment functions in `app.py` trigger reruns.
- **Do not read `controller.state.*` directly from view code.** Always go through `controller.get_snapshot()`.
- **Do not emit events from the UI thread.** Events flow worker → inbox → controller only.
- **Do not add blocking calls to view render functions.** Views must return quickly; long work belongs in the worker thread.
- **Do not set `data.num_workers` to a non-zero value on Windows** — it will deadlock inside Streamlit's daemon thread. On Linux (university cluster) `num_workers=6` works fine.
- **Do not use `st.cache_data` on objects that store experiment state** — state changes won't invalidate the cache.
