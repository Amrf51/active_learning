# Implementation Tasks: MVC Architecture for Active Learning Framework

## Overview
Transform the existing monolithic backend into an MVC + multiprocessing architecture with a Streamlit dashboard for the Bachelor Thesis: "Visual and Interactive Active Learning for Vehicle Image Classification."

## Thesis Objectives Mapping
- **Pipeline Development** → Phases 1-4 (Config, Backend, App Skeleton, MVC Core)
- **Interactive Visualization** → Phase 5 (Streamlit View with Gallery of Uncertainty)
- **Strategy Comparison** → Phase 5 (Strategy comparison dashboard)
- **Real-time Feedback** → Phase 5 (Live-Training visualization)

## Reference
- `specs/code_review_and_revised_plan.md` - Architecture decisions and code examples
- `specs/README.md` - Thesis requirements and objectives

---

## Phase 1: Configuration & Protocol Infrastructure

- [x] 1. Create YAML-based configuration system
  - [x] 1.1 Create `configs/` directory structure
  - [x] 1.2 Create `configs/default.yaml` with all configuration sections (experiment, data, model, training, active_learning, checkpoint, logging)
  - [x] 1.3 Create `configs/quick_test.yaml` for fast debugging
  - [x] 1.4 Create `config.py` with dataclasses (ExperimentConfig, DataConfig, ModelConfig, TrainingConfig, ALConfig, CheckpointConfig, Config)
  - [x] 1.5 Implement `load_config()` with YAML loading and layered merging (default → experiment → runtime overrides)
  - [x] 1.6 Implement `_deep_merge()` helper for nested dict merging
  - [x] 1.7 Add dotted-key support for runtime overrides (e.g., "training.epochs": 10)
  - [x] 1.8 Add config validation method — check value ranges, strategy names, path existence

- [x] 2. Create protocol.py for message types and events
  - [x] 2.1 Define message type constants (INIT_MODEL, RUN_CYCLE, QUERY, ANNOTATE, PROGRESS_UPDATE, TRAIN_COMPLETE, etc.)
  - [x] 2.2 Define event name constants (model_ready, training_done, query_done, stop_requested, worker_error)
  - [x] 2.3 Create message builder helpers for type-safe queue messages
  - [x] 2.4 Create helper to initialize mp.Event dict
  - [x] 2.5 Create `backend/__init__.py` — make backend a Python package for proper imports

---

## Phase 2: Backend Modifications (Minimal Changes)

- [x] 3. Update models.py for MVC compatibility
  - [x] 3.1 Modify `get_model()` to accept flat args instead of config object: `get_model(name, num_classes, pretrained=True, device="cpu")`
  - [x] 3.2 Add `search_timm_models(query, pretrained_only=True)` for UI model browser
  - [x] 3.3 Add `get_model_families()` returning curated model groups for UI dropdown
  - [x] 3.4 Add `get_model_card(model_name)` returning parameter count and pretrained status

- [-] 4. Update state.py dataclasses
  - [x] 4.1 Ensure all dataclasses have `to_dict()` or `model_dump()` methods for queue serialization
  - [x] 4.2 Verify EpochMetrics, CycleMetrics, QueriedImage, ProbeImage all serialize correctly

- [x] 5. Update data_manager.py and trainer.py
  - [x] 5.1 Add `_label_cache` dict to data_manager `__init__`
  - [x] 5.2 Add `_get_label(idx)` method with caching
  - [x] 5.3 Update `get_samples_by_class()` to use cached labels instead of loading images
  - [x] 5.4 Update trainer.py `reset_model_weights()` to use new `get_model()` signature

---

## Phase 3: Application Skeleton (Entry Point)

- [x] 6. Create app.py skeleton (Entry Point) — MUST BE BEFORE Controller/Model
  - [x] 6.1 Initialize multiprocessing context (spawn method for CUDA compatibility)
  - [x] 6.2 Create mp.Event dict for all events using protocol helper
  - [x] 6.3 Create task_queue and result_queue
  - [x] 6.4 Placeholder for Controller initialization (filled in Phase 4)
  - [x] 6.5 Placeholder for worker process spawn (filled in Phase 4)
  - [x] 6.6 Store queues and events in st.session_state
  - [x] 6.7 Basic Streamlit page config and layout structure
  - [x] 6.8 Handle graceful shutdown on app close

---

## Phase 4: MVC Core Components

- [x] 7. Create controller.py (State Machine + Dispatch)
  - [x] 7.1 Define AppState enum (IDLE, INITIALIZING, TRAINING, QUERYING, ANNOTATING, ERROR)
  - [x] 7.2 Create Controller class with state machine logic
  - [x] 7.3 Implement `dispatch_init_model(config_overrides)` - sends INIT_MODEL to worker
  - [x] 7.4 Implement `dispatch_run_cycle(cycle_num)` - sends RUN_CYCLE to worker
  - [x] 7.5 Implement `dispatch_query()` - sends QUERY to worker
  - [x] 7.6 Implement `dispatch_annotate(annotations)` - sends ANNOTATE to worker
  - [x] 7.7 Implement `dispatch_stop()` - sets stop_requested event
  - [x] 7.8 Implement `poll_results()` - non-blocking check of result_queue
  - [x] 7.9 Implement `save_state()` / `load_state()` for state.json persistence
  - [x] 7.10 Implement state transition validation (e.g., can't train while querying)

- [x] 8. Create worker.py (Worker Process)
  - [x] 8.1 Create `worker_loop(task_queue, result_queue, events, config_dict)` main loop
  - [x] 8.2 Implement `_build_al_loop(payload, config)` - initializes ActiveLearningLoop inside worker
  - [x] 8.3 Implement INIT_MODEL handler - builds model, data_manager, trainer, AL loop
  - [x] 8.4 Implement RUN_CYCLE handler - calls AL loop step methods with queue progress reporting
  - [x] 8.5 Implement QUERY handler - calls `al_loop.query_samples()`, converts QueriedImage to dicts
  - [x] 8.6 Implement ANNOTATE handler - calls `al_loop.receive_annotations()`
  - [x] 8.7 Implement SHUTDOWN handler for graceful termination
  - [x] 8.8 Add stop_requested event checking in training loop
  - [x] 8.9 Add error handling with worker_error event and error messages to queue
  - [x] 8.10 Create `_queried_to_dict(img)` helper for QueriedImage serialization

- [x] 9. Wire up app.py with Controller and Worker
  - [x] 9.1 Import and initialize Controller with queues and events
  - [x] 9.2 Start worker process with daemon=True
  - [x] 9.3 Store controller in st.session_state
  - [x] 9.4 Call view.render() for UI

---

## Phase 5: Streamlit Views

- [ ] 10. Create views/ directory structure
  - [ ] 10.1 Create `views/__init__.py`
  - [ ] 10.2 Create `views/router.py` — main render() function that dispatches to state-specific views

- [ ] 11. Implement sidebar configuration (views/sidebar.py)
  - [ ] 11.1 Model selection with curated families dropdown
  - [ ] 11.2 TIMM model search expander with text input
  - [ ] 11.3 Model info display (parameters, pretrained status)
  - [ ] 11.4 Training hyperparameters (epochs, batch_size, learning_rate sliders)
  - [ ] 11.5 AL settings (strategy dropdown, query size slider, reset mode)
  - [ ] 11.6 Data settings (data directory input, split ratios)

- [ ] 12. Implement state-specific views (views/states.py)
  - [ ] 12.1 `render_idle_view()` - start experiment button, config display
  - [ ] 12.2 `render_initializing_view()` - loading spinner, model info
  - [ ] 12.3 `render_querying_view()` - query progress indicator
  - [ ] 12.4 `render_error_view()` - error display with retry option

- [ ] 13. Implement training visualization (views/training.py) — THESIS REQUIREMENT
  - [ ] 13.1 Real-time loss/accuracy line charts using st.line_chart or plotly
  - [ ] 13.2 Progress bar for epochs within cycle
  - [ ] 13.3 Cycle progress indicator
  - [ ] 13.4 Pool statistics display (labeled/unlabeled counts)
  - [ ] 13.5 Best model metrics display
  - [ ] 13.6 Stop training button

- [ ] 14. Implement "Gallery of Uncertainty" (views/gallery.py) — THESIS REQUIREMENT
  - [ ] 14.1 Grid display of queried images (3-4 columns) with uncertainty ranking
  - [ ] 14.2 Per-image card showing: image, model prediction, confidence %, uncertainty score, selection reason
  - [ ] 14.3 Visual uncertainty indicator (color-coded border or badge based on uncertainty level)
  - [ ] 14.4 Class selection dropdown for each image (all 196 Stanford Cars classes)
  - [ ] 14.5 "Use Model Prediction" quick button
  - [ ] 14.6 "Use Ground Truth" button (for simulation mode)
  - [ ] 14.7 "Auto-label All" button for batch simulation
  - [ ] 14.8 Submit annotations button
  - [ ] 14.9 Annotation accuracy feedback after submission (correct/incorrect count)

- [ ] 15. Implement results dashboard (views/results.py) — THESIS REQUIREMENT
  - [ ] 15.1 Cycle-by-cycle metrics table (Accuracy, F1-Score, Precision, Recall)
  - [ ] 15.2 Test accuracy progression line chart
  - [ ] 15.3 F1-Score progression line chart
  - [ ] 15.4 Pool size progression chart (labeled vs unlabeled over cycles)
  - [ ] 15.5 Best cycle highlight with metrics summary
  - [ ] 15.6 Per-class performance breakdown (expandable section)
  - [ ] 15.7 Export results button (JSON download with all metrics)
  - [ ] 15.8 Probe image tracker — show how predictions change across cycles for fixed validation images
  - [ ] 15.9 Confusion matrix heatmap (top-20 most confused class pairs)

- [ ] 16. Implement Strategy Comparison View (views/comparison.py) — THESIS REQUIREMENT
  - [ ] 16.1 Multi-experiment tracking in session state
  - [ ] 16.2 Side-by-side comparison table (Random vs Entropy vs Margin vs LC)
  - [ ] 16.3 Overlay line chart comparing accuracy curves across strategies
  - [ ] 16.4 Statistical summary (final accuracy, improvement %, cycles to reach threshold)
  - [ ] 16.5 Export comparison report (CSV/JSON)

- [ ] 17. Implement Dataset Explorer (views/explorer.py) — THESIS REQUIREMENT
  - [ ] 17.1 Browse labeled pool by class
  - [ ] 17.2 Browse unlabeled pool samples
  - [ ] 17.3 View queried images history per cycle
  - [ ] 17.4 Filter by uncertainty score range
  - [ ] 17.5 Display image metadata (path, true label, predicted label, confidence)
  - [ ] 17.6 Visualize class distribution in labeled/unlabeled pools (bar chart)

---

## Phase 6: Session State Management

- [ ] 18. Session state management
  - [ ] 18.1 Initialize session state on first run
  - [ ] 18.2 Store current_cycle, current_epoch, metrics_history
  - [ ] 18.3 Store queried_images for annotation view
  - [ ] 18.4 Store experiment_config for persistence
  - [ ] 18.5 Store experiment_history for strategy comparison (multiple runs)
  - [ ] 18.6 Implement session state reset for new experiment

---

## Phase 7: Integration & Testing

- [ ] 19. Integration testing
  - [ ] 19.1 Test config loading with overrides
  - [ ] 19.2 Test worker process startup and shutdown
  - [ ] 19.3 Test message passing through queues
  - [ ] 19.4 Test event signaling (stop, completion)
  - [ ] 19.5 Test state persistence and resume

- [ ] 20. End-to-end workflow testing
  - [ ] 20.1 Test full cycle: init → train → query → annotate → next cycle
  - [ ] 20.2 Test stop button during training
  - [ ] 20.3 Test error recovery
  - [ ] 20.4 Test with quick_test.yaml config
  - [ ] 20.5 Test strategy comparison workflow (run multiple experiments)

---

## Phase 8: Polish & Documentation

- [ ] 21. Error handling improvements
  - [ ] 21.1 Add timeout handling for worker responses
  - [ ] 21.2 Add worker health check mechanism
  - [ ] 21.3 Add graceful degradation for GPU unavailability
  - [ ] 21.4 Add user-friendly error messages in UI

- [ ] 22. Documentation
  - [ ] 22.1 Update README with new architecture overview
  - [ ] 22.2 Add docstrings to all new modules
  - [ ] 22.3 Create usage guide for Streamlit dashboard
  - [ ] 22.4 Document configuration options
  - [ ] 22.5 Add screenshots of key UI features for thesis


---

## File Summary

| File | Action | Description |
|------|--------|-------------|
| `configs/default.yaml` | NEW | Base configuration |
| `configs/quick_test.yaml` | NEW | Fast debugging config |
| `config.py` | NEW | YAML config loading + validation |
| `protocol.py` | NEW | Message types + Event registry |
| `backend/__init__.py` | NEW | Package init for backend modules |
| `controller.py` | NEW | State machine + dispatch |
| `worker.py` | NEW | Worker process loop (renamed from model.py for clarity) |
| `app.py` | NEW | Entry point + Streamlit bootstrap |
| `views/__init__.py` | NEW | Views package init |
| `views/router.py` | NEW | Main view dispatcher |
| `views/sidebar.py` | NEW | Sidebar configuration controls |
| `views/states.py` | NEW | Idle, initializing, error views |
| `views/training.py` | NEW | Live training visualization |
| `views/gallery.py` | NEW | Gallery of Uncertainty (annotation) |
| `views/results.py` | NEW | Results dashboard + probe tracker + confusion matrix |
| `views/comparison.py` | NEW | Strategy comparison view |
| `views/explorer.py` | NEW | Dataset explorer |
| `models.py` | MODIFY | Add TIMM search, flat args |
| `state.py` | MODIFY | Ensure serialization methods |
| `data_manager.py` | MODIFY | Add label caching |
| `trainer.py` | MODIFY | Update reset_model_weights() for new get_model() signature |
| `strategies.py` | KEEP | No changes |
| `active_loop.py` | KEEP | No changes |
| `dataloader.py` | KEEP | No changes |

---

## Thesis Requirements Checklist

| Thesis Requirement | Task Reference | Status |
|--------------------|----------------|--------|
| AL Pipeline with PyTorch | Existing backend + Tasks 7-8 | ✅ Covered |
| Streamlit GUI | Tasks 10-17 | ✅ Covered |
| Strategy Comparison (LC, Margin, Entropy, Random) | Task 16 | ✅ Covered |
| Live-Training visualization | Task 13 | ✅ Covered |
| Gallery of Uncertainty | Task 14 | ✅ Covered |
| ResNet/MobileNet support | Task 3 (TIMM) | ✅ Covered |
| Stanford Cars dataset | Existing dataloader.py | ✅ Covered |
| F1-Score display | Task 15.3 | ✅ Covered |
| Dataset Explorer | Task 17 | ✅ Covered |
| Experiment export | Tasks 15.7, 16.5 | ✅ Covered |
| Probe image tracking | Task 15.8 | ✅ Covered |
| Confusion matrix | Task 15.9 | ✅ Covered |

---

## Issues Fixed Summary

| Issue | Fix Applied |
|-------|-------------|
| 🔴 Phase ordering wrong | Moved app.py to Phase 3 (Task 6), before Controller/Worker |
| 🔴 trainer.py breaking change | Added Task 5.4 to update reset_model_weights() |
| 🔴 Relative imports will break | Added Task 2.5 to create backend/__init__.py package |
| 🟡 views/ subfolder inconsistency | Added Task 10 for views/ directory structure with router |
| 🟡 Missing probe images view | Added Task 15.8 for probe image tracker |
| 🟡 Missing confusion matrix | Added Task 15.9 for confusion matrix heatmap |
| 🟡 Missing config validation | Added Task 1.8 for config validation method |
