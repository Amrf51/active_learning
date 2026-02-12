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

## Phase 5: Streamlit Views (MWP Scope)

**Goal:** Get one full AL experiment running end-to-end with visualization.

**MWP Scope:**
- sidebar.py → Config controls (model, strategy, epochs, query size)
- training.py → Progress bar + live loss/accuracy chart
- gallery.py → Image grid with predictions + annotation submit
- results.py → Simple table of cycle metrics

**Deferred (not essential for thesis demo):**
- comparison.py → Run one strategy at a time, compare manually
- explorer.py → Not essential for thesis demo
- Error handling UI → Let it crash, fix later

---

- [-] 10. Create views/ directory structure
  - [ ] 10.1 Create `views/__init__.py`
  - [ ] 10.2 Create `views/router.py` — main render() function that dispatches to state-specific views

- [ ] 11. Implement sidebar configuration (views/sidebar.py)
  - [ ] 11.1 Model selection dropdown (curated families: ResNet, MobileNet, EfficientNet)
  - [ ] 11.2 Strategy dropdown (entropy, margin, least_confidence, random)
  - [ ] 11.3 Training hyperparameters (epochs slider, batch_size, learning_rate)
  - [ ] 11.4 AL settings (query size slider, num_cycles)
  - [ ] 11.5 Start Experiment button (dispatches first cycle)
  - [ ] 11.6 Stop button (sets stop_requested event)

- [ ] 12. Implement training visualization (views/training.py)
  - [ ] 12.1 Progress bar for current epoch within cycle
  - [ ] 12.2 Cycle progress indicator (Cycle X/N)
  - [ ] 12.3 Live loss/accuracy line chart (updates each epoch via polling)
  - [ ] 12.4 Pool statistics display (labeled/unlabeled counts)
  - [ ] 12.5 Current metrics display (train_loss, train_acc, val_acc)

- [ ] 13. Implement "Gallery of Uncertainty" (views/gallery.py)
  - [ ] 13.1 Grid display of queried images (3-4 columns)
  - [ ] 13.2 Per-image card: image thumbnail, prediction, confidence %, uncertainty score
  - [ ] 13.3 "Auto-label All (Ground Truth)" button for batch simulation
  - [ ] 13.4 Submit annotations button (dispatches ANNOTATE, triggers next cycle)
  - [ ] 13.5 Annotation feedback after submission (X/Y correct)

- [ ] 14. Implement results dashboard (views/results.py)
  - [ ] 14.1 Cycle-by-cycle metrics table (Cycle, Labeled, Val Acc, Test Acc, F1)
  - [ ] 14.2 Test accuracy progression line chart
  - [ ] 14.3 Best cycle highlight with summary

- [ ] 15. Wire up app.py with view router
  - [ ] 15.1 Import and call `views.router.render()` in main()
  - [ ] 15.2 Pass controller and session state to router
  - [ ] 15.3 Implement polling loop for progress updates during training

---

## Phase 6: End-to-End Testing (MWP Validation)

- [ ] 16. Manual end-to-end testing
  - [ ] 16.1 Start app with `streamlit run app.py`
  - [ ] 16.2 Configure experiment via sidebar
  - [ ] 16.3 Click Start → verify training progress updates
  - [ ] 16.4 After training → verify gallery shows queried images
  - [ ] 16.5 Click Auto-label → Submit → verify next cycle starts
  - [ ] 16.6 Complete 3+ cycles → verify results table populates
  - [ ] 16.7 Test with quick_test.yaml config for fast iteration

---

## Phase 7: Post-MWP Enhancements (DEFERRED)

These tasks are deferred until MWP is working:

- [ ]* 17. Strategy Comparison View (views/comparison.py)
  - [ ]* 17.1 Multi-experiment tracking in session state
  - [ ]* 17.2 Side-by-side comparison table
  - [ ]* 17.3 Overlay line chart comparing strategies

- [ ]* 18. Dataset Explorer (views/explorer.py)
  - [ ]* 18.1 Browse labeled/unlabeled pools by class
  - [ ]* 18.2 View queried images history per cycle
  - [ ]* 18.3 Class distribution visualization

- [ ]* 19. Advanced Results Features
  - [ ]* 19.1 Probe image tracker
  - [ ]* 19.2 Confusion matrix heatmap
  - [ ]* 19.3 Per-class performance breakdown
  - [ ]* 19.4 Export results button (JSON download)

- [ ]* 20. Error Handling & Polish
  - [ ]* 20.1 Timeout handling for worker responses
  - [ ]* 20.2 Worker health check mechanism
  - [ ]* 20.3 User-friendly error messages in UI
  - [ ]* 20.4 Graceful degradation for GPU unavailability

- [ ]* 21. Documentation
  - [ ]* 21.1 Update README with architecture overview
  - [ ]* 21.2 Usage guide for Streamlit dashboard
  - [ ]* 21.3 Screenshots for thesis


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
