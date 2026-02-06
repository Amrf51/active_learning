# Implementation Plan: MVC Active Learning Dashboard

## Overview

This plan implements the MVC architecture for the Active Learning Dashboard using **multiprocessing with pipes** for process isolation. The ActiveLearning service runs in a separate process, communicating with the controller via bidirectional pipes. The existing `backend/` code remains unchanged.

## Tasks

- [x] 1. Set up MVC folder structure and core data types
  - [x] 1.1 Create controller/ and model/ directories with __init__.py files
    - Create `controller/__init__.py` and `model/__init__.py`
    - _Requirements: 8.1, 8.3_
  
  - [x] 1.2 Implement Phase enum and WorldState dataclass
    - Create `model/world_state.py` with Phase enum and WorldState dataclass
    - Include all fields: experiment_id, phase, current_cycle, epoch_metrics, queried_images, etc.
    - _Requirements: 7.1, 7.3, 7.4, 7.5_
  
  - [x] 1.3 Implement Event system
    - Create `controller/events.py` with EventType enum and Event dataclass
    - Include: CREATE_EXPERIMENT, LOAD_EXPERIMENT, START_CYCLE, PAUSE, STOP, SUBMIT_ANNOTATIONS
    - _Requirements: 8.1, 8.3_

- [x] 2. Implement ExperimentManager (persistence layer)
  - [x] 2.1 Create SQLite database initialization
    - Create `model/experiment_manager.py` with ExperimentManager class
    - Implement `_init_db()` to create experiments, cycle_results, epoch_metrics tables
    - _Requirements: 9.1, 9.5_
  
  - [x] 2.2 Implement experiment CRUD operations
    - Implement `create_experiment()`, `list_experiments()`, `load_experiment()`, `delete_experiment()`
    - Generate unique experiment IDs
    - Create experiment folder structure on creation
    - _Requirements: 1.1, 1.2, 2.1, 2.2, 2.3_
  
  - [ ] 2.2.1 Update experiment folder naming to use user-provided name
    - Modify `create_experiment()` to use experiment_name for folder instead of UUID
    - Update `delete_experiment()` to use experiment_name for folder lookup
    - Update all methods that reference experiment folders (save_checkpoint, save_confusion_matrix, etc.)
    - Ensure experiment_name is sanitized for filesystem use (remove special characters)
    - Handle potential name collisions (append suffix if folder exists)
    - _Requirements: 1.3, 9.3, 9.4_
  
  - [ ]* 2.3 Write property tests for experiment management
    - **Property 1: Experiment ID Uniqueness**
    - **Property 2: Experiment Persistence Round-Trip**
    - **Property 3: Experiment Listing Completeness**
    - **Property 4: Experiment Deletion Removes from List**
    - **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.3**
  
  - [x] 2.4 Implement cycle results persistence
    - Implement `save_cycle_result()` and `get_cycle_results()` with pagination
    - _Requirements: 4.2, 6.5, 10.4_
  
  - [ ]* 2.5 Write property tests for cycle results
    - **Property 8: Cycle Results Persistence Round-Trip**
    - **Property 13: Pagination Correctness**
    - **Validates: Requirements 4.2, 6.5**
  
  - [x] 2.6 Implement artifact persistence (checkpoints, confusion matrices)
    - Implement `save_checkpoint()`, `load_checkpoint()`, `save_confusion_matrix()`, `load_confusion_matrix()`
    - Use folder-based storage: experiments/{experiment_name}/checkpoints/, experiments/{experiment_name}/results/
    - _Requirements: 9.2, 9.3, 9.4_
  
  - [ ]* 2.7 Write property tests for artifact persistence
    - **Property 14: Checkpoint Persistence Round-Trip**
    - **Property 15: Confusion Matrix Persistence Round-Trip**
    - **Validates: Requirements 9.3, 9.4**

- [ ] 3. Checkpoint - Ensure persistence layer tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement ModelHandler (backend orchestration)
  - [x] 4.1 Create ModelHandler class with WorldState management
    - Create `model/model_handler.py` with ModelHandler class
    - Initialize WorldState and ExperimentManager
    - Implement `handle_event()` dispatcher
    - _Requirements: 7.2, 8.2_
  
  - [x] 4.2 Implement experiment creation and loading handlers
    - Handle CREATE_EXPERIMENT: create via ExperimentManager, update WorldState
    - Handle LOAD_EXPERIMENT: load via ExperimentManager, restore WorldState
    - _Requirements: 1.1, 1.2, 1.6, 2.2_
  
  - [x] 4.3 Implement training cycle handlers
    - Handle START_CYCLE: set phase to TRAINING, prepare ActiveLearningLoop
    - Implement `train_epoch()`: train one epoch, update WorldState.epoch_metrics
    - Transition to QUERYING after training, then AWAITING_ANNOTATION after querying
    - _Requirements: 3.2, 3.3, 3.4, 3.5_
  
  - [ ]* 4.4 Write property tests for training cycle
    - **Property 5: Epoch Tracking Consistency**
    - **Property 6: Phase Transition After Training**
    - **Property 7: Queried Images Population**
    - **Validates: Requirements 3.2, 3.3, 3.4, 3.5**
  
  - [x] 4.5 Implement annotation submission handler
    - Handle SUBMIT_ANNOTATIONS: pass to ActiveLearningLoop, save cycle results
    - Increment current_cycle, transition to COMPLETED if final cycle
    - _Requirements: 4.1, 4.2, 4.3, 4.4_
  
  - [ ]* 4.6 Write property tests for annotation submission
    - **Property 9: Cycle Counter Increment**
    - **Property 10: Completion Phase Transition**
    - **Validates: Requirements 4.3, 4.4**
  
  - [x] 4.7 Implement pause and stop handlers
    - Handle PAUSE: set flag to pause training loop
    - Handle STOP: terminate training, save current state
    - Update WorldState.phase accordingly
    - _Requirements: 5.1, 5.2, 5.3_
  
  - [ ]* 4.8 Write property test for control actions
    - **Property 11: Phase Reflects Control Actions**
    - **Validates: Requirements 5.3**

- [ ] 5. Checkpoint - Ensure model layer tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement Multiprocessing Infrastructure (NEW)
  - [x] 6.1 Update WorldState for picklability
    - Add `updated_at: float` field for state versioning
    - Ensure all fields are picklable (no lambdas, no open handles)
    - Add unit test to verify WorldState is picklable
    (provide the file to test but dont run the commands, i will test them manually)
    - _Requirements: 7.1_
  
  - [x] 6.2 Update Event system for picklability
    - Add `timestamp: float` field to Event dataclass
    - Add SHUTDOWN event type for graceful service termination
    - Add unit test to verify Event is picklable
    (provide the file to test but dont run the commands, i will test them manually)
    - _Requirements: 8.1_
  
  - [x] 6.3 Create BackgroundWorker class
    - Create `controller/background_worker.py`
    - Implement `start()`: spawn service process with Pipe
    - Implement `send_event()`: write event to pipe
    - Implement `poll_state()`: non-blocking read from pipe
    - Implement `is_alive()`: check process health
    - Implement `shutdown()`: graceful termination with timeout
    - _Requirements: 3.1_
  
  - [x] 6.4 Create ActiveLearningService class
    - Create `services/__init__.py`
    - Create `services/active_learning_service.py`
    - Implement `run_service_loop()`: entry point for Process
    - Implement `run()`: main event loop with pipe polling
    - Implement `_handle_event()`: delegate to ModelHandler
    - Implement `_train_one_epoch()`: train and push state
    - Implement `_push_state()`: send WorldState via pipe
    - _Requirements: 3.1, 7.2_
  
  - [x] 6.5 Write property tests for multiprocessing

    - **Property 16: Service Process Lifecycle**
    - **Property 17: Pipe Communication Integrity**
    - **Property 18: State Push After Event**
    (provide the testing files but dont run the commands, i will test them manually)
    - **Validates: Requirements 3.1, 7.2, 8.2**

- [x] 7. Update ExperimentController for Multiprocessing
  - [x] 7.1 Refactor ExperimentController to use BackgroundWorker
    - Replace direct ModelHandler with BackgroundWorker
    - Add `_cached_state: WorldState` for fast local reads
    - Update `dispatch()` to use `worker.send_event()`
    - Update `get_state()` to return cached state
    - Add `poll_updates()` to receive state from pipe
    - Add `drain_updates()` to get latest state
    - Add `shutdown()` for graceful service termination
    - _Requirements: 7.2, 8.2_
  
  - [x] 7.2 Add service health monitoring
    - Implement `is_service_alive()` check
    - Set error state if service dies unexpectedly
    - _Requirements: 7.5_
  
  - [x] 7.3 Remove threading code
    - Remove `_training_thread` and related methods
    - Remove `_state_lock` (no longer needed with process isolation)
    - Training now runs in service process, not thread
    - _Requirements: 3.1_

- [ ] 8. Checkpoint - Ensure multiprocessing infrastructure works
  - Test service start/shutdown lifecycle
  - Test event sending and state receiving via pipe
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Update Configuration Page to use MVC
  - [x] 9.1 Initialize controller in session state
    - Create ExperimentController on app start
    - Store in st.session_state for persistence across reruns
    - _Requirements: 8.4_
  
  - [x] 9.2 Update experiment creation to dispatch events
    - Replace direct backend calls with controller.dispatch(Event(CREATE_EXPERIMENT, {...}))
    - Read state via controller.get_state() for UI updates
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 8.4_

- [x] 10. Update Active Learning Page for Multiprocessing
  - [x] 10.1 Update training controls with events
    - Start button: dispatch START_CYCLE event (training runs in service process)
    - Pause button: dispatch PAUSE event
    - Stop button: dispatch STOP event
    - Remove `start_training_async()` call (no longer needed)
    - _Requirements: 3.1, 5.1, 5.2, 8.4_
  
  - [x] 10.2 Implement live metrics display via state polling
    - Use `@st.fragment(run_every=2.0)` decorator for efficient auto-refresh during training
    - Call `controller.poll_updates()` before `get_state()` inside the fragment
    - Display epoch_metrics as loss/accuracy curves using st.line_chart or plotly
    - Show current_cycle, current_epoch progress with st.metric
    - Fragment auto-refreshes only the metrics section, avoiding full page flicker
    - _Requirements: 3.2, 3.6_
  
  - [x] 10.3 Implement annotation interface
    - Display queried_images from WorldState in grid
    - Collect annotations and dispatch SUBMIT_ANNOTATIONS event
    - _Requirements: 4.1, 4.5_

- [x] 11. Update Results Page to use MVC
  - [x] 11.1 Implement experiment selector
    - Load experiment list via ExperimentManager
    - Dispatch LOAD_EXPERIMENT on selection
    - _Requirements: 2.1, 2.4_
  
  - [x] 11.2 Implement results visualization
    - Display cycle metrics from ExperimentManager
    - Show confusion matrix heatmaps
    - _Requirements: 6.1, 6.4_
  
  - [x] 11.3 Implement export functionality
    - Export cycle metrics to CSV/JSON
    - _Requirements: 6.3_
  
  - [ ]* 11.4 Write property test for export round-trip
    - **Property 12: Export Data Round-Trip**
    - **Validates: Requirements 6.3**

- [x] 12. Implement Dataset Explorer Page
  - [x] 12.1 Create Dataset Explorer page structure
    - Create `pages/4_Dataset_Explorer.py`
    - Add pool selector (labeled/unlabeled)
    - _Requirements: 11.1, 11.2_
  
  - [x] 12.2 Implement pool visualization with pagination
    - Display images in grid with lazy loading
    - Show image metadata (path, label if labeled)
    - Implement pagination for large pools
    - _Requirements: 11.1, 11.2, 11.4_
  
  - [x] 12.3 Implement pool statistics display
    - Show total counts for labeled/unlabeled pools
    - Display class distribution chart for labeled pool
    - _Requirements: 11.3_
  
  - [x] 12.4 Implement filtering functionality
    - Add filter by class name dropdown
    - Add search by filename
    - _Requirements: 11.5_

- [x] 13. Update pages to fit the new flow
  - [x] 13.1 check configuration.py
    - check compatibility with the required components 
    - imports
    - update required parts

  - [x] 13.2 check results.py
    - check compatibility with the required components 
    - imports
    - update required parts
  - [x] 13.3 Backend integration with the MVC

  - [x] 13.4 Check the whole project to ensure that readme.md is fulfilled
  

- [x] 14. Update Dashboard Entry Point

 
  - [x] 14.1 Add service lifecycle management
    - Initialize controller on app start
    - Add cleanup on app shutdown (atexit handler)
    - Handle service restart if it dies
    - _Requirements: 3.1_
  
  - [x] 14.2 Add service health indicator
    - Show service status in sidebar
    - Display error if service is not alive
    - _Requirements: 7.5_

  - [x] 14.3 Check dashboard.py to find any matching or import errors accros with the mvc as well as the integration



## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- The existing `backend/` code remains unchanged
- All view interactions go through controller.dispatch() - views never call backend directly
- Property tests use Hypothesis with minimum 100 iterations
- Each property test references its design document property number
- **Multiprocessing key points:**
  - Service process runs ModelHandler and backend components
  - Controller caches WorldState locally for fast UI reads
  - Pipe communication requires picklable data only
  - Service pushes state after each event/epoch (controller doesn't poll)
