# Implementation Plan: Active Learning Dashboard Migration

## Overview

This implementation plan migrates the Active Learning Dashboard from a JSON file-based worker pattern to an MVC event-driven architecture. The migration follows the design_v2 specification with:

- **Hybrid WorldState/SQLite approach**: In-memory WorldState for fast UI access (~1ms), SQLite for persistence and pagination (~50ms)
- **Event-driven communication**: Pipe-based IPC replaces file-based polling
- **Clean separation of concerns**: View, Controller, Model, Service layers
- **Single-user operation**: Simplified architecture with multi-tab detection
- **Backend preservation**: All existing backend components (trainer.py, active_loop.py, etc.) remain unchanged

## Architecture Summary

```
VIEW (Streamlit) ──dispatch(Event)──► CONTROLLER ──Pipe──► SERVICE
                                          │                   │
                                          ▼                   │
                                    ┌─────────────┐           │
                                    │ WorldState  │ ◄─────────┘
                                    │ (in-memory) │   Events
                                    └──────┬──────┘
                                           │ sync
                                           ▼
                                    ┌─────────────┐
                                    │   SQLite    │
                                    │(persistent) │
                                    └─────────────┘
```

## Task Dependencies

```
Phase 1: Foundation
    │
    ▼
Phase 2: Model Layer (WorldState + SQLite + Schemas)
    │
    ├───────────────────┐
    ▼                   ▼
Phase 3: Service    Phase 4: Controller (needs Model)
    │                   │
    └───────┬───────────┘
            ▼
    Phase 5: View Layer (needs Controller)
            │
            ▼
    Phase 6: Error Handling & Validation
            │
            ▼
    Phase 7: Migration & Compatibility
            │
            ▼
    Phase 8: Performance & Cleanup
```

---

## Tasks

### Phase 1: Foundation

- [x] 1. Create foundation structure and event system
  - [x] 1.1 Create new directory structure
    - Create directories: controller/, model/, services/, backend/
    - Move existing files: trainer.py, active_loop.py, data_manager.py, strategies.py, models.py, dataloader.py → backend/
    - Create __init__.py files for each package
    - _Requirements: None (infrastructure)_
  
  - [x] 1.2 Define Event system in controller/events.py
    - Create EventType enum with all event types (VIEW→CONTROLLER, CONTROLLER→SERVICE, SERVICE→CONTROLLER)
    - Create Event dataclass with type, payload, timestamp, source
    - _Requirements: 2.1, 2.2_

---

### Phase 2: Model Layer

**Dependencies:** Phase 1 complete

- [x] 2. Implement Model Layer

  - [x] 2.1 Create WorldState dataclass in model/world_state.py
    - Define in-memory experiment state structure:
      - Identity: experiment_id, experiment_name
      - Configuration: config dict
      - Phase: phase, error_message
      - Progress: current_cycle, total_cycles, current_epoch, epochs_per_cycle
      - Metrics: current_metrics, epoch_history (current cycle only)
      - Query: queried_images (current batch only)
      - UI flag: pending_updates
    - Add thread-safe Lock for listener thread updates
    - Implement reset() method for new experiments
    - Implement restore_from_db(experiment_data) for session recovery
    - _Requirements: 1.1, 1.3, 1.7, 7.1_
  
  - [x] 2.2 Create DatabaseManager in model/database.py
    - Implement SQLite connection with context manager
    - Create schema initialization (_init_schema method):
      - experiments table (id, name, config_json, phase, current_cycle, timestamps)
      - cycle_summaries table (experiment_id, cycle, metrics, timestamps)
      - epoch_metrics table (experiment_id, cycle, epoch, losses, accuracies)
      - pool_items table (experiment_id, image_path, pool_type, class_idx)
      - queried_images table (experiment_id, cycle, image_path, predictions, annotations)
    - Implement CRUD methods:
      - insert_experiment(), update_experiment_phase(), get_active_experiment()
      - insert_epoch_metrics(), get_epoch_metrics_paginated()
      - insert_cycle_summary(), get_cycle_summaries()
      - insert_pool_items(), get_pool_items_paginated(), get_pool_count()
    - Add indexes for performance (experiment_id, cycle, pool_type)
    - _Requirements: 1.2, 1.5, 7.2, 7.5_
  
  - [x] 2.3 Create data schemas in model/schemas.py
    - Define dataclasses:
      - EpochMetrics (epoch, train_loss, val_loss, train_acc, val_acc, lr)
      - CycleSummary (cycle, labeled_count, unlabeled_count, best_val_acc, test_acc, test_f1)
      - ExperimentConfig (dataset, model_type, strategy, hyperparameters)
      - ValidationResult (is_valid, error_message)
    - Add to_dict() and from_dict() methods for serialization
    - _Requirements: 7.1, 7.2_
  
  - [ ]* 2.4 Write property test for WorldState thread safety
    - Test concurrent updates from multiple threads
    - Verify state consistency after parallel modifications
    - **Property 1: State Update Consistency**
    - **Validates: Requirements 1.3, 2.4, 7.3**
  
  - [ ]* 2.5 Write unit tests for DatabaseManager
    - Test schema creation on fresh database
    - Test CRUD operations for all tables
    - Test pagination with various page sizes
    - Test index performance with bulk data
    - _Requirements: 1.2, 1.5_
  
  - [ ]* 2.6 Write unit tests for data schemas
    - Test serialization round-trip (to_dict → from_dict)
    - Test validation edge cases (empty values, invalid types)
    - _Requirements: 7.1, 7.2_

---

### Phase 3: Service Layer

**Dependencies:** Phase 1 complete, Phase 2.3 (schemas) complete

- [x] 3. Implement Service Layer (ActiveLearningService)

  - [x] 3.1 Create ActiveLearningService in services/al_service.py
    - Create run_active_learning_service(pipe, config) entry point function
    - Implement ActiveLearningService class:
      - __init__(pipe, config): Store pipe, config, initialize flags
      - run(): Main loop with try/except, send SERVICE_READY on start
      - _process_commands(): Poll pipe, handle incoming commands
      - _handle_command(event): Route to appropriate handler
    - Integrate existing backend components:
      - Import and initialize Trainer, ALDataManager, ActiveLearningLoop
      - Store as instance variables for use in training
    - Implement command handlers:
      - CMD_START_CYCLE → _execute_training_cycle()
      - CMD_PAUSE → set _is_paused flag
      - CMD_RESUME → clear _is_paused flag
      - CMD_STOP → set _should_stop flag
      - CMD_ANNOTATIONS → _process_annotations()
      - CMD_SHUTDOWN → graceful exit
    - _Requirements: 3.1, 3.2, 6.1_
  
  - [x] 3.2 Implement training cycle execution
    - _execute_training_cycle() method:
      - Call al_loop.prepare_cycle()
      - Loop through epochs with pause/stop checks
      - Call trainer.train_single_epoch() for each epoch
      - Send EPOCH_COMPLETE event after each epoch with metrics
      - After training: call al_loop.query_samples()
      - Send QUERY_READY event with queried images
    - _process_annotations(payload) method:
      - Extract annotations from payload
      - Call al_loop.receive_annotations()
      - Call al_loop.finalize_cycle()
      - Send CYCLE_COMPLETE event with results
    - _Requirements: 6.2, 6.3_
  
  - [x] 3.3 Add error handling and graceful shutdown
    - Wrap run() in try/except, send SERVICE_ERROR on exception
    - Include traceback in error payload for debugging
    - Implement _shutdown() for cleanup before exit
    - Handle EOFError on pipe (parent died)
    - Handle BrokenPipeError on send (pipe closed)
    - _Requirements: 6.4, 9.1, 9.4_
  
  - [ ]* 3.4 Write property test for service communication
    - Test that all training progress emits events
    - Test event payload contains required fields
    - **Property 6: Service Communication**
    - **Validates: Requirements 2.3, 6.2**
  
  - [ ]* 3.5 Write unit tests for command handling
    - Test each command type triggers correct behavior
    - Test pause/resume flow
    - Test graceful shutdown
    - _Requirements: 6.1, 6.4_

---

### Phase 4: Controller Layer

**Dependencies:** Phase 2 complete, Phase 3 complete

- [x] 4. Checkpoint - Ensure Model and Service tests pass
  - Run all Phase 2 and Phase 3 tests
  - Verify no import errors between components
  - Ask user if questions arise

- [x] 5. Implement Controller Layer

  - [x] 5.1 Create ServiceManager in controller/service_manager.py
    - Implement process lifecycle management:
      - spawn_service(config, event_callback): Create Pipe, spawn Process with daemon=True
      - _terminate_existing_service(): Graceful shutdown sequence (send CMD_SHUTDOWN → join with timeout → terminate → kill)
      - is_alive(): Check process status
      - send_command(event): Send event via pipe
      - shutdown(): Full cleanup
    - Implement listener thread:
      - _start_listener_thread(): Create daemon thread
      - _listen_for_events(): Loop with pipe.poll(), call event_callback
      - Handle EOFError, BrokenPipeError gracefully
    - Add process health monitoring:
      - Detect unexpected process death in listener
      - Emit SERVICE_ERROR event on crash detection
    - _Requirements: 3.1, 3.2, 3.3, 3.4_
  
  - [x] 5.2 Create ModelHandler in controller/model_handler.py
    - __init__(world_state, db_manager): Store references
    - State accessors (from WorldState - fast):
      - get_status() → {phase, current_cycle, total_cycles, error_message}
      - get_training_progress() → {current_epoch, metrics, epoch_history, progress_percentage}
      - get_queried_images() → list of queried images
      - has_pending_updates() → bool
      - clear_pending_updates() → None
    - State mutators:
      - set_phase(phase)
      - set_error(message)
      - set_pending_updates(value)
      - update_current_metrics(metrics)
      - set_queried_images(images)
      - finalize_cycle(results)
    - Database accessors (from SQLite - paginated):
      - get_results_history(page, limit) → paginated experiments
      - get_epoch_history(experiment_id, cycle) → epoch metrics
      - get_pool_page(pool_type, page, limit) → paginated pool items
    - Persistence methods:
      - persist_epoch_metrics(metrics) → write to SQLite
      - initialize_experiment(config) → create in DB, reset WorldState
    - Validation:
      - validate_config(config) → ValidationResult
    - _Requirements: 5.2, 7.4, 7.5_
  
  - [x] 5.3 Create EventDispatcher in controller/dispatcher.py
    - __init__(model_handler, service_manager): Store references, register handlers
    - _register_handlers(): Return dict mapping EventType → handler method
    - dispatch(event): Route to handler, catch exceptions, log errors
    - View event handlers:
      - _handle_initialize(event): Validate config, init experiment, spawn service
      - _handle_start_cycle(event): Set phase, send CMD_START_CYCLE
      - _handle_pause(event): Set phase, send CMD_PAUSE
      - _handle_resume(event): Set phase, send CMD_RESUME
      - _handle_stop(event): Set phase, send CMD_STOP
      - _handle_annotations(event): Save annotations, send CMD_ANNOTATIONS
    - Service event handlers:
      - _handle_service_ready(event): Set phase to IDLE
      - _handle_epoch_complete(event): Update WorldState, persist to SQLite, set pending_updates
      - _handle_cycle_complete(event): Finalize cycle, set phase to IDLE
      - _handle_query_ready(event): Set queried images, set phase to AWAITING_ANNOTATION
      - _handle_service_error(event): Set error state
    - Data access methods (delegate to ModelHandler):
      - get_status(), get_training_progress(), get_queried_images(), etc.
    - Lifecycle:
      - shutdown(): Terminate service, cleanup
    - _Requirements: 2.2, 5.1, 5.3, 5.4_
  
  - [x] 5.4 Create SessionManager in controller/session_manager.py
    - __init__(): Generate unique session_id, set session_file path
    - acquire_session() → bool:
      - Check if session file exists and is recent (<30s heartbeat)
      - If active session exists, return False
      - Write our session with heartbeat timestamp
      - Return True
    - update_heartbeat(): Update timestamp in session file
    - release_session(): Delete session file
    - Integration with Streamlit:
      - Call acquire_session() at dashboard start
      - Show warning if another session active
      - Periodic heartbeat update (can use st.fragment)
    - _Requirements: 10.5, 10.6_
  
  - [x] 5.5 Create controller initialization in controller/__init__.py
    - get_controller() → EventDispatcher:
      - Check st.session_state for existing controller
      - If not exists, call _create_controller()
      - Return controller
    - _create_controller() → EventDispatcher:
      - Initialize WorldState
      - Initialize DatabaseManager with path
      - Check for active experiment, restore if found
      - Create ModelHandler(world_state, db_manager)
      - Create ServiceManager()
      - Create and return EventDispatcher(model_handler, service_manager)
    - get_session_manager() → SessionManager:
      - Similar singleton pattern for session manager
    - reset_controller(): For error recovery, cleanup and recreate
    - _Requirements: 5.4_
  
  - [x] 5.6 Write property test for event dispatch performance
    - Test dispatch completes within 10ms for all event types
    - **Property 2: Event Dispatch Performance**
    - **Validates: Requirements 2.2**
  
  - [ ]* 5.7 Write property test for data formatting
    - Test ModelHandler returns correctly formatted view data
    - **Property 8: Data Formatting**
    - **Validates: Requirements 5.2**
  
  - [ ]* 5.8 Write unit tests for ServiceManager
    - Test process spawn and termination
    - Test graceful vs forced shutdown
    - Test crash detection
    - _Requirements: 3.1, 3.3, 3.4_

---

### Phase 5: View Layer

**Dependencies:** Phase 5 (Controller) complete

- [ ] 6. Implement View Layer refactoring

  - [ ] 6.1 Refactor dashboard.py entry point
    - Import get_controller, get_session_manager
    - Add session check at start:
      - Call session_manager.acquire_session()
      - If False: show error, st.stop()
    - Minimal page setup (title, navigation)
    - Remove any worker management code
    - _Requirements: 3.5, 10.5_
  
  - [ ] 6.2 Refactor Configuration page (pages/1_Configuration.py)
    - Remove all business logic, file operations, direct state access
    - Get controller via get_controller()
    - Request config view: ctrl.get_experiment_config_view()
    - Render form with available options
    - On submit: dispatch(Event(INITIALIZE_EXPERIMENT, payload))
    - Show validation errors from controller response
    - _Requirements: 4.1, 4.2, 4.3, 4.5, 11.1_
  
  - [ ] 6.3 Refactor Active Learning page (pages/2_Active_Learning.py)
    - Remove direct state access and business logic
    - Get controller via get_controller()
    - Display current status: ctrl.get_status()
    - Control buttons dispatch events:
      - "Start Cycle" → dispatch(Event(START_CYCLE))
      - "Pause" → dispatch(Event(PAUSE_TRAINING))
      - "Resume" → dispatch(Event(RESUME_TRAINING))
      - "Stop" → dispatch(Event(STOP_EXPERIMENT))
    - Live training progress with @st.fragment(run_every=2.0):
      - Check ctrl.has_pending_updates()
      - If True: get metrics, render chart, clear flag
      - This is NOT file polling (checks in-memory flag)
    - Annotation interface:
      - Display ctrl.get_queried_images()
      - On confirm: dispatch(Event(SUBMIT_ANNOTATIONS, annotations))
    - _Requirements: 4.1, 4.2, 4.4, 2.6_
  
  - [ ] 6.4 Refactor Results page (pages/3_Results.py)
    - Remove direct state file access
    - Get controller via get_controller()
    - Request paginated history: ctrl.get_results_history(page, limit)
    - Render experiment list with pagination controls
    - Display cycle metrics and charts
    - Add export functionality if needed
    - _Requirements: 4.1, 4.3_
  
  - [ ]* 6.5 Write property test for UI response times
    - Test page load < 2 seconds
    - Test UI update < 500ms
    - Test action feedback < 100ms
    - **Property 3: UI Response Time Performance**
    - **Validates: Requirements 10.1, 10.2, 10.3**

---

### Phase 6: Error Handling & Validation

**Dependencies:** Phase 5 (View) complete

- [ ] 7. Checkpoint - Ensure View layer tests pass
  - Run integration tests for View → Controller → Service flow
  - Verify events dispatch correctly
  - Ask user if questions arise

- [ ] 8. Implement error handling and recovery

  - [ ] 8.1 Add comprehensive error handling to EventDispatcher
    - Catch exceptions in all handlers, log with context
    - On service crash: detect via listener thread, set ERROR phase
    - Implement automatic service restart:
      - Track restart attempts
      - Use exponential backoff (2^attempt seconds)
      - Max 3 retries before giving up
    - On communication failure: retry with backoff
    - Preserve experiment state in SQLite before error transitions
    - _Requirements: 9.1, 9.2, 9.3, 9.4_
  
  - [ ] 8.2 Add error UI components to View layer
    - Create error display component:
      - Show clear error message from WorldState.error_message
      - Explain what went wrong in user-friendly terms
      - Provide recovery suggestions
    - Add "Retry" button → calls ctrl.retry_last_action()
    - Add "Reset Experiment" button → calls reset_controller()
    - For multi-tab warning:
      - Display prominent warning banner
      - Suggest closing other tabs
      - Provide "Check Again" button
    - _Requirements: 9.5, 10.5_
  
  - [ ]* 8.3 Write property test for error handling
    - Test all error categories trigger appropriate response
    - Test recovery mechanisms work correctly
    - **Property 10: Error Handling and Recovery**
    - **Validates: Requirements 5.5, 9.2, 9.3, 9.5**

- [ ] 9. Implement business rule validation

  - [ ] 9.1 Add experiment configuration validation in ModelHandler
    - Validate required fields are present
    - Validate numeric ranges:
      - initial_samples >= 10
      - batch_size >= 1
      - epochs_per_cycle >= 1
      - samples_per_cycle >= 1
    - Validate string enums (strategy, model_type)
    - Return ValidationResult with specific error messages
    - _Requirements: 5.3, 11.1, 11.5_
  
  - [ ] 9.2 Add configuration persistence
    - Save configurations to SQLite for reuse
    - Load previous configurations as templates
    - _Requirements: 11.2, 11.4_
  
  - [ ]* 9.3 Write property test for validation
    - Test all validation rules catch invalid input
    - Test valid configurations pass
    - **Property 9: Business Rule Validation**
    - **Validates: Requirements 5.3, 11.1**

---

### Phase 7: Migration & Compatibility

**Dependencies:** Phase 6 complete

- [ ] 10. Checkpoint - Ensure error handling tests pass
  - Run all error handling tests
  - Test error recovery flows manually
  - Ask user if questions arise

- [ ] 11. Implement backward compatibility and migration

  - [ ] 11.1 Create MigrationManager in model/migration.py
    - LEGACY_STATE_FILE = "experiment_state.json"
    - MIGRATION_MARKER = ".migrated"
    - check_and_migrate(experiment_dir) → bool:
      - Check for .migrated marker (skip if exists)
      - Check for legacy JSON file (skip if not exists)
      - Call _migrate_json_to_sqlite()
      - Create .migrated marker
      - Rename JSON to .json.backup
    - _migrate_json_to_sqlite(json_path) → bool:
      - Load JSON file
      - Insert experiment record to SQLite
      - Migrate cycle_results to cycle_summaries table
      - Migrate epoch history to epoch_metrics table
      - Migrate pool information if present
    - detect_legacy_format(experiment_dir) → bool:
      - Check if JSON exists and no marker
    - _Requirements: 8.1, 8.2, 8.4_
  
  - [ ] 11.2 Integrate migration into controller initialization
    - In _create_controller():
      - Check for legacy experiments in data directory
      - Call MigrationManager.check_and_migrate() for each
      - Log migration results
    - _Requirements: 8.1_
  
  - [ ] 11.3 Ensure model checkpoint compatibility
    - Verify existing .pth checkpoint files work with new architecture
    - Test loading checkpoints in ActiveLearningService
    - Document checkpoint file location expectations
    - _Requirements: 8.5_
  
  - [ ]* 11.4 Write unit tests for migration
    - Test JSON to SQLite conversion with sample data
    - Test migration preserves all data correctly
    - Test migration creates backup file
    - Test migration marker prevents re-migration
    - Test backward compatibility for loading both formats
    - _Requirements: 8.1, 8.2, 8.3_

---

### Phase 8: Performance, Logging & Cleanup

**Dependencies:** Phase 7 complete

- [ ] 12. Implement logging and monitoring

  - [ ] 12.1 Add comprehensive event logging
    - Configure logging at module level in each component
    - Log all event dispatch with: timestamp, event_type, source
    - Log state transitions: phase changes, cycle changes
    - Log errors with: exception type, message, traceback, context
    - _Requirements: 12.1, 12.2_
  
  - [ ] 12.2 Configure log levels and rotation
    - Development: DEBUG level, console output
    - Production: INFO level, file output
    - Implement log rotation (e.g., 10MB max, 5 backup files)
    - Add log configuration in config.py
    - _Requirements: 12.4, 12.5_
  
  - [ ] 12.3 Add performance monitoring
    - Log timing for: event dispatch, database queries, service communication
    - Add optional performance metrics to WorldState for debugging
    - _Requirements: 12.3_
  
  - [ ]* 12.4 Write property test for logging
    - Test all events are logged with required fields
    - Test errors include stack traces
    - **Property 14: Event Logging**
    - **Validates: Requirements 12.1, 12.2**

- [ ] 13. Performance optimization and testing

  - [ ] 13.1 Optimize WorldState access
    - Ensure all WorldState reads are direct attribute access
    - No file I/O in hot paths
    - Profile and verify < 1ms access time
    - _Requirements: 1.4_
  
  - [ ] 13.2 Optimize SQLite queries
    - Ensure indexes are used for common queries
    - Use LIMIT/OFFSET for all paginated queries
    - Profile and verify < 50ms query time
    - _Requirements: 1.5_
  
  - [ ] 13.3 Optimize UI responsiveness
    - Ensure state updates don't block UI rendering
    - Fragment only updates small portion of page
    - Profile page load times
    - _Requirements: 10.4_
  
  - [ ]* 13.4 Write property test for data access performance
    - Test WorldState access < 1ms
    - Test SQLite queries < 50ms
    - **Property 4: Data Access Performance**
    - **Validates: Requirements 1.4, 1.5**
  
  - [ ]* 13.5 Write property test for UI responsiveness
    - Test state updates don't block UI
    - **Property 12: UI Responsiveness**
    - **Validates: Requirements 10.4**

- [ ] 14. Final cleanup and integration

  - [ ] 14.1 Remove deprecated files
    - Delete: state.py (replaced by model/world_state.py + model/database.py)
    - Delete: run_worker.py (replaced by services/al_service.py)
    - Delete: worker_command_loop.py (absorbed into al_service.py)
    - Update .gitignore if needed
    - _Requirements: 1.6, 2.5_
  
  - [ ] 14.2 Update all imports throughout project
    - Update pages/* to use new controller imports
    - Update any remaining references to old modules
    - Run import verification script
    - _Requirements: None (infrastructure)_
  
  - [ ] 14.3 Update documentation
    - Update README.md with new architecture overview
    - Document new file structure
    - Add setup/installation instructions
    - Document configuration options
    - _Requirements: None (documentation)_
  
  - [ ]* 14.4 Write integration tests for complete workflows
    - Test: Initialize → Start → Train → Query → Annotate → Complete cycle
    - Test: Service crash → Detection → Error display → Retry → Recovery
    - Test: Multiple cycles end-to-end
    - Test: Application restart → State restoration from SQLite
    - _Requirements: 8.3, 9.1_

- [ ] 15. Final checkpoint - Complete system validation
  - Run all unit tests
  - Run all property tests
  - Run all integration tests
  - Manual testing of complete workflow
  - Performance benchmarking
  - Ask user if questions arise

---

## Requirement Coverage Matrix

| Requirement | Tasks |
|-------------|-------|
| 1.1 WorldState initialization | 2.1 |
| 1.2 SQLite initialization | 2.2 |
| 1.3 State update consistency | 2.1, 2.4 |
| 1.4 WorldState access < 1ms | 13.1, 13.4 |
| 1.5 SQLite access < 50ms | 2.2, 13.2, 13.4 |
| 1.6 Eliminate FileLock/JSON | 14.1 |
| 1.7 Restore from SQLite | 2.1, 5.5 |
| 2.1 View dispatches events | 1.2, 6.2, 6.3, 6.4 |
| 2.2 Event dispatch < 10ms | 5.3, 5.6 |
| 2.3 Service emits events | 3.2 |
| 2.4 Events update state | 5.3 |
| 2.5 Eliminate file polling | 14.1 |
| 2.6 Lightweight UI refresh | 6.3 |
| 3.1 Auto-launch service | 5.1 |
| 3.2 Pipe-based IPC | 3.1, 5.1 |
| 3.3 Restart on failure | 5.1, 8.1 |
| 3.4 Graceful termination | 5.1 |
| 3.5 No manual worker start | 6.1 |
| 4.1 View only display logic | 6.2, 6.3, 6.4 |
| 4.2 View dispatches events | 6.2, 6.3 |
| 4.3 View requests data | 6.2, 6.4 |
| 4.4 View no direct Model access | 6.3 |
| 4.5 View no file I/O | 6.2 |
| 5.1 Controller coordinates | 5.3 |
| 5.2 Controller formats data | 5.2, 5.7 |
| 5.3 Controller validates | 9.1, 9.3 |
| 5.4 Controller manages lifecycle | 5.3, 5.5 |
| 5.5 Controller handles errors | 8.1 |
| 6.1 Service separate process | 3.1 |
| 6.2 Service emits progress | 3.2 |
| 6.3 Service emits completion | 3.2 |
| 6.4 Service handles interruption | 3.3 |
| 6.5 Service isolation | 3.1 |
| 7.1 WorldState typed structures | 2.1, 2.3 |
| 7.2 SQLite schemas | 2.2, 2.3 |
| 7.3 Data integrity validation | 2.4 |
| 7.4 Current state from WorldState | 5.2 |
| 7.5 Historical data from SQLite | 2.2, 5.2 |
| 8.1 Migrate JSON to SQLite | 11.1, 11.2 |
| 8.2 Support both formats | 11.1, 11.4 |
| 8.3 Identical behavior | 11.4, 14.4 |
| 8.4 Preserve config options | 11.1 |
| 8.5 Checkpoint compatibility | 11.3 |
| 9.1 Auto-restart on crash | 8.1 |
| 9.2 Retry with backoff | 8.1 |
| 9.3 Log errors, continue | 8.1, 12.1 |
| 9.4 Preserve state on error | 3.3, 8.1 |
| 9.5 Clear error messages | 8.2 |
| 10.1 Page load < 2s | 6.5 |
| 10.2 UI update < 500ms | 6.5 |
| 10.3 Action feedback < 100ms | 6.5 |
| 10.4 Non-blocking updates | 13.3, 13.5 |
| 10.5 Multi-tab warning | 5.4, 8.2 |
| 10.6 Single-user optimized | 5.4 |
| 11.1 Validate before start | 9.1 |
| 11.2 Persist configurations | 9.2 |
| 11.4 Configuration templates | 9.2 |
| 11.5 Validation messages | 9.1, 9.3 |
| 12.1 Log events with timestamps | 12.1 |
| 12.2 Log errors with traces | 12.1 |
| 12.3 Log performance metrics | 12.3 |
| 12.4 Log levels | 12.2 |
| 12.5 Log rotation | 12.2 |

---

## Notes

- Tasks marked with `*` are optional property/unit tests that can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints (Tasks 4, 7, 10, 15) ensure incremental validation
- Property tests validate universal correctness properties from design document
- Unit tests validate specific examples and edge cases
- All existing backend components (trainer.py, active_loop.py, data_manager.py, strategies.py, models.py, dataloader.py) remain unchanged
- The migration follows the phased approach outlined in design_v2

## Estimated Timeline

| Phase | Tasks | Estimated Duration |
|-------|-------|-------------------|
| Phase 1: Foundation | 1 | 0.5 day |
| Phase 2: Model Layer | 2 | 1.5 days |
| Phase 3: Service Layer | 3 | 1.5 days |
| Phase 4: Controller Layer | 4-5 | 2 days |
| Phase 5: View Layer | 6-7 | 1.5 days |
| Phase 6: Error Handling | 8-9 | 1 day |
| Phase 7: Migration | 10-11 | 1 day |
| Phase 8: Performance & Cleanup | 12-15 | 1.5 days |
| **Total** | | **~10-11 days** |

*Note: Timeline assumes focused development. Add buffer for testing, debugging, and unexpected issues.*
