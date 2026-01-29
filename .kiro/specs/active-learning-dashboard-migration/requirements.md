# Requirements Document

## Introduction

This document specifies the requirements for migrating an Active Learning Dashboard from a JSON file-based worker pattern to an MVC event-driven architecture. The migration aims to eliminate performance bottlenecks, improve maintainability, and establish clear separation of concerns while preserving all existing functionality.

## Glossary

- **Dashboard**: The Streamlit-based web interface for active learning experiments
- **Worker_Pattern**: Current architecture using separate processes with file-based communication
- **MVC_Architecture**: Target Model-View-Controller event-driven architecture
- **WorldState**: Hybrid in-memory/SQLite data structure containing current experiment state
- **EventDispatcher**: Component that routes events between layers
- **ActiveLearningService**: Service layer component managing ML training processes
- **ViewLayer**: Streamlit pages containing only UI logic
- **ControllerLayer**: Components handling event routing and business logic coordination
- **ModelLayer**: Data structures and state management components
- **ServiceLayer**: Background processing and external system integration

## Requirements

### Requirement 1: State Management Migration

**User Story:** As a system architect, I want to migrate from JSON file-based state to a hybrid WorldState/SQLite architecture, so that the dashboard has both fast access and reliable persistence.

#### Acceptance Criteria

1. WHEN the system starts, THE WorldState SHALL initialize current experiment state in-memory
2. WHEN the system starts, THE SQLite database SHALL be initialized with schema for persistent storage
3. WHEN state changes occur, THE Controller SHALL update both WorldState (immediate) and SQLite (persistent)
4. WHEN UI components request current state, THE System SHALL provide it from WorldState within 1ms
5. WHEN UI components request paginated/historical data, THE System SHALL query SQLite within 50ms
6. THE System SHALL eliminate all FileLock usage and JSON file dependencies
7. WHEN the application restarts, THE System SHALL restore WorldState from SQLite

### Requirement 2: Event-Driven Communication

**User Story:** As a developer, I want event-driven communication between components, so that the system is responsive and eliminates file-based polling.

#### Acceptance Criteria

1. WHEN a user action occurs, THE ViewLayer SHALL dispatch an event to the ControllerLayer
2. WHEN an event is dispatched, THE EventDispatcher SHALL route it to the appropriate handler within 10ms
3. WHEN background processes complete tasks, THE ServiceLayer SHALL emit events via Pipe to update state
4. WHEN events are processed, THE System SHALL update WorldState and SQLite appropriately
5. THE System SHALL eliminate all file-based polling mechanisms
6. THE System MAY use lightweight periodic UI refresh (≤2s interval) for live training visualization, using in-memory flag checks only

### Requirement 3: Process Architecture Migration

**User Story:** As a system administrator, I want the worker process to be managed as a service, so that process lifecycle is automated and communication is reliable.

#### Acceptance Criteria

1. WHEN the dashboard starts, THE ServiceManager SHALL automatically launch the ActiveLearningService process
2. WHEN communication is needed, THE System SHALL use Pipe-based IPC instead of file-based commands
3. WHEN the service process fails, THE ServiceManager SHALL detect failure and restart the process
4. WHEN the dashboard shuts down, THE ServiceManager SHALL gracefully terminate all service processes
5. THE User SHALL never need to manually start or manage worker processes

### Requirement 4: View Layer Refactoring

**User Story:** As a UI developer, I want Streamlit pages to contain only presentation logic, so that the UI is maintainable and testable.

#### Acceptance Criteria

1. WHEN rendering UI components, THE ViewLayer SHALL only handle display logic and user input collection
2. WHEN user interactions occur, THE ViewLayer SHALL dispatch events without executing business logic
3. WHEN displaying data, THE ViewLayer SHALL request formatted data from the ControllerLayer
4. THE ViewLayer SHALL NOT directly access the ModelLayer or ServiceLayer
5. THE ViewLayer SHALL NOT contain any file I/O, computation, or state management logic

### Requirement 5: Controller Layer Implementation

**User Story:** As a system architect, I want a dedicated controller layer, so that business logic is centralized and components are loosely coupled.

#### Acceptance Criteria

1. WHEN events are received, THE ControllerLayer SHALL coordinate between ModelLayer and ServiceLayer
2. WHEN UI requests data, THE ControllerLayer SHALL format and provide view-appropriate data structures
3. WHEN business rules need enforcement, THE ControllerLayer SHALL validate and enforce them
4. THE ControllerLayer SHALL manage the lifecycle of active learning experiments
5. THE ControllerLayer SHALL handle error conditions and provide appropriate user feedback

### Requirement 6: Service Layer Architecture

**User Story:** As a developer, I want background processing isolated in a service layer, so that heavy computations don't block the UI.

#### Acceptance Criteria

1. WHEN training is initiated, THE ActiveLearningService SHALL execute in a separate process
2. WHEN training progresses, THE ActiveLearningService SHALL emit progress events via Pipe communication
3. WHEN training completes, THE ActiveLearningService SHALL emit completion events with results
4. THE ActiveLearningService SHALL handle training interruption and resumption gracefully
5. THE ActiveLearningService SHALL maintain isolation from UI components

### Requirement 7: Data Structure Migration

**User Story:** As a developer, I want well-defined data structures, so that state management is type-safe and predictable.

#### Acceptance Criteria

1. THE WorldState SHALL define current experiment state using typed data structures in-memory
2. THE SQLite database SHALL define persistent schemas for EpochMetrics, CycleSummary, and ExperimentConfig tables
3. WHEN state updates occur, THE System SHALL validate data integrity using schema validation and SQLite constraints
4. WHEN components access current state, THE System SHALL provide it from WorldState for immediate access
5. WHEN components access historical data, THE System SHALL query SQLite with appropriate indexing
6. THE System SHALL maintain backward compatibility with existing data formats during migration

### Requirement 8: Backward Compatibility

**User Story:** As a user, I want existing experiments to continue working, so that migration doesn't disrupt ongoing research.

#### Acceptance Criteria

1. WHEN migrating existing JSON state files, THE System SHALL convert them to the new format
2. WHEN loading historical experiment data, THE System SHALL support both old and new formats
3. WHEN users access existing functionality, THE System SHALL provide identical behavior
4. THE System SHALL preserve all existing configuration options and experiment parameters
5. THE System SHALL maintain compatibility with existing model checkpoints and data files

### Requirement 9: Error Handling and Recovery

**User Story:** As a user, I want robust error handling, so that system failures don't cause data loss or require manual intervention.

#### Acceptance Criteria

1. WHEN service processes crash, THE System SHALL automatically restart them and restore state
2. WHEN communication failures occur, THE System SHALL retry operations with exponential backoff
3. WHEN invalid events are received, THE System SHALL log errors and continue operation
4. WHEN critical errors occur, THE System SHALL preserve experiment state and notify the user
5. THE System SHALL provide clear error messages and recovery suggestions to users

### Requirement 10: Performance Requirements

**User Story:** As a user, I want fast dashboard response times, so that I can efficiently monitor and control experiments.

#### Acceptance Criteria

1. WHEN loading dashboard pages, THE System SHALL render within 2 seconds
2. WHEN updating UI components, THE System SHALL reflect changes within 500ms
3. WHEN processing user actions, THE System SHALL provide feedback within 100ms
4. THE System SHALL handle state updates without blocking the UI
5. WHEN multiple browser tabs are opened, THE System SHALL warn the user and recommend single-tab usage
6. THE System SHALL be optimized for single-user operation

### Requirement 11: Configuration Management

**User Story:** As a researcher, I want flexible experiment configuration, so that I can easily set up different active learning scenarios.

#### Acceptance Criteria

1. WHEN configuring experiments, THE System SHALL validate all parameters before starting
2. WHEN saving configurations, THE System SHALL persist them for reuse in future experiments
3. WHEN loading configurations, THE System SHALL restore all experiment parameters correctly
4. THE System SHALL support configuration templates for common experiment types
5. THE System SHALL provide clear validation messages for invalid configuration values

### Requirement 12: Monitoring and Logging

**User Story:** As a developer, I want comprehensive logging, so that I can debug issues and monitor system health.

#### Acceptance Criteria

1. WHEN events are processed, THE System SHALL log event details with timestamps
2. WHEN errors occur, THE System SHALL log stack traces and context information
3. WHEN performance issues arise, THE System SHALL log timing and resource usage metrics
4. THE System SHALL provide different log levels for development and production use
5. THE System SHALL rotate log files to prevent disk space issues