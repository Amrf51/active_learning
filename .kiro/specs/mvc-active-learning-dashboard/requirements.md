# Requirements Document

## Introduction

This document defines the requirements for the MVC Architecture implementation of an Interactive Active Learning Dashboard for Vehicle Classification. The system enables users to configure, run, and analyze active learning experiments using a Streamlit-based interface with PyTorch backend.

## Glossary

- **WorldState**: In-memory dataclass holding the current state of the application for fast UI reads
- **Event**: A command object representing a user action dispatched to the controller
- **ExperimentController**: The controller component that routes events and manages background threads
- **ModelHandler**: The model layer orchestrator that processes events and updates WorldState
- **ExperimentManager**: The persistence layer handling SQLite database and file artifacts
- **Phase**: An enumeration representing the current state of an experiment (IDLE, TRAINING, QUERYING, etc.)
- **Active_Learning_Loop**: The existing backend component that orchestrates training and querying cycles
- **Cycle**: One iteration of the active learning process (train → query → annotate)
- **Epoch**: One complete pass through the training data during model training

## Requirements

### Requirement 1: Experiment Creation and Configuration

**User Story:** As a researcher, I want to create and configure active learning experiments, so that I can set up training runs with specific parameters.

#### Acceptance Criteria

1. WHEN a user provides experiment configuration (dataset path, model, strategy, parameters) THEN THE ExperimentController SHALL create a new experiment with a unique ID
2. WHEN an experiment is created THEN THE ExperimentManager SHALL persist the configuration to SQLite and create the experiment folder structure
3. WHEN an experiment is created THEN THE experiment folder SHALL be named using the user-provided experiment name (e.g., "mvc_test2") instead of the auto-generated UUID
4. WHEN a user selects a dataset path THEN THE Configuration_Page SHALL scan and display available images
5. THE Configuration_Page SHALL provide dropdown selection for model architecture (ResNet-18, ResNet-50, MobileNetV2)
6. THE Configuration_Page SHALL provide dropdown selection for AL strategy (Random, Uncertainty, Entropy, Margin)
7. WHEN experiment creation fails THEN THE WorldState SHALL contain an error message describing the failure

### Requirement 2: Experiment Loading and Management

**User Story:** As a researcher, I want to load and manage existing experiments, so that I can resume work or review past experiments.

#### Acceptance Criteria

1. WHEN a user requests the experiment list THEN THE ExperimentManager SHALL return all experiments from SQLite
2. WHEN a user loads an experiment THEN THE ModelHandler SHALL restore WorldState from persisted data
3. WHEN a user deletes an experiment THEN THE ExperimentManager SHALL remove the database entry and experiment folder
4. THE Results_Page SHALL provide an experiment selector dropdown for viewing past experiments

### Requirement 3: Training Cycle Execution

**User Story:** As a researcher, I want to run training cycles with live progress updates, so that I can monitor the active learning process.

#### Acceptance Criteria

1. WHEN a user starts a training cycle THEN THE ExperimentController SHALL spawn a background thread for training
2. WHILE training is in progress THEN THE WorldState SHALL be updated after each epoch with current metrics
3. WHEN an epoch completes THEN THE WorldState.epoch_metrics list SHALL contain the new EpochMetrics
4. WHEN training completes THEN THE ModelHandler SHALL transition to the QUERYING phase
5. WHEN querying completes THEN THE WorldState.queried_images SHALL contain the samples selected by the AL strategy
6. THE Active_Learning_Page SHALL display live training metrics (loss/accuracy curves) by polling WorldState

### Requirement 4: Annotation Submission

**User Story:** As a researcher, I want to annotate queried samples and submit them, so that the active learning loop can continue.

#### Acceptance Criteria

1. WHEN a user submits annotations THEN THE ModelHandler SHALL pass them to the Active_Learning_Loop
2. WHEN annotations are submitted THEN THE ExperimentManager SHALL save the cycle results to SQLite
3. WHEN annotations are submitted THEN THE WorldState.current_cycle SHALL increment
4. WHEN all cycles are completed THEN THE WorldState.phase SHALL transition to COMPLETED
5. THE Active_Learning_Page SHALL display queried images in a grid with annotation interface

### Requirement 5: Experiment Pause and Stop

**User Story:** As a researcher, I want to pause or stop training, so that I can control experiment execution.

#### Acceptance Criteria

1. WHEN a user pauses training THEN THE ExperimentController SHALL suspend the background thread
2. WHEN a user stops training THEN THE ExperimentController SHALL terminate the background thread and save current state
3. WHEN training is paused or stopped THEN THE WorldState.phase SHALL reflect the current state

### Requirement 6: Results Visualization and Export

**User Story:** As a researcher, I want to view and export experiment results, so that I can analyze and share findings.

#### Acceptance Criteria

1. WHEN a user views results THEN THE Results_Page SHALL display performance metrics over cycles
2. WHEN a user selects multiple experiments THEN THE Results_Page SHALL display comparison charts
3. WHEN a user requests export THEN THE Results_Page SHALL generate CSV or JSON files with cycle metrics
4. THE Results_Page SHALL display confusion matrix heatmaps for each cycle
5. WHEN cycle results are queried THEN THE ExperimentManager SHALL return data with pagination support

### Requirement 7: WorldState Management

**User Story:** As a system component, I want fast in-memory state access, so that the UI can update without I/O delays.

#### Acceptance Criteria

1. THE WorldState SHALL be a dataclass containing experiment identity, phase, progress, pool sizes, and metrics
2. WHEN the UI polls for state THEN THE ExperimentController.get_state() SHALL return WorldState without database queries
3. THE WorldState SHALL contain queried_images list for the annotation interface
4. THE WorldState SHALL contain probe_images list for prediction tracking across cycles
5. IF an error occurs THEN THE WorldState.error_message SHALL contain a description of the error

### Requirement 8: Event System

**User Story:** As a system architect, I want all user actions routed through events, so that the UI is decoupled from the backend.

#### Acceptance Criteria

1. THE Event dataclass SHALL contain a type (EventType enum) and payload dictionary
2. WHEN a view dispatches an event THEN THE ExperimentController SHALL route it to the ModelHandler
3. THE EventType enum SHALL include: CREATE_EXPERIMENT, LOAD_EXPERIMENT, START_CYCLE, PAUSE, STOP, SUBMIT_ANNOTATIONS
4. Views SHALL NOT call backend services directly; all actions go through controller.dispatch()

### Requirement 9: Persistence Layer

**User Story:** As a system component, I want reliable data persistence, so that experiments survive application restarts.

#### Acceptance Criteria

1. THE ExperimentManager SHALL use SQLite for metadata and cycle results
2. THE ExperimentManager SHALL use folder-based storage for large artifacts (checkpoints, confusion matrices)
3. WHEN a checkpoint is saved THEN THE ExperimentManager SHALL write it to experiments/{experiment_name}/checkpoints/cycle_{n}.pth
4. WHEN a confusion matrix is saved THEN THE ExperimentManager SHALL write it to experiments/{experiment_name}/results/confusion_matrix_{n}.npy
5. THE SQLite schema SHALL include tables for experiments, cycle_results, and epoch_metrics

### Requirement 10: Scalability for Large Datasets

**User Story:** As a researcher, I want the system to handle large datasets (16k+ images), so that I can run experiments at scale.

#### Acceptance Criteria

1. THE system SHALL use lazy loading for images (loaded only when displayed)
2. THE ALDataManager SHALL use index-based pools without copying image data
3. THE Trainer SHALL process images in configurable batches
4. WHEN querying cycle results THEN THE ExperimentManager SHALL support LIMIT/OFFSET pagination
5. THE system SHALL cache only the current cycle's query images to disk

### Requirement 11: Dataset Explorer

**User Story:** As a researcher, I want to inspect the labeled and unlabeled data pools, so that I can understand the current state of my dataset and verify annotations.

#### Acceptance Criteria

1. THE Dataset_Explorer_Page SHALL display images from the labeled pool with their assigned labels
2. THE Dataset_Explorer_Page SHALL display images from the unlabeled pool
3. THE Dataset_Explorer_Page SHALL show pool statistics (counts, class distribution)
4. WHEN viewing large pools THEN THE Dataset_Explorer_Page SHALL use pagination with lazy loading
5. THE Dataset_Explorer_Page SHALL provide filtering by class name or filename
