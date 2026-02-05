# MVC Architecture Design - Active Learning Dashboard

## Overview

A practical MVC architecture for the Streamlit-based Active Learning dashboard. This design prioritizes simplicity while ensuring scalability from 400 to 16,000+ images.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           VIEW LAYER (Streamlit)                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │  Configuration  │  │ Active Learning │  │    Results      │          │
│  │     Page        │  │     Page        │  │     Page        │          │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘          │
│           │ reads              │ dispatches         │ reads             │
│           ▼                    ▼                    ▼                   │
│                    ┌───────────────────────┐                            │
│                    │   st.session_state    │                            │
│                    │   (WorldState ref)    │                            │
│                    └───────────┬───────────┘                            │
└────────────────────────────────┼────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────┐
│                         CONTROLLER LAYER                                 │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    ExperimentController                          │    │
│  │  • dispatch(event: Event) → routes to ModelHandler               │    │
│  │  • get_state() → returns WorldState (fast read)                  │    │
│  │  • start_background_task() → spawns training thread              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                 │                                        │
│  ┌──────────────────────────────┴──────────────────────────────────┐    │
│  │  Events: START_CYCLE | PAUSE | STOP | SUBMIT_ANNOTATIONS        │    │
│  │          CREATE_EXPERIMENT | LOAD_EXPERIMENT                     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────────┐
│                            MODEL LAYER                                   │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐     │
│  │   WorldState   │  │  ModelHandler  │  │   ExperimentManager    │     │
│  │  (In-Memory)   │  │ (Orchestrator) │  │  (Folder + SQLite)     │     │
│  └────────────────┘  └────────────────┘  └────────────────────────┘     │
│                                 │                                        │
│  ┌──────────────────────────────┴──────────────────────────────────┐    │
│  │                    Backend Services (existing)                   │    │
│  │  ActiveLearningLoop | Trainer | ALDataManager | Strategies       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. WorldState (In-Memory State)

Single source of truth for UI. Fast reads, no I/O.

```python
@dataclass
class WorldState:
    """In-memory state for fast UI reads."""
    # Identity
    experiment_id: Optional[str] = None
    experiment_name: Optional[str] = None
    
    # Phase
    phase: Phase = Phase.IDLE  # IDLE|TRAINING|QUERYING|AWAITING_ANNOTATION|COMPLETED
    
    # Progress
    current_cycle: int = 0
    total_cycles: int = 0
    current_epoch: int = 0
    epochs_per_cycle: int = 0
    
    # Pool sizes
    labeled_count: int = 0
    unlabeled_count: int = 0
    
    # Live metrics (updated during training)
    epoch_metrics: List[EpochMetrics] = field(default_factory=list)
    
    # Queried images for annotation
    queried_images: List[QueriedImage] = field(default_factory=list)
    
    # Probe images for prediction tracking
    probe_images: List[ProbeImage] = field(default_factory=list)
    
    # Error
    error_message: Optional[str] = None
```

### 2. Event System (Command Pattern)

All user actions go through events. Views never call backend directly.

```python
class EventType(Enum):
    CREATE_EXPERIMENT = "create_experiment"
    LOAD_EXPERIMENT = "load_experiment"
    START_CYCLE = "start_cycle"
    PAUSE = "pause"
    STOP = "stop"
    SUBMIT_ANNOTATIONS = "submit_annotations"

@dataclass
class Event:
    type: EventType
    payload: Dict[str, Any] = field(default_factory=dict)
```

### 3. ExperimentController

Routes events to ModelHandler. Manages background training thread.

```python
class ExperimentController:
    def __init__(self, experiments_dir: Path):
        self._model_handler = ModelHandler(experiments_dir)
        self._training_thread: Optional[Thread] = None
    
    def dispatch(self, event: Event) -> None:
        """Route event to model handler."""
        self._model_handler.handle_event(event)
    
    def get_state(self) -> WorldState:
        """Fast in-memory state read."""
        return self._model_handler.world_state
    
    def start_training_async(self) -> None:
        """Start training in background thread."""
        self._training_thread = Thread(target=self._run_training)
        self._training_thread.start()
```

### 4. ModelHandler

Orchestrates backend operations. Updates WorldState after each step.

```python
class ModelHandler:
    def __init__(self, experiments_dir: Path):
        self.experiments_dir = experiments_dir
        self.world_state = WorldState()
        self.exp_manager = ExperimentManager(experiments_dir)
        self._active_loop: Optional[ActiveLearningLoop] = None
    
    def handle_event(self, event: Event) -> None:
        """Process event and update state."""
        if event.type == EventType.CREATE_EXPERIMENT:
            self._create_experiment(event.payload)
        elif event.type == EventType.START_CYCLE:
            self._start_cycle()
        # ... etc
    
    def train_epoch(self) -> EpochMetrics:
        """Train one epoch, update WorldState."""
        metrics = self._active_loop.train_single_epoch(self.world_state.current_epoch)
        self.world_state.epoch_metrics.append(metrics)
        self.world_state.current_epoch += 1
        return metrics
```

### 5. ExperimentManager

Handles persistence: SQLite for metadata, folders for artifacts.

```python
class ExperimentManager:
    def __init__(self, experiments_dir: Path):
        self.experiments_dir = experiments_dir
        self.db_path = experiments_dir / "experiments.db"
        self._init_db()
    
    # Experiment CRUD
    def create_experiment(self, config: dict) -> str: ...
    def list_experiments(self) -> List[dict]: ...
    def load_experiment(self, exp_id: str) -> dict: ...
    def delete_experiment(self, exp_id: str) -> bool: ...
    
    # Cycle results
    def save_cycle_result(self, exp_id: str, cycle: CycleMetrics) -> None: ...
    def get_cycle_results(self, exp_id: str) -> List[CycleMetrics]: ...
    
    # Artifacts (files)
    def save_checkpoint(self, exp_id: str, cycle: int, state_dict: dict) -> Path: ...
    def save_confusion_matrix(self, exp_id: str, cycle: int, cm: np.ndarray) -> Path: ...
```

## Folder Structure

```
project_root/
├── dashboard.py                    # Entry point
├── pages/
│   ├── 1_Configuration.py          # Dataset, model, strategy selection
│   ├── 2_Active_Learning.py        # Training control, annotation UI
│   └── 3_Results.py                # Metrics, comparison, export
├── controller/
│   ├── __init__.py
│   ├── experiment_controller.py    # Event routing, thread management
│   └── events.py                   # EventType enum, Event dataclass
├── model/
│   ├── __init__.py
│   ├── world_state.py              # WorldState dataclass
│   ├── model_handler.py            # Backend orchestration
│   └── experiment_manager.py       # SQLite + folder persistence
├── backend/                        # Existing (unchanged)
│   ├── active_loop.py
│   ├── data_manager.py
│   ├── dataloader.py
│   ├── models.py
│   ├── state.py
│   ├── strategies.py
│   └── trainer.py
└── experiments/                    # Experiment storage
    ├── experiments.db              # SQLite: metadata, cycle results
    └── {experiment_id}/            # Per-experiment folder
        ├── config.yaml
        ├── checkpoints/
        │   └── cycle_{n}.pth
        ├── queries/
        │   └── cycle_{n}/          # Cached query images
        └── results/
            └── confusion_matrix_{n}.npy
```

## Database Schema (SQLite)

```sql
-- Experiments table
CREATE TABLE experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    config JSON NOT NULL,
    status TEXT DEFAULT 'created',  -- created|running|completed|error
    dataset_path TEXT,
    model_name TEXT,
    strategy TEXT
);

-- Cycle results table
CREATE TABLE cycle_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT REFERENCES experiments(id),
    cycle INTEGER NOT NULL,
    labeled_count INTEGER,
    unlabeled_count INTEGER,
    epochs_trained INTEGER,
    best_val_accuracy REAL,
    test_accuracy REAL,
    test_f1 REAL,
    test_precision REAL,
    test_recall REAL,
    per_class_metrics JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(experiment_id, cycle)
);

-- Epoch metrics table (for live training visualization)
CREATE TABLE epoch_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT REFERENCES experiments(id),
    cycle INTEGER NOT NULL,
    epoch INTEGER NOT NULL,
    train_loss REAL,
    train_accuracy REAL,
    val_loss REAL,
    val_accuracy REAL,
    UNIQUE(experiment_id, cycle, epoch)
);
```

## Data Flow Examples

### Starting a Training Cycle

```
1. User clicks "Start Cycle"
2. View: controller.dispatch(Event(START_CYCLE))
3. Controller: model_handler.handle_event(event)
4. ModelHandler:
   a. world_state.phase = Phase.TRAINING
   b. active_loop.prepare_cycle(cycle_num)
   c. For each epoch:
      - metrics = active_loop.train_single_epoch(epoch)
      - world_state.epoch_metrics.append(metrics)
      - world_state.current_epoch = epoch
   d. world_state.phase = Phase.QUERYING
   e. queried = active_loop.query_samples()
   f. world_state.queried_images = queried
   g. world_state.phase = Phase.AWAITING_ANNOTATION
5. View: Polls controller.get_state(), updates display
```

### Submitting Annotations

```
1. User clicks "Submit Annotations"
2. View: controller.dispatch(Event(SUBMIT_ANNOTATIONS, {annotations: [...]}))
3. Controller: model_handler.handle_event(event)
4. ModelHandler:
   a. active_loop.receive_annotations(annotations)
   b. exp_manager.save_cycle_result(exp_id, cycle_metrics)
   c. world_state.current_cycle += 1
   d. world_state.phase = Phase.IDLE (or COMPLETED if last cycle)
5. View: Shows updated pool sizes, ready for next cycle
```

## Scalability Considerations

### For Large Datasets (16k+ images)

1. **Lazy Loading**: Images loaded only when displayed
2. **Index-Based Pools**: ALDataManager uses indices, no data copying
3. **Batch Predictions**: Trainer processes in batches
4. **SQLite Pagination**: Query results with LIMIT/OFFSET
5. **Cached Queries**: Only current cycle's query images cached to disk

### Memory Management

- WorldState holds only essential data (counts, current metrics)
- Large arrays (confusion matrices) saved to disk immediately
- Query image cache cleared after each cycle

## Threading Strategy

PyTorch releases GIL during tensor operations, allowing concurrent UI updates.

```python
class ExperimentController:
    def start_training_async(self):
        """Run training in background thread."""
        def training_loop():
            while self._model_handler.world_state.phase == Phase.TRAINING:
                self._model_handler.train_epoch()
                # WorldState updated, UI can poll
        
        self._thread = Thread(target=training_loop, daemon=True)
        self._thread.start()
```

UI polls `controller.get_state()` on each Streamlit rerun to show live progress.

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Command Pattern (Events) | Decouples UI from backend, enables logging/undo |
| In-Memory WorldState | Fast UI reads without I/O |
| SQLite for metadata | Scalable, queryable, ACID-compliant |
| Folder-based artifacts | Large files (checkpoints, matrices) stay out of DB |
| Threading (not multiprocessing) | Simpler, PyTorch releases GIL |
| Existing backend unchanged | Reuse proven ActiveLearningLoop, Trainer, etc. |

## Phase Enum

```python
class Phase(Enum):
    IDLE = "idle"                           # Ready to start
    INITIALIZING = "initializing"           # Setting up experiment
    TRAINING = "training"                   # Training in progress
    QUERYING = "querying"                   # Running AL strategy
    AWAITING_ANNOTATION = "awaiting_annotation"  # Waiting for user
    COMPLETED = "completed"                 # All cycles done
    ERROR = "error"                         # Something went wrong
```

## View Responsibilities

### Configuration Page
- Dataset path selection + scanning
- Model architecture dropdown (ResNet-18, ResNet-50, MobileNetV2)
- Strategy dropdown (Random, Uncertainty, Entropy, Margin)
- Training parameters (epochs, batch size, learning rate)
- Create experiment button → dispatches CREATE_EXPERIMENT

### Active Learning Page
- Cycle progress display (current/total)
- Live training metrics (loss/accuracy curves)
- Control buttons: Start, Pause, Stop
- Query image grid with annotation interface
- Probe image predictions across cycles
- Submit annotations button → dispatches SUBMIT_ANNOTATIONS

### Results Page
- Experiment selector dropdown
- Performance over cycles chart
- Cycle metrics table
- Multi-experiment comparison
- Confusion matrix heatmap
- Export buttons (CSV, JSON)

## Migration from Current Code

1. **Create MVC folders**: `controller/`, `model/`
2. **Implement WorldState**: Simple dataclass
3. **Implement Events**: Enum + dataclass
4. **Implement ExperimentManager**: SQLite wrapper
5. **Implement ModelHandler**: Wraps existing backend
6. **Implement Controller**: Event routing
7. **Update Views**: Use controller.dispatch() and controller.get_state()

The existing `backend/` code (ActiveLearningLoop, Trainer, ALDataManager, strategies) remains unchanged.


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Experiment ID Uniqueness

*For any* set of experiment configurations, when experiments are created, each experiment SHALL have a unique ID that does not collide with any existing experiment ID.

**Validates: Requirements 1.1**

### Property 2: Experiment Persistence Round-Trip

*For any* valid experiment configuration, creating an experiment and then loading it by ID SHALL return an equivalent configuration.

**Validates: Requirements 1.2, 2.2**

### Property 3: Experiment Listing Completeness

*For any* set of created experiments, calling list_experiments() SHALL return all created experiments (no experiments are lost).

**Validates: Requirements 2.1**

### Property 4: Experiment Deletion Removes from List

*For any* existing experiment, after deletion, the experiment SHALL NOT appear in the experiment list.

**Validates: Requirements 2.3**

### Property 5: Epoch Tracking Consistency

*For any* training run, the length of WorldState.epoch_metrics SHALL equal WorldState.current_epoch after each epoch completes.

**Validates: Requirements 3.2, 3.3**

### Property 6: Phase Transition After Training

*For any* completed training phase, the WorldState.phase SHALL transition to QUERYING.

**Validates: Requirements 3.4**

### Property 7: Queried Images Population

*For any* completed querying phase, WorldState.queried_images SHALL be non-empty (assuming unlabeled pool is non-empty).

**Validates: Requirements 3.5**

### Property 8: Cycle Results Persistence Round-Trip

*For any* submitted annotations, saving cycle results and then querying them SHALL return equivalent metrics.

**Validates: Requirements 4.2**

### Property 9: Cycle Counter Increment

*For any* annotation submission, WorldState.current_cycle SHALL increment by exactly 1.

**Validates: Requirements 4.3**

### Property 10: Completion Phase Transition

*For any* experiment where current_cycle equals total_cycles after annotation submission, WorldState.phase SHALL be COMPLETED.

**Validates: Requirements 4.4**

### Property 11: Phase Reflects Control Actions

*For any* pause or stop action, WorldState.phase SHALL reflect the corresponding state (not TRAINING).

**Validates: Requirements 5.3**

### Property 12: Export Data Round-Trip

*For any* cycle metrics, exporting to CSV/JSON and parsing back SHALL produce equivalent data.

**Validates: Requirements 6.3**

### Property 13: Pagination Correctness

*For any* set of cycle results and pagination parameters (limit, offset), the returned subset SHALL match the expected slice of the full result set.

**Validates: Requirements 6.5, 10.4**

### Property 14: Checkpoint Persistence Round-Trip

*For any* saved checkpoint, loading from the expected path SHALL return the original state dictionary.

**Validates: Requirements 9.3**

### Property 15: Confusion Matrix Persistence Round-Trip

*For any* saved confusion matrix, loading from the expected path SHALL return an equivalent numpy array.

**Validates: Requirements 9.4**

## Error Handling

### Experiment Creation Errors
- Invalid dataset path → WorldState.error_message set, phase remains IDLE
- Missing required configuration → WorldState.error_message set, experiment not created
- Database write failure → WorldState.error_message set, rollback any partial state

### Training Errors
- Model initialization failure → WorldState.phase = ERROR, error_message describes issue
- Out of memory during training → Graceful degradation, save checkpoint, set error state
- Backend exception → Catch, log, set WorldState.error_message, transition to ERROR phase

### Persistence Errors
- SQLite connection failure → Retry with exponential backoff, then error state
- Disk full during checkpoint save → Log warning, continue without checkpoint
- Corrupted experiment data → Skip corrupted entry, log warning, continue with valid data

### Recovery Strategy
- All errors populate WorldState.error_message for UI display
- Partial state is saved when possible before error transition
- User can retry failed operations after addressing the issue

## Testing Strategy

### Dual Testing Approach

This project uses both unit tests and property-based tests for comprehensive coverage:

- **Unit tests**: Verify specific examples, edge cases, and error conditions
- **Property tests**: Verify universal properties across randomly generated inputs

### Property-Based Testing Configuration

- **Library**: Hypothesis (Python)
- **Minimum iterations**: 100 per property test
- **Tag format**: `# Feature: mvc-active-learning-dashboard, Property {N}: {property_text}`

### Test Organization

```
tests/
├── unit/
│   ├── test_world_state.py         # WorldState dataclass tests
│   ├── test_events.py              # Event creation and validation
│   ├── test_experiment_manager.py  # CRUD operations, edge cases
│   └── test_model_handler.py       # Event handling, state transitions
├── property/
│   ├── test_experiment_properties.py   # Properties 1-4 (experiment management)
│   ├── test_training_properties.py     # Properties 5-7 (training cycle)
│   ├── test_annotation_properties.py   # Properties 8-10 (annotation flow)
│   ├── test_persistence_properties.py  # Properties 12-15 (data persistence)
│   └── test_control_properties.py      # Property 11 (pause/stop)
└── integration/
    └── test_full_cycle.py          # End-to-end cycle tests
```

### Unit Test Focus Areas

- WorldState field initialization and defaults
- Event creation with various payloads
- ExperimentManager CRUD edge cases (empty list, duplicate names)
- Phase transition validation
- Error message population

### Property Test Implementation

Each correctness property maps to a single property-based test using Hypothesis:

```python
from hypothesis import given, strategies as st

# Feature: mvc-active-learning-dashboard, Property 2: Experiment Persistence Round-Trip
@given(config=experiment_config_strategy())
def test_experiment_persistence_round_trip(config):
    """For any valid config, create then load returns equivalent config."""
    manager = ExperimentManager(tmp_path)
    exp_id = manager.create_experiment(config)
    loaded = manager.load_experiment(exp_id)
    assert loaded['config'] == config
```

### Test Data Generators (Hypothesis Strategies)

```python
# Experiment configuration generator
@st.composite
def experiment_config_strategy(draw):
    return {
        'name': draw(st.text(min_size=1, max_size=50)),
        'dataset_path': draw(st.text(min_size=1, max_size=200)),
        'model_name': draw(st.sampled_from(['resnet18', 'resnet50', 'mobilenetv2'])),
        'strategy': draw(st.sampled_from(['random', 'uncertainty', 'entropy', 'margin'])),
        'epochs_per_cycle': draw(st.integers(min_value=1, max_value=100)),
        'batch_size': draw(st.integers(min_value=1, max_value=128)),
        'learning_rate': draw(st.floats(min_value=1e-6, max_value=1.0)),
    }

# Cycle metrics generator
@st.composite  
def cycle_metrics_strategy(draw):
    return CycleMetrics(
        cycle=draw(st.integers(min_value=0, max_value=100)),
        labeled_count=draw(st.integers(min_value=0, max_value=10000)),
        unlabeled_count=draw(st.integers(min_value=0, max_value=10000)),
        best_val_accuracy=draw(st.floats(min_value=0.0, max_value=1.0)),
        test_accuracy=draw(st.floats(min_value=0.0, max_value=1.0)),
    )
```
