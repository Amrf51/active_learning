# MVC Architecture Design - Active Learning Dashboard

## Overview

A practical MVC architecture for the Streamlit-based Active Learning dashboard using **multiprocessing with pipes** for process isolation and scalability. The ActiveLearning service runs in a separate process, communicating with the controller via bidirectional pipes.

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
│                    │   (cached WorldState) │                            │
│                    └───────────┬───────────┘                            │
└────────────────────────────────┼────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────┐
│                         CONTROLLER LAYER                                 │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    ExperimentController                          │    │
│  │  • dispatch(event) → sends to service via pipe                   │    │
│  │  • get_state() → returns cached WorldState                       │    │
│  │  • poll_updates() → receives state from service                  │    │
│  │  INBOX: receives WorldState updates from service                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                 │                                        │
│  ┌──────────────────────────────┴──────────────────────────────────┐    │
│  │                      BackgroundWorker                            │    │
│  │  • start() → spawns service process                              │    │
│  │  • send_event() → writes to pipe                                 │    │
│  │  • poll_state() → reads from pipe (non-blocking)                 │    │
│  │  • is_alive() → checks process health                            │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                          ┌───────┴───────┐
                          │     PIPE      │  (bidirectional)
                          │  Event →      │
                          │  ← WorldState │
                          └───────┬───────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────────┐
│                    SERVICE PROCESS (multiprocessing)                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                  ActiveLearningService                           │    │
│  │  INBOX: receives Events from controller                          │    │
│  │  • run_loop() → main event loop                                  │    │
│  │  • handle_event() → processes events                             │    │
│  │  • push_state() → sends WorldState to controller                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                 │                                        │
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

## Process Communication Pattern

```
┌──────────────────┐                              ┌──────────────────┐
│    CONTROLLER    │                              │     SERVICE      │
│                  │                              │                  │
│  ┌────────────┐  │      Event (pickle)          │  ┌────────────┐  │
│  │   INBOX    │◄─┼──────────────────────────────┼──│   OUTBOX   │  │
│  │ WorldState │  │                              │  │ WorldState │  │
│  └────────────┘  │                              │  └────────────┘  │
│                  │                              │                  │
│  ┌────────────┐  │      Event (pickle)          │  ┌────────────┐  │
│  │   OUTBOX   │──┼──────────────────────────────┼─►│   INBOX    │  │
│  │   Event    │  │                              │  │   Event    │  │
│  └────────────┘  │                              │  └────────────┘  │
└──────────────────┘                              └──────────────────┘
```

## Core Components

### 1. WorldState (In-Memory State)

Single source of truth for UI. Must be **picklable** for pipe transport.

```python
@dataclass
class WorldState:
    """In-memory state for fast UI reads. Must be picklable."""
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
    
    # Class distribution (for Dataset Explorer)
    class_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Live metrics (updated during training)
    epoch_metrics: List[EpochMetrics] = field(default_factory=list)
    
    # Queried images for annotation (paths only, not image data)
    queried_images: List[QueriedImage] = field(default_factory=list)
    
    # Probe images for prediction tracking
    probe_images: List[ProbeImage] = field(default_factory=list)
    
    # Error
    error_message: Optional[str] = None
    
    # Timestamp for state versioning
    updated_at: float = field(default_factory=time.time)
```

### 2. Event System (Command Pattern)

All user actions go through events. Events must be **picklable** for pipe transport.

```python
class EventType(Enum):
    # Experiment lifecycle
    CREATE_EXPERIMENT = "create_experiment"
    LOAD_EXPERIMENT = "load_experiment"
    DELETE_EXPERIMENT = "delete_experiment"
    
    # Training control
    START_CYCLE = "start_cycle"
    PAUSE = "pause"
    STOP = "stop"
    CONTINUE = "continue"
    
    # Annotation
    SUBMIT_ANNOTATIONS = "submit_annotations"
    
    # Data queries
    GET_POOL_DATA = "get_pool_data"
    
    # Service control
    SHUTDOWN = "shutdown"  # Graceful service termination

@dataclass
class Event:
    """Picklable event for pipe transport."""
    type: EventType
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
```

### 3. BackgroundWorker (Process Manager)

Manages the service process lifecycle and pipe communication.

```python
class BackgroundWorker:
    """Manages the ActiveLearning service process."""
    
    def __init__(self):
        self._process: Optional[Process] = None
        self._pipe: Optional[Connection] = None  # Parent end of pipe
        self._is_started = False
    
    def start(self, experiments_dir: Path) -> None:
        """Spawn the service process with pipe connection."""
        if self._is_started:
            return
        
        parent_conn, child_conn = Pipe(duplex=True)
        self._pipe = parent_conn
        self._process = Process(
            target=run_service_loop,
            args=(child_conn, experiments_dir),
            daemon=False  # Allow graceful shutdown
        )
        self._process.start()
        self._is_started = True
    
    def send_event(self, event: Event) -> None:
        """Send event to service via pipe."""
        if self._pipe and self._is_started:
            self._pipe.send(event)
    
    def poll_state(self, timeout: float = 0.0) -> Optional[WorldState]:
        """Non-blocking check for state updates from service."""
        if self._pipe and self._pipe.poll(timeout):
            return self._pipe.recv()
        return None
    
    def is_alive(self) -> bool:
        """Check if service process is running."""
        return self._process is not None and self._process.is_alive()
    
    def shutdown(self, timeout: float = 5.0) -> None:
        """Graceful shutdown of service process."""
        if self._pipe and self._is_started:
            self.send_event(Event(EventType.SHUTDOWN))
            self._process.join(timeout=timeout)
            if self._process.is_alive():
                self._process.terminate()
            self._pipe.close()
            self._is_started = False
```

### 4. ExperimentController

Routes events via BackgroundWorker. Caches WorldState locally for fast reads.

```python
class ExperimentController:
    """Controller that communicates with service process via pipes."""
    
    def __init__(self, experiments_dir: Path):
        self.experiments_dir = Path(experiments_dir)
        self._worker = BackgroundWorker()
        self._cached_state = WorldState()  # Local cache for fast UI reads
        self._worker.start(experiments_dir)
    
    def dispatch(self, event: Event) -> None:
        """Send event to service process via pipe."""
        self._worker.send_event(event)
    
    def get_state(self) -> WorldState:
        """Return cached state. Call poll_updates() first for fresh data."""
        return self._cached_state
    
    def poll_updates(self) -> bool:
        """Poll for state updates from service. Returns True if updated."""
        new_state = self._worker.poll_state()
        if new_state:
            self._cached_state = new_state
            return True
        return False
    
    def drain_updates(self) -> WorldState:
        """Drain all pending updates and return latest state."""
        while self.poll_updates():
            pass
        return self._cached_state
    
    def is_service_alive(self) -> bool:
        """Check if service process is running."""
        return self._worker.is_alive()
    
    def shutdown(self) -> None:
        """Graceful shutdown of service."""
        self._worker.shutdown()
```

### 5. ActiveLearningService (Service Process)

Runs in a separate process. Handles events and pushes state updates.

```python
def run_service_loop(pipe: Connection, experiments_dir: Path) -> None:
    """Main loop running in separate process."""
    service = ActiveLearningService(pipe, experiments_dir)
    service.run()


class ActiveLearningService:
    """Service that runs in a separate process."""
    
    def __init__(self, pipe: Connection, experiments_dir: Path):
        self._pipe = pipe
        self._model_handler = ModelHandler(experiments_dir)
        self._running = True
    
    def run(self) -> None:
        """Main event loop."""
        while self._running:
            # Check for incoming events (INBOX)
            if self._pipe.poll(timeout=0.1):
                event = self._pipe.recv()
                self._handle_event(event)
            
            # If training, run one epoch and push state
            if self._model_handler.world_state.phase == Phase.TRAINING:
                self._train_one_epoch()
    
    def _handle_event(self, event: Event) -> None:
        """Process event from controller."""
        if event.type == EventType.SHUTDOWN:
            self._running = False
            return
        
        # Delegate to ModelHandler
        self._model_handler.handle_event(event)
        
        # Push updated state to controller
        self._push_state()
    
    def _train_one_epoch(self) -> None:
        """Train one epoch and push state update."""
        try:
            self._model_handler.train_epoch()
            self._push_state()
        except Exception as e:
            self._model_handler.world_state.phase = Phase.ERROR
            self._model_handler.world_state.error_message = str(e)
            self._push_state()
    
    def _push_state(self) -> None:
        """Send current WorldState to controller."""
        self._model_handler.world_state.updated_at = time.time()
        self._pipe.send(self._model_handler.world_state)
```

### 6. ModelHandler

Orchestrates backend operations. Updates WorldState after each step.
Runs inside the service process.

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
        elif event.type == EventType.PAUSE:
            self._pause()
        elif event.type == EventType.STOP:
            self._stop()
        # ... etc
    
    def train_epoch(self) -> EpochMetrics:
        """Train one epoch, update WorldState."""
        metrics = self._active_loop.train_single_epoch(self.world_state.current_epoch)
        self.world_state.epoch_metrics.append(metrics)
        self.world_state.current_epoch += 1
        return metrics
```

### 7. ExperimentManager

Handles persistence: SQLite for metadata, folders for artifacts.

**Experiment Folder Naming**: Experiment folders are named using the user-provided experiment name (e.g., "mvc_test2") instead of the auto-generated UUID. The UUID is still used as the primary key in the database for uniqueness, but the folder structure uses the human-readable name for easier navigation.

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
│   ├── 3_Results.py                # Metrics, comparison, export
│   └── 4_Dataset_Explorer.py       # Inspect labeled/unlabeled pools
├── controller/
│   ├── __init__.py
│   ├── experiment_controller.py    # Event routing, state caching
│   ├── background_worker.py        # NEW: Process management, pipe I/O
│   └── events.py                   # EventType enum, Event dataclass
├── services/                       # NEW: Service process
│   ├── __init__.py
│   └── active_learning_service.py  # Main service loop
├── model/
│   ├── __init__.py
│   ├── world_state.py              # WorldState dataclass (picklable)
│   ├── model_handler.py            # Backend orchestration
│   ├── schemas.py                  # Data schemas
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
    └── {experiment_name}/          # Per-experiment folder (named by user, e.g., "mvc_test2")
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

### Starting a Training Cycle (Multiprocessing)

```
1. User clicks "Start Cycle"
2. View: controller.dispatch(Event(START_CYCLE))
3. Controller: worker.send_event(event) → writes to pipe
4. Service Process (INBOX receives event):
   a. model_handler.handle_event(event)
   b. world_state.phase = Phase.TRAINING
   c. active_loop.prepare_cycle(cycle_num)
   d. service.push_state() → sends WorldState via pipe
5. Service Process (training loop):
   a. For each epoch:
      - metrics = model_handler.train_epoch()
      - world_state.epoch_metrics.append(metrics)
      - service.push_state() → sends WorldState via pipe
   b. world_state.phase = Phase.QUERYING
   c. queried = active_loop.query_samples()
   d. world_state.queried_images = queried
   e. world_state.phase = Phase.AWAITING_ANNOTATION
   f. service.push_state() → sends WorldState via pipe
6. View: controller.poll_updates() → receives WorldState from pipe
7. View: Updates display with latest state
```

### Submitting Annotations (Multiprocessing)

```
1. User clicks "Submit Annotations"
2. View: controller.dispatch(Event(SUBMIT_ANNOTATIONS, {annotations: [...]}))
3. Controller: worker.send_event(event) → writes to pipe
4. Service Process (INBOX receives event):
   a. model_handler.handle_event(event)
   b. active_loop.receive_annotations(annotations)
   c. exp_manager.save_cycle_result(exp_id, cycle_metrics)
   d. world_state.current_cycle += 1
   e. world_state.phase = Phase.IDLE (or COMPLETED if last cycle)
   f. service.push_state() → sends WorldState via pipe
5. View: controller.poll_updates() → receives WorldState from pipe
6. View: Shows updated pool sizes, ready for next cycle
```

### Service Lifecycle

```
1. Dashboard starts:
   - controller = ExperimentController(experiments_dir)
   - controller._worker.start() → spawns service process
   - Service process initializes ModelHandler, enters run loop

2. During operation:
   - View dispatches events → pipe → Service INBOX
   - Service processes events, trains epochs
   - Service pushes WorldState → pipe → Controller INBOX
   - View polls controller.get_state() for display

3. Dashboard closes:
   - controller.shutdown()
   - Sends SHUTDOWN event → pipe → Service
   - Service exits run loop gracefully
   - Process joins with timeout, terminate if needed
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

## Multiprocessing Strategy

### Why Multiprocessing over Threading

| Aspect | Threading | Multiprocessing (chosen) |
|--------|-----------|--------------------------|
| Process isolation | No - crash affects UI | Yes - service crash doesn't kill UI |
| Memory isolation | Shared memory | Separate memory spaces |
| GIL limitations | Affected by GIL | True parallelism |
| Scalability | Limited | Better for heavy workloads |
| Debugging | Easier | Requires more care |

### Pipe Communication

Using `multiprocessing.Pipe` for bidirectional communication:

```python
from multiprocessing import Pipe, Process

# Create bidirectional pipe
parent_conn, child_conn = Pipe(duplex=True)

# Parent (Controller) uses parent_conn
# Child (Service) uses child_conn
```

### Serialization Requirements

All data crossing the pipe must be **picklable**:

✅ **Picklable** (safe to send):
- Primitive types (int, float, str, bool)
- Dataclasses with primitive fields
- Lists, dicts with picklable contents
- Enums
- File paths (as strings)

❌ **Not Picklable** (must stay in service):
- PyTorch models
- DataLoaders
- Open file handles
- Database connections
- Lambda functions

### State Synchronization

```python
# Service pushes state after each significant change
def _push_state(self) -> None:
    self._model_handler.world_state.updated_at = time.time()
    self._pipe.send(self._model_handler.world_state)

# Controller polls for updates (non-blocking)
def poll_updates(self) -> bool:
    if self._worker._pipe.poll(timeout=0.0):
        self._cached_state = self._worker._pipe.recv()
        return True
    return False
```

### Error Handling in Multiprocessing

```python
class BackgroundWorker:
    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()
    
    def restart_if_dead(self) -> bool:
        """Restart service if it crashed."""
        if not self.is_alive() and self._is_started:
            self.start(self._experiments_dir)
            return True
        return False

class ExperimentController:
    def get_state(self) -> WorldState:
        # Check service health before returning state
        if not self._worker.is_alive():
            self._cached_state.phase = Phase.ERROR
            self._cached_state.error_message = "Service process died unexpectedly"
        return self._cached_state
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Command Pattern (Events) | Decouples UI from backend, enables logging/undo |
| Multiprocessing with Pipes | Process isolation, crash recovery, true parallelism |
| Cached WorldState in Controller | Fast UI reads without pipe I/O on every access |
| Picklable dataclasses | Required for pipe transport |
| Service pushes state | Controller doesn't need to request updates |
| SQLite for metadata | Scalable, queryable, ACID-compliant |
| Folder-based artifacts | Large files (checkpoints, matrices) stay out of DB |
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

### Dataset Explorer Page (NEW)
- View labeled pool: display images with their assigned labels
- View unlabeled pool: display images awaiting annotation
- Pool statistics: counts, class distribution charts
- Image preview with metadata (path, predicted class if available)
- Filter/search by class or filename
- Pagination for large datasets (lazy loading)

## Migration from Current Code

### Phase 1: Create Multiprocessing Infrastructure
1. **Create `services/` folder**: New package for service process
2. **Create `controller/background_worker.py`**: Process management
3. **Create `services/active_learning_service.py`**: Main service loop

### Phase 2: Update Existing Components
4. **Update `controller/events.py`**: Add SHUTDOWN event, timestamp field
5. **Update `model/world_state.py`**: Add updated_at field, ensure picklable
6. **Update `controller/experiment_controller.py`**: Use BackgroundWorker instead of direct ModelHandler

### Phase 3: Integration
7. **Update Views**: Use `controller.poll_updates()` before `get_state()`
8. **Update `dashboard.py`**: Handle service lifecycle (start/shutdown)

### Phase 4: Testing
9. **Add process lifecycle tests**: Start, shutdown, crash recovery
10. **Add pipe communication tests**: Event sending, state receiving

The existing `backend/` code (ActiveLearningLoop, Trainer, ALDataManager, strategies) remains unchanged.
The existing `model/model_handler.py` remains unchanged (runs inside service process).



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

### Property 16: Service Process Lifecycle

*For any* controller initialization, the service process SHALL be alive after start() and not alive after shutdown().

**Validates: Requirements 3.1 (background process management)**

### Property 17: Pipe Communication Integrity

*For any* event sent via pipe, the service SHALL receive an equivalent event (serialization round-trip).

**Validates: Requirements 8.2 (event routing)**

### Property 18: State Push After Event

*For any* event processed by the service, a WorldState update SHALL be pushed to the controller via pipe.

**Validates: Requirements 7.2 (state synchronization)**

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
│   ├── test_model_handler.py       # Event handling, state transitions
│   ├── test_background_worker.py   # NEW: Process lifecycle tests
│   └── test_serialization.py       # NEW: Pickle round-trip tests
├── property/
│   ├── test_experiment_properties.py   # Properties 1-4 (experiment management)
│   ├── test_training_properties.py     # Properties 5-7 (training cycle)
│   ├── test_annotation_properties.py   # Properties 8-10 (annotation flow)
│   ├── test_persistence_properties.py  # Properties 12-15 (data persistence)
│   ├── test_control_properties.py      # Property 11 (pause/stop)
│   └── test_multiprocessing_properties.py  # NEW: Properties 16-18 (process/pipe)
└── integration/
    ├── test_full_cycle.py          # End-to-end cycle tests
    └── test_service_integration.py # NEW: Full service process tests
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
