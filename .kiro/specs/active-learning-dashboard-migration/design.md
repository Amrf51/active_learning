# Design Document: Active Learning Dashboard Migration

## Overview

This design document specifies the migration from a JSON file-based worker pattern to an MVC event-driven architecture for the Active Learning Dashboard. The new architecture implements a hybrid WorldState/SQLite approach that provides both fast in-memory access for current experiment state and reliable persistence for historical data.

The migration transforms the current polling-based system into a responsive event-driven architecture with clear separation of concerns across View, Controller, Model, and Service layers.

---

## Assumptions and Constraints

### Single-User Operation

The system is designed for **single-user operation** to simplify architecture and reduce complexity. This assumption enables:

| Simplification | Benefit |
|----------------|---------|
| No session management | Reduced complexity |
| No concurrent access protection | Simple thread safety sufficient |
| One active experiment at a time | Straightforward state management |
| No user authentication | Focus on core functionality |

#### Multi-Tab Detection

When multiple browser tabs are detected, the system warns the user:

```python
class SessionManager:
    """Manages single-user session detection."""
    
    SESSION_KEY = "al_dashboard_session"
    
    def __init__(self):
        self._session_id = str(uuid.uuid4())
        self._session_file = Path.home() / ".al_dashboard_session"
    
    def acquire_session(self) -> bool:
        """
        Attempt to acquire exclusive session.
        Returns True if this is the only active session.
        """
        try:
            if self._session_file.exists():
                # Check if existing session is stale (>30 seconds old)
                existing_data = json.loads(self._session_file.read_text())
                last_heartbeat = datetime.fromisoformat(existing_data['heartbeat'])
                
                if datetime.now() - last_heartbeat < timedelta(seconds=30):
                    return False  # Another session is active
            
            # Write our session
            self._session_file.write_text(json.dumps({
                'session_id': self._session_id,
                'heartbeat': datetime.now().isoformat()
            }))
            return True
            
        except Exception:
            return True  # Fail open for robustness
    
    def update_heartbeat(self) -> None:
        """Update session heartbeat timestamp."""
        self._session_file.write_text(json.dumps({
            'session_id': self._session_id,
            'heartbeat': datetime.now().isoformat()
        }))
    
    def release_session(self) -> None:
        """Release the session on shutdown."""
        if self._session_file.exists():
            self._session_file.unlink()
```

**Usage in View Layer:**

```python
# In dashboard.py
def main():
    session_manager = get_session_manager()
    
    if not session_manager.acquire_session():
        st.error("⚠️ Another browser tab is already running the dashboard.")
        st.warning("Please close other tabs and refresh this page.")
        st.stop()
    
    # Normal dashboard rendering...
```

---

## Architecture

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           VIEW LAYER                                         │
│                      (Streamlit Pages)                                       │
│                                                                              │
│   ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                │
│   │ Configuration  │  │ Active Learning│  │    Results     │                │
│   │     Page       │  │     Page       │  │     Page       │                │
│   └───────┬────────┘  └───────┬────────┘  └───────┬────────┘                │
│           │                   │                   │                          │
│           └───────────────────┼───────────────────┘                          │
│                               │                                              │
│                    dispatch(Event) / get_data()                              │
│                               │                                              │
└───────────────────────────────┼──────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                         CONTROLLER LAYER                                      │
│                                                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │                      EventDispatcher                                 │    │
│   │                      (Singleton in st.session_state)                 │    │
│   │                                                                      │    │
│   │   ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐   │    │
│   │   │  ModelHandler   │  │ ServiceManager  │  │  SessionManager  │   │    │
│   │   │                 │  │                 │  │                  │   │    │
│   │   │ • get_status()  │  │ • spawn_service │  │ • acquire_session│   │    │
│   │   │ • get_metrics() │  │ • send_command  │  │ • check_session  │   │    │
│   │   │ • get_pool()    │  │ • terminate     │  │                  │   │    │
│   │   └────────┬────────┘  └────────┬────────┘  └──────────────────┘   │    │
│   │            │                    │                                   │    │
│   └────────────┼────────────────────┼───────────────────────────────────┘    │
│                │                    │                                         │
└────────────────┼────────────────────┼─────────────────────────────────────────┘
                 │                    │
                 ▼                    │ Pipe (bidirectional)
┌────────────────────────────────┐    │
│         MODEL LAYER            │    │
│                                │    │
│  ┌──────────────────────────┐  │    │
│  │      WorldState          │  │    │
│  │    (In-Memory Cache)     │  │    │
│  │                          │  │    │
│  │  • phase                 │  │    │
│  │  • current_cycle         │  │    │
│  │  • current_metrics       │  │    │
│  │  • pending_updates flag  │  │    │
│  └────────────┬─────────────┘  │    │
│               │ sync           │    │
│               ▼                │    │
│  ┌──────────────────────────┐  │    │
│  │    SQLite Database       │  │    │
│  │    (Persistent)          │  │    │
│  │                          │  │    │
│  │  • experiments table     │  │    │
│  │  • epochs table          │  │    │
│  │  • cycles table          │  │    │
│  │  • pool_items table      │  │    │
│  └──────────────────────────┘  │    │
│                                │    │
└────────────────────────────────┘    │
                                      │
══════════════════════════════════════╪════════════════════════════════════════
                PROCESS BOUNDARY      │
══════════════════════════════════════╪════════════════════════════════════════
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                          SERVICE LAYER                                        │
│                      [SEPARATE PROCESS]                                       │
│                                                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │                   ActiveLearningService                              │    │
│   │                                                                      │    │
│   │   INBOX ◄──── Commands (START_CYCLE, PAUSE, STOP, ANNOTATIONS)      │    │
│   │   OUTBOX ────► Events (EPOCH_COMPLETE, QUERY_READY, ERROR)          │    │
│   │                                                                      │    │
│   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │    │
│   │   │   Trainer   │  │ DataManager │  │  AL Loop    │                 │    │
│   │   │ (unchanged) │  │ (unchanged) │  │ (unchanged) │                 │    │
│   │   └─────────────┘  └─────────────┘  └─────────────┘                 │    │
│   │                                                                      │    │
│   └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Event Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EVENT FLOW DIAGRAM                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  USER ACTION                                                                │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────┐                                                       │
│  │   VIEW LAYER    │                                                       │
│  │                 │                                                       │
│  │  st.button() ───┼──► dispatch(Event(START_CYCLE))                       │
│  │                 │                                                       │
│  └─────────────────┘                                                       │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      CONTROLLER LAYER                                │   │
│  │                                                                      │   │
│  │  EventDispatcher.dispatch(event)                                     │   │
│  │       │                                                              │   │
│  │       ├──► ModelHandler.update_state()                              │   │
│  │       │         │                                                    │   │
│  │       │         ├──► WorldState.phase = TRAINING (immediate)        │   │
│  │       │         └──► SQLite.update_experiment() (persistent)        │   │
│  │       │                                                              │   │
│  │       └──► ServiceManager.send_command(CMD_START_CYCLE)             │   │
│  │                 │                                                    │   │
│  └─────────────────┼────────────────────────────────────────────────────┘   │
│                    │                                                        │
│                    ▼ Pipe                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      SERVICE LAYER                                   │   │
│  │                                                                      │   │
│  │  ActiveLearningService._handle_command()                            │   │
│  │       │                                                              │   │
│  │       └──► _execute_training_cycle()                                │   │
│  │                 │                                                    │   │
│  │                 ├──► Train epoch 1 ──► pipe.send(EPOCH_COMPLETE)    │   │
│  │                 ├──► Train epoch 2 ──► pipe.send(EPOCH_COMPLETE)    │   │
│  │                 └──► Query samples ──► pipe.send(QUERY_READY)       │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                    │                                                        │
│                    ▼ Pipe (events back)                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                 CONTROLLER (Listener Thread)                         │   │
│  │                                                                      │   │
│  │  _handle_service_event(EPOCH_COMPLETE)                              │   │
│  │       │                                                              │   │
│  │       ├──► WorldState.current_metrics = new_metrics                 │   │
│  │       ├──► WorldState.pending_updates = True  ◄── Flag for UI      │   │
│  │       └──► SQLite.insert_epoch_metrics()                            │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Live Update Strategy

### The Challenge

Streamlit's execution model reruns the entire script on user interaction. True "push" updates from server to client aren't natively supported. However, we need to show live training progress.

### Solution: Lightweight Fragment Refresh

We use `@st.fragment` with a **flag-based check** - this is NOT file-based polling:

```python
# pages/2_Active_Learning.py

@st.fragment(run_every=2.0)
def training_progress_display():
    """
    Lightweight periodic refresh for training visualization.
    
    WHY THIS IS NOT POLLING:
    - Checks in-memory boolean flag (microseconds)
    - Reads from WorldState (in-memory), not files
    - Only refreshes this fragment, not entire page
    - No file I/O or JSON parsing
    """
    ctrl = get_controller()
    
    # Fast flag check (in-memory, ~1 microsecond)
    if not ctrl.has_pending_updates():
        return  # Nothing new, skip rendering
    
    # Get data from WorldState (in-memory, ~10 microseconds)
    progress = ctrl.get_training_progress()
    
    # Render progress
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Epoch", f"{progress.current_epoch}/{progress.total_epochs}")
        st.metric("Train Loss", f"{progress.train_loss:.4f}")
    with col2:
        st.metric("Val Accuracy", f"{progress.val_accuracy:.2%}")
        st.progress(progress.progress_percentage / 100)
    
    # Display loss chart
    if progress.epoch_history:
        st.line_chart(progress.loss_history_df)
    
    # Clear the flag
    ctrl.clear_pending_updates()
```

### Comparison: Old Polling vs New Approach

| Aspect | Old (File Polling) | New (Flag Check) |
|--------|-------------------|------------------|
| Read operation | 30MB JSON file | Boolean flag |
| Time per check | ~3 seconds | ~1 microsecond |
| Locks needed | FileLock | None |
| Scope | Entire page | Fragment only |
| Data source | Disk | RAM |

### Alternative: Manual Refresh

For users who prefer no automatic updates:

```python
def training_progress_manual():
    """Manual refresh version - fully event-driven."""
    ctrl = get_controller()
    
    if st.button("🔄 Refresh Progress"):
        st.rerun()
    
    progress = ctrl.get_training_progress()
    # ... render progress ...
```

---

## Controller Lifecycle Management

### Singleton Pattern with st.session_state

The Controller must survive Streamlit's script reruns:

```python
# controller/__init__.py

from typing import Optional
import streamlit as st

_controller_instance: Optional['EventDispatcher'] = None

def get_controller() -> 'EventDispatcher':
    """
    Get or create the Controller singleton.
    
    The Controller is stored in st.session_state to survive
    Streamlit reruns. This is the ONLY way to access the Controller.
    """
    if 'controller' not in st.session_state:
        st.session_state.controller = _create_controller()
    
    return st.session_state.controller


def _create_controller() -> 'EventDispatcher':
    """
    Create and initialize the Controller with all dependencies.
    Called once per session.
    """
    from .dispatcher import EventDispatcher
    from .model_handler import ModelHandler
    from .service_manager import ServiceManager
    from ..model.world_state import WorldState
    from ..model.database import DatabaseManager
    
    # Get paths
    db_path = _get_database_path()
    
    # Initialize components
    world_state = WorldState()
    db_manager = DatabaseManager(db_path)
    
    # Check for existing experiment to restore
    active_experiment = db_manager.get_active_experiment()
    if active_experiment:
        world_state.restore_from_db(active_experiment)
    
    # Create handlers
    model_handler = ModelHandler(world_state, db_manager)
    service_manager = ServiceManager()
    
    # Create and return dispatcher
    return EventDispatcher(
        model_handler=model_handler,
        service_manager=service_manager
    )


def _get_database_path() -> Path:
    """Get path to SQLite database."""
    data_dir = Path.home() / ".al_dashboard" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "experiments.db"


def reset_controller() -> None:
    """
    Reset the Controller (for testing or error recovery).
    """
    if 'controller' in st.session_state:
        # Cleanup existing controller
        ctrl = st.session_state.controller
        ctrl.shutdown()
        del st.session_state.controller
```

### Initialization Sequence

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONTROLLER INITIALIZATION SEQUENCE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. First Page Load                                                         │
│     │                                                                       │
│     ▼                                                                       │
│  2. get_controller() called                                                 │
│     │                                                                       │
│     ├──► Check st.session_state['controller']                              │
│     │         │                                                             │
│     │         └──► NOT FOUND                                               │
│     │                                                                       │
│     ▼                                                                       │
│  3. _create_controller()                                                    │
│     │                                                                       │
│     ├──► Create WorldState (empty)                                         │
│     ├──► Create DatabaseManager (connect to SQLite)                        │
│     │         │                                                             │
│     │         └──► Check for active experiment                             │
│     │                   │                                                   │
│     │                   └──► If found: restore WorldState                  │
│     │                                                                       │
│     ├──► Create ModelHandler(WorldState, DatabaseManager)                  │
│     ├──► Create ServiceManager()                                           │
│     └──► Create EventDispatcher(ModelHandler, ServiceManager)              │
│                                                                             │
│  4. Store in st.session_state['controller']                                │
│     │                                                                       │
│     ▼                                                                       │
│  5. Return Controller                                                       │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                             │
│  6. Subsequent Page Loads / Reruns                                          │
│     │                                                                       │
│     ▼                                                                       │
│  7. get_controller() called                                                 │
│     │                                                                       │
│     ├──► Check st.session_state['controller']                              │
│     │         │                                                             │
│     │         └──► FOUND ──► Return existing Controller                    │
│     │                                                                       │
│     └──► (Same instance survives reruns)                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Process Lifecycle Management

### Daemon Process Configuration

The Service process is configured as a daemon to ensure cleanup:

```python
# controller/service_manager.py

from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from threading import Thread
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ServiceManager:
    """
    Manages the ActiveLearningService process lifecycle.
    
    Key responsibilities:
    - Spawn service process with Pipe communication
    - Monitor service health
    - Handle graceful shutdown
    - Restart on failure
    """
    
    def __init__(self):
        self._process: Optional[Process] = None
        self._pipe: Optional[Connection] = None
        self._listener_thread: Optional[Thread] = None
        self._event_callback = None
        self._shutdown_requested = False
    
    def spawn_service(self, config: dict, event_callback) -> bool:
        """
        Spawn the ActiveLearningService process.
        
        Args:
            config: Experiment configuration
            event_callback: Callback for handling service events
            
        Returns:
            True if service started successfully
        """
        # Terminate any existing service first
        self._terminate_existing_service()
        
        self._event_callback = event_callback
        self._shutdown_requested = False
        
        try:
            # Create bidirectional pipe
            parent_conn, child_conn = Pipe()
            self._pipe = parent_conn
            
            # Spawn process as DAEMON
            # Daemon processes are automatically terminated when parent exits
            self._process = Process(
                target=run_active_learning_service,
                args=(child_conn, config),
                daemon=True,
                name="ActiveLearningService"
            )
            self._process.start()
            
            logger.info(f"Service process started with PID: {self._process.pid}")
            
            # Start listener thread for service events
            self._start_listener_thread()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to spawn service: {e}")
            return False
    
    def _terminate_existing_service(self) -> None:
        """Gracefully terminate existing service if running."""
        if not self._process or not self._process.is_alive():
            return
        
        logger.info("Terminating existing service...")
        
        try:
            # Step 1: Send graceful shutdown command
            if self._pipe:
                self._pipe.send(Event(EventType.CMD_SHUTDOWN))
            
            # Step 2: Wait for graceful shutdown (max 5 seconds)
            self._process.join(timeout=5.0)
            
            # Step 3: Force terminate if still running
            if self._process.is_alive():
                logger.warning("Service didn't shutdown gracefully, terminating...")
                self._process.terminate()
                self._process.join(timeout=2.0)
            
            # Step 4: Kill if still running (last resort)
            if self._process.is_alive():
                logger.error("Service didn't terminate, killing...")
                self._process.kill()
                
        except Exception as e:
            logger.error(f"Error during service termination: {e}")
        
        finally:
            self._process = None
            self._pipe = None
    
    def _start_listener_thread(self) -> None:
        """Start background thread to listen for service events."""
        self._listener_thread = Thread(
            target=self._listen_for_events,
            daemon=True,
            name="ServiceEventListener"
        )
        self._listener_thread.start()
    
    def _listen_for_events(self) -> None:
        """
        Background thread: Listen for events from service.
        
        This thread blocks on pipe.recv() waiting for events.
        No polling - just blocking I/O.
        """
        while not self._shutdown_requested:
            try:
                # Check if service is still alive
                if not self.is_alive():
                    logger.warning("Service process died unexpectedly")
                    if self._event_callback:
                        self._event_callback(Event(
                            EventType.SERVICE_ERROR,
                            {"message": "Service process died unexpectedly"}
                        ))
                    break
                
                # Wait for event (with timeout to check shutdown flag)
                if self._pipe and self._pipe.poll(timeout=1.0):
                    event = self._pipe.recv()
                    
                    if self._event_callback:
                        self._event_callback(event)
                        
            except EOFError:
                logger.info("Service pipe closed")
                break
            except Exception as e:
                logger.error(f"Error receiving service event: {e}")
    
    def send_command(self, event: 'Event') -> bool:
        """Send command to service process."""
        if not self._pipe:
            logger.error("Cannot send command: no pipe connection")
            return False
        
        try:
            self._pipe.send(event)
            return True
        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            return False
    
    def is_alive(self) -> bool:
        """Check if service process is running."""
        return self._process is not None and self._process.is_alive()
    
    def shutdown(self) -> None:
        """Shutdown the service manager and cleanup resources."""
        self._shutdown_requested = True
        self._terminate_existing_service()
        
        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=2.0)
```

### Process Lifecycle Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SERVICE PROCESS LIFECYCLE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SPAWN                                                                      │
│  ─────                                                                      │
│                                                                             │
│  User clicks "Initialize"                                                   │
│       │                                                                     │
│       ▼                                                                     │
│  ServiceManager.spawn_service(config)                                       │
│       │                                                                     │
│       ├──► _terminate_existing_service()  ◄── Always cleanup first         │
│       │                                                                     │
│       ├──► Create Pipe (bidirectional)                                     │
│       │                                                                     │
│       ├──► Process(daemon=True)  ◄── Key: daemon process                   │
│       │         │                                                           │
│       │         └──► Starts ActiveLearningService                          │
│       │                                                                     │
│       └──► Start listener thread                                           │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                             │
│  NORMAL OPERATION                                                           │
│  ────────────────                                                           │
│                                                                             │
│  Controller ◄─────── Pipe ────────► Service                                │
│       │                                 │                                   │
│       │ CMD_START_CYCLE ───────────►   │                                   │
│       │                                 ├──► Execute training               │
│       │ ◄─────────── EPOCH_COMPLETE    │                                   │
│       │ ◄─────────── EPOCH_COMPLETE    │                                   │
│       │ ◄─────────── QUERY_READY       │                                   │
│       │                                 │                                   │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                             │
│  GRACEFUL SHUTDOWN                                                          │
│  ─────────────────                                                          │
│                                                                             │
│  User closes browser OR new experiment                                      │
│       │                                                                     │
│       ▼                                                                     │
│  ServiceManager._terminate_existing_service()                               │
│       │                                                                     │
│       ├──► Send CMD_SHUTDOWN via pipe                                      │
│       │                                                                     │
│       ├──► process.join(timeout=5)  ◄── Wait for graceful exit            │
│       │         │                                                           │
│       │         └──► Service saves state and exits                         │
│       │                                                                     │
│       └──► If still alive: process.terminate()                             │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                             │
│  CRASH RECOVERY                                                             │
│  ──────────────                                                             │
│                                                                             │
│  Service crashes unexpectedly                                               │
│       │                                                                     │
│       ▼                                                                     │
│  Listener thread detects: is_alive() == False                              │
│       │                                                                     │
│       ├──► Emit SERVICE_ERROR event                                        │
│       │                                                                     │
│       └──► Controller sets phase=ERROR, shows user message                 │
│                                                                             │
│  User can click "Retry" to spawn new service                               │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                             │
│  BROWSER CLOSE (Daemon Behavior)                                            │
│  ───────────────────────────────                                            │
│                                                                             │
│  User closes browser abruptly                                               │
│       │                                                                     │
│       ▼                                                                     │
│  Streamlit server process ends                                              │
│       │                                                                     │
│       ▼                                                                     │
│  Daemon child process automatically killed by OS                           │
│       │                                                                     │
│       └──► No orphan processes                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Migration Component

### JSON to SQLite Migration

For backward compatibility with existing experiments:

```python
# model/migration.py

from pathlib import Path
from typing import Optional, Dict, Any
import json
import logging
from datetime import datetime

from .database import DatabaseManager

logger = logging.getLogger(__name__)


class MigrationManager:
    """
    Handles migration from legacy JSON state files to SQLite.
    
    Supports:
    - Detection of legacy format
    - One-time conversion to SQLite
    - Preservation of all historical data
    """
    
    LEGACY_STATE_FILE = "experiment_state.json"
    MIGRATION_MARKER = ".migrated"
    
    def __init__(self, db_manager: DatabaseManager):
        self._db = db_manager
    
    def check_and_migrate(self, experiment_dir: Path) -> bool:
        """
        Check for legacy format and migrate if needed.
        
        Args:
            experiment_dir: Path to experiment directory
            
        Returns:
            True if migration successful or not needed
        """
        legacy_file = experiment_dir / self.LEGACY_STATE_FILE
        migration_marker = experiment_dir / self.MIGRATION_MARKER
        
        # Already migrated
        if migration_marker.exists():
            return True
        
        # No legacy file
        if not legacy_file.exists():
            return True
        
        # Perform migration
        logger.info(f"Migrating legacy experiment: {experiment_dir}")
        
        try:
            success = self._migrate_json_to_sqlite(legacy_file)
            
            if success:
                # Create migration marker
                migration_marker.write_text(datetime.now().isoformat())
                
                # Rename old file (keep as backup)
                backup_file = legacy_file.with_suffix('.json.backup')
                legacy_file.rename(backup_file)
                
                logger.info(f"Migration complete. Backup saved to {backup_file}")
            
            return success
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
    
    def _migrate_json_to_sqlite(self, json_path: Path) -> bool:
        """
        Convert JSON state file to SQLite records.
        
        Args:
            json_path: Path to legacy JSON file
            
        Returns:
            True if migration successful
        """
        with open(json_path, 'r') as f:
            old_state = json.load(f)
        
        # Extract experiment info
        experiment_id = old_state.get('experiment_id', 'migrated_experiment')
        experiment_name = old_state.get('experiment_name', 'Migrated Experiment')
        created_at = old_state.get('created_at')
        
        # Create experiment record
        config = old_state.get('config', {})
        db_experiment_id = self._db.insert_experiment(
            experiment_id=experiment_id,
            name=experiment_name,
            config=config,
            created_at=created_at
        )
        
        # Migrate cycle results
        for cycle_result in old_state.get('cycle_results', []):
            self._migrate_cycle(db_experiment_id, cycle_result)
        
        # Migrate current cycle epochs (if any)
        current_cycle = old_state.get('current_cycle', 0)
        for epoch_data in old_state.get('current_cycle_epochs', []):
            self._db.insert_epoch_metrics(
                experiment_id=db_experiment_id,
                cycle=current_cycle,
                epoch=epoch_data.get('epoch', 0),
                train_loss=epoch_data.get('train_loss'),
                val_loss=epoch_data.get('val_loss'),
                train_accuracy=epoch_data.get('train_accuracy'),
                val_accuracy=epoch_data.get('val_accuracy'),
                learning_rate=epoch_data.get('learning_rate')
            )
        
        # Migrate pool information
        self._migrate_pools(db_experiment_id, old_state)
        
        return True
    
    def _migrate_cycle(self, experiment_id: int, cycle_data: Dict[str, Any]) -> None:
        """Migrate a single cycle's data."""
        cycle_num = cycle_data.get('cycle', 0)
        
        # Insert cycle summary
        self._db.insert_cycle_summary(
            experiment_id=experiment_id,
            cycle=cycle_num,
            labeled_count=cycle_data.get('labeled_pool_size'),
            unlabeled_count=cycle_data.get('unlabeled_pool_size'),
            best_val_accuracy=cycle_data.get('best_val_accuracy'),
            test_accuracy=cycle_data.get('test_accuracy'),
            test_f1=cycle_data.get('test_f1')
        )
        
        # Insert epoch history for this cycle
        for epoch_data in cycle_data.get('epoch_history', []):
            self._db.insert_epoch_metrics(
                experiment_id=experiment_id,
                cycle=cycle_num,
                epoch=epoch_data.get('epoch', 0),
                train_loss=epoch_data.get('train_loss'),
                val_loss=epoch_data.get('val_loss'),
                train_accuracy=epoch_data.get('train_accuracy'),
                val_accuracy=epoch_data.get('val_accuracy'),
                learning_rate=epoch_data.get('learning_rate')
            )
    
    def _migrate_pools(self, experiment_id: int, old_state: Dict[str, Any]) -> None:
        """Migrate pool information (if stored in old format)."""
        # This depends on how pools were stored in the old format
        # Adapt as needed for your specific legacy format
        pass
    
    def detect_legacy_format(self, experiment_dir: Path) -> bool:
        """Check if experiment uses old JSON format."""
        legacy_file = experiment_dir / self.LEGACY_STATE_FILE
        migration_marker = experiment_dir / self.MIGRATION_MARKER
        
        return legacy_file.exists() and not migration_marker.exists()
```

### Migration Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MIGRATION FLOW                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Application Start                                                          │
│       │                                                                     │
│       ▼                                                                     │
│  MigrationManager.check_and_migrate(experiment_dir)                        │
│       │                                                                     │
│       ├──► Check for .migrated marker                                      │
│       │         │                                                           │
│       │         └──► EXISTS ──► Skip (already migrated)                    │
│       │                                                                     │
│       ├──► Check for experiment_state.json                                 │
│       │         │                                                           │
│       │         └──► NOT EXISTS ──► Skip (new experiment)                  │
│       │                                                                     │
│       ▼                                                                     │
│  Legacy file found - perform migration                                      │
│       │                                                                     │
│       ├──► Read JSON file                                                  │
│       │                                                                     │
│       ├──► Create experiment record in SQLite                              │
│       │                                                                     │
│       ├──► Migrate cycle_results[]                                         │
│       │         │                                                           │
│       │         ├──► Insert cycle summaries                                │
│       │         └──► Insert epoch metrics                                  │
│       │                                                                     │
│       ├──► Migrate current_cycle_epochs[]                                  │
│       │                                                                     │
│       ├──► Create .migrated marker                                         │
│       │                                                                     │
│       └──► Rename JSON to .json.backup                                     │
│                                                                             │
│  Result:                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  experiment_dir/                                                     │   │
│  │  ├── experiment_state.json.backup  (preserved)                      │   │
│  │  ├── .migrated                      (marker)                        │   │
│  │  └── (SQLite DB now has all data)                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Components and Interfaces

### View Layer Components

#### Streamlit Pages
**Responsibility**: Pure UI presentation and user input collection

```python
# pages/1_Configuration.py

import streamlit as st
from controller import get_controller
from controller.events import Event, EventType


def render_configuration_page():
    """Render experiment configuration form."""
    ctrl = get_controller()
    
    st.header("Experiment Configuration")
    
    # Get current config and available options from controller
    config_view = ctrl.get_experiment_config_view()
    
    # Render form fields (pure UI)
    with st.form("experiment_config"):
        dataset = st.selectbox(
            "Dataset",
            options=config_view.available_datasets,
            index=0 if not config_view.current else 
                  config_view.available_datasets.index(config_view.current.dataset)
        )
        
        model_type = st.selectbox(
            "Model Architecture",
            options=config_view.available_models
        )
        
        strategy = st.selectbox(
            "Sampling Strategy",
            options=config_view.available_strategies
        )
        
        col1, col2 = st.columns(2)
        with col1:
            initial_samples = st.number_input("Initial Samples", min_value=10, value=100)
            batch_size = st.number_input("Batch Size", min_value=1, value=32)
        with col2:
            samples_per_cycle = st.number_input("Samples per Cycle", min_value=1, value=50)
            epochs_per_cycle = st.number_input("Epochs per Cycle", min_value=1, value=10)
        
        submitted = st.form_submit_button("Initialize Experiment")
    
    # Dispatch event on submit (no business logic here)
    if submitted:
        ctrl.dispatch(Event(
            type=EventType.INITIALIZE_EXPERIMENT,
            payload={
                "dataset": dataset,
                "model_type": model_type,
                "strategy": strategy,
                "initial_samples": initial_samples,
                "samples_per_cycle": samples_per_cycle,
                "batch_size": batch_size,
                "epochs_per_cycle": epochs_per_cycle
            }
        ))
        st.success("Experiment initialized!")
        st.rerun()


# Page entry point
render_configuration_page()
```

**Interface Contract**:
- MUST only handle UI rendering and user input
- MUST dispatch events for all user actions
- MUST NOT access Model or Service layers directly
- MUST request formatted data from Controller

### Controller Layer Components

#### EventDispatcher
**Responsibility**: Route events between components and coordinate system responses

```python
# controller/dispatcher.py

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Callable, Optional
from enum import Enum
import logging

from .events import Event, EventType
from .model_handler import ModelHandler
from .service_manager import ServiceManager

logger = logging.getLogger(__name__)


class EventDispatcher:
    """
    Central coordinator for the application.
    Routes events to appropriate handlers and manages component lifecycle.
    """
    
    def __init__(
        self,
        model_handler: ModelHandler,
        service_manager: ServiceManager
    ):
        self._model_handler = model_handler
        self._service_manager = service_manager
        self._handlers = self._register_handlers()
    
    def _register_handlers(self) -> Dict[EventType, Callable]:
        """Register event handlers."""
        return {
            # View -> Controller events
            EventType.INITIALIZE_EXPERIMENT: self._handle_initialize,
            EventType.START_CYCLE: self._handle_start_cycle,
            EventType.PAUSE_TRAINING: self._handle_pause,
            EventType.STOP_EXPERIMENT: self._handle_stop,
            EventType.SUBMIT_ANNOTATIONS: self._handle_annotations,
            
            # Service -> Controller events
            EventType.SERVICE_READY: self._handle_service_ready,
            EventType.EPOCH_COMPLETE: self._handle_epoch_complete,
            EventType.CYCLE_COMPLETE: self._handle_cycle_complete,
            EventType.QUERY_READY: self._handle_query_ready,
            EventType.SERVICE_ERROR: self._handle_service_error,
        }
    
    def dispatch(self, event: Event) -> None:
        """
        Dispatch event to appropriate handler.
        
        Args:
            event: Event to dispatch
        """
        handler = self._handlers.get(event.type)
        
        if handler:
            try:
                logger.debug(f"Dispatching event: {event.type}")
                handler(event)
            except Exception as e:
                logger.error(f"Error handling event {event.type}: {e}")
                self._model_handler.set_error(str(e))
        else:
            logger.warning(f"No handler for event type: {event.type}")
    
    # === View Event Handlers ===
    
    def _handle_initialize(self, event: Event) -> None:
        """Handle experiment initialization."""
        config = event.payload
        
        # Validate configuration
        validation = self._model_handler.validate_config(config)
        if not validation.is_valid:
            self._model_handler.set_error(validation.error_message)
            return
        
        # Initialize experiment in model
        self._model_handler.initialize_experiment(config)
        
        # Spawn service process
        success = self._service_manager.spawn_service(
            config=config,
            event_callback=self.dispatch
        )
        
        if not success:
            self._model_handler.set_error("Failed to start service process")
    
    def _handle_start_cycle(self, event: Event) -> None:
        """Handle start training cycle."""
        self._model_handler.set_phase("TRAINING")
        self._service_manager.send_command(Event(EventType.CMD_START_CYCLE))
    
    def _handle_pause(self, event: Event) -> None:
        """Handle pause training."""
        self._model_handler.set_phase("PAUSED")
        self._service_manager.send_command(Event(EventType.CMD_PAUSE))
    
    def _handle_stop(self, event: Event) -> None:
        """Handle stop experiment."""
        self._model_handler.set_phase("STOPPED")
        self._service_manager.send_command(Event(EventType.CMD_STOP))
    
    def _handle_annotations(self, event: Event) -> None:
        """Handle annotation submission."""
        annotations = event.payload.get('annotations', [])
        self._model_handler.save_annotations(annotations)
        self._service_manager.send_command(Event(
            EventType.CMD_ANNOTATIONS,
            payload={'annotations': annotations}
        ))
    
    # === Service Event Handlers ===
    
    def _handle_service_ready(self, event: Event) -> None:
        """Handle service ready notification."""
        self._model_handler.set_phase("IDLE")
        logger.info("Service is ready")
    
    def _handle_epoch_complete(self, event: Event) -> None:
        """Handle epoch completion from service."""
        metrics = event.payload
        
        # Update WorldState (fast, for UI)
        self._model_handler.update_current_metrics(metrics)
        
        # Persist to SQLite (durable)
        self._model_handler.persist_epoch_metrics(metrics)
        
        # Set flag for UI refresh
        self._model_handler.set_pending_updates(True)
    
    def _handle_cycle_complete(self, event: Event) -> None:
        """Handle cycle completion from service."""
        results = event.payload
        
        self._model_handler.finalize_cycle(results)
        self._model_handler.set_phase("IDLE")
    
    def _handle_query_ready(self, event: Event) -> None:
        """Handle query samples ready."""
        images = event.payload.get('images', [])
        
        self._model_handler.set_queried_images(images)
        self._model_handler.set_phase("AWAITING_ANNOTATION")
    
    def _handle_service_error(self, event: Event) -> None:
        """Handle service error."""
        error_msg = event.payload.get('message', 'Unknown error')
        logger.error(f"Service error: {error_msg}")
        
        self._model_handler.set_error(error_msg)
        self._model_handler.set_phase("ERROR")
    
    # === Data Access Methods (for View) ===
    
    def get_status(self) -> dict:
        """Get current experiment status."""
        return self._model_handler.get_status()
    
    def get_training_progress(self) -> dict:
        """Get current training progress."""
        return self._model_handler.get_training_progress()
    
    def get_experiment_config_view(self) -> 'ExperimentConfigView':
        """Get experiment configuration view."""
        return self._model_handler.get_experiment_config_view()
    
    def get_queried_images(self) -> list:
        """Get current queried images for annotation."""
        return self._model_handler.get_queried_images()
    
    def get_results_history(self, page: int = 0, limit: int = 50) -> dict:
        """Get paginated results history."""
        return self._model_handler.get_results_history(page, limit)
    
    def has_pending_updates(self) -> bool:
        """Check if there are pending UI updates."""
        return self._model_handler.has_pending_updates()
    
    def clear_pending_updates(self) -> None:
        """Clear pending updates flag."""
        self._model_handler.clear_pending_updates()
    
    def shutdown(self) -> None:
        """Shutdown the controller and cleanup resources."""
        self._service_manager.shutdown()
```

#### ModelHandler
**Responsibility**: Provide view-appropriate data and manage state updates

```python
# controller/model_handler.py

from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from ..model.world_state import WorldState
from ..model.database import DatabaseManager


@dataclass
class ExperimentConfigView:
    """View model for experiment configuration."""
    current: Optional[dict]
    available_datasets: List[str]
    available_models: List[str]
    available_strategies: List[str]


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    error_message: Optional[str] = None


class ModelHandler:
    """
    Provides view-appropriate data and manages state updates.
    
    This is the ONLY component that directly accesses WorldState and Database.
    """
    
    def __init__(self, world_state: WorldState, db_manager: DatabaseManager):
        self._world_state = world_state
        self._db = db_manager
    
    # === State Accessors (from WorldState - fast) ===
    
    def get_status(self) -> dict:
        """Get current experiment status from WorldState."""
        return {
            "phase": self._world_state.phase,
            "current_cycle": self._world_state.current_cycle,
            "total_cycles": self._world_state.total_cycles,
            "current_epoch": self._world_state.current_epoch,
            "error_message": self._world_state.error_message
        }
    
    def get_training_progress(self) -> dict:
        """Get current training progress from WorldState."""
        ws = self._world_state
        return {
            "current_epoch": ws.current_epoch,
            "total_epochs": ws.epochs_per_cycle,
            "train_loss": ws.current_metrics.train_loss if ws.current_metrics else None,
            "val_loss": ws.current_metrics.val_loss if ws.current_metrics else None,
            "val_accuracy": ws.current_metrics.val_accuracy if ws.current_metrics else None,
            "progress_percentage": (ws.current_epoch / ws.epochs_per_cycle * 100) if ws.epochs_per_cycle else 0,
            "epoch_history": ws.epoch_history,
            "loss_history_df": self._build_loss_history_df()
        }
    
    def get_queried_images(self) -> list:
        """Get current queried images from WorldState."""
        return self._world_state.queried_images
    
    def has_pending_updates(self) -> bool:
        """Check pending updates flag."""
        return self._world_state.pending_updates
    
    def clear_pending_updates(self) -> None:
        """Clear pending updates flag."""
        self._world_state.pending_updates = False
    
    # === State Mutators ===
    
    def set_phase(self, phase: str) -> None:
        """Set experiment phase."""
        self._world_state.phase = phase
        self._db.update_experiment_phase(self._world_state.experiment_id, phase)
    
    def set_error(self, message: str) -> None:
        """Set error state."""
        self._world_state.error_message = message
        self._world_state.phase = "ERROR"
    
    def set_pending_updates(self, value: bool) -> None:
        """Set pending updates flag."""
        self._world_state.pending_updates = value
    
    def update_current_metrics(self, metrics: dict) -> None:
        """Update current metrics in WorldState."""
        self._world_state.current_metrics = metrics
        self._world_state.current_epoch = metrics.get('epoch', 0)
        self._world_state.epoch_history.append(metrics)
    
    def persist_epoch_metrics(self, metrics: dict) -> None:
        """Persist epoch metrics to SQLite."""
        self._db.insert_epoch_metrics(
            experiment_id=self._world_state.experiment_id,
            cycle=self._world_state.current_cycle,
            **metrics
        )
    
    def set_queried_images(self, images: list) -> None:
        """Set queried images in WorldState."""
        self._world_state.queried_images = images
    
    def finalize_cycle(self, results: dict) -> None:
        """Finalize cycle in both WorldState and SQLite."""
        # Update WorldState
        self._world_state.current_cycle += 1
        self._world_state.epoch_history = []  # Clear for next cycle
        self._world_state.queried_images = []
        
        # Persist to SQLite
        self._db.insert_cycle_summary(
            experiment_id=self._world_state.experiment_id,
            cycle=results.get('cycle'),
            **results
        )
    
    # === Database Accessors (from SQLite - paginated) ===
    
    def get_results_history(self, page: int, limit: int) -> dict:
        """Get paginated results history from SQLite."""
        experiments = self._db.get_experiments_paginated(page, limit)
        total = self._db.get_experiments_count()
        
        return {
            "experiments": experiments,
            "total": total,
            "page": page,
            "has_next": (page + 1) * limit < total
        }
    
    def get_experiment_config_view(self) -> ExperimentConfigView:
        """Get experiment configuration view."""
        return ExperimentConfigView(
            current=self._world_state.config,
            available_datasets=["CIFAR-10", "MNIST", "Custom"],
            available_models=["ResNet18", "ResNet50", "VGG16"],
            available_strategies=["uncertainty", "margin", "entropy", "random"]
        )
    
    # === Validation ===
    
    def validate_config(self, config: dict) -> ValidationResult:
        """Validate experiment configuration."""
        errors = []
        
        if config.get('initial_samples', 0) < 10:
            errors.append("Initial samples must be at least 10")
        
        if config.get('batch_size', 0) < 1:
            errors.append("Batch size must be at least 1")
        
        if errors:
            return ValidationResult(is_valid=False, error_message="; ".join(errors))
        
        return ValidationResult(is_valid=True)
    
    # === Initialization ===
    
    def initialize_experiment(self, config: dict) -> None:
        """Initialize new experiment."""
        # Create in database
        experiment_id = self._db.insert_experiment(config)
        
        # Initialize WorldState
        self._world_state.reset()
        self._world_state.experiment_id = experiment_id
        self._world_state.config = config
        self._world_state.total_cycles = config.get('max_cycles', 10)
        self._world_state.epochs_per_cycle = config.get('epochs_per_cycle', 10)
        self._world_state.phase = "INITIALIZING"
```

### Model Layer Components

#### WorldState
**Responsibility**: In-memory current experiment state for fast access

```python
# model/world_state.py

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from threading import Lock


@dataclass
class EpochMetrics:
    """Metrics for a single epoch."""
    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    learning_rate: Optional[float] = None


@dataclass
class WorldState:
    """
    In-memory current experiment state.
    
    Provides fast access for UI components.
    Synchronized with SQLite for persistence.
    """
    
    # Identity
    experiment_id: Optional[int] = None
    experiment_name: str = ""
    
    # Configuration
    config: Optional[Dict[str, Any]] = None
    
    # Phase
    phase: str = "IDLE"
    error_message: Optional[str] = None
    
    # Progress
    current_cycle: int = 0
    total_cycles: int = 0
    current_epoch: int = 0
    epochs_per_cycle: int = 0
    
    # Current metrics (latest values)
    current_metrics: Optional[Dict[str, Any]] = None
    
    # Epoch history (current cycle only)
    epoch_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Queried images (current batch)
    queried_images: List[Dict[str, Any]] = field(default_factory=list)
    
    # UI update flag
    pending_updates: bool = False
    
    # Thread safety
    _lock: Lock = field(default_factory=Lock, repr=False)
    
    def reset(self) -> None:
        """Reset to initial state."""
        self.experiment_id = None
        self.experiment_name = ""
        self.config = None
        self.phase = "IDLE"
        self.error_message = None
        self.current_cycle = 0
        self.total_cycles = 0
        self.current_epoch = 0
        self.epochs_per_cycle = 0
        self.current_metrics = None
        self.epoch_history = []
        self.queried_images = []
        self.pending_updates = False
    
    def restore_from_db(self, experiment_data: dict) -> None:
        """Restore state from database record."""
        self.experiment_id = experiment_data.get('id')
        self.experiment_name = experiment_data.get('name', '')
        self.config = experiment_data.get('config')
        self.phase = experiment_data.get('phase', 'IDLE')
        self.current_cycle = experiment_data.get('current_cycle', 0)
        self.total_cycles = experiment_data.get('total_cycles', 0)
```

#### Database Schema
**Responsibility**: Persistent storage for historical data and configuration

```sql
-- Experiments table
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT UNIQUE,
    name TEXT,
    config_json TEXT NOT NULL,
    phase TEXT NOT NULL DEFAULT 'IDLE',
    current_cycle INTEGER DEFAULT 0,
    total_cycles INTEGER DEFAULT 0,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Cycle summaries table
CREATE TABLE IF NOT EXISTS cycle_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    cycle INTEGER NOT NULL,
    labeled_count INTEGER,
    unlabeled_count INTEGER,
    best_val_accuracy REAL,
    test_accuracy REAL,
    test_f1 REAL,
    test_precision REAL,
    test_recall REAL,
    completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id),
    UNIQUE(experiment_id, cycle)
);

-- Epoch metrics table
CREATE TABLE IF NOT EXISTS epoch_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    cycle INTEGER NOT NULL,
    epoch INTEGER NOT NULL,
    train_loss REAL,
    val_loss REAL,
    train_accuracy REAL,
    val_accuracy REAL,
    learning_rate REAL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

-- Pool items table (for large datasets)
CREATE TABLE IF NOT EXISTS pool_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    image_path TEXT NOT NULL,
    pool_type TEXT NOT NULL,  -- 'labeled', 'unlabeled', 'val', 'test'
    class_idx INTEGER,
    class_name TEXT,
    added_at_cycle INTEGER,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

-- Queried images table
CREATE TABLE IF NOT EXISTS queried_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    cycle INTEGER NOT NULL,
    image_path TEXT NOT NULL,
    ground_truth INTEGER,
    predicted_class INTEGER,
    confidence REAL,
    uncertainty_score REAL,
    user_annotation INTEGER,
    annotated_at TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_experiments_phase ON experiments(phase);
CREATE INDEX IF NOT EXISTS idx_cycle_summaries_experiment ON cycle_summaries(experiment_id);
CREATE INDEX IF NOT EXISTS idx_epoch_metrics_experiment_cycle ON epoch_metrics(experiment_id, cycle);
CREATE INDEX IF NOT EXISTS idx_pool_items_experiment_type ON pool_items(experiment_id, pool_type);
CREATE INDEX IF NOT EXISTS idx_queried_images_experiment_cycle ON queried_images(experiment_id, cycle);
```

### Service Layer Components

#### ActiveLearningService
**Responsibility**: Execute training in isolated process with event communication

```python
# services/al_service.py

from multiprocessing.connection import Connection
from typing import Optional, Dict, Any
from threading import Thread, Event as ThreadEvent
import logging
import traceback

from ..controller.events import Event, EventType

logger = logging.getLogger(__name__)


def run_active_learning_service(pipe: Connection, config: dict) -> None:
    """
    Entry point for service process.
    Called by ServiceManager.spawn_service().
    """
    service = ActiveLearningService(pipe, config)
    service.run()


class ActiveLearningService:
    """
    Executes Active Learning training in an isolated process.
    
    Communicates with Controller via Pipe:
    - Receives: Commands (START_CYCLE, PAUSE, STOP, etc.)
    - Sends: Events (EPOCH_COMPLETE, QUERY_READY, etc.)
    """
    
    def __init__(self, pipe: Connection, config: dict):
        self._pipe = pipe
        self._config = config
        
        # Control flags
        self._should_stop = False
        self._is_paused = False
        
        # Backend components (initialized on first use)
        self._trainer = None
        self._data_manager = None
        self._al_loop = None
    
    def run(self) -> None:
        """Main service loop."""
        try:
            self._initialize_components()
            self._send_event(EventType.SERVICE_READY, {"status": "ready"})
            
            while not self._should_stop:
                self._process_commands()
                
        except Exception as e:
            logger.error(f"Service error: {e}\n{traceback.format_exc()}")
            self._send_event(EventType.SERVICE_ERROR, {
                "message": str(e),
                "traceback": traceback.format_exc()
            })
    
    def _process_commands(self) -> None:
        """Process incoming commands from Controller."""
        try:
            if self._pipe.poll(timeout=0.5):
                event = self._pipe.recv()
                self._handle_command(event)
        except EOFError:
            logger.info("Pipe closed, shutting down")
            self._should_stop = True
    
    def _handle_command(self, event: Event) -> None:
        """Handle command from Controller."""
        logger.debug(f"Received command: {event.type}")
        
        if event.type == EventType.CMD_START_CYCLE:
            self._execute_training_cycle()
        elif event.type == EventType.CMD_PAUSE:
            self._is_paused = True
        elif event.type == EventType.CMD_RESUME:
            self._is_paused = False
        elif event.type == EventType.CMD_STOP:
            self._should_stop = True
        elif event.type == EventType.CMD_ANNOTATIONS:
            self._process_annotations(event.payload)
        elif event.type == EventType.CMD_SHUTDOWN:
            self._shutdown()
    
    def _execute_training_cycle(self) -> None:
        """Execute one Active Learning cycle."""
        cycle = self._al_loop.current_cycle
        
        # Prepare cycle
        self._al_loop.prepare_cycle()
        
        # Training loop
        for epoch in range(self._config.get('epochs_per_cycle', 10)):
            # Check for stop/pause
            if self._should_stop:
                break
            
            while self._is_paused:
                if self._pipe.poll(timeout=0.5):
                    event = self._pipe.recv()
                    self._handle_command(event)
                if self._should_stop:
                    break
            
            # Train one epoch
            metrics = self._trainer.train_single_epoch()
            
            # Send progress event
            self._send_event(EventType.EPOCH_COMPLETE, {
                "cycle": cycle,
                "epoch": epoch,
                "train_loss": metrics['train_loss'],
                "val_loss": metrics['val_loss'],
                "train_accuracy": metrics['train_accuracy'],
                "val_accuracy": metrics['val_accuracy']
            })
        
        # Query phase
        queried_images = self._al_loop.query_samples()
        
        self._send_event(EventType.QUERY_READY, {
            "cycle": cycle,
            "images": queried_images
        })
    
    def _process_annotations(self, payload: dict) -> None:
        """Process user annotations and continue cycle."""
        annotations = payload.get('annotations', [])
        
        # Update data manager with annotations
        self._al_loop.receive_annotations(annotations)
        
        # Finalize cycle
        results = self._al_loop.finalize_cycle()
        
        self._send_event(EventType.CYCLE_COMPLETE, results)
    
    def _send_event(self, event_type: EventType, payload: dict) -> None:
        """Send event to Controller."""
        try:
            self._pipe.send(Event(type=event_type, payload=payload))
        except Exception as e:
            logger.error(f"Failed to send event: {e}")
    
    def _initialize_components(self) -> None:
        """Initialize backend components."""
        from ..backend.trainer import Trainer
        from ..backend.data_manager import ALDataManager
        from ..backend.active_loop import ActiveLearningLoop
        
        # Initialize (your existing code)
        self._data_manager = ALDataManager(self._config)
        self._trainer = Trainer(self._config)
        self._al_loop = ActiveLearningLoop(
            trainer=self._trainer,
            data_manager=self._data_manager,
            config=self._config
        )
    
    def _shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Service shutting down...")
        self._should_stop = True
        # Save any pending state if needed
```

---

## Data Models

### Event Definitions

```python
# controller/events.py

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, Any, Optional


class EventType(Enum):
    """All events in the system."""
    
    # View → Controller
    INITIALIZE_EXPERIMENT = auto()
    START_CYCLE = auto()
    PAUSE_TRAINING = auto()
    RESUME_TRAINING = auto()
    STOP_EXPERIMENT = auto()
    SUBMIT_ANNOTATIONS = auto()
    
    # Controller → Service (Commands)
    CMD_START_CYCLE = auto()
    CMD_PAUSE = auto()
    CMD_RESUME = auto()
    CMD_STOP = auto()
    CMD_ANNOTATIONS = auto()
    CMD_SHUTDOWN = auto()
    
    # Service → Controller
    SERVICE_READY = auto()
    EPOCH_COMPLETE = auto()
    TRAINING_COMPLETE = auto()
    QUERY_READY = auto()
    CYCLE_COMPLETE = auto()
    EXPERIMENT_COMPLETE = auto()
    SERVICE_ERROR = auto()


@dataclass
class Event:
    """An event in the system."""
    type: EventType
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
```

---

## Correctness Properties

*Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: State Update Consistency
*For any* state change event, when the Controller processes it, both WorldState and SQLite SHALL be updated consistently with the same data.
**Validates: Requirements 1.3, 2.4, 7.3**

### Property 2: Event Dispatch Performance
*For any* event dispatched to the EventDispatcher, the routing to the appropriate handler SHALL complete within 10ms.
**Validates: Requirements 2.2**

### Property 3: UI Response Time Performance
*For any* user action or UI update request, the system SHALL provide feedback or complete the update within the specified time limits (100ms for actions, 500ms for updates, 2s for page loads).
**Validates: Requirements 10.1, 10.2, 10.3**

### Property 4: Data Access Performance
*For any* data request, current state access from WorldState SHALL complete within 1ms and historical data queries from SQLite SHALL complete within 50ms.
**Validates: Requirements 1.4, 1.5**

### Property 5: Event-Driven Communication
*For any* user action in the ViewLayer, an event SHALL be dispatched to the ControllerLayer without executing business logic.
**Validates: Requirements 2.1, 4.2**

### Property 6: Service Communication
*For any* background task completion in the ServiceLayer, progress or completion events SHALL be emitted via Pipe communication to update system state.
**Validates: Requirements 2.3, 6.2**

### Property 7: Single-User Enforcement
*For any* session attempt when another session is active, the system SHALL warn the user and recommend single-tab usage.
**Validates: Requirements 10.5, 10.6**

### Property 8: Process Lifecycle Management
*For any* service spawn request, existing processes SHALL be terminated before new process creation, and daemon configuration SHALL ensure cleanup on parent exit.
**Validates: Requirements 3.1, 3.4**

### Property 9: Migration Compatibility
*For any* legacy JSON state file, the system SHALL detect and migrate it to SQLite format while preserving all historical data.
**Validates: Requirements 8.1, 8.2**

### Property 10: Error Handling and Recovery
*For any* error condition, the system SHALL handle it gracefully with appropriate logging, user feedback, and recovery mechanisms.
**Validates: Requirements 5.5, 9.2, 9.3, 9.5**

---

## Error Handling

### Error Categories and Responses

| Error Type | Detection | Response | Recovery |
|------------|-----------|----------|----------|
| Service Crash | Listener thread detects `is_alive() == False` | Set phase=ERROR, notify user | User clicks "Retry" |
| Communication Failure | `BrokenPipeError` or timeout | Retry with backoff, then ERROR | Auto-restart service |
| Validation Error | ModelHandler.validate_config() | Show error message | User corrects input |
| Database Error | SQLite exception | Fallback to WorldState, log error | Auto-retry on next access |
| Multi-Tab Conflict | SessionManager detection | Show warning, block operation | User closes other tabs |

### Error Recovery Strategy

```python
class ErrorRecoveryManager:
    """Centralized error recovery handling."""
    
    MAX_RETRIES = 3
    BACKOFF_BASE = 2  # seconds
    
    def handle_service_failure(self, error: Exception) -> None:
        """Handle service process failures."""
        logger.error(f"Service failure: {error}")
        
        if self._is_recoverable(error):
            self._attempt_restart_with_backoff()
        else:
            self._notify_user_critical_failure(error)
    
    def handle_communication_failure(self, error: Exception) -> bool:
        """
        Handle communication failures with retry.
        Returns True if recovered, False if failed.
        """
        for attempt in range(self.MAX_RETRIES):
            wait_time = self.BACKOFF_BASE ** attempt
            logger.info(f"Retry attempt {attempt + 1} in {wait_time}s")
            
            time.sleep(wait_time)
            
            if self._attempt_reconnect():
                return True
        
        return False
```

---

## Testing Strategy

### Property-Based Testing

```python
from hypothesis import given, strategies as st
import pytest


@given(st.dictionaries(st.text(), st.integers()))
def test_state_update_consistency(config_data):
    """
    Property 1: State Update Consistency
    
    For any state change, WorldState and SQLite must be consistent.
    """
    # Setup
    world_state = WorldState()
    db = DatabaseManager(":memory:")
    handler = ModelHandler(world_state, db)
    
    # Action
    handler.initialize_experiment(config_data)
    
    # Verify consistency
    ws_config = world_state.config
    db_config = db.get_experiment(world_state.experiment_id)['config']
    
    assert ws_config == db_config


@given(st.integers(min_value=1, max_value=100))
def test_data_access_performance(num_epochs):
    """
    Property 4: Data Access Performance
    
    WorldState access < 1ms, SQLite access < 50ms.
    """
    # ... performance test implementation
```

### Integration Testing

```python
def test_complete_training_cycle():
    """End-to-end test of training cycle."""
    # Initialize
    ctrl = create_test_controller()
    
    # Configure experiment
    ctrl.dispatch(Event(EventType.INITIALIZE_EXPERIMENT, {...}))
    assert ctrl.get_status()['phase'] == 'IDLE'
    
    # Start training
    ctrl.dispatch(Event(EventType.START_CYCLE))
    
    # Wait for completion (with timeout)
    wait_for_phase(ctrl, 'AWAITING_ANNOTATION', timeout=60)
    
    # Submit annotations
    ctrl.dispatch(Event(EventType.SUBMIT_ANNOTATIONS, {...}))
    
    # Verify cycle complete
    wait_for_phase(ctrl, 'IDLE', timeout=10)
    assert ctrl.get_status()['current_cycle'] == 1
```

---

## File Structure

```
project/
├── dashboard.py                    # Entry point
│
├── pages/                          # VIEW LAYER
│   ├── __init__.py
│   ├── 1_Configuration.py
│   ├── 2_Active_Learning.py
│   └── 3_Results.py
│
├── controller/                     # CONTROLLER LAYER
│   ├── __init__.py                 # get_controller()
│   ├── events.py                   # Event, EventType
│   ├── dispatcher.py               # EventDispatcher
│   ├── model_handler.py            # ModelHandler
│   ├── service_manager.py          # ServiceManager
│   └── session_manager.py          # SessionManager (multi-tab detection)
│
├── model/                          # MODEL LAYER
│   ├── __init__.py
│   ├── world_state.py              # WorldState
│   ├── database.py                 # DatabaseManager
│   ├── migration.py                # MigrationManager
│   └── schemas.py                  # Data structures
│
├── services/                       # SERVICE LAYER
│   ├── __init__.py
│   └── al_service.py               # ActiveLearningService
│
├── backend/                        # EXISTING (UNCHANGED)
│   ├── __init__.py
│   ├── trainer.py
│   ├── active_loop.py
│   ├── data_manager.py
│   ├── strategies.py
│   ├── models.py
│   └── dataloader.py
│
└── config.py                       # Configuration
```

---

*End of Design Document*
