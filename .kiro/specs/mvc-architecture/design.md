# Design Document: Active Learning Framework

## Project Overview

This is a **Visual and Interactive Active Learning Framework** for vehicle image classification, built as a Bachelor Thesis project. The system enables researchers to:

- Run Active Learning experiments with real-time visualization
- Compare different sampling strategies (Entropy, Margin, Least Confidence, Random)
- Interactively annotate uncertain samples through a "Gallery of Uncertainty"
- Track model performance across AL cycles

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         STREAMLIT APPLICATION                            │
│                                                                         │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────────────┐│
│  │   SIDEBAR   │    │    VIEWS     │    │      RESULT DASHBOARD       ││
│  │             │    │              │    │                             ││
│  │ • Model     │    │ • Training   │    │ • Accuracy Charts           ││
│  │ • Strategy  │    │ • Gallery    │    │ • Strategy Comparison       ││
│  │ • Params    │    │ • Explorer   │    │ • Confusion Matrix          ││
│  └─────────────┘    └──────────────┘    └─────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                │ User Actions
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           CONTROLLER                                     │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     STATE MACHINE                                │   │
│  │                                                                  │   │
│  │   IDLE ──► TRAINING ──► QUERYING ──► ANNOTATING ──┐             │   │
│  │     ▲                                              │             │   │
│  │     └──────────────────────────────────────────────┘             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  • Dispatch commands to Worker                                          │
│  • Poll results from Worker                                             │
│  • Persist state to state.json                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
              task_queue              result_queue
              mp.Events               mp.Events
                    │                       │
                    ▼                       │
┌─────────────────────────────────────────────────────────────────────────┐
│                      WORKER PROCESS (GPU)                                │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   ACTIVE LEARNING LOOP                           │   │
│  │                                                                  │   │
│  │   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │   │
│  │   │  MODEL   │   │ TRAINER  │   │   DATA   │   │ STRATEGY │    │   │
│  │   │  (TIMM)  │   │          │   │ MANAGER  │   │          │    │   │
│  │   └──────────┘   └──────────┘   └──────────┘   └──────────┘    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## MVC Pattern Implementation

### Model (Worker Process)

The **Worker Process** runs in a separate process for GPU isolation:

- Builds all AL components at startup (eager initialization)
- Handles training, querying, and annotation processing
- Communicates via message queues (no shared state)
- Reports progress updates for real-time visualization

### View (Streamlit UI)

The **Streamlit Dashboard** provides interactive visualization:

- Sidebar for configuration (model, strategy, hyperparameters)
- Live training charts (loss, accuracy per epoch)
- Gallery of Uncertainty for annotation
- Results dashboard with strategy comparison

### Controller (State Machine)

The **Controller** orchestrates the application:

- Manages state transitions (IDLE → TRAINING → QUERYING → ANNOTATING)
- Dispatches commands to Worker via task_queue
- Polls results from Worker via result_queue
- Persists state for experiment resumption

---

## Inter-Process Communication

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    COMMUNICATION LAYERS                                  │
│                                                                         │
│  LAYER 1: mp.Event() — Lightweight Signals                              │
│  ─────────────────────────────────────────                              │
│  • worker_initialized  → Worker ready                                   │
│  • training_done       → Training complete                              │
│  • stop_requested      → User clicked stop                              │
│                                                                         │
│  LAYER 2: mp.Queue (dict messages) — Rich Data                          │
│  ─────────────────────────────────────────────                          │
│  • PROGRESS_UPDATE     → {epoch, loss, accuracy}                        │
│  • QUERY_COMPLETE      → {queried_images: [...]}                        │
│  • CYCLE_COMPLETE      → {metrics: {...}}                               │
│                                                                         │
│  LAYER 3: Filesystem — Persistence                                      │
│  ─────────────────────────────────────                                  │
│  • state.json          → Application state                              │
│  • config.yaml         → Experiment configuration                       │
│  • checkpoints/*.pth   → Model weights                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Active Learning Cycle Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ACTIVE LEARNING CYCLE                                │
│                                                                         │
│   ┌─────────┐                                                           │
│   │  START  │                                                           │
│   └────┬────┘                                                           │
│        │                                                                │
│        ▼                                                                │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  1. PREPARE CYCLE                                                │  │
│   │     • Reset model weights (pretrained/head_only/none)           │  │
│   │     • Create training DataLoader from labeled pool              │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│        │                                                                │
│        ▼                                                                │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  2. TRAINING                                                     │  │
│   │     • Train for N epochs on labeled data                        │  │
│   │     • Validate on fixed validation set                          │  │
│   │     • Early stopping if no improvement                          │  │
│   │     • Send PROGRESS_UPDATE after each epoch                     │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│        │                                                                │
│        ▼                                                                │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  3. EVALUATION                                                   │  │
│   │     • Evaluate on test set                                      │  │
│   │     • Compute accuracy, F1, precision, recall                   │  │
│   │     • Save best model checkpoint                                │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│        │                                                                │
│        ▼                                                                │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  4. QUERY (Sampling Strategy)                                    │  │
│   │     • Run inference on unlabeled pool                           │  │
│   │     • Apply strategy (entropy/margin/LC/random)                 │  │
│   │     • Select top-K most uncertain samples                       │  │
│   │     • Build QueriedImage objects with full info                 │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│        │                                                                │
│        ▼                                                                │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  5. ANNOTATION (Human-in-the-Loop)                               │  │
│   │     • Display "Gallery of Uncertainty" in UI                    │  │
│   │     • User provides labels (or auto-label with ground truth)    │  │
│   │     • Move annotated samples to labeled pool                    │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│        │                                                                │
│        ▼                                                                │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  6. REPEAT                                                       │  │
│   │     • Continue until all cycles complete                        │  │
│   │     • Or unlabeled pool exhausted                               │  │
│   └─────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Pool Management

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      DATA POOL ARCHITECTURE                              │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    TRAINING DATASET                              │   │
│  │                    (Stanford Cars)                               │   │
│  │                                                                  │   │
│  │   ┌─────────────────────┐    ┌─────────────────────────────┐   │   │
│  │   │    LABELED POOL     │    │      UNLABELED POOL         │   │   │
│  │   │                     │    │                             │   │   │
│  │   │  Initial: 100       │    │  Initial: ~15,000           │   │   │
│  │   │  Grows each cycle   │◄───│  Shrinks each cycle         │   │   │
│  │   │                     │    │                             │   │   │
│  │   │  [idx: 5, 12, 99]   │    │  [idx: 0, 1, 2, 3, ...]     │   │   │
│  │   └─────────────────────┘    └─────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────┐    ┌─────────────────────┐                    │
│  │   VALIDATION SET    │    │      TEST SET       │                    │
│  │   (Fixed, 15%)      │    │   (Fixed, 15%)      │                    │
│  └─────────────────────┘    └─────────────────────┘                    │
│                                                                         │
│  Key Design: Index-based management (no data copying)                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Sampling Strategies

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     SAMPLING STRATEGIES                                  │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  ENTROPY SAMPLING                                                │   │
│  │  ─────────────────                                               │   │
│  │  Score = -Σ p(y) × log(p(y))                                    │   │
│  │  High entropy = uncertainty spread across classes                │   │
│  │  Best for: Multi-class ambiguity                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  MARGIN SAMPLING                                                 │   │
│  │  ───────────────                                                 │   │
│  │  Score = P(top1) - P(top2)                                      │   │
│  │  Small margin = can't decide between two classes                 │   │
│  │  Best for: Similar class pairs (BMW vs Audi)                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  LEAST CONFIDENCE                                                │   │
│  │  ────────────────                                                │   │
│  │  Score = 1 - max(p(y))                                          │   │
│  │  Low confidence = model unsure about top prediction              │   │
│  │  Best for: General uncertainty                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  RANDOM (Baseline)                                               │   │
│  │  ────────────────                                                │   │
│  │  Uniform random selection                                        │   │
│  │  Used for comparison baseline                                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## UI Mockups

### Main Dashboard Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│  🎯 Active Learning Framework                                            │
├─────────────────┬───────────────────────────────────────────────────────┤
│                 │                                                       │
│  ⚙️ CONFIG      │   📊 TRAINING PROGRESS                                │
│  ─────────────  │   ────────────────────                                │
│                 │                                                       │
│  Model:         │   Cycle 3/10  ████████░░░░░░░░░░░░  30%              │
│  [ResNet-50 ▼]  │                                                       │
│                 │   ┌─────────────────────────────────────────────┐    │
│  Strategy:      │   │  Loss                    Accuracy           │    │
│  [Entropy   ▼]  │   │   ╲                         ╱               │    │
│                 │   │    ╲                       ╱                │    │
│  Epochs: [5]    │   │     ╲_____               ╱                  │    │
│  Batch:  [32]   │   │           ╲_____   _____╱                   │    │
│  LR: [1e-4]     │   │                 ╲_╱                         │    │
│                 │   │                                             │    │
│  Query Size:    │   │  Epoch 1  2  3  4  5                        │    │
│  [50]           │   └─────────────────────────────────────────────┘    │
│                 │                                                       │
│  Reset Mode:    │   Pool Status:                                        │
│  [Pretrained▼]  │   Labeled: 250 (1.5%)  │  Unlabeled: 15,935 (98.5%) │
│                 │                                                       │
│  ─────────────  │   Current Metrics:                                    │
│  [▶ Start]      │   Val Acc: 67.2%  │  Test Acc: 65.8%  │  F1: 0.64   │
│  [⏹ Stop]       │                                                       │
│                 │                                                       │
└─────────────────┴───────────────────────────────────────────────────────┘
```

### Gallery of Uncertainty (Annotation View)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  🔍 Gallery of Uncertainty — Cycle 3                                     │
│  ───────────────────────────────────────────────────────────────────────│
│                                                                         │
│  50 samples selected by Entropy strategy                                │
│                                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │  [Image 1]  │  │  [Image 2]  │  │  [Image 3]  │  │  [Image 4]  │   │
│  │             │  │             │  │             │  │             │   │
│  │  🔴 High    │  │  🟠 Medium  │  │  🟠 Medium  │  │  🟡 Low     │   │
│  │  Entropy    │  │  Entropy    │  │  Entropy    │  │  Entropy    │   │
│  │             │  │             │  │             │  │             │   │
│  │  Pred: BMW  │  │  Pred: Audi │  │  Pred: Ford │  │  Pred: Honda│   │
│  │  Conf: 34%  │  │  Conf: 41%  │  │  Conf: 45%  │  │  Conf: 52%  │   │
│  │             │  │             │  │             │  │             │   │
│  │  [BMW    ▼] │  │  [Audi   ▼] │  │  [Ford   ▼] │  │  [Honda  ▼] │   │
│  │  [✓ Use GT] │  │  [✓ Use GT] │  │  [✓ Use GT] │  │  [✓ Use GT] │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
│                                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │  [Image 5]  │  │  [Image 6]  │  │  [Image 7]  │  │  [Image 8]  │   │
│  │     ...     │  │     ...     │  │     ...     │  │     ...     │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│  [🏷️ Auto-Label All (Ground Truth)]    [✅ Submit Annotations]          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Results Dashboard

```
┌─────────────────────────────────────────────────────────────────────────┐
│  📈 Results Dashboard                                                    │
│  ───────────────────────────────────────────────────────────────────────│
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Test Accuracy Progression                                       │   │
│  │                                                                  │   │
│  │  80% ─                                          ___●            │   │
│  │       │                                    ___●                 │   │
│  │  70% ─│                               ___●                      │   │
│  │       │                          ___●                           │   │
│  │  60% ─│                     ___●                                │   │
│  │       │                ___●                                     │   │
│  │  50% ─│           ___●                                          │   │
│  │       │      ___●                                               │   │
│  │  40% ─│ ___●                                                    │   │
│  │       └──────────────────────────────────────────────────────   │   │
│  │         1    2    3    4    5    6    7    8    9   10          │   │
│  │                         Cycle                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Cycle Summary                                                   │   │
│  │  ─────────────────────────────────────────────────────────────  │   │
│  │  Cycle │ Labeled │ Val Acc │ Test Acc │ F1    │ Precision      │   │
│  │  ──────┼─────────┼─────────┼──────────┼───────┼────────────    │   │
│  │    1   │   100   │  45.2%  │  43.8%   │ 0.42  │   0.44         │   │
│  │    2   │   150   │  52.1%  │  50.3%   │ 0.49  │   0.51         │   │
│  │    3   │   200   │  58.7%  │  56.9%   │ 0.55  │   0.57         │   │
│  │   ...  │   ...   │   ...   │   ...    │  ...  │   ...          │   │
│  │   10   │   550   │  78.4%  │  76.2%   │ 0.75  │   0.77         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Best Cycle: 10 | Test Accuracy: 76.2% | Improvement: +32.4%           │
│                                                                         │
│  [📥 Export Results (JSON)]                                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Strategy Comparison View

```
┌─────────────────────────────────────────────────────────────────────────┐
│  📊 Strategy Comparison                                                  │
│  ───────────────────────────────────────────────────────────────────────│
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Accuracy Comparison                                             │   │
│  │                                                                  │   │
│  │  80% ─                                          ●── Entropy     │   │
│  │       │                                    ●────○── Margin      │   │
│  │  70% ─│                               ●────○                    │   │
│  │       │                          ●────○────□── LC              │   │
│  │  60% ─│                     ●────○────□                         │   │
│  │       │                ●────○────□────◇── Random               │   │
│  │  50% ─│           ●────○────□────◇                              │   │
│  │       │      ●────○────□────◇                                   │   │
│  │  40% ─│ ●────○────□────◇                                        │   │
│  │       └──────────────────────────────────────────────────────   │   │
│  │         1    2    3    4    5    6    7    8    9   10          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Final Results                                                   │   │
│  │  ─────────────────────────────────────────────────────────────  │   │
│  │  Strategy        │ Final Acc │ Improvement │ vs Random          │   │
│  │  ────────────────┼───────────┼─────────────┼──────────────      │   │
│  │  Entropy         │   76.2%   │   +32.4%    │   +4.1%            │   │
│  │  Margin          │   75.1%   │   +31.3%    │   +3.0%            │   │
│  │  Least Conf.     │   74.3%   │   +30.5%    │   +2.2%            │   │
│  │  Random          │   72.1%   │   +28.3%    │     —              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Conclusion: Entropy sampling outperforms Random by 4.1% after 10      │
│  cycles, confirming the value of uncertainty-based selection.          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration System

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CONFIGURATION LAYERS                                  │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Layer 1: configs/default.yaml (Base)                            │   │
│  │  ─────────────────────────────────────                           │   │
│  │  All default values for the framework                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼ merge                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Layer 2: configs/experiment.yaml (Experiment-specific)          │   │
│  │  ─────────────────────────────────────────────────               │   │
│  │  Override specific settings for an experiment                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼ merge                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Layer 3: Runtime Overrides (UI/CLI)                             │   │
│  │  ───────────────────────────────────                             │   │
│  │  User changes from Streamlit sidebar                             │   │
│  │  Supports dotted keys: "training.epochs": 10                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Final Config Object                                             │   │
│  │  ───────────────────                                             │   │
│  │  Validated, device-resolved, ready for use                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Startup Sequence

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      STARTUP SEQUENCE                                    │
│                                                                         │
│  app.py (Main Process)                    worker.py (Worker Process)    │
│  ─────────────────────                    ──────────────────────────    │
│                                                                         │
│  1. Load config from YAML                                               │
│     │                                                                   │
│  2. Create mp.Queue (task, result)                                      │
│     │                                                                   │
│  3. Create mp.Event dict                                                │
│     │                                                                   │
│  4. Spawn worker process ─────────────────► 5. Rebuild config from dict │
│     │                                           │                       │
│     │                                       6. Build AL components:     │
│     │                                          • Model (TIMM)           │
│     │                                          • DataManager            │
│     │                                          • Trainer                │
│     │                                          • Strategy               │
│     │                                           │                       │
│     │                                       7. Set worker_initialized   │
│     │                                           │                       │
│  8. Wait for worker_initialized ◄───────────────┘                       │
│     │                                                                   │
│  9. Initialize Controller                                               │
│     │                                                                   │
│  10. Render Streamlit UI                                                │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│  Worker is now ready to receive commands via task_queue                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
al-framework/
│
├── app.py                    # Entry point — init queues, events, worker
├── controller.py             # State machine, dispatch, persist state
├── worker.py                 # Worker process (wraps AL loop)
│
├── active_loop.py            # AL orchestrator (inside worker)
├── trainer.py                # Training/eval logic (inside worker)
├── data_manager.py           # Pool management (inside worker)
├── dataloader.py             # Dataset + transforms
├── strategies.py             # Query strategies
├── models.py                 # TIMM model creation
├── state.py                  # Dataclasses for state
├── protocol.py               # Message types, events
├── config.py                 # YAML config loading
│
├── views/                    # Streamlit UI components (Phase 5)
│   ├── router.py             # Main view dispatcher
│   ├── sidebar.py            # Configuration controls
│   ├── training.py           # Live training visualization
│   ├── gallery.py            # Gallery of Uncertainty
│   ├── results.py            # Results dashboard
│   ├── comparison.py         # Strategy comparison
│   └── explorer.py           # Dataset explorer
│
├── configs/
│   ├── default.yaml          # Base configuration
│   └── quick_test.yaml       # Fast debugging config
│
├── experiments/              # Per-experiment outputs
├── checkpoints/              # Model weights
└── state.json                # Persisted application state
```

---

## Current Implementation Status

| Component | Status | Description |
|-----------|--------|-------------|
| Configuration System | ✅ Complete | YAML-based with validation |
| Protocol Layer | ✅ Complete | Message types, events, builders |
| Backend Modules | ✅ Complete | Trainer, DataManager, Strategies |
| Worker Process | ✅ Complete | Eager init, message handling |
| Controller | ✅ Complete | State machine, dispatch, persist |
| App Entry Point | ✅ Complete | Queues, events, worker spawn |
| Streamlit Views | ⏳ Phase 5 | UI components pending |

---

## Key Design Decisions

1. **Eager Initialization**: Worker builds all AL components at startup, not lazily on first command. This simplifies debugging and ensures the worker is ready immediately.

2. **Index-Based Pool Management**: DataManager uses index lists, not data copying. This is memory-efficient for large datasets.

3. **Message-Based Communication**: All inter-process communication uses dict messages via mp.Queue. No shared state between processes.

4. **Event Signaling**: mp.Event for fast boolean signals (stop, ready). Queues for rich data transfer.

5. **Config Layering**: YAML files with merge semantics allow experiment-specific overrides without duplicating defaults.

6. **State Persistence**: Controller saves state to JSON for experiment resumption. Worker is stateless (rebuilds from config).

---

## Thesis Requirements Coverage

| Requirement | Implementation |
|-------------|----------------|
| AL Pipeline with PyTorch | ✅ ActiveLearningLoop + Trainer |
| Streamlit GUI | ⏳ Phase 5 Views |
| Strategy Comparison | ⏳ views/comparison.py |
| Live-Training Visualization | ⏳ views/training.py |
| Gallery of Uncertainty | ⏳ views/gallery.py |
| ResNet/MobileNet Support | ✅ TIMM integration |
| Stanford Cars Dataset | ✅ dataloader.py |
| F1-Score Display | ⏳ views/results.py |
| Dataset Explorer | ⏳ views/explorer.py |
| Probe Image Tracking | ✅ ActiveLearningLoop |
| Confusion Matrix | ⏳ views/results.py |