# Code Review & Revised Architecture Plan
## Your Backend Files vs. Our Agreed MVC Design

---

## Verdict: STRONG Foundation, But Needs Restructuring

Your backend code is well-written. The individual modules are solid engineering. But they were designed for a **monolithic single-process** execution model, not the **MVC + multiprocessing** architecture we agreed on. Here's the detailed breakdown.

---

## 1. File-by-File Assessment

### ✅ `strategies.py` — EXCELLENT, Keep As-Is

**What's good:**
- Clean interface: `fn(model, loader, n_samples, device) -> np.ndarray`
- Registry pattern with `STRATEGIES` dict and `get_strategy()`
- All 4 strategies implemented correctly (entropy, margin, LC, random)
- Returns relative indices (into unlabeled loader), not absolute — correct design

**One issue:** The strategies take a `model` and `loader` and run inference internally. In our MVC design, inference happens inside the worker process. This is actually **fine** — the strategies will be called **inside** `model.py` (the worker), not from the controller. No change needed.

**Integration point:**
```python
# Inside model.py worker process:
from strategies import get_strategy

strategy_fn = get_strategy(payload["strategy"])
query_indices = strategy_fn(model, unlabeled_loader, n_query, device)
```

---

### ✅ `models.py` — GOOD, Minor Improvements

**What's good:**
- TIMM integration via `timm.create_model()` — exactly what you wanted
- `AVAILABLE_ARCHITECTURES` registry for UI dropdown population
- Freeze/unfreeze utilities for transfer learning

**Issues to fix:**

1. **`get_model()` takes a config object** — but the worker receives flat dict payloads from the queue. You need either:
   - Option A: Reconstruct a config object in the worker (adds coupling)
   - Option B: Make `get_model()` accept simple args (better for MVC)

```python
# CURRENT: tightly coupled to config object
def get_model(config, device="cpu"):
    model = timm.create_model(config.name, pretrained=config.pretrained, ...)

# PROPOSED: flexible, works with both config object AND flat args
def get_model(name: str, num_classes: int, pretrained: bool = True, device: str = "cpu"):
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    return model.to(device)
```

2. **Missing: TIMM model search for UI.** Add a function the View can call to let users browse models:

```python
def search_timm_models(query: str, pretrained_only: bool = True) -> list:
    """Search TIMM model registry. Used by Streamlit sidebar for model selection."""
    results = timm.list_models(f"*{query}*", pretrained=pretrained_only)
    return results[:20]  # cap for UI dropdown performance
```

---

### ⚠️ `state.py` — NEEDS RETHINKING (Your Question #1)

**Current design:** Dataclasses used as transport objects within a single process.

**Problem for MVC:** These objects cannot cross the `mp.Queue` boundary as-is. `mp.Queue` serializes objects with pickle. Dataclasses *can* be pickled, but this creates tight coupling — the View would need to import backend dataclasses, violating MVC separation.

**Your Question: "What about using Events instead of JSON state?"**

Here's the answer — use **both**, for different purposes:

```
┌──────────────────────────────────────────────────────────────┐
│                  STATE MANAGEMENT LAYERS                      │
│                                                              │
│  Layer 1: mp.Event() — Lightweight process synchronization   │
│  ─────────────────────────────────────────────────────────── │
│  Purpose: "Has something happened?" (boolean signals)        │
│  Examples:                                                   │
│    model_ready = mp.Event()     # Worker: model loaded?      │
│    training_done = mp.Event()   # Worker: training finished? │
│    stop_requested = mp.Event()  # View: user clicked stop?   │
│                                                              │
│  Layer 2: mp.Queue (dict messages) — Rich data transfer      │
│  ─────────────────────────────────────────────────────────── │
│  Purpose: "What exactly happened?" (metrics, indices, etc.)  │
│  Examples:                                                   │
│    {"type": "PROGRESS", "epoch": 3, "loss": 0.42}           │
│    {"type": "QUERY_RESULT", "indices": [5, 12, 99]}         │
│                                                              │
│  Layer 3: Filesystem (JSON/YAML) — Persistence & Resume      │
│  ─────────────────────────────────────────────────────────── │
│  Purpose: "What was the state when we stopped?"              │
│  Examples:                                                   │
│    state.json    → labeled/unlabeled indices, iteration      │
│    config.yaml   → experiment hyperparameters                │
│    checkpoint.pt → model weights                             │
└──────────────────────────────────────────────────────────────┘
```

**Concrete implementation:**

```python
# In app.py — initialization
import multiprocessing as mp

# Events (Layer 1) — fast boolean signals
events = {
    "model_ready":    mp.Event(),
    "training_done":  mp.Event(),
    "query_done":     mp.Event(),
    "stop_requested": mp.Event(),
    "worker_error":   mp.Event(),
}

# Queues (Layer 2) — rich data
task_queue = mp.Queue(maxsize=10)
result_queue = mp.Queue(maxsize=100)

# Pass both to worker
worker = mp.Process(
    target=worker_loop,
    args=(task_queue, result_queue, events, config),
    daemon=True,
)
```

```python
# In model.py (worker) — using Events for fast signaling
def _run_training(model, payload, result_queue, events):
    for epoch in range(payload["epochs"]):
        
        # Check if user wants to stop (non-blocking!)
        if events["stop_requested"].is_set():
            result_queue.put({"type": "TRAIN_STOPPED", "epoch": epoch})
            return
        
        train_one_epoch(...)
        
        # Send rich data via Queue
        result_queue.put({"type": "PROGRESS", "epoch": epoch, "loss": loss})
    
    # Signal completion via Event (instant, no queue overhead)
    events["training_done"].set()
    
    # Also send detailed results via Queue
    result_queue.put({"type": "TRAIN_COMPLETE", "metrics": {...}})
```

```python
# In view.py — polling with Events is cleaner
def _render_training_view():
    events = st.session_state.events
    
    # Fast check: is training done? (no queue parsing needed)
    if events["training_done"].is_set():
        events["training_done"].clear()  # reset for next cycle
        st.session_state.status = "QUERYING"
        st.rerun()
    
    # Rich check: any progress updates?
    try:
        msg = result_queue.get_nowait()
        if msg["type"] == "PROGRESS":
            update_charts(msg)
    except queue.Empty:
        pass
    
    # Also handle stop button
    if st.button("⏹ Stop Training"):
        events["stop_requested"].set()  # instant signal to worker
```

**So the answer to your question:** Events and JSON serve different purposes. Events give you **instant cross-process signaling** (stop button, completion detection). Queues give you **rich data transfer**. JSON files give you **persistence**. Use all three.

**What happens to `state.py`?** Keep the dataclasses for internal backend use (within `active_loop.py` and `trainer.py`), but **convert to dicts** before putting on the queue:

```python
# In active_loop.py (runs inside worker process)
cycle_metrics = CycleMetrics(...)  # internal dataclass

# Convert to dict before sending to controller
result_queue.put({
    "type": "CYCLE_COMPLETE",
    "payload": cycle_metrics.model_dump()  # ← dict, not dataclass
})
```

---

### ⚠️ `trainer.py` — GOOD Logic, Needs Interface Adapter

**What's good:**
- `train_single_epoch()` returns `EpochMetrics` — perfect for step-by-step dashboard control
- `validate()`, `evaluate()` are clean
- `get_predictions_for_indices()` is exactly what the annotation gallery needs
- Model reset logic (`reset_model_weights`) handles all three modes well
- Checkpoint save/load

**Issue: No queue awareness.** The trainer currently returns values via function return. For our MVC design, it needs to **push progress to the queue**. But we should NOT modify the Trainer class itself — instead, we wrap it in the worker:

```python
# model.py (worker) — wraps Trainer with queue updates
def _run_training(trainer, data_manager, config, task_id, result_queue, events):
    """Wraps Trainer.train_single_epoch with queue progress reporting."""
    
    trainer.reset_model_weights(mode=config["reset_mode"])
    
    train_loader = data_manager.get_labeled_loader(batch_size=config["batch_size"])
    val_loader = ...  # from fixed val set
    
    for epoch in range(1, config["epochs"] + 1):
        if events["stop_requested"].is_set():
            break
        
        # Call existing Trainer method — unchanged!
        metrics = trainer.train_single_epoch(train_loader, val_loader, epoch)
        
        # Wrap result and push to queue — this is the adapter layer
        result_queue.put({
            "type": "PROGRESS_UPDATE",
            "payload": {
                "task_id": task_id,
                "epoch": epoch,
                "total_epochs": config["epochs"],
                **metrics.to_dict(),  # EpochMetrics → dict
            }
        })
    
    # Final evaluation
    test_metrics = trainer.evaluate(test_loader, class_names=class_names)
    result_queue.put({
        "type": "TRAIN_COMPLETE",
        "payload": {"task_id": task_id, "metrics": test_metrics}
    })
    events["training_done"].set()
```

**This way `trainer.py` stays pure** — no queue imports, no Streamlit awareness. The worker is the adapter.

---

### ⚠️ `active_loop.py` — BIGGEST RESTRUCTURING NEEDED

**What's good:**
- `run_cycle()` and `run_all_cycles()` for batch mode
- Step-by-step methods: `prepare_cycle()`, `train_single_epoch()`, `query_samples()`, `receive_annotations()` — this is exactly what interactive mode needs
- Probe image tracking (`_initialize_probe_images`, `_update_probe_predictions`) — great for thesis visualization
- `_build_queried_images()` creates rich data for the annotation gallery

**The Problem:** `ActiveLearningLoop` is currently a **God Object** that orchestrates everything. In MVC, its responsibilities split across two processes:

```
CURRENT (monolithic):
  ActiveLearningLoop owns: trainer + data_manager + strategy + orchestration

MVC SPLIT:
  Controller (Process 1): orchestration + state transitions
  Worker/Model (Process 2): trainer + data_manager + strategy execution
```

**How to split it:**

The `ActiveLearningLoop` class should live **inside the worker process** and be the worker's internal orchestration engine. The Controller doesn't need to know about it — it just sends commands and receives results.

```python
# model.py (worker process)
from active_loop import ActiveLearningLoop

def worker_loop(task_queue, result_queue, events, config):
    al_loop = None  # initialized on INIT_MODEL
    
    while True:
        msg = task_queue.get()
        
        if msg["type"] == "INIT_MODEL":
            # Build everything inside worker
            al_loop = _build_al_loop(msg["payload"], config)
            events["model_ready"].set()
            result_queue.put({"type": "MODEL_READY", ...})
        
        elif msg["type"] == "RUN_CYCLE":
            cycle_num = msg["payload"]["cycle_num"]
            
            # Use AL loop's step methods with queue reporting
            al_loop.prepare_cycle(cycle_num)
            
            for epoch in range(1, config["epochs"] + 1):
                if events["stop_requested"].is_set():
                    break
                metrics = al_loop.train_single_epoch(epoch)
                result_queue.put({"type": "PROGRESS_UPDATE", "payload": metrics.to_dict()})
            
            test_metrics = al_loop.run_evaluation()
            cycle_metrics = al_loop.finalize_cycle(test_metrics)
            events["training_done"].set()
            result_queue.put({"type": "TRAIN_COMPLETE", "payload": cycle_metrics.model_dump()})
        
        elif msg["type"] == "QUERY":
            queried_images = al_loop.query_samples()
            # Convert QueriedImage dataclasses to dicts for queue
            result_queue.put({
                "type": "QUERY_RESULT",
                "payload": {
                    "images": [_queried_to_dict(img) for img in queried_images]
                }
            })
            events["query_done"].set()
        
        elif msg["type"] == "ANNOTATE":
            result = al_loop.receive_annotations(msg["payload"]["annotations"])
            result_queue.put({"type": "ANNOTATE_DONE", "payload": result})
```

---

### ✅ `data_manager.py` — GOOD, Keep Inside Worker

**What's good:**
- Index-based pool management (no data copying) — correct design
- `PoolSubset` for creating DataLoaders from index lists
- `update_labeled_pool()` and `update_labeled_pool_with_annotations()` work correctly
- State save/load for persistence
- Annotation history tracking with accuracy comparison

**Where it lives in MVC:** Inside the worker process. The Controller only knows about index lists (from `state.json`). The DataManager handles the actual PyTorch Dataset operations.

**One concern:** `get_samples_by_class()` iterates the entire pool calling `dataset[idx]` — this is O(n) with actual data loading. For large datasets, cache labels separately:

```python
# Add to __init__:
self._label_cache = {}

def _get_label(self, idx):
    if idx not in self._label_cache:
        _, label = self.dataset[idx]
        self._label_cache[idx] = int(label)
    return self._label_cache[idx]
```

---

### ✅ `dataloader.py` — GOOD, Minor Issue

**What's good:**
- `ImageFolderWithIndex` applies different transforms based on split — smart design
- `SplitSubset` for clean subset access
- Standard ImageNet normalization

**Issue: `ImageFolderWithIndex.__getitem__` gets a PIL Image from ImageFolder.** But ImageFolder with no transform returns PIL. Your code assumes `self.dataset[idx]` returns PIL, which is correct only if ImageFolder has `transform=None`. Verify this is the case (it should be since you don't pass transforms to ImageFolder's constructor).

---

## 2. Flexible Config with YAML (Your Question #2)

Replace the flat dict/dataclass config with **YAML files** + Pydantic validation:

### `configs/default.yaml`
```yaml
# ──────────────────────────────────────
# Active Learning Framework Configuration
# ──────────────────────────────────────

experiment:
  name: "al_stanford_cars"
  seed: 42
  device: "auto"             # "auto", "cuda", "cpu"
  exp_dir: "experiments/"

data:
  data_dir: "data/raw/stanford_cars"
  val_split: 0.15
  test_split: 0.15
  augmentation: true
  num_workers: 4
  image_size: 224

model:
  name: "resnet50"           # Any TIMM model name
  pretrained: true
  num_classes: null           # auto-detected from dataset if null

training:
  epochs: 5
  batch_size: 32
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  optimizer: "adamw"          # "adam", "adamw", "sgd"
  early_stopping_patience: 3

active_learning:
  num_cycles: 10
  initial_pool_size: 100
  batch_size_al: 50           # samples per query
  sampling_strategy: "entropy" # "entropy", "margin", "least_confidence", "random"
  uncertainty_method: "entropy"
  reset_mode: "pretrained"    # "pretrained", "head_only", "none"

checkpoint:
  save_best_model: true
  save_best_per_cycle: true
  save_every_n_epochs: 5

logging:
  level: "INFO"
  log_to_file: true
```

### `configs/quick_test.yaml` (for debugging — small, fast)
```yaml
experiment:
  name: "quick_test"

model:
  name: "mobilenetv3_small_100"

training:
  epochs: 2
  batch_size: 16

active_learning:
  num_cycles: 3
  initial_pool_size: 50
  batch_size_al: 20
```

### `config.py` — Pydantic validation + YAML merging
```python
"""
config.py — YAML-based configuration with validation and merging.

Usage:
    config = load_config()                          # loads default.yaml
    config = load_config("configs/quick_test.yaml") # merges over default
    config = load_config(overrides={"training.epochs": 10})  # CLI/UI override
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import torch
import logging

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"


@dataclass
class ExperimentConfig:
    name: str = "al_experiment"
    seed: int = 42
    device: str = "auto"
    exp_dir: str = "experiments/"


@dataclass
class DataConfig:
    data_dir: str = "data/raw/stanford_cars"
    val_split: float = 0.15
    test_split: float = 0.15
    augmentation: bool = True
    num_workers: int = 4
    image_size: int = 224


@dataclass
class ModelConfig:
    name: str = "resnet50"
    pretrained: bool = True
    num_classes: Optional[int] = None


@dataclass
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    early_stopping_patience: int = 3


@dataclass
class ALConfig:
    num_cycles: int = 10
    initial_pool_size: int = 100
    batch_size_al: int = 50
    sampling_strategy: str = "entropy"
    uncertainty_method: str = "entropy"
    reset_mode: str = "pretrained"


@dataclass
class CheckpointConfig:
    save_best_model: bool = True
    save_best_per_cycle: bool = True
    save_every_n_epochs: int = 5


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    active_learning: ALConfig = field(default_factory=ALConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    def resolve_device(self):
        if self.experiment.device == "auto":
            self.experiment.device = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self) -> dict:
        """Serialize entire config for queue transport (no dataclass objects)."""
        import dataclasses
        return dataclasses.asdict(self)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _dict_to_config(d: dict) -> Config:
    """Convert nested dict to Config dataclass."""
    return Config(
        experiment=ExperimentConfig(**d.get("experiment", {})),
        data=DataConfig(**d.get("data", {})),
        model=ModelConfig(**d.get("model", {})),
        training=TrainingConfig(**d.get("training", {})),
        active_learning=ALConfig(**d.get("active_learning", {})),
        checkpoint=CheckpointConfig(**d.get("checkpoint", {})),
    )


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[dict] = None
) -> Config:
    """
    Load config with layered merging:
      1. default.yaml (base)
      2. config_path yaml (experiment-specific overrides)
      3. overrides dict (runtime/UI overrides)
    """
    # Layer 1: defaults
    with open(DEFAULT_CONFIG) as f:
        base = yaml.safe_load(f)

    # Layer 2: experiment config
    if config_path:
        with open(config_path) as f:
            experiment = yaml.safe_load(f) or {}
        base = _deep_merge(base, experiment)

    # Layer 3: runtime overrides (from Streamlit UI, CLI args, etc.)
    if overrides:
        # Support dotted keys: {"training.epochs": 10}
        expanded = {}
        for key, value in overrides.items():
            parts = key.split(".")
            d = expanded
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = value
        base = _deep_merge(base, expanded)

    config = _dict_to_config(base)
    config.resolve_device()

    logger.info(f"Config loaded: {config.experiment.name} | "
                f"Model: {config.model.name} | "
                f"Strategy: {config.active_learning.sampling_strategy}")

    return config
```

**How the View uses it for live overrides:**

```python
# In view.py sidebar:
overrides = {
    "model.name": st.selectbox("Model", timm_models),
    "training.epochs": st.slider("Epochs", 1, 20, 5),
    "training.learning_rate": st.select_slider("LR", [1e-5, 1e-4, 1e-3]),
    "active_learning.sampling_strategy": st.selectbox("Strategy", strategies),
    "active_learning.batch_size_al": st.slider("Query Size", 10, 200, 50),
}

# Controller merges overrides with base config before dispatching
controller.dispatch_train(overrides=overrides)
```

---

## 3. TIMM Integration for Model Search (Your Question #3)

Your `models.py` already uses TIMM for `create_model`. Here's what to add for the **UI model browser**:

```python
# Add to models.py

import timm

def search_timm_models(query: str = "", pretrained_only: bool = True) -> list:
    """
    Search TIMM model registry for UI dropdown.

    Args:
        query: Search term (e.g. "resnet", "mobile", "efficient")
        pretrained_only: Only show models with pretrained weights

    Returns:
        List of model name strings
    """
    pattern = f"*{query}*" if query else "*"
    results = timm.list_models(pattern, pretrained=pretrained_only)
    return results


def get_model_families() -> dict:
    """Get curated model families for UI dropdown groups."""
    return {
        "ResNet (Recommended)": [
            "resnet18", "resnet34", "resnet50", "resnet101",
        ],
        "MobileNet (Lightweight)": [
            "mobilenetv2_100", "mobilenetv3_small_100", "mobilenetv3_large_100",
        ],
        "EfficientNet": [
            "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
        ],
        "DenseNet": [
            "densenet121", "densenet169",
        ],
    }


def get_model_card(model_name: str) -> dict:
    """
    Get info about a TIMM model for UI display.
    Useful for showing parameter count before the user commits.
    """
    try:
        model = timm.create_model(model_name, pretrained=False, num_classes=10)
        total_params = sum(p.numel() for p in model.parameters())
        del model  # free memory

        return {
            "name": model_name,
            "parameters": total_params,
            "parameters_human": f"{total_params / 1e6:.1f}M",
            "has_pretrained": model_name in timm.list_models(pretrained=True),
        }
    except Exception as e:
        return {"name": model_name, "error": str(e)}
```

**How this looks in the Streamlit sidebar:**

```python
# In view.py
from models import get_model_families, get_model_card, search_timm_models

with st.sidebar:
    st.header("🧠 Model Selection")

    # Option 1: Curated families (simple)
    families = get_model_families()
    family = st.selectbox("Model Family", list(families.keys()))
    model_name = st.selectbox("Architecture", families[family])

    # Option 2: TIMM search (advanced, behind expander)
    with st.expander("🔍 Search All TIMM Models"):
        search = st.text_input("Search", placeholder="e.g. resnet, vit, convnext")
        if search:
            results = search_timm_models(search)
            if results:
                model_name = st.selectbox("Results", results[:30])
            else:
                st.warning("No models found")

    # Show model info
    card = get_model_card(model_name)
    if "error" not in card:
        st.caption(f"Parameters: {card['parameters_human']} | "
                   f"Pretrained: {'✅' if card['has_pretrained'] else '❌'}")
```

---

## 4. Revised Architecture — Everything Together

```
al-framework/
│
├── app.py                    # Entry point — init queues, events, worker, streamlit
├── controller.py             # C — state machine, dispatch, persist state.json
├── model.py                  # M — worker process (wraps AL loop + trainer)
├── view.py                   # V — Streamlit UI rendering
│
├── active_loop.py            # ✅ KEEP — internal orchestrator INSIDE worker
├── trainer.py                # ✅ KEEP — training/eval logic INSIDE worker
├── data_manager.py           # ✅ KEEP — pool management INSIDE worker
├── dataloader.py             # ✅ KEEP — dataset + transforms INSIDE worker
├── strategies.py             # ✅ KEEP — query strategies INSIDE worker
├── models.py                 # ✅ KEEP + ADD TIMM search — used by worker + view
├── state.py                  # ✅ KEEP — dataclasses for backend internal use
│
├── protocol.py               # NEW — message types, Event names, builder helpers
├── config.py                 # NEW — YAML config loading + validation
│
├── configs/
│   ├── default.yaml          # Base config
│   ├── quick_test.yaml       # Fast debugging config
│   └── experiment_entropy.yaml  # Specific experiment configs
│
├── state.json                # Persisted AL state (controller saves/loads)
│
├── data/
│   └── raw/                  # Stanford Cars
├── checkpoints/              # Model .pt files
├── experiments/              # Per-experiment outputs
└── tests/
```

### Data Flow Summary

```
┌─ Process 1 (Main) ──────────────────────────────────────────────────┐
│                                                                      │
│  view.py                          controller.py                      │
│  ┌──────────┐                     ┌──────────────┐                  │
│  │ Streamlit │── button click ───►│ State Machine │                  │
│  │  sidebar  │                    │              │                   │
│  │  charts   │◄── st.rerun() ────│ state.json   │                   │
│  │  gallery  │                    │ iteration    │                   │
│  └──────────┘                     └──────┬───────┘                  │
│        ▲                                 │                           │
│        │ poll result_queue               │ task_queue.put()          │
│        │ check events                    │ events["stop"].set()      │
└────────┼─────────────────────────────────┼───────────────────────────┘
         │                                 │
    result_queue                      task_queue
    mp.Event()                        mp.Event()
         │                                 │
┌────────┼─────────────────────────────────┼───────────────────────────┐
│        ▼                                 ▼                           │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    model.py (worker_loop)                    │    │
│  │                                                             │    │
│  │   ┌─────────────┐   ┌──────────┐   ┌──────────────────┐   │    │
│  │   │ active_loop │──►│ trainer  │──►│ strategies.py    │   │    │
│  │   │   .py       │   │   .py    │   │ data_manager.py  │   │    │
│  │   └─────────────┘   └──────────┘   │ dataloader.py    │   │    │
│  │         │                           │ models.py (TIMM) │   │    │
│  │         │                           └──────────────────┘   │    │
│  │         └── result_queue.put(progress/results)             │    │
│  │         └── events["training_done"].set()                  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─ Process 2 (GPU Worker) ────────────────────────────────────────────┘
```

### What's New vs. Your Original Code

| Component | Status | Change |
|-----------|--------|--------|
| `strategies.py` | ✅ Keep | No changes |
| `models.py` | ✅ Keep + Extend | Add `search_timm_models()`, `get_model_card()`, make `get_model()` accept flat args |
| `state.py` | ✅ Keep | Add `.to_dict()` to all dataclasses for queue serialization |
| `trainer.py` | ✅ Keep | No changes — wrapped by `model.py` |
| `data_manager.py` | ✅ Keep | Add label cache for `get_samples_by_class()` |
| `dataloader.py` | ✅ Keep | No changes |
| `active_loop.py` | ✅ Keep | Lives inside worker; step methods called by `model.py` |
| `config.py` | 🔄 Replace | YAML-based with layered merging |
| `protocol.py` | 🆕 New | Message types + Event registry |
| `controller.py` | 🆕 New | State machine + dispatch |
| `model.py` | 🆕 New | Worker process loop wrapping your existing backend |
| `view.py` | 🆕 New | Streamlit UI |
| `app.py` | 🆕 New | Entry point |

**Bottom line:** Your 6 backend files are solid and need minimal changes. You need 4 new files (`app.py`, `controller.py`, `model.py`, `view.py`) plus 2 support files (`protocol.py`, `config.py`) to wrap them into the MVC architecture.
