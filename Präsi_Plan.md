# Thesis Presentation Plan: Active Learning for Vehicle Image Classification

## Context

You need to present your bachelor thesis project to your professor/supervisor. The project is a fully interactive Active Learning (AL) pipeline for vehicle image classification built with PyTorch + Streamlit. The presentation must cover the ML science in depth, the experiments and their results/metrics, and the architectural decisions driven by Streamlit's constraints.

---

## Part 1: Deep Learning Science

### 1.1 CNNs and Feature Hierarchy
- Convolutional layers: local receptive fields, parameter sharing, translation equivariance
- Feature hierarchy: edges/textures → parts (wheels, headlights) → whole objects (car models)
- Global average pooling (`model.global_pool`) as the bridge between features and classification — the code hooks into this layer for embedding extraction ([models.py:88](ml/models.py#L88))
- **Show**: ResNet-18 block diagram with residual connections (project default: `resnet18`)

### 1.2 Transfer Learning & Fine-Tuning
- **Pretrained weights**: ImageNet (1.2M images, 1000 classes) via `timm.create_model(name, pretrained=True)` at [models.py:26-28](ml/models.py#L26-L28)
- **Freeze-then-unfreeze strategy**: Cycle 1 freezes backbone so random head can learn without corrupting pretrained features; after `freeze_backbone_epochs` backbone unfreezes — [trainer.py:184-256](ml/trainer.py#L184-L256) and [trainer.py:387-396](ml/trainer.py#L387-L396)
- **Discriminative learning rates**: Backbone gets `backbone_lr_factor * lr` (0.1x default), head gets full lr — prevents catastrophic forgetting — [trainer.py:101-131](ml/trainer.py#L101-L131)
- **Diagram**: Show model with frozen backbone (blue) and trainable head (red), then color change after unfreeze

### 1.3 Loss Functions
- **CrossEntropy + label smoothing** (default): `label_smoothing=0.1` prevents overconfidence, critical since AL relies on confidence for sample selection — [trainer.py:58-60](ml/trainer.py#L58-L60)
- **Supervised Contrastive Loss** (Khosla 2020): Pulls same-class embeddings together, pushes different-class apart using temperature-scaled cosine similarity — [losses.py:14-66](ml/losses.py#L14-L66)
  - ProjectionHead (512→128 MLP with L2 normalization) — [losses.py:69-91](ml/losses.py#L69-L91)
  - Temperature τ=0.07 controls separation sharpness
- **Combined**: `(1-α)·CE + α·SupCon` with single forward pass via hook on global_pool — [trainer.py:296-312](ml/trainer.py#L296-L312)

### 1.4 Optimizers & Schedulers
- Adam/AdamW/SGD with discriminative param groups — [trainer.py:101-131](ml/trainer.py#L101-L131)
- Cosine annealing with linear warmup via `SequentialLR([LinearLR, CosineAnnealingLR])` — [trainer.py:148-182](ml/trainer.py#L148-L182)
- ReduceLROnPlateau as alternative (factor=0.5, patience=2)

### 1.5 Regularization (especially important with small AL pools)
- Label smoothing (0.1), weight decay (1e-4), gradient clipping (`grad_clip_norm=1.0`) — [trainer.py:318-321](ml/trainer.py#L318-L321)
- Data augmentation: RandomResizedCrop(224), RandomHorizontalFlip, RandomRotation(10), ColorJitter — [dataloader.py:77-84](ml/dataloader.py#L77-L84)
- Early stopping (patience=3) — [trainer.py:439-446](ml/trainer.py#L439-L446)

---

## Part 2: Active Learning Science

### 2.1 The AL Problem
- **Labeling cost**: Stanford Cars has 196 fine-grained classes — manual annotation is expensive
- **Pool-based AL**: Large unlabeled pool + learner selects samples to query. Implemented via `ALDataManager` with `_labeled_list` / `_unlabeled_list` — [data_manager.py:89-90](ml/data_manager.py#L89-L90)
- **The cycle**: init small pool → train → evaluate → query most informative samples → annotate → repeat

### 2.2 Uncertainty Sampling Strategies
All implemented in [strategies.py](ml/strategies.py):

| Strategy | Formula | Code | Intuition |
|----------|---------|------|-----------|
| Least Confidence | `U(x) = 1 - max P(y\|x)` | [strategies.py:21-63](ml/strategies.py#L21-L63) | Model's best guess is weak |
| Entropy | `U(x) = -Σ P(y\|x)·log P(y\|x)` | [strategies.py:66-109](ml/strategies.py#L66-L109) | Uncertainty spread across ALL classes |
| Margin | `U(x) = P(y₁\|x) - P(y₂\|x)` | [strategies.py:112-162](ml/strategies.py#L112-L162) | Can't decide between top-2 |
| Random | uniform sample | [strategies.py:165-193](ml/strategies.py#L165-L193) | Baseline (no AL) |

**Key insight**: For binary classification all three are equivalent. For 196 classes, they diverge — entropy captures information from all class probabilities, typically best for many-class problems.

### 2.3 Why AL Works
- Uncertainty sampling selects samples near **decision boundaries** — maximum information gain
- **Version space argument**: each labeled sample eliminates hypotheses; uncertain samples eliminate the most
- Random sampling wastes labels on "easy" samples the model already understands

### 2.4 Stratified Initialization
- `stratified_init=True` guarantees ≥1 sample per class in initial pool — [data_manager.py:100-154](ml/data_manager.py#L100-L154)
- Prevents the "cold start" problem where missing classes can never be queried

### 2.5 Weight Reset Modes
- `"continue"` (default): Freeze→unfreeze cycle 1, carry weights forward — fast but may accumulate bias
- `"pretrained"`: Reload ImageNet each cycle — independent runs, eliminates bias
- `"head_only"`: Keep backbone, reset classifier — middle ground
- Implementation: [trainer.py:184-256](ml/trainer.py#L184-L256)

---

## Part 3: Experiments to Run

### 3.1 Primary: Strategy Comparison (the central thesis question)
> "Does uncertainty-based AL outperform random sampling, and which uncertainty measure is most effective?"

| Parameter | Value |
|-----------|-------|
| Dataset | Stanford Cars (196 classes) |
| Model | ResNet-18, pretrained |
| Initial pool | 100 (stratified) |
| Query batch | 50 per cycle |
| Cycles | 10 |
| Seed | 42 |

**4 runs**: entropy, least_confidence, margin, random

**Key metric**: Accuracy vs. labeled pool size curves. The gap between AL strategies and random = "label savings."

### 3.2 Secondary: Model Architecture Comparison
Fix strategy (entropy), vary model:
- MobileNetV3-Small (~2.5M params)
- ResNet-18 (~11.7M params)
- ResNet-50 (~25.6M params)
- EfficientNet-B0 (~5.3M params)

**Hypothesis**: Larger models may overfit more with tiny labeled pools.

### 3.3 Ablations
- **Reset mode**: "continue" vs. "pretrained" vs. "head_only" (entropy, ResNet-18)
- **Initial pool size**: 50 vs. 100 vs. 200
- **Query batch size**: 25 vs. 50 vs. 100

### 3.4 Calibration Analysis
- ECE evolution across cycles (does more data improve calibration?)
- Raw ECE vs. temperature-scaled ECE
- This is a unique angle most AL papers overlook

### 3.5 Quick Smoke Test
Use `configs/quick_test.yaml` (4-class, MobileNetV3-Small, 3 cycles) for pipeline verification before running full experiments.

---

## Part 4: Results & Metrics Explanation

### 4.1 Classification Metrics
| Metric | Definition | AL Interpretation | Implementation |
|--------|-----------|-------------------|----------------|
| **Accuracy** | Fraction correct | Plot vs. pool size; steeper slope = better strategy | `accuracy_score()` in [trainer.py:533](ml/trainer.py#L533) |
| **Precision** (weighted) | Of predicted class X, how many are actually X? | May diverge from accuracy if strategy creates class imbalance | [trainer.py:534-536](ml/trainer.py#L534-L536) |
| **Recall** (weighted) | Of actual class X, how many did model find? | Rare classes may have low recall if under-sampled | Same as above |
| **F1** (weighted) | Harmonic mean of P & R | Most robust single metric for imbalanced data | Same as above |

### 4.2 Calibration Metrics
- **ECE** (Expected Calibration Error): 15 equal-width confidence bins, `Σ |avg_confidence - avg_accuracy|` weighted by bin size — [trainer.py:547-559](ml/trainer.py#L547-L559)
  - Good: ECE < 0.05, Bad: ECE > 0.15
  - Critical for AL because sample selection depends on model confidence
- **Temperature-scaled ECE**: Learn scalar T via LBFGS on validation set, then `softmax(logits/T)` — [trainer.py:613-642](ml/trainer.py#L613-L642)
  - Shows how much post-hoc calibration helps

### 4.3 Visual Analytics
- **Confusion matrices**: Per-cycle heatmaps saved as `.npy` — [trainer.py:538-545](ml/trainer.py#L538-L545). Show evolution from noisy to sharp diagonal.
- **UMAP embeddings**: 2D projections of backbone features (cosine, n_neighbors=15, min_dist=0.1) — [embeddings.py:22-56](ml/embeddings.py#L22-L56). Color by class or pool membership (labeled=blue, unlabeled=gray, queried=orange).
  - Show cycle 1 (messy) vs. cycle 10 (well-separated clusters)
  - Queried points should cluster near decision boundaries
- **Probe images**: 12 fixed validation samples tracked across cycles — [active_loop.py:106-217](core/active_loop.py#L106-L217). Show prediction evolution from wrong/uncertain → correct/confident.
- **Query summaries**: Class distribution of queried batches, uncertainty statistics (min/max/mean/std) — reveals strategy biases.

### 4.4 How to Read AL Learning Curves
- X-axis: number of labeled samples (not cycles)
- Y-axis: test accuracy (or F1)
- Multiple lines: one per strategy
- The gap between entropy and random = "label savings"
- Ideally run 3 seeds and show mean ± std for statistical significance

---

## Part 5: Architecture Presentation

### 5.1 The Fundamental Streamlit Constraint
- Streamlit reruns the entire script on every interaction — no persistent state
- Training takes minutes/hours — cannot run inside a Streamlit callback
- **Solution**: Background daemon thread + communication protocol

### 5.2 Key Architectural Patterns

| Pattern | Why | Where |
|---------|-----|-------|
| **Controller singleton** via `@st.cache_resource` | Survives page reruns | [app.py:60-66](app.py#L60-L66) |
| **Two-channel communication** | Asymmetric needs: rare commands (UI→worker) vs. continuous events (worker→UI) | Inbox: [events.py:62-118](core/events.py#L62-L118), cmd queue: [controller.py:219](core/controller.py#L219) |
| **Immutable events** (frozen dataclass + MappingProxyType) | Prevents race conditions on shared payloads | [events.py:46-59](core/events.py#L46-L59) |
| **Atomic snapshots** (deepcopy under lock) | Views get consistent state, not partial updates | [experiment_state.py:100-127](core/experiment_state.py#L100-L127) |
| **Adaptive polling** (0.5s/1.5s/off) | Fast for user-facing states, slow for training, off when idle | [app.py:39-57](app.py#L39-L57) |
| **AppState enum** drives everything | Views, buttons, polling all key off one state field | [experiment_state.py:19-30](core/experiment_state.py#L19-L30) |
| **Query token** (UUID per query) | Prevents stale annotation submissions after restart | [worker.py:475](core/worker.py#L475), [controller.py:266-292](core/controller.py#L266-L292) |
| **num_workers=0** enforced | PyTorch DataLoader multiprocessing deadlocks in daemon thread on Windows/Streamlit | [controller.py:33-34](core/controller.py#L33-L34) |
| **Incremental artifact persistence** | Crash at cycle 8 still saves cycles 1-7 | [worker.py:257-264](core/worker.py#L257-L264) |
| **Heartbeat watchdog** (120s timeout) | Detects stalled threads, warns user | [router.py:16-36](views/router.py#L16-L36) |

### 5.3 State Machine Diagram
States: IDLE → INITIALIZING → TRAINING → QUERYING → ANNOTATING → WAITING_STEP → FINISHED (+ ERROR, STOPPING)
- All transitions go through `controller.dispatch()`
- Views render based on `snap["app_state"]`

### 5.4 Communication Sequence (one full cycle)
```
User clicks Start → Controller spawns worker thread
Worker: build_al_loop → emit CYCLE_STARTED
Worker: train N epochs → emit EPOCH_DONE (×N)
Worker: evaluate → emit EVAL_COMPLETE
Worker: query → emit NEW_IMAGES (with query_token)
UI: renders gallery → User selects labels → Submit
Controller: validates token → command_queue.put(SUBMIT_ANNOTATIONS)
Worker: receive_annotations → emit ANNOTATIONS_APPLIED
Worker: finalize_cycle → loop to next cycle
```

---

## Part 6: Concept-to-Code Mapping Table (for slides)

| Scientific Concept | Implementation | File |
|---|---|---|
| Softmax for uncertainty | `F.softmax(outputs, dim=1)` | [strategies.py:50,95,141](ml/strategies.py) |
| Shannon entropy | `-(probs * torch.log(probs + 1e-10)).sum(dim=1)` | [strategies.py:97](ml/strategies.py#L97) |
| Transfer learning | `timm.create_model(name, pretrained=True)` | [models.py:26-28](ml/models.py#L26-L28) |
| Discriminative LR | Two param groups with different `lr` | [trainer.py:101-131](ml/trainer.py#L101-L131) |
| Label smoothing | `nn.CrossEntropyLoss(label_smoothing=0.1)` | [trainer.py:58-60](ml/trainer.py#L58-L60) |
| Cosine annealing + warmup | `SequentialLR([LinearLR, CosineAnnealingLR])` | [trainer.py:163-173](ml/trainer.py#L163-L173) |
| Pool-based AL (index mgmt) | `_labeled_list` / `_unlabeled_list` (zero-copy) | [data_manager.py:89-90](ml/data_manager.py#L89-L90) |
| Temperature calibration | LBFGS on scalar T minimizing NLL | [trainer.py:613-642](ml/trainer.py#L613-L642) |
| SupCon + projection head | Hook on `global_pool`, project to 128D, contrastive loss | [losses.py:14-91](ml/losses.py#L14-L91) |
| UMAP embeddings | `umap.UMAP(metric="cosine", n_components=2)` | [embeddings.py:49-56](ml/embeddings.py#L49-L56) |

---

## Suggested Presentation Order (20-25 min)

1. **Problem + Live Demo** (3 min): Vehicle classification labeling cost, 1-minute Streamlit walkthrough
2. **Active Learning Science** (4 min): AL problem, cycle, uncertainty strategies, why it works
3. **Deep Learning Science** (4 min): Transfer learning, freeze/unfreeze, loss functions, regularization
4. **Science → Code Mapping** (3 min): Mapping table, data flow trace through one cycle
5. **Experiments + Results** (5 min): Strategy comparison curves, UMAP evolution, calibration analysis
6. **Architecture** (4 min): Streamlit constraints, threading model, state machine, event system
7. **Q&A buffer** (2 min)

---

## Verification

- Run `configs/quick_test.yaml` end-to-end to verify pipeline works before full experiments
- Run 4 strategy comparison experiments with `configs/default.yaml` (change `active_learning.sampling_strategy` per run)
- Check `experiments/` directory for output files: `al_cycle_results.json`, confusion matrices, embeddings
- Verify results dashboard loads and displays all charts correctly
