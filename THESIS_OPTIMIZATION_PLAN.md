# Thesis Optimization Plan (Final)

## Scope

This plan covers three categories of work:

1. **Transfer learning fix** — the model isn't genuinely being trained on your dataset; fix this first
2. **Training foundation fixes** — scheduler, clipping, smoothing, calibration
3. **New visualization & loss features** — feature extraction, SupCon loss, UMAP, Datashader

Everything maps to the thesis requirements from the README:
- Compare sampling strategies with a baseline (Random)
- Compare neural network architectures
- Visual presentation of results with live training feedback
- Comprehensive evaluation pipeline to meet scientific standards
- Deeper insights into the functionality and dynamics of AL approaches

**Removed from previous plan** (out of scope): Mixup/CutMix, adaptive reset strategy, CoreSet/BADGE, multi-seed analysis script, per-class progression tracking, class-weighted loss.

---

## Priority Legend

| Label | Meaning |
|---|---|
| **P0** | Must-have. Without this, the model isn't properly training on your data. |
| **P1** | High impact. Produces a strong thesis figure or serves the four new features. |
| **P2** | Polish. Improves quality but thesis works without it. |

| Effort | Meaning |
|---|---|
| **S** | Under 50 lines, single file |
| **M** | 50–200 lines, 2–3 files |
| **L** | 200+ lines, touches 4+ files or adds a new module |

---

## Part A: Training Pipeline Fixes

---

### A0 — Fix Transfer Learning: Proper Freeze/Unfreeze + Discriminative Learning Rates
**Priority:** P0 · **Effort:** L · **This is the most important item in the entire plan.**

#### The Problem

The model is not genuinely being trained on the Stanford Cars dataset across active learning cycles. Here is what currently happens with each reset mode:

**`reset_mode = "pretrained"` (the default):**
Every cycle, `prepare_cycle()` calls `reset_model_weights("pretrained")` which calls `timm.create_model()` with `pretrained=True`. This loads a brand-new ResNet50 with fresh ImageNet weights. The backbone has never seen a car. The head is randomly initialized (because `num_classes=196 ≠ 1000`). Everything the model learned in previous cycles is destroyed. Cycle 10 has no memory of cycles 1–9. You're running 10 independent experiments from the same starting point with increasing data, not training one model over time.

**`reset_mode = "head_only"`:**
The backbone keeps its weights from the previous cycle, but the head is reset to random. Then `_create_optimizer()` creates an optimizer over `self.model.parameters()` — ALL parameters, backbone included, at the SAME learning rate. From epoch 1, the random head produces garbage gradients that flow backward into the backbone and damage the features learned in previous cycles. The freeze functions in `models.py` (`freeze_backbone`, `unfreeze_backbone`, `freeze_backbone_unfreeze_head`) exist but are **never called anywhere in the pipeline** — they are dead code.

**`reset_mode = "none"`:**
Closest to correct — both backbone and head carry forward. But the optimizer treats all parameters identically (one LR for everything), there's no warmup, and if the model overfit to the previous cycle's small pool, that overfitting carries forward too.

**The core issue across all modes:** `_create_optimizer()` at `trainer.py:74` passes `self.model.parameters()` as a single flat group. Every layer — from the first conv filter that detects edges to the classification head that maps to 196 classes — gets the exact same learning rate. This makes proper transfer learning impossible.

#### What Correct Training Looks Like

For a model that genuinely learns car-specific features across AL cycles:

**Cycle 1 (cold start from ImageNet):**
1. Load pretrained backbone + random head.
2. **Phase 1 — Head warmup (first N epochs):** Freeze the entire backbone. Only the head trains. This lets the head learn to map ImageNet features → 196 car classes without corrupting the backbone. Fast and stable (~100K parameters vs ~23M).
3. **Phase 2 — Full fine-tuning (remaining epochs):** Unfreeze the backbone. Train everything, but backbone gets a **10x lower learning rate** than the head. The head continues learning quickly; the backbone slowly adapts its features from "generic objects" toward "car-specific details."
4. Evaluate. Query uncertain images. **Keep the model weights.**

**Cycle 2+ (warm start from previous cycle):**
1. **Don't reload ImageNet.** Don't reset the backbone. The model carries forward.
2. Optionally reset only the **optimizer state** (clears stale momentum) and the **tracking variables** (best accuracy, patience).
3. Apply brief warmup (1–2 epochs at reduced LR), then continue fine-tuning.
4. The backbone progressively moves further from ImageNet → car-specific features. UMAP makes this visible across cycles.

#### Implementation

**File: `config.py`**

```python
@dataclass
class TrainingConfig:
    # ... existing fields ...
    backbone_lr_factor: float = 0.1      # backbone LR = learning_rate × this factor
    freeze_backbone_epochs: int = 2       # freeze backbone for first N epochs per cycle

@dataclass
class ALConfig:
    # ... existing fields ...
    reset_mode: str = "continue"          # NEW default: "continue", "pretrained", "head_only", "none"
```

**File: `trainer.py` — Replace `_create_optimizer()`**

```python
def _create_optimizer(self):
    """Create optimizer with separate parameter groups for backbone and head."""
    name = self.config.training.optimizer.lower()
    lr = self.config.training.learning_rate
    wd = self.config.training.weight_decay
    backbone_lr = lr * self.config.training.backbone_lr_factor

    # Identify head module (TIMM models use fc, classifier, or head)
    head_module = (getattr(self.model, 'fc', None) or
                   getattr(self.model, 'classifier', None) or
                   getattr(self.model, 'head', None))

    if head_module is None:
        # Fallback: treat everything as one group
        params = [{"params": self.model.parameters(), "lr": lr}]
    else:
        head_param_ids = set(id(p) for p in head_module.parameters())
        backbone_params = [p for p in self.model.parameters() if id(p) not in head_param_ids]
        head_params = [p for p in self.model.parameters() if id(p) in head_param_ids]

        params = [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": lr},
        ]

    if name == "adam":
        return optim.Adam(params, weight_decay=wd)
    elif name == "sgd":
        return optim.SGD(params, momentum=0.9, weight_decay=wd)
    elif name == "adamw":
        return optim.AdamW(params, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {name}")
```

**File: `trainer.py` — Add freeze/unfreeze methods**

```python
def freeze_backbone(self):
    """Freeze all backbone parameters (head stays trainable)."""
    head_module = (getattr(self.model, 'fc', None) or
                   getattr(self.model, 'classifier', None) or
                   getattr(self.model, 'head', None))
    head_param_ids = set(id(p) for p in head_module.parameters()) if head_module else set()

    for param in self.model.parameters():
        if id(param) not in head_param_ids:
            param.requires_grad = False

def unfreeze_backbone(self):
    """Unfreeze all backbone parameters."""
    for param in self.model.parameters():
        param.requires_grad = True
```

**File: `trainer.py` — Replace `reset_model_weights()`**

```python
def reset_model_weights(self, mode: str = "continue", cycle: int = 1):
    """
    Reset model weights for a new AL cycle.

    Modes:
        "continue"  — Keep all weights. Reset optimizer + tracking only.
                       Cycle 1: loads pretrained, freezes backbone for warmup.
                       Cycle 2+: keeps everything, brief warmup.
        "pretrained" — Reload ImageNet weights every cycle (independent experiments).
        "head_only"  — Keep backbone, reset head. Freeze backbone for warmup.
        "none"       — Legacy mode. Keep everything, reset optimizer + tracking.
    """
    if mode == "continue":
        if cycle == 1:
            # First cycle: backbone is already pretrained from build_al_loop().
            # Freeze backbone so the random head can warm up.
            self.freeze_backbone()
            self._backbone_frozen = True
        else:
            # Subsequent cycles: keep all weights, just reset optimizer state.
            self.unfreeze_backbone()
            self._backbone_frozen = False
        self.optimizer = self._create_optimizer()
        self._reset_tracking()
        return

    if mode == "pretrained":
        self.model = get_model(
            name=self.config.model.name,
            num_classes=self.config.model.num_classes,
            pretrained=self.config.model.pretrained,
            device=self.device
        )
        # Freeze backbone for warmup (random head needs safe epochs)
        self.freeze_backbone()
        self._backbone_frozen = True
        self.optimizer = self._create_optimizer()
        self._reset_tracking()
        return

    if mode == "head_only":
        # Reset head to random, keep backbone
        head_module = (getattr(self.model, 'fc', None) or
                       getattr(self.model, 'classifier', None) or
                       getattr(self.model, 'head', None))
        if head_module is not None:
            if isinstance(head_module, nn.Sequential):
                for layer in head_module:
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
            else:
                head_module.reset_parameters()
        # Freeze backbone for warmup
        self.freeze_backbone()
        self._backbone_frozen = True
        self.optimizer = self._create_optimizer()
        self._reset_tracking()
        return

    if mode == "none":
        self.optimizer = self._create_optimizer()
        self._reset_tracking()
        self._backbone_frozen = False
        return

    raise ValueError(f"Unknown reset mode: {mode}")
```

**File: `trainer.py` — Modify `train_single_epoch()` to handle unfreeze after warmup**

```python
def train_single_epoch(self, train_loader, val_loader, epoch_num):
    # Unfreeze backbone after warmup period
    freeze_epochs = self.config.training.freeze_backbone_epochs
    if getattr(self, '_backbone_frozen', False) and epoch_num > freeze_epochs:
        self.unfreeze_backbone()
        self._backbone_frozen = False
        # Recreate optimizer to include backbone params with lower LR
        self.optimizer = self._create_optimizer()
        logger.info(f"Epoch {epoch_num}: backbone unfrozen, discriminative LR active")

    # ... rest of existing training logic unchanged ...
```

**File: `active_loop.py` — Pass cycle number to reset**

```python
def prepare_cycle(self, cycle_num: int) -> Dict:
    # ...
    reset_mode = self.config.active_learning.reset_mode
    self.trainer.reset_model_weights(mode=reset_mode, cycle=cycle_num)
    # ...
```

**File: `config.py` — Update validation**

Add `"continue"` to `valid_reset_modes`.

#### What This Enables for the Thesis

- UMAP across cycles now shows **genuine feature space evolution** — the same model's backbone slowly specializing from ImageNet → car features
- Probe images show the model's prediction on the same image genuinely improving over cycles (not resetting each time)
- SupCon loss (B2) can build on previous cycles' clusters instead of starting from scratch
- The "learning curve" across cycles reflects actual model learning, not just the effect of more data

---

### A1 — Learning Rate Scheduler + Warmup
**Priority:** P0 · **Effort:** M

**Problem:** LR is flat. Combined with A0's freeze/unfreeze, the scheduler needs to handle the warmup period (frozen backbone) and the fine-tuning period (unfrozen backbone) correctly.

**Implementation:**

| File | Change |
|---|---|
| `config.py` → `TrainingConfig` | Add `scheduler: str = "cosine"`, `warmup_epochs: int = 2` |
| `trainer.py:__init__()` | `self.scheduler = self._create_scheduler()` |
| `trainer.py` | New `_create_scheduler()` — supports `cosine` (CosineAnnealingLR) and `plateau` (ReduceLROnPlateau). Warmup via `SequentialLR` combining `LinearLR` + main scheduler. |
| `trainer.py:train_single_epoch()` | After validation: `self.scheduler.step()` |
| `trainer.py:reset_model_weights()` | Recreate scheduler alongside optimizer |

**Note:** The `warmup_epochs` in the scheduler and the `freeze_backbone_epochs` from A0 can be the same value. During the frozen phase, the scheduler warms up the head's LR. When the backbone unfreezes, the scheduler continues with the main schedule at full LR for the head and `backbone_lr_factor × LR` for the backbone.

---

### A2 — Gradient Clipping
**Priority:** P0 · **Effort:** S

**Problem:** No gradient clipping between `loss.backward()` and `optimizer.step()`. Early cycles with tiny datasets produce gradient spikes.

**Implementation:**

| File | Change |
|---|---|
| `config.py` → `TrainingConfig` | Add `grad_clip_norm: float = 1.0` |
| `trainer.py:train_epoch()` | Between backward and step: `torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip_norm)` |

---

### A3 — Best-Model Restoration After Early Stopping
**Priority:** P0 · **Effort:** S

**Problem:** Early stopping saves the best checkpoint but never reloads it. Evaluation and querying use the last epoch's model (which is worse than the best).

**Verified:** `grep` confirms `load_checkpoint` is never called after `should_stop_early()` triggers. The best model sits on disk unused.

**Implementation:**

| File | Change |
|---|---|
| `trainer.py` | New method `restore_best_model()` — loads `best_model.pth` if it exists and `best_val_accuracy > 0` |
| `active_loop.py:run_cycle()` | After epoch loop: `self.trainer.restore_best_model()` before `run_evaluation()` |
| `worker.py:run_experiment()` | After line 361 (early stop break): `al_loop.trainer.restore_best_model()` before evaluation |

---

### A4 — Label Smoothing
**Priority:** P1 · **Effort:** S

**Problem:** Hard one-hot targets → overconfident predictions → unreliable uncertainty scores for AL strategies.

**Implementation:**

| File | Change |
|---|---|
| `config.py` → `TrainingConfig` | Add `label_smoothing: float = 0.1` |
| `trainer.py:__init__()` | `self.criterion = nn.CrossEntropyLoss(label_smoothing=self.config.training.label_smoothing)` |

---

### A5 — Stratified Initial Pool Selection
**Priority:** P1 · **Effort:** M

**Problem:** Random initial pool can miss entire classes. With 196 classes and 100 samples, the model starts blind to many classes.

**Implementation:**

| File | Change |
|---|---|
| `config.py` → `ALConfig` | Add `stratified_init: bool = True` |
| `data_manager.py:__init__()` | When `stratified_init=True`: group indices by class via `_get_label()`, sample ≥1 per class, distribute remaining budget randomly. |

---

### A6 — Calibration Metrics (ECE)
**Priority:** P1 · **Effort:** M

**Problem:** No measurement of whether the model's confidence matches reality. The thesis compares uncertainty strategies but never validates that confidence is meaningful.

**Implementation:**

| File | Change |
|---|---|
| `trainer.py:evaluate()` | Collect probability vectors. Compute ECE: 15 bins, |avg_confidence − avg_accuracy| per bin, weighted by bin size. |
| `state.py` → `CycleMetrics` | Add `ece: Optional[float] = None` |
| `active_loop.py:finalize_cycle()` | Pass ECE into CycleMetrics |

---

### A7 — MC Dropout for Uncertainty Estimation
**Priority:** P1 · **Effort:** M

**Problem:** All strategies use a single forward pass with dropout off. Single-pass softmax is overconfident and unreliable.

**Implementation:**

| File | Change |
|---|---|
| `strategies.py` | New function `mc_dropout_sampling()`: `model.train()`, N forward passes, predictive entropy from mean prediction. |
| `strategies.py` → `STRATEGIES` | Register as `"mc_dropout"` |
| `config.py` → `ALConfig` | Add `mc_dropout_passes: int = 10`. Add `"mc_dropout"` to valid strategies. |

---

## Part B: New Visualization & Loss Features

---

### B1 — Feature Extraction Hooks
**Priority:** P0 for Part B · **Effort:** M

**What:** Extract the penultimate-layer embedding vector from any TIMM model. Foundation for B2, B3, B4.

**Implementation:**

| File | Change |
|---|---|
| `models.py` | New `extract_features(model, dataloader, device) -> (embeddings, labels)` via forward hook on `model.global_pool` |
| `models.py` | New `get_feature_dim(model) -> int` |
| `trainer.py` | New `get_embeddings(dataloader) -> (embeddings, labels)` |
| `active_loop.py:finalize_cycle()` | Extract + save embeddings as `.npz` per cycle |
| `state.py` → `CycleMetrics` | Add `embeddings_path: Optional[str] = None` |

**Hook mechanism:**

```python
def extract_features(model, dataloader, device):
    features, labels_list = [], []
    hook_output = {}

    def hook_fn(module, input, output):
        hook_output['features'] = output.detach()

    handle = model.global_pool.register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            _ = model(images)
            feat = hook_output['features'].view(hook_output['features'].size(0), -1)
            features.append(feat.cpu().numpy())
            labels_list.extend(labels.numpy())

    handle.remove()
    return np.vstack(features), np.array(labels_list)
```

---

### B2 — Supervised Contrastive Loss (SupCon)
**Priority:** P1 · **Effort:** L

**What:** Alternative loss that shapes embedding geometry directly. Pulls same-class embeddings together, pushes different-class apart.

**Implementation:**

| File | Change |
|---|---|
| **New: `losses.py`** | `SupConLoss` class + `ProjectionHead` class |
| `models.py` | New `create_model_with_projection_head(name, num_classes, embed_dim, proj_dim=128)` |
| `config.py` → `TrainingConfig` | Add `loss_fn: str = "cross_entropy"`, `supcon_temperature: float = 0.07`, `supcon_weight: float = 0.5` |
| `trainer.py:__init__()` | Loss selection: `"cross_entropy"`, `"supcon"` (two-phase), `"combined"` (α×CE + (1-α)×SupCon) |
| `trainer.py:train_epoch()` | When SupCon active: extract embeddings via projection head, compute combined loss |

**SupCon loss core:**

```python
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        similarity = torch.matmul(features, features.T) / self.temperature

        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        mask.fill_diagonal_(0)

        logits_max, _ = similarity.max(dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        logits_mask = 1.0 - torch.eye(features.size(0), device=features.device)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        pos_count = mask.sum(1)
        mean_log_prob = (mask * log_prob).sum(1) / (pos_count + 1e-12)
        return -mean_log_prob[pos_count > 0].mean()
```

**Projection head:**

```python
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=1)
```

---

### B3 — UMAP Embedding Visualization
**Priority:** P1 · **Effort:** M

**What:** Project high-D embeddings (from B1) to 2D using UMAP. Save coordinates + metadata per cycle.

**Implementation:**

| File | Change |
|---|---|
| **New: `embeddings.py`** | `compute_umap_projection()`, `save_cycle_embeddings()` |
| `active_loop.py:finalize_cycle()` | After extracting embeddings, run UMAP, save `.npz` |
| `requirements.txt` | Add `umap-learn` |

**Key functions:**

```python
def compute_umap_projection(embeddings, n_neighbors=15, min_dist=0.1, metric="cosine"):
    import umap
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                         metric=metric, n_components=2, random_state=42)
    return reducer.fit_transform(embeddings)

def save_cycle_embeddings(exp_dir, cycle, coords_2d, labels, pool_membership, uncertainty=None):
    path = exp_dir / "embeddings" / f"cycle_{cycle}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, coords=coords_2d, labels=labels,
                        pool=pool_membership, uncertainty=uncertainty or np.array([]))
    return str(path)
```

**Thesis figures this enables:**
- Feature space at cycle 1 vs. 5 vs. 10 (clusters tightening as the model learns car features)
- Query overlay (strategy selections sitting on cluster boundaries)
- CE vs. SupCon side-by-side (tighter clusters with SupCon)
- Pool coverage (labeled vs. unlabeled coloring)

---

### B4 — Datashader / Plotly Rendering
**Priority:** P2 · **Effort:** M

**What:** Render UMAP scatter plots at scale (~16K points). Datashader for density, Plotly scattergl for interactive.

**Implementation:**

| File | Change |
|---|---|
| `embeddings.py` | Add `render_embedding_datashader()` and `render_embedding_plotly()` |
| `requirements.txt` | Add `datashader`, `colorcet` |

**Two rendering paths:**

```python
def render_embedding_datashader(coords, labels, width=800, height=600):
    import datashader as ds, datashader.transfer_functions as tf, colorcet as cc
    df = pd.DataFrame({'x': coords[:,0], 'y': coords[:,1], 'label': pd.Categorical(labels)})
    canvas = ds.Canvas(plot_width=width, plot_height=height)
    agg = canvas.points(df, 'x', 'y', agg=ds.count_cat('label'))
    return tf.set_background(tf.shade(agg, color_key=cc.glasbey_category10, how='eq_hist'), "white")

def render_embedding_plotly(coords, labels, title="Feature Space (UMAP)"):
    import plotly.graph_objects as go
    fig = go.Figure(go.Scattergl(x=coords[:,0], y=coords[:,1], mode='markers',
        marker=dict(size=3, color=labels, colorscale='Viridis', opacity=0.6)))
    fig.update_layout(title=title, xaxis_title="UMAP-1", yaxis_title="UMAP-2",
                      template="plotly_white", width=800, height=600)
    return fig
```

Plotly scattergl handles ~16K points well. Datashader needed only above ~50K.

---

## Implementation Order

```
Step 1:  A0 (transfer learning fix)
         → The model now genuinely trains on your dataset across cycles.
         → This is the single biggest change. ~2–3 days.

Step 2:  A1 (scheduler+warmup) + A2 (grad clip) + A3 (best-model restore)
         → Training pipeline is now correct and stable. ~1–2 days.

Step 3:  A4 (label smoothing) + A5 (stratified init)
         → Better calibration and starting conditions. ~1 day.

Step 4:  B1 (feature extraction hooks)
         → Foundation for all visualization. ~1 day.

Step 5:  B2 (SupCon loss + losses.py)
         → Alternative training objective. ~2–3 days.

Step 6:  B3 (UMAP projection + embeddings.py)
         → Feature space visualization. ~1–2 days.

Step 7:  B4 (Datashader/Plotly rendering)
         → Embedding plots at scale. ~1 day.

Step 8:  A6 (ECE calibration) + A7 (MC Dropout)
         → Calibration metrics and advanced strategy. ~2 days.

Step 9:  Dashboard integration + experimental runs
         → Wire into results view, run strategy comparisons.
```

**Total estimated time: ~2–3 weeks of focused work.**

---

## Files Changed Summary

| File | Items |
|---|---|
| `config.py` | A0, A1, A2, A4, A5, A6, A7, B2 |
| `trainer.py` | A0, A1, A2, A3, A4, A6, B1, B2 |
| `models.py` | A0 (wire freeze functions), B1, B2 |
| `strategies.py` | A7 |
| `data_manager.py` | A5 |
| `active_loop.py` | A0, A3, A6, B1, B3 |
| `worker.py` | A0, A3 |
| `state.py` | A6, B1 |
| `requirements.txt` | B3, B4 |
| **New: `losses.py`** | B2 |
| **New: `embeddings.py`** | B3, B4 |

---

## Thesis Chapter Mapping

| Chapter | Items |
|---|---|
| **Theory / Background** | A0 (transfer learning, progressive unfreezing), A7 (Bayesian uncertainty), B2 (contrastive learning), B3 (dimensionality reduction) |
| **System Design** | A0 (proper fine-tuning pipeline), A1–A5 (training optimizations), B1 (feature extraction), B2 (SupCon integration) |
| **Experiments / Evaluation** | A0 (genuine model learning across cycles), A6 (calibration), A7 (MC Dropout vs. standard), B2 (CE vs. SupCon), B3 (UMAP evolution) |
| **Discussion** | A0 (why reset mode matters), A6 (calibration analysis), B3 (visual interpretation), B2+B3 (SupCon impact on feature space) |

---

## Verification Checklist

Before starting the thesis, confirm all of these work:

- [ ] A single model's feature space visibly evolves across 10 AL cycles (UMAP figures)
- [ ] Probe images show the same model improving predictions over cycles (not resetting)
- [ ] Training curves show warmup + cosine decay (not flat LR)
- [ ] Best-model checkpoint is restored before evaluation (not last-epoch model)
- [ ] ECE is tracked per cycle (calibration improves as labeled pool grows)
- [ ] Strategy comparison: Random vs. Entropy vs. Margin vs. MC Dropout (convergence curves)
- [ ] Architecture comparison: ResNet18 vs. ResNet50 vs. MobileNetV3 (same strategy)
- [ ] SupCon vs. CrossEntropy: UMAP side-by-side showing tighter clusters
- [ ] Embedding plots render cleanly for ~16K points
- [ ] Confusion matrices saved per cycle
- [ ] All results reproducible via saved config.yaml per run
