# ml/ — Machine Learning Components

## Module Responsibilities

| File | Role |
|---|---|
| `models.py` | TIMM model loader; backbone feature dimension helper |
| `trainer.py` | Training loop, validation, evaluation, checkpointing |
| `data_manager.py` | Labeled/unlabeled pool management (index-based, no data copy) |
| `dataloader.py` | ImageFolder loading, train/val/test splits, augmentation |
| `strategies.py` | Uncertainty sampling strategy registry |
| `embeddings.py` | UMAP projection saved per cycle for the results dashboard |
| `losses.py` | SupConLoss (Khosla 2020) + ProjectionHead |

## TIMM Model Usage

All models are loaded via `timm.create_model(name, pretrained=True, num_classes=n)`. The `name` field in `ModelConfig` must be a valid TIMM model identifier. Models download to `~/.cache/huggingface/hub` on first use — internet access required.

`get_feature_dim(model)` reads `model.num_features` (TIMM's standard attribute) for the backbone embedding dimension used by `ProjectionHead`. Do not hardcode embedding dimensions.

## Adding a Sampling Strategy

1. Write a function with this exact signature:
   ```python
   def my_strategy(
       model: torch.nn.Module,
       unlabeled_loader: DataLoader,
       n_samples: int,
       device: str = "cuda",
       heartbeat_fn: Optional[Callable[[], None]] = None,
   ) -> np.ndarray:
   ```
   Return value: **relative indices into `unlabeled_loader`** (0 to `len(unlabeled_pool)-1`), not absolute dataset indices. `ALDataManager` handles conversion.

2. Call `heartbeat_fn()` periodically inside the inference loop. This updates the worker heartbeat so the UI does not show a stale-thread warning during long query passes.

3. Add the name → function mapping to the `STRATEGIES` dict in `strategies.py`.

4. Add the name to `valid_strategies` in `config.py` (`Config.validate()`).

5. Add a human-readable description to `_STRATEGY_DESCRIPTIONS` in `active_loop.py`.

## Trainer Patterns

- `Trainer` manages optimizer, scheduler, criterion, and checkpointing. One instance per run.
- `train_single_epoch(train_loader)` → returns `EpochMetrics`. Called once per epoch by `active_loop.py`.
- `reset_for_new_cycle(mode)` handles weight reuse between cycles. Modes from `ALConfig.reset_mode`:
  - `"continue"` — keep all weights as-is
  - `"pretrained"` — reload original pretrained weights
  - `"head_only"` — re-initialize classifier head only
  - `"none"` — freeze everything
- Early stopping is checked by the loop in `active_loop.py` after each `EPOCH_DONE` event, not inside `Trainer`.
- Checkpoints save to `{exp_dir}/checkpoints/`. Best model tracked by `val_accuracy`.

## ALDataManager — Index Convention

Pools are index lists into the train split of the dataset — no data is duplicated.

```python
manager.get_labeled_loader(batch_size)       # DataLoader over labeled pool
manager.get_unlabeled_loader(batch_size)     # DataLoader over unlabeled pool
manager.update_labeled_pool(abs_indices)     # Move abs indices from unlabeled → labeled
manager.unlabeled_to_absolute(rel_indices)   # Convert relative → absolute indices
```

**Critical:** Strategy functions return **relative indices** (position within the unlabeled pool). `active_loop.py` converts them to absolute dataset indices via `manager.unlabeled_to_absolute()` before calling `update_labeled_pool()`. Never pass relative indices to `update_labeled_pool`.

## Loss Functions

Three `loss_fn` options (set in `TrainingConfig`):
- `"cross_entropy"` — standard CE with label smoothing
- `"supcon"` — Supervised Contrastive Loss only (uses `ProjectionHead`)
- `"combined"` — `(1 - supcon_weight) * CE + supcon_weight * SupCon`

`ProjectionHead` and `SupConLoss` are instantiated in `Trainer.__init__()` only when `loss_fn` is `"supcon"` or `"combined"`. The projection head parameters are included in the optimizer param groups.

## UMAP Embeddings

`build_cycle_embeddings(model, labeled_loader, unlabeled_loader, device, exp_dir, cycle)` runs at the end of each cycle evaluation. It:
1. Extracts backbone features via `extract_features()`
2. Subsamples the unlabeled pool to `UMAP_UNLABELED_SAMPLE_LIMIT = 2000` for performance
3. Computes a 2D UMAP projection
4. Saves to `{exp_dir}/cycle_{n}_embeddings.npz`

The results dashboard reads these `.npz` files directly from disk.
