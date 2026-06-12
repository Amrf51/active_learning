"""
embeddings.py — UMAP projection and persistence for cycle-level embeddings.

Usage in active_loop.py:
    path = build_cycle_embeddings(trainer, data_manager, exp_dir, cycle, rng, ...)

build_cycle_embeddings() projects with a run-wide shared UMAP reducer
(project_with_shared_reducer) so coordinates stay aligned across cycles, and
saves synchronously so the .npz exists when it returns.
"""

from pathlib import Path
from typing import Callable, List, Optional
import pickle
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Maximum number of unlabeled samples to include in the UMAP projection.
# Keeps computation time reasonable for large pools (Stanford Cars ~16K).
UMAP_UNLABELED_SAMPLE_LIMIT = 2000

# Filename for the persisted UMAP reducer, shared across all cycles of a run so
# that 2-D coordinates live in the same space and the evolution view is coherent.
REDUCER_FILENAME = "umap_reducer.pkl"


def compute_umap_projection(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """Project high-D embeddings to 2D using a freshly-fitted UMAP reducer.

    Args:
        embeddings:    [N, D] float array of backbone feature vectors
        n_neighbors:   UMAP neighborhood size
        min_dist:      UMAP minimum distance between points
        metric:        Distance metric (cosine works best for L2-normalised features)
        random_state:  Reproducibility seed

    Returns:
        [N, 2] float array of 2-D coordinates
    """
    coords, _ = _fit_reducer(
        embeddings,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return coords


def _fit_reducer(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
):
    """Fit a UMAP reducer and return (coords_2d, fitted_reducer)."""
    try:
        import umap  # umap-learn
    except ImportError:
        raise ImportError(
            "umap-learn is required for embedding visualisation. "
            "Install it with: pip install umap-learn"
        )

    # n_neighbors must be < n_samples; clamp for tiny initial pools.
    n_neighbors = max(2, min(n_neighbors, len(embeddings) - 1))
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=2,
        random_state=random_state,
    )
    coords = reducer.fit_transform(embeddings).astype(np.float32)
    return coords, reducer


def project_with_shared_reducer(embeddings: np.ndarray, exp_dir: Path) -> np.ndarray:
    """Project embeddings into a 2-D space that is consistent across cycles.

    The first cycle fits a UMAP reducer and persists it to
    ``{exp_dir}/embeddings/{REDUCER_FILENAME}``. Later cycles load that reducer
    and only ``transform`` their points, so the axes stay fixed and the evolution
    view is coherent rather than rotating/flipping each cycle.

    Falls back to a fresh fit if the persisted reducer cannot be loaded or used.
    """
    reducer_path = Path(exp_dir) / "embeddings" / REDUCER_FILENAME

    if reducer_path.exists():
        try:
            with open(reducer_path, "rb") as f:
                reducer = pickle.load(f)
            return reducer.transform(embeddings).astype(np.float32)
        except Exception:
            logger.warning(
                "Failed to reuse persisted UMAP reducer at %s — refitting (cycles "
                "may not be aligned).", reducer_path, exc_info=True,
            )

    # First cycle (or fallback): fit a new reducer and persist it.
    coords, reducer = _fit_reducer(embeddings)
    try:
        reducer_path.parent.mkdir(parents=True, exist_ok=True)
        with open(reducer_path, "wb") as f:
            pickle.dump(reducer, f)
    except Exception:
        logger.warning("Failed to persist UMAP reducer to %s.", reducer_path, exc_info=True)
    return coords


def save_cycle_embeddings(
    exp_dir: Path,
    cycle: int,
    coords_2d: np.ndarray,
    labels: np.ndarray,
    pool_membership: np.ndarray,
    uncertainty_scores: Optional[np.ndarray] = None,
) -> str:
    """Persist UMAP 2-D coordinates + metadata for one AL cycle.

    Args:
        exp_dir:           Experiment root directory
        cycle:             Cycle number (1-indexed)
        coords_2d:         [N, 2] UMAP coordinates
        labels:            [N] integer class labels
        pool_membership:   [N] integer array — 0=labeled, 1=unlabeled, 2=queried
        uncertainty_scores: Optional [N] uncertainty values

    Returns:
        Absolute path to the saved .npz file (for CycleMetrics.embeddings_path)
    """
    path = Path(exp_dir) / "embeddings" / f"cycle_{cycle}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs = dict(
        coords=coords_2d,
        labels=labels,
        pool=pool_membership,
    )
    if uncertainty_scores is not None:
        save_kwargs["uncertainty"] = uncertainty_scores

    np.savez_compressed(path, **save_kwargs)
    logger.info(f"Cycle {cycle} embeddings saved → {path}")
    return str(path)


def build_cycle_embeddings(
    trainer,
    data_manager,
    exp_dir: Path,
    cycle: int,
    rng: np.random.Generator,
    queried_abs_indices: Optional[List[int]] = None,
    heartbeat_fn: Optional[Callable[[], None]] = None,
) -> Optional[str]:
    """High-level helper called from active_loop.finalize_cycle().

    Extracts embeddings for the labeled pool + a capped sample of the
    unlabeled pool, projects them with a run-wide shared UMAP reducer, and
    saves the .npz synchronously before returning.

    The projection runs synchronously (not in a background thread) so the
    .npz file is guaranteed to exist when this returns — a fire-and-forget
    daemon thread was previously killed on process exit, leaving the results
    dashboard with no embeddings to show. ``heartbeat_fn`` keeps the worker
    heartbeat fresh during the now-blocking fit.

    Args:
        queried_abs_indices: Absolute dataset indices of samples queried in
            the *previous* cycle.  These are now in the labeled pool and will
            be marked as pool=2 ("Queried this cycle") in the saved .npz.
        heartbeat_fn: Optional callback to keep worker heartbeat fresh
            during embedding extraction.

    Returns:
        Path to the saved .npz file, or None if umap-learn is unavailable or
        the projection failed.
    """
    try:
        import umap  # noqa: F401 — check availability before heavy work
    except ImportError:
        logger.warning("umap-learn not installed — skipping embedding projection.")
        return None

    num_workers = trainer.config.data.num_workers
    batch_size = trainer.config.training.batch_size

    # Labeled pool
    labeled_loader = data_manager.get_labeled_loader(
        batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    emb_labeled, lbl_labeled = trainer.get_embeddings(labeled_loader)
    pool_labeled = np.zeros(len(lbl_labeled), dtype=np.int8)  # 0 = labeled
    if heartbeat_fn:
        heartbeat_fn()

    # Unlabeled pool (capped)
    unlabeled_indices = data_manager._unlabeled_list
    emb_unlabeled = np.empty((0, emb_labeled.shape[1]), dtype=emb_labeled.dtype)
    lbl_unlabeled = np.empty(0, dtype=lbl_labeled.dtype)
    unlabeled_loader = None
    if len(unlabeled_indices) > 0:
        if len(unlabeled_indices) > UMAP_UNLABELED_SAMPLE_LIMIT:
            sample_idx = rng.choice(
                len(unlabeled_indices), size=UMAP_UNLABELED_SAMPLE_LIMIT, replace=False
            )
            sampled = [unlabeled_indices[i] for i in sample_idx]
        else:
            sampled = unlabeled_indices

        from .data_manager import PoolSubset
        from torch.utils.data import DataLoader
        subset = PoolSubset(data_manager.dataset, sampled)
        unlabeled_loader = DataLoader(
            subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, persistent_workers=num_workers > 0,
        )
        emb_unlabeled, lbl_unlabeled = trainer.get_embeddings(unlabeled_loader)
        if heartbeat_fn:
            heartbeat_fn()

    pool_unlabeled = np.ones(len(lbl_unlabeled), dtype=np.int8)  # 1 = unlabeled
    all_embeddings = np.vstack([emb_labeled, emb_unlabeled])
    all_labels = np.concatenate([lbl_labeled, lbl_unlabeled])
    all_pool = np.concatenate([pool_labeled, pool_unlabeled])

    # Mark previously-queried samples (now in labeled pool) as pool=2
    if queried_abs_indices:
        queried_set = set(queried_abs_indices)
        labeled_indices = data_manager.get_labeled_indices()
        marked = 0
        for i, abs_idx in enumerate(labeled_indices):
            if abs_idx in queried_set:
                all_pool[i] = 2
                marked += 1
        if marked:
            logger.info("Marked %d queried points as pool=2 in cycle %d UMAP", marked, cycle)

    # Compute entropy-based uncertainty for every point (second inference pass).
    # Labeled points typically have low entropy; unlabeled points span the full range,
    # making the colormap useful for showing where the AL strategy focuses.
    uncertainty_scores: Optional[np.ndarray] = None
    try:
        _, prob_labeled = trainer.get_predictions_for_loader(labeled_loader)
        if unlabeled_loader is not None:
            _, prob_unlabeled = trainer.get_predictions_for_loader(unlabeled_loader)
            all_probs = np.vstack([prob_labeled, prob_unlabeled])
        else:
            all_probs = prob_labeled
        uncertainty_scores = trainer.compute_uncertainty_scores(all_probs, method="entropy")
        if heartbeat_fn:
            heartbeat_fn()
    except Exception:
        logger.warning("Uncertainty score computation failed — UMAP will be saved without uncertainty.", exc_info=True)

    # Project + save synchronously so the .npz is guaranteed present on return.
    # A run-wide shared reducer keeps coordinates aligned across cycles.
    try:
        if heartbeat_fn:
            heartbeat_fn()
        coords_2d = project_with_shared_reducer(all_embeddings, exp_dir)
        if heartbeat_fn:
            heartbeat_fn()
        return save_cycle_embeddings(
            exp_dir, cycle, coords_2d, all_labels, all_pool, uncertainty_scores
        )
    except Exception:
        logger.exception("UMAP projection failed for cycle %d — no embeddings saved.", cycle)
        return None
