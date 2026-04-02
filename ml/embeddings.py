"""
embeddings.py — UMAP projection and persistence for cycle-level embeddings.

Usage in active_loop.py:
    coords_2d = compute_umap_projection(embeddings)
    path = save_cycle_embeddings(exp_dir, cycle, coords_2d, labels, pool_membership)
"""

from pathlib import Path
from typing import List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Maximum number of unlabeled samples to include in the UMAP projection.
# Keeps computation time reasonable for large pools (Stanford Cars ~16K).
UMAP_UNLABELED_SAMPLE_LIMIT = 2000


def compute_umap_projection(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """Project high-D embeddings to 2D using UMAP.

    Args:
        embeddings:    [N, D] float array of backbone feature vectors
        n_neighbors:   UMAP neighborhood size
        min_dist:      UMAP minimum distance between points
        metric:        Distance metric (cosine works best for L2-normalised features)
        random_state:  Reproducibility seed

    Returns:
        [N, 2] float array of 2-D coordinates
    """
    try:
        import umap  # umap-learn
    except ImportError:
        raise ImportError(
            "umap-learn is required for embedding visualisation. "
            "Install it with: pip install umap-learn"
        )

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=2,
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings).astype(np.float32)


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
) -> Optional[str]:
    """High-level helper called from active_loop.finalize_cycle().

    Extracts embeddings for the labeled pool + a capped sample of the
    unlabeled pool, runs UMAP, and saves the result.

    Args:
        queried_abs_indices: Absolute dataset indices of samples queried in
            the *previous* cycle.  These are now in the labeled pool and will
            be marked as pool=2 ("Queried this cycle") in the saved .npz.

    Returns:
        Path to the .npz file, or None if umap-learn is not installed.
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

    # Unlabeled pool (capped)
    unlabeled_indices = data_manager._unlabeled_list
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
        subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    emb_unlabeled, lbl_unlabeled = trainer.get_embeddings(unlabeled_loader)
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

    coords_2d = compute_umap_projection(all_embeddings)
    return save_cycle_embeddings(exp_dir, cycle, coords_2d, all_labels, all_pool)
