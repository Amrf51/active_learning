"""
ALDataManager - Manages labeled and unlabeled pools for Active Learning.

Responsibilities:
- Maintain index-based separation of labeled/unlabeled pools
- Initialize with random labeled set
- Move samples between pools (simulated or real annotation)
- Provide DataLoaders for each pool
- Save/load state for reproducibility
- Support user annotations from dashboard
"""

import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class PoolSubset(Dataset):
    """
    A subset that wraps a dataset and selects specific indices.
    Delegates __getitem__ to parent, preserving transforms.
    """
    
    def __init__(self, dataset: Dataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return self.dataset[actual_idx]


class ALDataManager:
    """
    Manages labeled and unlabeled data pools for Active Learning.
    
    Key design: Index-based pool management (no data copying).
    
    Usage:
        1. Pass the training dataset (already has correct transforms)
        2. Manager splits into labeled/unlabeled pools
        3. Query returns indices into unlabeled pool
        4. update_labeled_pool() moves indices to labeled pool
    """
    
    def __init__(
        self,
        dataset: Dataset,
        initial_pool_size: int = 50,
        seed: int = 42,
        exp_dir: Optional[Path] = None,
        stratified_init: bool = True,
    ):
        """
        Initialize AL Data Manager.

        Args:
            dataset: PyTorch dataset (training pool only, not val/test)
            initial_pool_size: Number of samples to start in labeled pool
            seed: Random seed for reproducibility
            exp_dir: Experiment directory for saving state (optional)
            stratified_init: If True, guarantee ≥1 sample per class in the
                initial labeled pool (falls back to random when budget is
                smaller than the number of classes).
        """
        self.dataset = dataset
        self.initial_pool_size = initial_pool_size
        self.seed = seed
        self.stratified_init = stratified_init
        self.exp_dir = Path(exp_dir) if exp_dir else None

        self.total_samples = len(dataset)

        # Label cache for efficient class-based operations
        self._label_cache = {}

        np.random.seed(seed)

        labeled, unlabeled = self._build_initial_pools(initial_pool_size)
        self._labeled_list = labeled
        self._unlabeled_list = unlabeled

        self.query_history = []
        self.annotation_history = []

        logger.info(f"ALDataManager initialized:")
        logger.info(f"  Total samples: {self.total_samples}")
        logger.info(f"  Initial labeled: {len(self._labeled_list)}")
        logger.info(f"  Initial unlabeled: {len(self._unlabeled_list)}")
    
    def _build_initial_pools(self, n_initial: int):
        """
        Build labeled / unlabeled index lists for pool initialization.

        When stratified_init=True and the budget is large enough, guarantees
        at least one sample per class before filling the rest randomly.
        Falls back to a plain random shuffle if the budget is smaller than
        the number of unique classes.

        Returns:
            (labeled_list, unlabeled_list) — both are Python lists of ints.
        """
        n_initial = min(n_initial, self.total_samples)
        all_indices = np.arange(self.total_samples)

        if not self.stratified_init:
            np.random.shuffle(all_indices)
            return all_indices[:n_initial].tolist(), all_indices[n_initial:].tolist()

        # Group indices by class label (uses fast label cache path)
        class_to_indices: dict = {}
        for idx in all_indices:
            label = self._get_label(int(idx))
            class_to_indices.setdefault(label, []).append(int(idx))

        num_classes = len(class_to_indices)

        if n_initial < num_classes:
            # Budget too small to place one per class — fall back to random
            logger.warning(
                f"Stratified init: budget ({n_initial}) < num_classes ({num_classes}). "
                "Falling back to random initialization."
            )
            np.random.shuffle(all_indices)
            return all_indices[:n_initial].tolist(), all_indices[n_initial:].tolist()

        # Pick exactly one random sample per class
        labeled_set = []
        remaining_pool = []
        for indices in class_to_indices.values():
            pick = int(np.random.choice(indices))
            labeled_set.append(pick)
            remaining_pool.extend([i for i in indices if i != pick])

        # Fill remaining budget randomly from the rest
        extra_needed = n_initial - len(labeled_set)
        np.random.shuffle(remaining_pool)
        labeled_set.extend(remaining_pool[:extra_needed])
        unlabeled_list = remaining_pool[extra_needed:]

        logger.info(
            f"Stratified init: {num_classes} classes covered, "
            f"{len(labeled_set)} labeled, {len(unlabeled_list)} unlabeled."
        )
        return labeled_set, unlabeled_list

    def _get_label(self, idx: int) -> int:
        """
        Get label for a dataset index with caching.

        This avoids loading images when we only need labels for
        class distribution calculations.
        
        Args:
            idx: Absolute index in the dataset
            
        Returns:
            Label as integer
        """
        if idx not in self._label_cache:
            label = None

            # Fast path for SplitSubset/ImageFolderWithIndex wrapper used by this project.
            if hasattr(self.dataset, "indices") and hasattr(self.dataset, "parent"):
                try:
                    actual_idx = self.dataset.indices[idx]
                    parent = self.dataset.parent
                    if hasattr(parent, "dataset") and hasattr(parent.dataset, "samples"):
                        label = int(parent.dataset.samples[actual_idx][1])
                except Exception:  # pylint: disable=broad-exception-caught
                    label = None

            # Generic Subset(dataset=..., indices=...) fast path.
            if label is None and hasattr(self.dataset, "dataset") and hasattr(self.dataset, "indices"):
                try:
                    actual_idx = self.dataset.indices[idx]
                    inner = self.dataset.dataset
                    if hasattr(inner, "targets"):
                        label = int(inner.targets[actual_idx])
                    elif hasattr(inner, "samples"):
                        label = int(inner.samples[actual_idx][1])
                except Exception:  # pylint: disable=broad-exception-caught
                    label = None

            # Direct ImageFolder-like datasets.
            if label is None and hasattr(self.dataset, "targets"):
                try:
                    label = int(self.dataset.targets[idx])
                except Exception:  # pylint: disable=broad-exception-caught
                    label = None
            if label is None and hasattr(self.dataset, "samples"):
                try:
                    label = int(self.dataset.samples[idx][1])
                except Exception:  # pylint: disable=broad-exception-caught
                    label = None

            # Slow fallback: run __getitem__ (loads image + transforms).
            if label is None:
                _, fetched = self.dataset[idx]
                label = int(fetched)

            self._label_cache[idx] = label
        return self._label_cache[idx]
    
    def get_pool_info(self) -> Dict:
        """Get current pool statistics."""
        labeled = len(self._labeled_list)
        unlabeled = len(self._unlabeled_list)
        
        return {
            "total": self.total_samples,
            "labeled": labeled,
            "unlabeled": unlabeled,
            "labeled_percentage": 100.0 * labeled / self.total_samples if self.total_samples > 0 else 0,
            "unlabeled_percentage": 100.0 * unlabeled / self.total_samples if self.total_samples > 0 else 0,
            "num_queries": len(self.query_history),
        }
    
    def get_labeled_loader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4
    ) -> DataLoader:
        """Get DataLoader for labeled samples."""
        subset = PoolSubset(self.dataset, self._labeled_list)
        
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def get_unlabeled_loader(
        self,
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 4
    ) -> DataLoader:
        """Get DataLoader for unlabeled samples."""
        subset = PoolSubset(self.dataset, self._unlabeled_list)
        
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def get_unlabeled_indices(self) -> List[int]:
        """Get list of current unlabeled indices (absolute)."""
        return self._unlabeled_list.copy()
    
    def get_labeled_indices(self) -> List[int]:
        """Get list of current labeled indices (absolute)."""
        return self._labeled_list.copy()
    
    def get_image_info(self, image_id: int) -> Dict:
        """
        Get full information about an image for display.
        
        Args:
            image_id: Absolute index in the dataset
            
        Returns:
            Dict with image_id, path, and label
        """
        try:
            if hasattr(self.dataset, 'parent'):
                parent = self.dataset.parent
                if hasattr(parent, 'dataset') and hasattr(parent.dataset, 'samples'):
                    actual_idx = self.dataset.indices[image_id] if image_id < len(self.dataset.indices) else image_id
                    path, label = parent.dataset.samples[actual_idx]
                    return {"image_id": image_id, "path": path, "label": label}
            
            if hasattr(self.dataset, 'dataset'):
                inner = self.dataset.dataset
                if hasattr(inner, 'samples'):
                    path, label = inner.samples[image_id]
                    return {"image_id": image_id, "path": path, "label": label}
            
            if hasattr(self.dataset, 'samples'):
                path, label = self.dataset.samples[image_id]
                return {"image_id": image_id, "path": path, "label": label}
            
            _, label = self.dataset[image_id]
            return {"image_id": image_id, "path": f"image_{image_id}", "label": int(label)}
            
        except Exception as e:
            logger.warning(f"Could not get image info for {image_id}: {e}")
            return {"image_id": image_id, "path": f"image_{image_id}", "label": -1}
    
    def get_ground_truth(self, image_id: int) -> int:
        """
        Get ground truth label for an image.
        
        Args:
            image_id: Absolute index in the dataset
            
        Returns:
            Ground truth label (int)
        """
        return self._get_label(image_id)
    
    def update_labeled_pool(
        self,
        query_indices: Union[np.ndarray, List[int], torch.Tensor]
    ) -> List[int]:
        """
        Move queried samples from unlabeled to labeled pool.
        
        Args:
            query_indices: Indices INTO the unlabeled pool (0 to len-1)
            
        Returns:
            List of absolute dataset indices that were moved
        """
        if isinstance(query_indices, torch.Tensor):
            query_indices = query_indices.cpu().numpy()
        if isinstance(query_indices, np.ndarray):
            query_indices = query_indices.tolist()
        
        valid_query_indices = [i for i in query_indices if 0 <= i < len(self._unlabeled_list)]
        
        if len(valid_query_indices) < len(query_indices):
            logger.warning(
                f"Some query indices out of bounds. "
                f"Requested: {len(query_indices)}, Valid: {len(valid_query_indices)}"
            )
        
        absolute_indices = [self._unlabeled_list[i] for i in valid_query_indices]
        
        self._labeled_list.extend(absolute_indices)
        
        absolute_set = set(absolute_indices)
        self._unlabeled_list = [i for i in self._unlabeled_list if i not in absolute_set]
        
        self.query_history.append({
            "query_size": len(absolute_indices),
            "labeled_pool_size": len(self._labeled_list),
            "unlabeled_pool_size": len(self._unlabeled_list),
        })
        
        logger.info(
            f"Pool updated: +{len(absolute_indices)} labeled | "
            f"Labeled: {len(self._labeled_list)}, Unlabeled: {len(self._unlabeled_list)}"
        )
        
        return absolute_indices
    
    def update_labeled_pool_with_annotations(
        self,
        annotations: List[Dict]
    ) -> Dict:
        """
        Move queried samples to labeled pool using user-provided annotations.
        
        In this educational simulation, we use ground truth labels for
        actual training, but log user annotations for analysis.
        
        Args:
            annotations: List of dicts with 'image_id' and 'user_label' keys
            
        Returns:
            Dict with annotation summary stats
        """
        if not annotations:
            return {
                "moved_count": 0,
                "annotation_accuracy": 0.0,
                "correct_count": 0,
                "total_count": 0
            }
        
        image_ids = [a["image_id"] for a in annotations]

        # Build reverse index for O(1) lookups instead of O(n) list.index()
        unlabeled_index = {idx: pos for pos, idx in enumerate(self._unlabeled_list)}

        query_indices = []
        for img_id in image_ids:
            pos = unlabeled_index.get(img_id)
            if pos is not None:
                query_indices.append(pos)
            else:
                logger.warning(f"Image {img_id} not in unlabeled pool, skipping")
        
        moved = self.update_labeled_pool(query_indices)
        
        correct_count = 0
        annotation_details = []
        
        for a in annotations:
            img_id = a["image_id"]
            user_label = a["user_label"]
            ground_truth = self.get_ground_truth(img_id)
            is_correct = (user_label == ground_truth)
            
            if is_correct:
                correct_count += 1
            
            annotation_details.append({
                "image_id": img_id,
                "user_label": user_label,
                "ground_truth": ground_truth,
                "correct": is_correct
            })
        
        total = len(annotations)
        accuracy = correct_count / total if total > 0 else 0.0
        
        self.annotation_history.append({
            "annotations": annotation_details,
            "accuracy": accuracy,
            "correct": correct_count,
            "total": total
        })
        
        logger.info(
            f"Annotations processed: {correct_count}/{total} correct ({accuracy:.1%})"
        )
        
        return {
            "moved_count": len(moved),
            "annotation_accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total,
            "details": annotation_details
        }
    
    def get_samples_by_class(self, pool: str = "labeled") -> Dict[int, List[int]]:
        """
        Get indices grouped by class label.
        
        Args:
            pool: "labeled" or "unlabeled"
            
        Returns:
            Dict mapping class label to list of indices
        """
        if pool == "labeled":
            indices = self._labeled_list
        else:
            indices = self._unlabeled_list
        
        by_class = {}
        for idx in indices:
            label = self._get_label(idx)
            if label not in by_class:
                by_class[label] = []
            by_class[label].append(idx)
        
        return by_class
    
    def get_class_distribution(self, pool: str = "labeled") -> Dict[int, int]:
        """
        Get class distribution for a pool.
        
        Args:
            pool: "labeled" or "unlabeled"
            
        Returns:
            Dict mapping class label to count
        """
        by_class = self.get_samples_by_class(pool)
        return {k: len(v) for k, v in by_class.items()}
    
    def save_state(self, path: Optional[Path] = None) -> Dict:
        """Save pool state to JSON."""
        state = {
            "labeled_indices": self._labeled_list,
            "unlabeled_indices": self._unlabeled_list,
            "query_history": self.query_history,
            "annotation_history": self.annotation_history,
            "pool_info": self.get_pool_info(),
            "initial_pool_size": self.initial_pool_size,
            "seed": self.seed,
            "total_samples": self.total_samples,
        }
        
        if path is None and self.exp_dir:
            path = self.exp_dir / "al_pool_state.json"
        
        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(state, f, indent=2)
            logger.info(f"Pool state saved to {path}")
        
        return state
    
    def load_state(self, path_or_state: Union[Path, str, Dict]) -> None:
        """Load pool state from JSON file or dict."""
        if isinstance(path_or_state, dict):
            state = path_or_state
        else:
            path = Path(path_or_state)
            if not path.exists():
                raise FileNotFoundError(f"State file not found: {path}")
            with open(path, "r") as f:
                state = json.load(f)
        
        self._labeled_list = state["labeled_indices"]
        self._unlabeled_list = state["unlabeled_indices"]
        self.query_history = state.get("query_history", [])
        self.annotation_history = state.get("annotation_history", [])
        
        logger.info(
            f"Pool state loaded: Labeled={len(self._labeled_list)}, "
            f"Unlabeled={len(self._unlabeled_list)}"
        )
    
    def reset(self) -> None:
        """Reset pools to initial state."""
        np.random.seed(self.seed)

        labeled, unlabeled = self._build_initial_pools(self.initial_pool_size)
        self._labeled_list = labeled
        self._unlabeled_list = unlabeled
        self.query_history = []
        self.annotation_history = []

        logger.info("Pools reset to initial state")
    
    def get_query_history(self) -> List[Dict]:
        """Get history of all queries."""
        return self.query_history.copy()
    
    def get_annotation_history(self) -> List[Dict]:
        """Get history of all annotations."""
        return self.annotation_history.copy()
    
    def get_annotation_summary(self) -> Dict:
        """Get summary statistics for all annotations."""
        if not self.annotation_history:
            return {
                "total_annotations": 0,
                "total_correct": 0,
                "overall_accuracy": 0.0,
                "cycles": 0
            }
        
        total_correct = sum(h["correct"] for h in self.annotation_history)
        total_annotations = sum(h["total"] for h in self.annotation_history)
        
        return {
            "total_annotations": total_annotations,
            "total_correct": total_correct,
            "overall_accuracy": total_correct / total_annotations if total_annotations > 0 else 0.0,
            "cycles": len(self.annotation_history),
            "per_cycle_accuracy": [h["accuracy"] for h in self.annotation_history]
        }
