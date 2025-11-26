"""
ALDataManager - Manages labeled and unlabeled pools for Active Learning.

Responsibilities:
- Maintain index-based separation of labeled/unlabeled pools
- Initialize with random labeled set
- Move samples between pools (simulated annotation)
- Provide DataLoaders for each pool
- Save/load state for reproducibility
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
        exp_dir: Optional[Path] = None
    ):
        """
        Initialize AL Data Manager.
        
        Args:
            dataset: PyTorch dataset (training pool only, not val/test)
            initial_pool_size: Number of samples to start in labeled pool
            seed: Random seed for reproducibility
            exp_dir: Experiment directory for saving state (optional)
        """
        self.dataset = dataset
        self.initial_pool_size = initial_pool_size
        self.seed = seed
        self.exp_dir = Path(exp_dir) if exp_dir else None
        
        self.total_samples = len(dataset)
        
        # Set seed and shuffle
        np.random.seed(seed)
        
        all_indices = np.arange(self.total_samples)
        np.random.shuffle(all_indices)
        
        n_initial = min(initial_pool_size, self.total_samples)
        
        # Store as lists for consistent ordering
        self._labeled_list = all_indices[:n_initial].tolist()
        self._unlabeled_list = all_indices[n_initial:].tolist()
        
        self.query_history = []
        
        logger.info(f"ALDataManager initialized:")
        logger.info(f"  Total samples: {self.total_samples}")
        logger.info(f"  Initial labeled: {len(self._labeled_list)}")
        logger.info(f"  Initial unlabeled: {len(self._unlabeled_list)}")
    
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
        """Get list of current unlabeled indices."""
        return self._unlabeled_list.copy()
    
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
        # Convert to list
        if isinstance(query_indices, torch.Tensor):
            query_indices = query_indices.cpu().numpy()
        if isinstance(query_indices, np.ndarray):
            query_indices = query_indices.tolist()
        
        # Validate indices
        valid_query_indices = [i for i in query_indices if 0 <= i < len(self._unlabeled_list)]
        
        if len(valid_query_indices) < len(query_indices):
            logger.warning(
                f"Some query indices out of bounds. "
                f"Requested: {len(query_indices)}, Valid: {len(valid_query_indices)}"
            )
        
        # Convert to absolute indices
        absolute_indices = [self._unlabeled_list[i] for i in valid_query_indices]
        
        # Add to labeled pool
        self._labeled_list.extend(absolute_indices)
        
        # Remove from unlabeled pool
        absolute_set = set(absolute_indices)
        self._unlabeled_list = [i for i in self._unlabeled_list if i not in absolute_set]
        
        # Record history
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
    
    def save_state(self, path: Optional[Path] = None) -> Dict:
        """Save pool state to JSON."""
        state = {
            "labeled_indices": self._labeled_list,
            "unlabeled_indices": self._unlabeled_list,
            "query_history": self.query_history,
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
        
        logger.info(
            f"Pool state loaded: Labeled={len(self._labeled_list)}, "
            f"Unlabeled={len(self._unlabeled_list)}"
        )
    
    def reset(self) -> None:
        """Reset pools to initial state."""
        np.random.seed(self.seed)
        
        all_indices = np.arange(self.total_samples)
        np.random.shuffle(all_indices)
        
        n_initial = min(self.initial_pool_size, self.total_samples)
        
        self._labeled_list = all_indices[:n_initial].tolist()
        self._unlabeled_list = all_indices[n_initial:].tolist()
        self.query_history = []
        
        logger.info(f"Pools reset to initial state")
    
    def get_query_history(self) -> List[Dict]:
        """Get history of all queries."""
        return self.query_history.copy()