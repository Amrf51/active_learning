"""
ALDataManager - Manages Labeled and Unlabeled data indices for Active Learning.

Responsibilities:
- Maintain separation of labeled and unlabeled pools
- Initialize with random labeled set
- Move samples between pools as they're queried
- Provide DataLoaders for labeled/unlabeled data
- Save/load pool state for reproducibility
"""

import json
from pathlib import Path
from typing import List
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
import logging

logger = logging.getLogger(__name__)


class ALDataManager:
    """
    Manages splitting and movement of data indices for Active Learning.
    
    Instead of copying data, we maintain indices to the original dataset.
    This allows efficient memory usage and easy pool management.
    """
    
    def __init__(self, 
                 dataset,
                 n_initial_samples: int = 50,
                 seed: int = 42,
                 exp_dir: Path = None):
        """
        Initialize ALDataManager.
        
        Args:
            dataset: PyTorch Dataset (e.g., ImageFolder or any torch.utils.data.Dataset)
            n_initial_samples: Number of samples to initialize labeled pool with
            seed: Random seed for reproducibility
            exp_dir: Experiment directory to save state (optional)
        """
        self.dataset = dataset
        self.n_initial_samples = n_initial_samples
        self.seed = seed
        self.exp_dir = Path(exp_dir) if exp_dir else None
        
        # Total number of samples in dataset
        self.total_samples = len(dataset)
        
        # Pool indices (not the data itself, just indices)
        self.labeled_indices = []
        self.unlabeled_indices = list(range(self.total_samples))
        
        # Track querying history
        self.query_history = []
        
        # Initialize labeled pool with random samples
        self._initialize_labeled_pool()
        
        logger.info("ALDataManager initialized")
        logger.info(f"  Total samples: {self.total_samples}")
        logger.info(f"  Labeled pool: {len(self.labeled_indices)}")
        logger.info(f"  Unlabeled pool: {len(self.unlabeled_indices)}")
    
    def _initialize_labeled_pool(self):
        """Randomly select initial labeled pool from all data."""
        np.random.seed(self.seed)
        
        # Random indices for initial labeled set
        initial_indices = np.random.choice(
            self.total_samples,
            size=min(self.n_initial_samples, self.total_samples),
            replace=False
        )
        
        # Convert to sorted list for consistency
        self.labeled_indices = sorted(initial_indices.tolist())
        
        # Remaining indices are unlabeled
        self.unlabeled_indices = [
            i for i in range(self.total_samples)
            if i not in self.labeled_indices
        ]
        
        logger.info(
            f"Initialized labeled pool with {len(self.labeled_indices)} samples "
            f"(random seed={self.seed})"
        )
    
    def get_labeled_dataset(self) -> Subset:
        """
        Get Subset dataset for labeled data.
        
        Returns:
            torch.utils.data.Subset: Subset of the dataset containing only labeled indices
        """
        return Subset(self.dataset, self.labeled_indices)
    
    def get_unlabeled_dataset(self) -> Subset:
        """
        Get Subset dataset for unlabeled data.
        
        Returns:
            torch.utils.data.Subset: Subset of the dataset containing only unlabeled indices
        """
        return Subset(self.dataset, self.unlabeled_indices)
    
    def get_labeled_loader(self, 
                          batch_size: int = 32, 
                          shuffle: bool = True,
                          num_workers: int = 4) -> DataLoader:
        """
        Create and return DataLoader for labeled data.
        
        Args:
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data
            num_workers: Number of workers for data loading
            
        Returns:
            torch.utils.data.DataLoader: DataLoader for labeled subset
        """
        labeled_dataset = self.get_labeled_dataset()
        
        return DataLoader(
            labeled_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    
    def get_unlabeled_loader(self,
                            batch_size: int = 32,
                            shuffle: bool = False,
                            num_workers: int = 4) -> DataLoader:
        """
        Create and return DataLoader for unlabeled data.
        
        Args:
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data
            num_workers: Number of workers for data loading
            
        Returns:
            torch.utils.data.DataLoader: DataLoader for unlabeled subset
        """
        unlabeled_dataset = self.get_unlabeled_dataset()
        
        return DataLoader(
            unlabeled_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    
    def update_labeled_pool(self, query_indices: np.ndarray) -> List[int]:
        """
        Move queried samples from unlabeled to labeled pool.
        
        This is the core operation of Active Learning:
        - Takes indices WITHIN the unlabeled pool (0 to len(unlabeled_indices))
        - Converts them to absolute indices in the full dataset
        - Moves them from unlabeled to labeled
        
        Args:
            query_indices: Indices into the unlabeled pool (numpy array or list)
                          Example: if unlabeled has 350 samples, indices are 0-349
            
        Returns:
            List of absolute indices that were moved to labeled pool
        """
        # Ensure query_indices is numpy array
        if isinstance(query_indices, list):
            query_indices = np.array(query_indices)
        
        # Validate indices
        if len(query_indices) > len(self.unlabeled_indices):
            logger.warning(
                f"Requested {len(query_indices)} samples but only "
                f"{len(self.unlabeled_indices)} available. Using available."
            )
            query_indices = query_indices[:len(self.unlabeled_indices)]
        
        # Convert from unlabeled pool indices to absolute dataset indices
        absolute_indices = [
            self.unlabeled_indices[i] for i in query_indices
        ]
        
        # Move to labeled pool
        self.labeled_indices.extend(absolute_indices)
        self.labeled_indices.sort()
        
        # Update unlabeled pool (remove the moved indices)
        self.unlabeled_indices = [
            i for i in self.unlabeled_indices
            if i not in absolute_indices
        ]
        
        # Track in history
        self.query_history.append({
            "queried_count": len(absolute_indices),
            "labeled_pool_size": len(self.labeled_indices),
            "unlabeled_pool_size": len(self.unlabeled_indices),
        })
        
        logger.info(f"Updated pools after query:")
        logger.info(f"  Queried: {len(absolute_indices)} samples")
        logger.info(f"  Labeled pool: {len(self.labeled_indices)}")
        logger.info(f"  Unlabeled pool: {len(self.unlabeled_indices)}")
        
        return absolute_indices
    
    def get_pool_info(self) -> dict:
        """
        Get current pool information.
        
        Returns:
            dict: Information about current pools
                {
                    "labeled": int (number of labeled samples),
                    "unlabeled": int (number of unlabeled samples),
                    "total": int (total samples),
                    "labeled_percentage": float (% of data that is labeled)
                }
        """
        return {
            "labeled": len(self.labeled_indices),
            "unlabeled": len(self.unlabeled_indices),
            "total": self.total_samples,
            "labeled_percentage": 100 * len(self.labeled_indices) / self.total_samples,
        }
    
    def save_state(self) -> dict:
        """
        Save pool state to JSON for reproducibility and resumption.
        
        Returns:
            dict: State dictionary (also saved to file if exp_dir provided)
        """
        state = {
            "labeled_indices": self.labeled_indices,
            "unlabeled_indices": self.unlabeled_indices,
            "query_history": self.query_history,
            "pool_info": self.get_pool_info(),
            "n_initial_samples": self.n_initial_samples,
            "seed": self.seed,
            "total_samples": self.total_samples,
        }
        
        # Save to file if experiment directory provided
        if self.exp_dir:
            state_file = self.exp_dir / "al_pool_state.json"
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
            logger.info(f"Pool state saved to {state_file}")
        
        return state
    
    def load_state(self, state: dict):
        """
        Load pool state from dict (e.g., from saved JSON).
        
        Args:
            state: State dictionary with labeled_indices, unlabeled_indices, etc.
        """
        self.labeled_indices = state.get("labeled_indices", [])
        self.unlabeled_indices = state.get("unlabeled_indices", [])
        self.query_history = state.get("query_history", [])
        logger.info("Pool state loaded from checkpoint")