"""Active Learning orchestration utilities."""

from pathlib import Path
from typing import Dict, List
import logging
import json
import numpy as np
from torch.utils.data import DataLoader, Subset

from src.dataloader import get_dataset_splits
from src.strategies import get_strategy
from src.models import extract_features

logger = logging.getLogger(__name__)


class ActiveLearningOrchestrator:
    """Manage active learning cycles, pools, and bookkeeping."""

    def __init__(self, config, device: str, base_exp_dir: Path):
        self.config = config
        self.device = device
        self.base_exp_dir = Path(base_exp_dir)
        np.random.seed(self.config.training.seed)

        self.train_dataset, self.val_dataset, self.test_dataset = get_dataset_splits(
            config.data, seed=config.training.seed
        )
        # random_split creates Subset objects with an "indices" field referencing the
        # underlying dataset. We keep track of those to build new subsets on demand.
        self.unlabeled_indices: List[int] = list(self.train_dataset.indices)
        self.labeled_indices: List[int] = []

        self.strategy_fn = get_strategy(
            config.active_learning.sampling_strategy,
            uncertainty_method=config.active_learning.uncertainty_method,
        )
        logger.info(
            "Active Learning orchestrator initialized with %d unlabeled samples",
            len(self.unlabeled_indices),
        )

    def initialize_pools(self, initial_pool_size: int) -> None:
        """Seed the labeled pool with an initial random subset."""
        if len(self.unlabeled_indices) == 0:
            logger.warning("No data available to initialize pools")
            return

        initial_size = min(initial_pool_size, len(self.unlabeled_indices))
        initial_indices = np.random.choice(
            self.unlabeled_indices, size=initial_size, replace=False
        ).tolist()
        self._label_samples(initial_indices)
        logger.info(
            "Initialized labeled pool with %d samples (unlabeled remaining: %d)",
            len(self.labeled_indices),
            len(self.unlabeled_indices),
        )

    def _build_loader(self, indices: List[int], batch_size: int, shuffle: bool) -> DataLoader:
        subset = Subset(self.train_dataset.dataset, indices)
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
        )

    def get_labeled_loader(self, batch_size: int) -> DataLoader:
        if not self.labeled_indices:
            raise ValueError("Labeled pool is empty. Initialize before training.")
        return self._build_loader(self.labeled_indices, batch_size, shuffle=True)

    def get_unlabeled_loader(self, batch_size: int) -> DataLoader:
        if not self.unlabeled_indices:
            raise ValueError("Unlabeled pool is empty. No queries can be made.")
        return self._build_loader(self.unlabeled_indices, batch_size, shuffle=False)

    def get_val_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
        )

    def get_test_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
        )

    def query_samples(self, model, n_samples: int) -> List[int]:
        """Run the configured strategy to select samples for labeling."""
        if len(self.unlabeled_indices) == 0:
            logger.info("No unlabeled samples left to query")
            return []

        model.to(self.device)
        if self.config.active_learning.sampling_strategy == "random":
            local_indices = self.strategy_fn(len(self.unlabeled_indices), n_samples)
        else:
            unlabeled_loader = self.get_unlabeled_loader(batch_size=self.config.training.batch_size)
            strategy_kwargs = {
                "model": model,
                "unlabeled_pool": unlabeled_loader,
                "n_samples": n_samples,
                "device": self.device,
            }
            if self.config.active_learning.sampling_strategy == "kmeans":
                strategy_kwargs["feature_fn"] = extract_features
            local_indices = self.strategy_fn(**strategy_kwargs)

        selected_indices = [self.unlabeled_indices[i] for i in local_indices]
        logger.info("Strategy selected %d samples to label", len(selected_indices))
        return selected_indices

    def _label_samples(self, dataset_indices: List[int]) -> None:
        for idx in dataset_indices:
            if idx in self.unlabeled_indices:
                self.unlabeled_indices.remove(idx)
            if idx not in self.labeled_indices:
                self.labeled_indices.append(idx)

    def move_to_labeled(self, dataset_indices: List[int]) -> None:
        """Move queried samples into the labeled pool (simulated labeling)."""
        self._label_samples(dataset_indices)
        logger.info(
            "Updated pools -> labeled: %d, unlabeled: %d",
            len(self.labeled_indices),
            len(self.unlabeled_indices),
        )

    def save_cycle_state(self, cycle_dir: Path, cycle_index: int) -> None:
        """Persist pool composition for the current cycle."""
        cycle_dir = Path(cycle_dir)
        cycle_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "cycle": cycle_index,
            "labeled_indices": self.labeled_indices,
            "unlabeled_indices": self.unlabeled_indices,
        }
        with open(cycle_dir / "pool_state.json", "w") as f:
            json.dump(state, f, indent=2)

    def summarize(self) -> Dict:
        return {
            "labeled": len(self.labeled_indices),
            "unlabeled": len(self.unlabeled_indices),
        }
