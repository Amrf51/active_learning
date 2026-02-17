"""
ActiveLearningLoop - Orchestrates Active Learning training cycles.

This module supports two execution modes:

1. Batch Mode (CLI): run_all_cycles() executes everything automatically
2. Interactive Mode (Dashboard): Individual methods for step-by-step control
   - prepare_cycle()
   - train_single_epoch()
   - run_validation()
   - run_evaluation()
   - query_samples()
   - receive_annotations()

The interactive mode allows the worker process to update state after each
step, enabling real-time visualization in the Streamlit dashboard.
"""

import json
import shutil
import inspect
import numpy as np
from pathlib import Path
from typing import Dict, List, Callable, Optional, Tuple
from torch.utils.data import DataLoader
import logging

from state import (
    EpochMetrics,
    CycleMetrics,
    QueriedImage,
    ProbeImage,
)

logger = logging.getLogger(__name__)


class ActiveLearningLoop:
    """
    Orchestrates the Active Learning training loop.
    
    Supports both batch execution (run_all_cycles) and step-by-step
    execution for interactive dashboard use.
    """
    
    def __init__(
        self,
        trainer,
        data_manager,
        strategy: Callable,
        val_loader: DataLoader,
        test_loader: DataLoader,
        exp_dir: Path,
        config,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize ActiveLearningLoop.
        
        Args:
            trainer: Trainer instance
            data_manager: ALDataManager instance
            strategy: Strategy function (model, loader, n_samples, device) -> indices
            val_loader: Fixed validation DataLoader
            test_loader: Fixed test DataLoader
            exp_dir: Experiment directory
            config: Config object
            class_names: List of class names for display
        """
        self.trainer = trainer
        self.data_manager = data_manager
        self.strategy = strategy
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.exp_dir = Path(exp_dir)
        self.config = config
        self.class_names = class_names or []
        
        self.cycle_results: List[CycleMetrics] = []
        self.current_cycle = 0
        self.current_train_loader: Optional[DataLoader] = None
        self.probe_images: List[ProbeImage] = []
        
        (self.exp_dir / "queries").mkdir(parents=True, exist_ok=True)
        
        al_config = config.active_learning
        logger.info("ActiveLearningLoop initialized:")
        logger.info(f"  Cycles: {al_config.num_cycles}")
        logger.info(f"  Samples per query: {al_config.batch_size_al}")
        logger.info(f"  Strategy: {al_config.sampling_strategy}")
        logger.info(f"  Reset mode: {al_config.reset_mode}")
    
    def _initialize_probe_images(self, n_probes: int = 12) -> List[ProbeImage]:
        """
        Initialize probe images using stratified sampling from validation set.
        
        Args:
            n_probes: Number of probe images to select (10-15 recommended)
            
        Returns:
            List of ProbeImage objects
        """
        from sklearn.model_selection import train_test_split
        import random
        
        # Get validation dataset indices and labels
        val_indices = []
        val_labels = []
        
        # Extract samples from validation loader
        for batch_idx, (images, labels) in enumerate(self.val_loader):
            batch_start = batch_idx * self.val_loader.batch_size
            for i, label in enumerate(labels):
                val_indices.append(batch_start + i)
                val_labels.append(label.item())
        
        # Convert to numpy for stratified sampling
        val_indices = np.array(val_indices)
        val_labels = np.array(val_labels)
        
        # Get unique classes and their counts
        unique_classes, class_counts = np.unique(val_labels, return_counts=True)
        
        # Calculate samples per class (at least 1 per class, distribute remaining)
        min_per_class = 1
        remaining_probes = n_probes - len(unique_classes)
        
        if remaining_probes < 0:
            # If we have more classes than probes, just take 1 per class up to n_probes
            samples_per_class = {cls: 1 for cls in unique_classes[:n_probes]}
        else:
            # Distribute remaining probes proportionally to class sizes
            samples_per_class = {cls: min_per_class for cls in unique_classes}
            
            # Distribute remaining samples proportionally
            total_samples = len(val_labels)
            for cls, count in zip(unique_classes, class_counts):
                if remaining_probes > 0:
                    additional = int(remaining_probes * (count / total_samples))
                    samples_per_class[cls] += additional
                    remaining_probes -= additional
        
        # Select probe images
        probe_images = []
        probe_id = 0
        
        for cls in unique_classes:
            if probe_id >= n_probes:
                break
                
            # Get indices for this class
            class_indices = val_indices[val_labels == cls]
            n_samples = min(samples_per_class[cls], len(class_indices))
            
            # Randomly select samples from this class
            selected_indices = np.random.choice(class_indices, size=n_samples, replace=False)
            
            for idx in selected_indices:
                if probe_id >= n_probes:
                    break
                
                # Get image info from dataset
                try:
                    # Access the underlying dataset from the validation loader
                    dataset = self.val_loader.dataset
                    _, label = dataset[idx]

                    image_path = f"val_sample_{idx}"
                    display_path = image_path
                    if (
                        hasattr(dataset, "indices")
                        and hasattr(dataset, "parent")
                        and hasattr(dataset.parent, "dataset")
                        and hasattr(dataset.parent.dataset, "samples")
                    ):
                        actual_idx = dataset.indices[idx] if idx < len(dataset.indices) else idx
                        sample_path, _ = dataset.parent.dataset.samples[actual_idx]
                        image_path = str(sample_path)
                        display_path = image_path

                    # Create probe image
                    probe_image = ProbeImage(
                        image_id=int(idx),
                        image_path=image_path,
                        display_path=display_path,
                        true_class=self.class_names[cls] if self.class_names else str(cls),
                        true_class_idx=int(cls),
                        probe_type="validation_stratified",
                        predictions_by_cycle={}
                    )
                    
                    probe_images.append(probe_image)
                    probe_id += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to create probe image for index {idx}: {e}")
                    continue
        
        logger.info(f"Initialized {len(probe_images)} probe images across {len(unique_classes)} classes")
        
        # Store probe images
        self.probe_images = probe_images
        
        return probe_images
    
    def _update_probe_predictions(self, cycle_num: int) -> None:
        """
        Update probe image predictions for the current cycle.
        
        Args:
            cycle_num: Current cycle number
        """
        if not self.probe_images:
            logger.warning("No probe images to update")
            return
        
        # Get predictions for all probe images
        probe_indices = [probe.image_id for probe in self.probe_images]
        
        try:
            # Get dataset from validation loader
            dataset = self.val_loader.dataset
            
            # Get predictions from trainer
            predictions = self.trainer.get_predictions_for_indices(
                probe_indices, dataset, self.class_names
            )
            
            # Update each probe image with new predictions
            for i, probe in enumerate(self.probe_images):
                if i < len(predictions):
                    pred = predictions[i]
                    
                    # Store prediction for this cycle
                    probe.predictions_by_cycle[cycle_num] = {
                        "predicted_class": pred.get("predicted_class", str(pred["predicted_label"])),
                        "confidence": pred["confidence"],
                        "probabilities": pred["probabilities"]
                    }
            
            logger.info(f"Updated probe predictions for cycle {cycle_num}")
            
        except Exception as e:
            logger.error(f"Failed to update probe predictions: {e}")
    
    def prepare_cycle(self, cycle_num: int) -> Dict:
        """
        Prepare for a new AL cycle.
        
        Resets model weights and prepares data loaders.
        
        Args:
            cycle_num: Cycle number (1-indexed)
            
        Returns:
            Dict with cycle preparation info
        """
        self.current_cycle = cycle_num
        
        # Initialize probe images on first cycle
        if cycle_num == 1 and not self.probe_images:
            self.probe_images = self._initialize_probe_images()
        
        reset_mode = self.config.active_learning.reset_mode
        self.trainer.reset_model_weights(mode=reset_mode)
        
        pool_info = self.data_manager.get_pool_info()
        
        self.current_train_loader = self.data_manager.get_labeled_loader(
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers
        )
        
        logger.info(f"Cycle {cycle_num} prepared:")
        logger.info(f"  Labeled: {pool_info['labeled']}")
        logger.info(f"  Unlabeled: {pool_info['unlabeled']}")
        logger.info(f"  Reset mode: {reset_mode}")
        
        return {
            "cycle": cycle_num,
            "labeled_count": pool_info["labeled"],
            "unlabeled_count": pool_info["unlabeled"],
            "reset_mode": reset_mode,
            "train_batches": len(self.current_train_loader),
        }
    
    def train_single_epoch(self, epoch_num: int) -> EpochMetrics:
        """
        Train one epoch within current cycle.
        
        Args:
            epoch_num: Epoch number (1-indexed within cycle)
            
        Returns:
            EpochMetrics for this epoch
        """
        if self.current_train_loader is None:
            raise RuntimeError("Call prepare_cycle() before training")
        
        metrics = self.trainer.train_single_epoch(
            self.current_train_loader,
            self.val_loader,
            epoch_num
        )
        
        return metrics
    
    def should_stop_early(self) -> bool:
        """Check if early stopping criteria is met."""
        return self.trainer.should_stop_early()
    
    def run_validation(self) -> Tuple[float, float]:
        """
        Run validation pass.
        
        Returns:
            Tuple of (val_loss, val_accuracy)
        """
        return self.trainer.validate(self.val_loader)
    
    def run_evaluation(self) -> Dict:
        """
        Run test set evaluation.
        
        Returns:
            Dict with test metrics
        """
        cm_path = self.exp_dir / "confusion_matrices" / f"cycle_{self.current_cycle}.npy"
        test_metrics = self.trainer.evaluate(
            self.test_loader,
            class_names=self.class_names,
            save_cm_path=cm_path,
        )
        
        if self.config.checkpoint.save_best_per_cycle:
            self.trainer.save_cycle_checkpoint(self.current_cycle)
        
        return test_metrics
    
    def _select_query_indices(
        self,
        heartbeat_fn: Optional[Callable[[], None]] = None,
    ) -> np.ndarray:
        """
        Run the configured strategy and return relative indices into unlabeled pool.

        Args:
            heartbeat_fn: Optional callback used to keep worker heartbeat fresh
                during potentially long query passes.

        Returns:
            Array of relative indices into current unlabeled pool.
        """
        al_config = self.config.active_learning

        pool_info = self.data_manager.get_pool_info()
        if pool_info["unlabeled"] == 0:
            logger.warning("No unlabeled samples remaining")
            return np.array([], dtype=int)

        n_query = min(al_config.batch_size_al, pool_info["unlabeled"])
        unlabeled_loader = self.data_manager.get_unlabeled_loader(
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers
        )

        supports_heartbeat = False
        try:
            supports_heartbeat = "heartbeat_fn" in inspect.signature(self.strategy).parameters
        except (TypeError, ValueError):
            supports_heartbeat = False

        if supports_heartbeat:
            query_indices = self.strategy(
                self.trainer.model,
                unlabeled_loader,
                n_query,
                self.trainer.device,
                heartbeat_fn=heartbeat_fn,
            )
        else:
            query_indices = self.strategy(
                self.trainer.model,
                unlabeled_loader,
                n_query,
                self.trainer.device,
            )

        query_indices = np.asarray(query_indices, dtype=int).reshape(-1)
        if query_indices.size == 0:
            return query_indices

        max_idx = len(self.data_manager.get_unlabeled_indices())
        valid = (query_indices >= 0) & (query_indices < max_idx)
        filtered = query_indices[valid]
        if filtered.size < query_indices.size:
            logger.warning(
                "Strategy returned %d out-of-range query indices; kept %d valid",
                int(query_indices.size - filtered.size),
                int(filtered.size),
            )
        return filtered

    def query_and_auto_annotate(
        self,
        heartbeat_fn: Optional[Callable[[], None]] = None,
    ) -> Dict[str, int]:
        """
        Query samples and apply ground-truth annotations without UI payload building.

        This is the fast path for auto-annotate mode.
        """
        query_indices = self._select_query_indices(heartbeat_fn=heartbeat_fn)
        if query_indices.size == 0:
            return {"queried_count": 0, "applied_count": 0}

        unlabeled_indices = self.data_manager.get_unlabeled_indices()
        absolute_indices: List[int] = []
        seen = set()

        for rel_idx in query_indices.tolist():
            if 0 <= rel_idx < len(unlabeled_indices):
                abs_idx = int(unlabeled_indices[rel_idx])
                if abs_idx not in seen:
                    seen.add(abs_idx)
                    absolute_indices.append(abs_idx)

        annotations = []
        for image_id in absolute_indices:
            annotations.append(
                {
                    "image_id": image_id,
                    "user_label": int(self.data_manager.get_ground_truth(image_id)),
                }
            )
            if heartbeat_fn is not None:
                heartbeat_fn()

        if not annotations:
            return {"queried_count": 0, "applied_count": 0}

        result = self.receive_annotations(annotations)
        applied = int(result.get("moved_count", 0))
        logger.info("Auto-annotated %d/%d queried samples", applied, len(annotations))
        return {"queried_count": len(annotations), "applied_count": applied}

    def query_samples(
        self,
        heartbeat_fn: Optional[Callable[[], None]] = None,
    ) -> List[QueriedImage]:
        """
        Apply AL strategy to select samples for annotation.
        
        Args:
            heartbeat_fn: Optional callback used during query phase.

        Returns:
            List of QueriedImage objects with full info for UI
        """
        query_indices = self._select_query_indices(heartbeat_fn=heartbeat_fn)
        if query_indices.size == 0:
            return []
        
        queried_images = self._build_queried_images(query_indices)
        
        self._cache_queried_images(queried_images)
        
        logger.info(f"Queried {len(queried_images)} samples for annotation")
        
        return queried_images
    
    def _build_queried_images(self, query_indices: np.ndarray) -> List[QueriedImage]:
        """
        Build QueriedImage objects with full information.
        
        Args:
            query_indices: Indices into unlabeled pool
            
        Returns:
            List of QueriedImage objects
        """
        queried_images = []
        unlabeled_indices = self.data_manager.get_unlabeled_indices()
        
        absolute_indices = [unlabeled_indices[i] for i in query_indices]
        
        predictions = self.trainer.get_predictions_for_indices(
            absolute_indices,
            self.data_manager.dataset,
            self.class_names
        )
        
        strategy_name = self.config.active_learning.sampling_strategy
        uncertainty_method = self.config.active_learning.uncertainty_method
        
        for i, (rel_idx, abs_idx) in enumerate(zip(query_indices, absolute_indices)):
            pred = predictions[i]
            
            img_info = self.data_manager.get_image_info(abs_idx)
            
            probs = pred["probabilities"]
            uncertainty = self._compute_uncertainty(probs, strategy_name, uncertainty_method)
            selection_reason = self._format_selection_reason(
                probs, uncertainty, strategy_name, uncertainty_method
            )
            
            prob_dict = {}
            for j, class_name in enumerate(self.class_names):
                prob_dict[class_name] = probs[j]
            
            queried_image = QueriedImage(
                image_id=abs_idx,
                image_path=img_info["path"],
                display_path="",
                ground_truth=img_info["label"],
                ground_truth_name=self.class_names[img_info["label"]] if self.class_names else str(img_info["label"]),
                model_probabilities=prob_dict,
                predicted_class=pred.get("predicted_class", str(pred["predicted_label"])),
                predicted_confidence=pred["confidence"],
                uncertainty_score=uncertainty,
                selection_reason=selection_reason
            )
            
            queried_images.append(queried_image)
        
        return queried_images
    
    def _compute_uncertainty(
        self,
        probs: List[float],
        strategy: str,
        method: str
    ) -> float:
        """Compute uncertainty score for a single sample."""
        probs = np.array(probs)
        
        if strategy in ["uncertainty", "least_confidence"]:
            if method == "entropy":
                return float(-np.sum(probs * np.log(probs + 1e-10)))
            else:
                return float(1.0 - probs.max())
        
        elif strategy == "entropy":
            return float(-np.sum(probs * np.log(probs + 1e-10)))
        
        elif strategy == "margin":
            sorted_probs = np.sort(probs)[::-1]
            return float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
        
        else:
            return float(1.0 - probs.max())
    
    def _format_selection_reason(
        self,
        probs: List[float],
        uncertainty: float,
        strategy: str,
        method: str
    ) -> str:
        """Format human-readable selection reason."""
        probs = np.array(probs)
        max_prob = probs.max()
        
        if strategy in ["uncertainty", "least_confidence"]:
            return f"Low confidence: {max_prob:.0%}"
        
        elif strategy == "entropy":
            return f"High entropy: {uncertainty:.2f}"
        
        elif strategy == "margin":
            sorted_probs = np.sort(probs)[::-1]
            margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 0
            return f"Small margin: {margin:.0%}"
        
        elif strategy == "random":
            return "Random selection"
        
        else:
            return f"Uncertainty: {uncertainty:.2f}"
    
    def _cache_queried_images(self, queried_images: List[QueriedImage]) -> None:
        """
        Copy queried images to experiment folder for display.
        
        Args:
            queried_images: List of QueriedImage objects (modified in place)
        """
        cache_dir = self.exp_dir / "queries" / f"cycle_{self.current_cycle}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        for img in queried_images:
            src_path = Path(img.image_path)
            if src_path.exists():
                dst_filename = f"{img.image_id}_{src_path.name}"
                dst_path = cache_dir / dst_filename
                
                try:
                    shutil.copy(src_path, dst_path)
                    img.display_path = str(dst_path)
                except Exception as e:
                    logger.warning(f"Failed to cache image {src_path}: {e}")
                    img.display_path = img.image_path
            else:
                logger.warning(f"Source image not found: {src_path}")
                img.display_path = img.image_path
    
    def receive_annotations(
        self,
        annotations: List[Dict]
    ) -> Dict:
        """
        Process user annotations and update pools.
        
        Args:
            annotations: List of dicts with image_id and user_label
            
        Returns:
            Dict with annotation processing summary
        """
        result = self.data_manager.update_labeled_pool_with_annotations(annotations)
        
        query_file = self.exp_dir / f"cycle_{self.current_cycle}_annotations.json"
        with open(query_file, "w") as f:
            json.dump({
                "cycle": self.current_cycle,
                "annotations": annotations,
                "summary": result
            }, f, indent=2)
        
        logger.info(
            f"Annotations processed: {result['moved_count']} samples added to labeled pool"
        )
        
        return result
    
    def finalize_cycle(self, test_metrics: Dict) -> CycleMetrics:
        """
        Create and store cycle metrics after completion.
        
        Args:
            test_metrics: Test evaluation metrics
            
        Returns:
            CycleMetrics for this cycle
        """
        pool_info = self.data_manager.get_pool_info()
        training_summary = self.trainer.get_training_summary()
        
        cycle_metrics = CycleMetrics(
            cycle=self.current_cycle,
            labeled_pool_size=pool_info["labeled"],
            unlabeled_pool_size=pool_info["unlabeled"],
            epochs_trained=training_summary["epochs_trained"],
            best_val_accuracy=training_summary["best_val_accuracy"],
            best_epoch=training_summary["best_epoch"],
            test_accuracy=test_metrics["test_accuracy"],
            test_f1=test_metrics["test_f1"],
            test_precision=test_metrics["test_precision"],
            test_recall=test_metrics["test_recall"],
            per_class_metrics=test_metrics.get("per_class"),
            confusion_matrix_path=test_metrics.get("confusion_matrix_path"),
        )
        
        self.cycle_results.append(cycle_metrics)
        
        # Update probe image predictions for this cycle
        self._update_probe_predictions(self.current_cycle)
        
        return cycle_metrics
    
    def run_cycle(self, cycle_num: int) -> Dict:
        """
        Execute one complete AL cycle (batch mode).
        
        Args:
            cycle_num: Cycle number (1-indexed)
            
        Returns:
            Dict with cycle results
        """
        al_config = self.config.active_learning
        
        logger.info(f"\n{'='*60}")
        logger.info(f"CYCLE {cycle_num}/{al_config.num_cycles}")
        logger.info(f"{'='*60}")
        
        self.prepare_cycle(cycle_num)
        
        epochs = self.config.training.epochs
        for epoch in range(1, epochs + 1):
            metrics = self.train_single_epoch(epoch)
            
            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {metrics.train_loss:.4f}, Train Acc: {metrics.train_accuracy:.4f}"
                + (f" | Val Acc: {metrics.val_accuracy:.4f}" if metrics.val_accuracy else "")
            )
            
            if self.should_stop_early():
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        test_metrics = self.run_evaluation()
        
        cycle_metrics = self.finalize_cycle(test_metrics)
        
        queried_images = []
        if cycle_num < al_config.num_cycles:
            pool_info = self.data_manager.get_pool_info()
            if pool_info["unlabeled"] > 0:
                queried_images = self.query_samples()
                
                simulated_annotations = [
                    {"image_id": img.image_id, "user_label": img.ground_truth}
                    for img in queried_images
                ]
                self.receive_annotations(simulated_annotations)
        
        logger.info(
            f"Cycle {cycle_num} complete | "
            f"Val Acc: {cycle_metrics.best_val_accuracy:.4f}, "
            f"Test Acc: {cycle_metrics.test_accuracy:.4f}"
        )
        
        return {
            "cycle": cycle_num,
            "metrics": cycle_metrics.model_dump(),
            "queried_count": len(queried_images),
        }
    
    def run_all_cycles(self) -> List[CycleMetrics]:
        """
        Execute all AL cycles (batch mode).
        
        Returns:
            List of CycleMetrics for all cycles
        """
        num_cycles = self.config.active_learning.num_cycles
        
        logger.info("\n" + "="*60)
        logger.info("STARTING ACTIVE LEARNING")
        logger.info("="*60)
        
        for cycle in range(1, num_cycles + 1):
            self.run_cycle(cycle)
        
        logger.info("\n" + "="*60)
        logger.info("ACTIVE LEARNING COMPLETE")
        logger.info("="*60)
        
        self._log_summary()
        self.persist_artifacts()
        
        return self.cycle_results
    
    def persist_artifacts(self) -> None:
        """Persist run artifacts used by the dashboard and experiment analysis."""
        self._save_results()
        self.data_manager.save_state(self.exp_dir / "al_pool_state.json")
        self.trainer.save_training_log()
    
    def _save_results(self):
        """Save cycle results to JSON."""
        results_file = self.exp_dir / "al_cycle_results.json"
        
        output = {
            "strategy": self.config.active_learning.sampling_strategy,
            "num_cycles": self.config.active_learning.num_cycles,
            "initial_pool_size": self.config.active_learning.initial_pool_size,
            "batch_size_al": self.config.active_learning.batch_size_al,
            "cycles": [c.model_dump() for c in self.cycle_results],
        }
        
        with open(results_file, "w") as f:
            json.dump(output, f, indent=2)
    
    def _log_summary(self):
        """Log summary of all cycles."""
        logger.info("\nAL CYCLE SUMMARY")
        logger.info("-" * 70)
        logger.info(f"{'Cycle':<6} {'Labeled':<8} {'Val Acc':<10} {'Test Acc':<10} {'Test F1':<10}")
        logger.info("-" * 70)
        
        for r in self.cycle_results:
            logger.info(
                f"{r.cycle:<6} {r.labeled_pool_size:<8} "
                f"{r.best_val_accuracy:<10.4f} {r.test_accuracy:<10.4f} "
                f"{r.test_f1:<10.4f}"
            )
        
        logger.info("-" * 70)
        
        best = max(self.cycle_results, key=lambda x: x.test_accuracy)
        logger.info(f"Best test accuracy: {best.test_accuracy:.4f} at cycle {best.cycle}")
        
        if len(self.cycle_results) > 1:
            first_acc = self.cycle_results[0].test_accuracy
            last_acc = self.cycle_results[-1].test_accuracy
            improvement = last_acc - first_acc
            pct = 100 * improvement / first_acc if first_acc > 0 else 0
            logger.info(f"Improvement: {improvement:+.4f} ({pct:+.1f}%)")
    
    def get_results(self) -> List[CycleMetrics]:
        """Get all cycle results."""
        return self.cycle_results.copy()
    
    def get_best_cycle(self) -> Optional[CycleMetrics]:
        """Get cycle with best test accuracy."""
        if not self.cycle_results:
            return None
        return max(self.cycle_results, key=lambda x: x.test_accuracy)
    
    def get_current_pool_info(self) -> Dict:
        """Get current pool statistics."""
        return self.data_manager.get_pool_info()
