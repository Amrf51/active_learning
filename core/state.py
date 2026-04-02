"""
State data structures for the Active Learning backend.

This module defines dataclasses used by the backend components
for representing training state and metrics.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EpochMetrics:
    """
    Training metrics for a single epoch.
    
    Used by the Trainer to return epoch-level training results.
    """
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "train_accuracy": self.train_accuracy,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_accuracy,
            "learning_rate": self.learning_rate
        }


@dataclass
class CycleMetrics:
    """
    Metrics for a completed Active Learning cycle.
    
    Used by the ActiveLearningLoop to track cycle-level results.
    """
    cycle: int
    labeled_pool_size: int
    unlabeled_pool_size: int
    epochs_trained: int
    best_val_accuracy: float
    best_epoch: int
    test_accuracy: float
    test_f1: float
    test_precision: float
    test_recall: float
    per_class_metrics: Optional[dict] = None
    confusion_matrix_path: Optional[str] = None
    ece: Optional[float] = None
    embeddings_path: Optional[str] = None

    def model_dump(self) -> dict:
        """Convert to dictionary for serialization (Pydantic-style method name)."""
        return {
            "cycle": self.cycle,
            "labeled_pool_size": self.labeled_pool_size,
            "unlabeled_pool_size": self.unlabeled_pool_size,
            "epochs_trained": self.epochs_trained,
            "best_val_accuracy": self.best_val_accuracy,
            "best_epoch": self.best_epoch,
            "test_accuracy": self.test_accuracy,
            "test_f1": self.test_f1,
            "test_precision": self.test_precision,
            "test_recall": self.test_recall,
            "per_class_metrics": self.per_class_metrics,
            "confusion_matrix_path": self.confusion_matrix_path,
            "ece": self.ece,
            "embeddings_path": self.embeddings_path,
        }


@dataclass
class QueriedImage:
    """
    Information about an image selected for annotation.
    
    Used by the ActiveLearningLoop to provide rich information
    about queried samples to the UI.
    """
    image_id: int
    image_path: str
    display_path: str
    ground_truth: int
    ground_truth_name: str
    model_probabilities: dict
    predicted_class: str
    predicted_confidence: float
    uncertainty_score: float
    selection_reason: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "display_path": self.display_path,
            "ground_truth": self.ground_truth,
            "ground_truth_name": self.ground_truth_name,
            "model_probabilities": self.model_probabilities,
            "predicted_class": self.predicted_class,
            "predicted_confidence": self.predicted_confidence,
            "uncertainty_score": self.uncertainty_score,
            "selection_reason": self.selection_reason
        }


@dataclass
class ProbeImage:
    """
    Information about a probe image for tracking model predictions.
    
    Used by the ActiveLearningLoop to track how model predictions
    change over cycles for a fixed set of validation images.
    """
    image_id: int
    image_path: str
    display_path: str
    true_class: str
    true_class_idx: int
    probe_type: str
    predictions_by_cycle: dict
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "display_path": self.display_path,
            "true_class": self.true_class,
            "true_class_idx": self.true_class_idx,
            "probe_type": self.probe_type,
            "predictions_by_cycle": self.predictions_by_cycle
        }


@dataclass
class QuerySummary:
    """
    Summary of a single query cycle: why and how images were selected.

    Persisted as cycle_N_query_summary.json for experiment tracking.
    """
    cycle: int
    strategy_name: str
    strategy_description: str
    pool_size_before_query: int
    n_queried: int
    uncertainty_stats: dict          # min, max, mean, std of queried batch
    queried_class_distribution: dict # class_name -> count
    labeled_class_distribution: dict # class_name -> count (before query applied)
    top_uncertain: list              # dicts: image_id, uncertainty_score, predicted_class, predicted_confidence
    queried_image_ids: list          # all queried absolute image IDs

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items()}
