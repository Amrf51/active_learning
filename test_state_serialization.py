"""
Verification script for state.py dataclass serialization.
Tests that all dataclasses can be serialized to dictionaries for queue communication.
"""

from state import EpochMetrics, CycleMetrics, QueriedImage, ProbeImage


def test_epoch_metrics_serialization():
    """Test EpochMetrics serialization."""
    metrics = EpochMetrics(
        epoch=1,
        train_loss=0.5,
        train_accuracy=0.85,
        val_loss=0.6,
        val_accuracy=0.82,
        learning_rate=0.001
    )
    
    # Test to_dict method
    data = metrics.to_dict()
    assert isinstance(data, dict), "to_dict() should return a dict"
    assert data["epoch"] == 1
    assert data["train_loss"] == 0.5
    assert data["train_accuracy"] == 0.85
    assert data["val_loss"] == 0.6
    assert data["val_accuracy"] == 0.82
    assert data["learning_rate"] == 0.001
    print("✓ EpochMetrics serialization works correctly")


def test_cycle_metrics_serialization():
    """Test CycleMetrics serialization."""
    metrics = CycleMetrics(
        cycle=1,
        labeled_pool_size=100,
        unlabeled_pool_size=900,
        epochs_trained=10,
        best_val_accuracy=0.85,
        best_epoch=7,
        test_accuracy=0.83,
        test_f1=0.82,
        test_precision=0.84,
        test_recall=0.81,
        per_class_metrics={"class_0": 0.9, "class_1": 0.8}
    )
    
    # Test model_dump method
    data = metrics.model_dump()
    assert isinstance(data, dict), "model_dump() should return a dict"
    assert data["cycle"] == 1
    assert data["labeled_pool_size"] == 100
    assert data["unlabeled_pool_size"] == 900
    assert data["epochs_trained"] == 10
    assert data["best_val_accuracy"] == 0.85
    assert data["best_epoch"] == 7
    assert data["test_accuracy"] == 0.83
    assert data["test_f1"] == 0.82
    assert data["test_precision"] == 0.84
    assert data["test_recall"] == 0.81
    assert data["per_class_metrics"] == {"class_0": 0.9, "class_1": 0.8}
    print("✓ CycleMetrics serialization works correctly")


def test_queried_image_serialization():
    """Test QueriedImage serialization."""
    image = QueriedImage(
        image_id=42,
        image_path="/path/to/image.jpg",
        display_path="image.jpg",
        ground_truth=5,
        ground_truth_name="sedan",
        model_probabilities={"sedan": 0.7, "suv": 0.2, "truck": 0.1},
        predicted_class="sedan",
        predicted_confidence=0.7,
        uncertainty_score=0.3,
        selection_reason="High entropy"
    )
    
    # Test to_dict method
    data = image.to_dict()
    assert isinstance(data, dict), "to_dict() should return a dict"
    assert data["image_id"] == 42
    assert data["image_path"] == "/path/to/image.jpg"
    assert data["display_path"] == "image.jpg"
    assert data["ground_truth"] == 5
    assert data["ground_truth_name"] == "sedan"
    assert data["model_probabilities"] == {"sedan": 0.7, "suv": 0.2, "truck": 0.1}
    assert data["predicted_class"] == "sedan"
    assert data["predicted_confidence"] == 0.7
    assert data["uncertainty_score"] == 0.3
    assert data["selection_reason"] == "High entropy"
    print("✓ QueriedImage serialization works correctly")


def test_probe_image_serialization():
    """Test ProbeImage serialization."""
    probe = ProbeImage(
        image_id=10,
        image_path="/path/to/probe.jpg",
        display_path="probe.jpg",
        true_class="sedan",
        true_class_idx=5,
        probe_type="validation",
        predictions_by_cycle={
            "cycle_0": {"predicted": "suv", "confidence": 0.6},
            "cycle_1": {"predicted": "sedan", "confidence": 0.8}
        }
    )
    
    # Test to_dict method
    data = probe.to_dict()
    assert isinstance(data, dict), "to_dict() should return a dict"
    assert data["image_id"] == 10
    assert data["image_path"] == "/path/to/probe.jpg"
    assert data["display_path"] == "probe.jpg"
    assert data["true_class"] == "sedan"
    assert data["true_class_idx"] == 5
    assert data["probe_type"] == "validation"
    assert data["predictions_by_cycle"] == {
        "cycle_0": {"predicted": "suv", "confidence": 0.6},
        "cycle_1": {"predicted": "sedan", "confidence": 0.8}
    }
    print("✓ ProbeImage serialization works correctly")


if __name__ == "__main__":
    print("Testing state.py dataclass serialization...\n")
    
    test_epoch_metrics_serialization()
    test_cycle_metrics_serialization()
    test_queried_image_serialization()
    test_probe_image_serialization()
    
    print("\n✅ All dataclasses serialize correctly!")
