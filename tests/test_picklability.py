"""Unit tests to verify WorldState and Event are picklable.

These tests ensure that all data structures used for multiprocessing
pipe transport can be serialized and deserialized correctly.

Run with: python -m pytest tests/test_picklability.py -v
"""

import pickle
import time
from typing import Dict, List

import pytest

from model.world_state import (
    WorldState,
    Phase,
    EpochMetrics,
    CycleMetrics,
    QueriedImage,
    ProbeImage,
)
from controller.events import Event, EventType


class TestWorldStatePicklability:
    """Tests for WorldState picklability."""

    def test_empty_world_state_is_picklable(self):
        """Test that a default WorldState can be pickled and unpickled."""
        state = WorldState()
        
        # Pickle and unpickle
        pickled = pickle.dumps(state)
        unpickled = pickle.loads(pickled)
        
        # Verify fields match
        assert unpickled.experiment_id == state.experiment_id
        assert unpickled.phase == state.phase
        assert unpickled.current_cycle == state.current_cycle
        assert unpickled.updated_at == state.updated_at

    def test_populated_world_state_is_picklable(self):
        """Test that a fully populated WorldState can be pickled."""
        state = WorldState(
            experiment_id="exp-123",
            experiment_name="Test Experiment",
            phase=Phase.TRAINING,
            current_cycle=2,
            total_cycles=5,
            current_epoch=3,
            epochs_per_cycle=10,
            labeled_count=100,
            unlabeled_count=900,
            class_distribution={"cat": 50, "dog": 50},
            error_message=None,
            updated_at=time.time(),
        )
        
        # Add epoch metrics
        state.epoch_metrics = [
            EpochMetrics(
                epoch=1,
                train_loss=0.5,
                train_accuracy=0.8,
                val_loss=0.6,
                val_accuracy=0.75,
                learning_rate=0.001,
            ),
            EpochMetrics(
                epoch=2,
                train_loss=0.4,
                train_accuracy=0.85,
                val_loss=0.55,
                val_accuracy=0.78,
                learning_rate=0.001,
            ),
        ]
        
        # Add queried images
        state.queried_images = [
            QueriedImage(
                image_id=1,
                image_path="/path/to/image1.jpg",
                display_path="image1.jpg",
                ground_truth=0,
                ground_truth_name="cat",
                model_probabilities={"cat": 0.6, "dog": 0.4},
                predicted_class="cat",
                predicted_confidence=0.6,
                uncertainty_score=0.4,
                selection_reason="high_uncertainty",
            ),
        ]
        
        # Add probe images
        state.probe_images = [
            ProbeImage(
                image_id=10,
                image_path="/path/to/probe.jpg",
                display_path="probe.jpg",
                true_class="dog",
                true_class_idx=1,
                probe_type="validation",
                predictions_by_cycle={1: {"cat": 0.3, "dog": 0.7}},
            ),
        ]
        
        # Pickle and unpickle
        pickled = pickle.dumps(state)
        unpickled = pickle.loads(pickled)
        
        # Verify all fields match
        assert unpickled.experiment_id == "exp-123"
        assert unpickled.experiment_name == "Test Experiment"
        assert unpickled.phase == Phase.TRAINING
        assert unpickled.current_cycle == 2
        assert unpickled.total_cycles == 5
        assert unpickled.current_epoch == 3
        assert unpickled.epochs_per_cycle == 10
        assert unpickled.labeled_count == 100
        assert unpickled.unlabeled_count == 900
        assert unpickled.class_distribution == {"cat": 50, "dog": 50}
        
        # Verify epoch metrics
        assert len(unpickled.epoch_metrics) == 2
        assert unpickled.epoch_metrics[0].epoch == 1
        assert unpickled.epoch_metrics[0].train_loss == 0.5
        
        # Verify queried images
        assert len(unpickled.queried_images) == 1
        assert unpickled.queried_images[0].image_id == 1
        assert unpickled.queried_images[0].predicted_class == "cat"
        
        # Verify probe images
        assert len(unpickled.probe_images) == 1
        assert unpickled.probe_images[0].true_class == "dog"

    def test_world_state_with_error_is_picklable(self):
        """Test that WorldState with error state can be pickled."""
        state = WorldState(
            phase=Phase.ERROR,
            error_message="Something went wrong",
        )
        
        pickled = pickle.dumps(state)
        unpickled = pickle.loads(pickled)
        
        assert unpickled.phase == Phase.ERROR
        assert unpickled.error_message == "Something went wrong"

    def test_phase_enum_is_picklable(self):
        """Test that all Phase enum values can be pickled."""
        for phase in Phase:
            pickled = pickle.dumps(phase)
            unpickled = pickle.loads(pickled)
            assert unpickled == phase

    def test_updated_at_field_exists(self):
        """Test that WorldState has updated_at field for state versioning."""
        state = WorldState()
        
        # Verify field exists and is a float
        assert hasattr(state, 'updated_at')
        assert isinstance(state.updated_at, float)
        
        # Verify it's set to current time by default
        now = time.time()
        assert abs(state.updated_at - now) < 1.0  # Within 1 second

    def test_touch_updates_timestamp(self):
        """Test that touch() method updates the timestamp."""
        state = WorldState()
        original_time = state.updated_at
        
        # Small delay to ensure time difference
        time.sleep(0.01)
        state.touch()
        
        assert state.updated_at > original_time


class TestEventPicklability:
    """Tests for Event picklability."""

    def test_simple_event_is_picklable(self):
        """Test that a simple Event can be pickled."""
        event = Event(type=EventType.START_CYCLE)
        
        pickled = pickle.dumps(event)
        unpickled = pickle.loads(pickled)
        
        assert unpickled.type == EventType.START_CYCLE
        assert unpickled.payload == {}

    def test_event_with_payload_is_picklable(self):
        """Test that Event with payload can be pickled."""
        event = Event(
            type=EventType.CREATE_EXPERIMENT,
            payload={
                "experiment_name": "Test",
                "config": {
                    "model_name": "resnet18",
                    "num_cycles": 5,
                    "epochs_per_cycle": 10,
                    "learning_rate": 0.001,
                },
            },
        )
        
        pickled = pickle.dumps(event)
        unpickled = pickle.loads(pickled)
        
        assert unpickled.type == EventType.CREATE_EXPERIMENT
        assert unpickled.payload["experiment_name"] == "Test"
        assert unpickled.payload["config"]["model_name"] == "resnet18"

    def test_all_event_types_are_picklable(self):
        """Test that all EventType enum values can be pickled."""
        for event_type in EventType:
            event = Event(type=event_type)
            pickled = pickle.dumps(event)
            unpickled = pickle.loads(pickled)
            assert unpickled.type == event_type

    def test_shutdown_event_exists(self):
        """Test that SHUTDOWN event type exists for graceful termination."""
        assert hasattr(EventType, 'SHUTDOWN')
        
        event = Event(type=EventType.SHUTDOWN)
        pickled = pickle.dumps(event)
        unpickled = pickle.loads(pickled)
        
        assert unpickled.type == EventType.SHUTDOWN

    def test_timestamp_field_exists(self):
        """Test that Event has timestamp field."""
        event = Event(type=EventType.PAUSE)
        
        # Verify field exists and is a float
        assert hasattr(event, 'timestamp')
        assert isinstance(event.timestamp, float)
        
        # Verify it's set to current time by default
        now = time.time()
        assert abs(event.timestamp - now) < 1.0  # Within 1 second

    def test_event_with_annotations_payload_is_picklable(self):
        """Test that Event with annotations payload can be pickled."""
        event = Event(
            type=EventType.SUBMIT_ANNOTATIONS,
            payload={
                "annotations": [
                    {"image_id": 1, "label": 0},
                    {"image_id": 2, "label": 1},
                    {"image_id": 3, "label": 0},
                ],
            },
        )
        
        pickled = pickle.dumps(event)
        unpickled = pickle.loads(pickled)
        
        assert unpickled.type == EventType.SUBMIT_ANNOTATIONS
        assert len(unpickled.payload["annotations"]) == 3


class TestBackendTypesPicklability:
    """Tests for backend state types picklability."""

    def test_epoch_metrics_is_picklable(self):
        """Test that EpochMetrics can be pickled."""
        metrics = EpochMetrics(
            epoch=5,
            train_loss=0.25,
            train_accuracy=0.92,
            val_loss=0.30,
            val_accuracy=0.88,
            learning_rate=0.0001,
        )
        
        pickled = pickle.dumps(metrics)
        unpickled = pickle.loads(pickled)
        
        assert unpickled.epoch == 5
        assert unpickled.train_loss == 0.25
        assert unpickled.train_accuracy == 0.92

    def test_cycle_metrics_is_picklable(self):
        """Test that CycleMetrics can be pickled."""
        metrics = CycleMetrics(
            cycle=3,
            labeled_pool_size=150,
            unlabeled_pool_size=850,
            epochs_trained=10,
            best_val_accuracy=0.89,
            best_epoch=8,
            test_accuracy=0.87,
            test_f1=0.86,
            test_precision=0.88,
            test_recall=0.85,
            per_class_metrics={"cat": {"precision": 0.9, "recall": 0.85}},
        )
        
        pickled = pickle.dumps(metrics)
        unpickled = pickle.loads(pickled)
        
        assert unpickled.cycle == 3
        assert unpickled.test_accuracy == 0.87
        assert unpickled.per_class_metrics["cat"]["precision"] == 0.9

    def test_queried_image_is_picklable(self):
        """Test that QueriedImage can be pickled."""
        image = QueriedImage(
            image_id=42,
            image_path="/data/images/img42.jpg",
            display_path="img42.jpg",
            ground_truth=1,
            ground_truth_name="dog",
            model_probabilities={"cat": 0.3, "dog": 0.7},
            predicted_class="dog",
            predicted_confidence=0.7,
            uncertainty_score=0.3,
            selection_reason="margin_sampling",
        )
        
        pickled = pickle.dumps(image)
        unpickled = pickle.loads(pickled)
        
        assert unpickled.image_id == 42
        assert unpickled.predicted_class == "dog"
        assert unpickled.model_probabilities["dog"] == 0.7

    def test_probe_image_is_picklable(self):
        """Test that ProbeImage can be pickled."""
        probe = ProbeImage(
            image_id=100,
            image_path="/data/probes/probe100.jpg",
            display_path="probe100.jpg",
            true_class="cat",
            true_class_idx=0,
            probe_type="validation",
            predictions_by_cycle={
                1: {"cat": 0.5, "dog": 0.5},
                2: {"cat": 0.7, "dog": 0.3},
            },
        )
        
        pickled = pickle.dumps(probe)
        unpickled = pickle.loads(pickled)
        
        assert unpickled.image_id == 100
        assert unpickled.true_class == "cat"
        assert unpickled.predictions_by_cycle[2]["cat"] == 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
