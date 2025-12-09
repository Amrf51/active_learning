"""
Unit tests for state management module.

Run with: pytest tests/test_state.py -v
"""

import pytest
import tempfile
import shutil
import time
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.state import (
    ExperimentPhase,
    ExperimentState,
    ExperimentConfig,
    EpochMetrics,
    CycleMetrics,
    QueriedImage,
    ProbeImage,
    UserAnnotation,
    AnnotationSubmission,
    StateManager,
    ExperimentManager,
    DatasetInfo,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def sample_config():
    """Create a sample ExperimentConfig."""
    return ExperimentConfig(
        model_name="resnet18",
        pretrained=True,
        num_classes=4,
        class_names=["bus", "car", "motorcycle", "truck"],
        epochs_per_cycle=20,
        batch_size=32,
        learning_rate=0.001,
        optimizer="adam",
        weight_decay=0.0001,
        early_stopping_patience=5,
        num_cycles=5,
        sampling_strategy="uncertainty",
        uncertainty_method="least_confidence",
        initial_pool_size=50,
        batch_size_al=20,
        reset_mode="pretrained",
        seed=42,
        data_dir="./data/raw/kaggle-vehicle/",
        val_split=0.15,
        test_split=0.15,
        augmentation=True,
    )


class TestExperimentState:
    """Tests for ExperimentState Pydantic model."""
    
    def test_create_minimal_state(self):
        """Test creating state with minimal required fields."""
        state = ExperimentState(
            experiment_id="test_001",
            experiment_name="Test Experiment",
            created_at=datetime.now(),
        )
        
        assert state.experiment_id == "test_001"
        assert state.phase == ExperimentPhase.IDLE
        assert state.current_cycle == 0
        assert len(state.cycle_results) == 0
    
    def test_create_full_state(self, sample_config):
        """Test creating state with all fields."""
        state = ExperimentState(
            experiment_id="test_002",
            experiment_name="Full Test",
            created_at=datetime.now(),
            phase=ExperimentPhase.TRAINING,
            worker_pid=12345,
            last_heartbeat=datetime.now(),
            config=sample_config,
            current_cycle=2,
            total_cycles=5,
            current_epoch=10,
            epochs_in_cycle=20,
            labeled_count=80,
            unlabeled_count=200,
            total_train_samples=280,
        )
        
        assert state.phase == ExperimentPhase.TRAINING
        assert state.config.model_name == "resnet18"
        assert state.labeled_count == 80
    
    def test_state_serialization(self, sample_config):
        """Test JSON serialization and deserialization."""
        state = ExperimentState(
            experiment_id="test_003",
            experiment_name="Serialization Test",
            created_at=datetime.now(),
            config=sample_config,
        )
        
        json_str = state.model_dump_json()
        loaded = ExperimentState.model_validate_json(json_str)
        
        assert loaded.experiment_id == state.experiment_id
        assert loaded.config.model_name == state.config.model_name
    
    def test_state_with_metrics(self):
        """Test state with epoch and cycle metrics."""
        epoch_metrics = [
            EpochMetrics(epoch=1, train_loss=1.5, train_accuracy=0.4, val_loss=1.6, val_accuracy=0.35),
            EpochMetrics(epoch=2, train_loss=1.2, train_accuracy=0.5, val_loss=1.4, val_accuracy=0.45),
        ]
        
        state = ExperimentState(
            experiment_id="test_004",
            experiment_name="Metrics Test",
            created_at=datetime.now(),
            current_cycle_epochs=epoch_metrics,
        )
        
        assert len(state.current_cycle_epochs) == 2
        assert state.current_cycle_epochs[0].train_loss == 1.5


class TestStateManager:
    """Tests for StateManager class."""
    
    def test_initialize_state(self, temp_dir):
        """Test initializing a new experiment state."""
        exp_dir = temp_dir / "exp_001"
        manager = StateManager(exp_dir)
        
        state = manager.initialize_state(
            experiment_id="exp_001",
            experiment_name="Test Experiment"
        )
        
        assert exp_dir.exists()
        assert (exp_dir / "experiment_state.json").exists()
        assert state.experiment_id == "exp_001"
        assert state.phase == ExperimentPhase.IDLE
    
    def test_read_state(self, temp_dir):
        """Test reading state from file."""
        exp_dir = temp_dir / "exp_002"
        manager = StateManager(exp_dir)
        
        original = manager.initialize_state("exp_002", "Read Test")
        
        loaded = manager.read_state()
        
        assert loaded.experiment_id == original.experiment_id
        assert loaded.experiment_name == original.experiment_name
    
    def test_write_state(self, temp_dir):
        """Test writing modified state."""
        exp_dir = temp_dir / "exp_003"
        manager = StateManager(exp_dir)
        
        state = manager.initialize_state("exp_003", "Write Test")
        state.phase = ExperimentPhase.TRAINING
        state.current_cycle = 2
        
        manager.write_state(state)
        
        loaded = manager.read_state()
        assert loaded.phase == ExperimentPhase.TRAINING
        assert loaded.current_cycle == 2
    
    def test_update_state(self, temp_dir):
        """Test partial state updates."""
        exp_dir = temp_dir / "exp_004"
        manager = StateManager(exp_dir)
        
        manager.initialize_state("exp_004", "Update Test")
        
        updated = manager.update_state(
            phase=ExperimentPhase.TRAINING,
            current_epoch=5,
            labeled_count=100
        )
        
        assert updated.phase == ExperimentPhase.TRAINING
        assert updated.current_epoch == 5
        assert updated.labeled_count == 100
        
        loaded = manager.read_state()
        assert loaded.phase == ExperimentPhase.TRAINING
    
    def test_update_heartbeat(self, temp_dir):
        """Test heartbeat updates."""
        exp_dir = temp_dir / "exp_005"
        manager = StateManager(exp_dir)
        
        manager.initialize_state("exp_005", "Heartbeat Test")
        
        before = datetime.now()
        manager.update_heartbeat()
        after = datetime.now()
        
        state = manager.read_state()
        assert state.last_heartbeat is not None
        assert before <= state.last_heartbeat <= after
    
    def test_add_epoch_metrics(self, temp_dir):
        """Test adding epoch metrics."""
        exp_dir = temp_dir / "exp_006"
        manager = StateManager(exp_dir)
        
        manager.initialize_state("exp_006", "Epoch Test")
        
        metrics1 = EpochMetrics(epoch=1, train_loss=1.5, train_accuracy=0.4)
        metrics2 = EpochMetrics(epoch=2, train_loss=1.2, train_accuracy=0.5)
        
        manager.add_epoch_metrics(metrics1)
        manager.add_epoch_metrics(metrics2)
        
        state = manager.read_state()
        assert len(state.current_cycle_epochs) == 2
        assert state.current_epoch == 2
    
    def test_finalize_cycle(self, temp_dir):
        """Test finalizing a cycle."""
        exp_dir = temp_dir / "exp_007"
        manager = StateManager(exp_dir)
        
        manager.initialize_state("exp_007", "Finalize Test")
        
        manager.add_epoch_metrics(EpochMetrics(epoch=1, train_loss=1.0, train_accuracy=0.5))
        manager.add_epoch_metrics(EpochMetrics(epoch=2, train_loss=0.8, train_accuracy=0.6))
        
        cycle_metrics = CycleMetrics(
            cycle=1,
            labeled_pool_size=50,
            unlabeled_pool_size=230,
            epochs_trained=2,
            best_val_accuracy=0.6,
            best_epoch=2,
            test_accuracy=0.55,
            test_f1=0.54,
            test_precision=0.56,
            test_recall=0.53,
        )
        
        manager.finalize_cycle(cycle_metrics)
        
        state = manager.read_state()
        assert len(state.cycle_results) == 1
        assert len(state.current_cycle_epochs) == 0
        assert state.cycle_results[0].test_accuracy == 0.55
        assert len(state.cycle_results[0].epoch_history) == 2
    
    def test_annotations_workflow(self, temp_dir):
        """Test annotation read/write/clear workflow."""
        exp_dir = temp_dir / "exp_008"
        manager = StateManager(exp_dir)
        
        manager.initialize_state("exp_008", "Annotation Test")
        
        assert manager.read_annotations() is None
        assert not manager.annotations_pending()
        
        submission = AnnotationSubmission(
            experiment_id="exp_008",
            cycle=1,
            annotations=[
                UserAnnotation(
                    image_id=10,
                    user_label=2,
                    user_label_name="motorcycle",
                    timestamp=datetime.now(),
                    was_correct=True
                ),
                UserAnnotation(
                    image_id=25,
                    user_label=0,
                    user_label_name="bus",
                    timestamp=datetime.now(),
                    was_correct=False
                ),
            ],
            submitted_at=datetime.now()
        )
        
        manager.write_annotations(submission)
        
        assert manager.annotations_pending()
        
        loaded = manager.read_annotations()
        assert loaded is not None
        assert len(loaded.annotations) == 2
        assert loaded.annotations[0].image_id == 10
        
        manager.clear_annotations()
        assert not manager.annotations_pending()
        assert manager.read_annotations() is None
    
    def test_is_worker_alive(self, temp_dir):
        """Test worker alive detection."""
        exp_dir = temp_dir / "exp_009"
        manager = StateManager(exp_dir)
        
        manager.initialize_state("exp_009", "Worker Test")
        
        assert not manager.is_worker_alive()
        
        manager.update_heartbeat()
        assert manager.is_worker_alive(timeout_seconds=30)
        
        old_time = datetime.now() - timedelta(seconds=60)
        manager.update_state(last_heartbeat=old_time)
        assert not manager.is_worker_alive(timeout_seconds=30)
    
    def test_concurrent_access(self, temp_dir):
        """Test concurrent read/write access with locking."""
        exp_dir = temp_dir / "exp_010"
        manager = StateManager(exp_dir)
        
        manager.initialize_state("exp_010", "Concurrent Test")
        
        def update_counter(n):
            for _ in range(n):
                state = manager.read_state()
                new_count = state.current_epoch + 1
                manager.update_state(current_epoch=new_count)
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(update_counter, 10) for _ in range(3)]
            for f in futures:
                f.result()
        
        final_state = manager.read_state()
        assert final_state.current_epoch == 30
    
    def test_state_not_found(self, temp_dir):
        """Test error when state file doesn't exist."""
        exp_dir = temp_dir / "nonexistent"
        manager = StateManager(exp_dir)
        
        with pytest.raises(FileNotFoundError):
            manager.read_state()


class TestExperimentManager:
    """Tests for ExperimentManager class."""
    
    def test_create_experiment(self, temp_dir):
        """Test creating a new experiment."""
        manager = ExperimentManager(temp_dir)
        
        exp_dir = manager.create_experiment("test_exp")
        
        assert exp_dir.exists()
        assert (exp_dir / "checkpoints").exists()
        assert (exp_dir / "queries").exists()
        assert (exp_dir / "probes").exists()
    
    def test_create_experiment_custom_id(self, temp_dir):
        """Test creating experiment with custom ID."""
        manager = ExperimentManager(temp_dir)
        
        exp_dir = manager.create_experiment("test", experiment_id="custom_123")
        
        assert exp_dir.name == "custom_123"
    
    def test_set_and_get_active(self, temp_dir):
        """Test setting and getting active experiment."""
        manager = ExperimentManager(temp_dir)
        
        assert manager.get_active() is None
        
        exp_dir = manager.create_experiment("active_test")
        manager.set_active(exp_dir)
        
        active = manager.get_active()
        assert active is not None
        assert active.experiment_id == exp_dir.name
    
    def test_clear_active(self, temp_dir):
        """Test clearing active experiment."""
        manager = ExperimentManager(temp_dir)
        
        exp_dir = manager.create_experiment("clear_test")
        manager.set_active(exp_dir)
        
        manager.clear_active()
        
        assert manager.get_active() is None
    
    def test_list_experiments(self, temp_dir):
        """Test listing all experiments."""
        manager = ExperimentManager(temp_dir)
        
        exp1 = manager.create_experiment("exp1", experiment_id="exp_001")
        exp2 = manager.create_experiment("exp2", experiment_id="exp_002")
        
        state_manager1 = StateManager(exp1)
        state_manager1.initialize_state("exp_001", "Experiment 1")
        
        experiments = manager.list_experiments()
        
        assert len(experiments) == 2
        
        exp1_info = next(e for e in experiments if e["experiment_id"] == "exp_001")
        assert exp1_info["has_state"] is True
        assert exp1_info["phase"] == "IDLE"
    
    def test_is_experiment_running(self, temp_dir):
        """Test checking if experiment is running."""
        manager = ExperimentManager(temp_dir)
        
        assert not manager.is_experiment_running()
        
        exp_dir = manager.create_experiment("running_test")
        state_manager = StateManager(exp_dir)
        state_manager.initialize_state("running_test", "Running Test")
        
        manager.set_active(exp_dir)
        
        assert not manager.is_experiment_running()
        
        state_manager.update_state(phase=ExperimentPhase.TRAINING)
        assert manager.is_experiment_running()
        
        state_manager.update_state(phase=ExperimentPhase.COMPLETED)
        assert not manager.is_experiment_running()


class TestQueriedImage:
    """Tests for QueriedImage model."""
    
    def test_create_queried_image(self):
        """Test creating a queried image."""
        img = QueriedImage(
            image_id=42,
            image_path="/data/images/car_001.jpg",
            display_path="/exp/queries/cycle_1/42_car_001.jpg",
            ground_truth=1,
            ground_truth_name="car",
            model_probabilities={
                "bus": 0.1,
                "car": 0.4,
                "motorcycle": 0.2,
                "truck": 0.3
            },
            predicted_class="car",
            predicted_confidence=0.4,
            uncertainty_score=0.6,
            selection_reason="Low confidence: 40%"
        )
        
        assert img.image_id == 42
        assert img.model_probabilities["car"] == 0.4
        assert img.uncertainty_score == 0.6


class TestProbeImage:
    """Tests for ProbeImage model."""
    
    def test_create_probe_with_predictions(self):
        """Test creating a probe image with predictions across cycles."""
        probe = ProbeImage(
            image_id=100,
            image_path="/data/images/truck_050.jpg",
            display_path="/exp/probes/probe_100_hard.jpg",
            true_class="truck",
            true_class_idx=3,
            probe_type="hard",
            predictions_by_cycle={
                0: {"bus": 0.3, "car": 0.3, "motorcycle": 0.2, "truck": 0.2},
                1: {"bus": 0.2, "car": 0.2, "motorcycle": 0.1, "truck": 0.5},
                2: {"bus": 0.1, "car": 0.1, "motorcycle": 0.05, "truck": 0.75},
            }
        )
        
        assert probe.probe_type == "hard"
        assert len(probe.predictions_by_cycle) == 3
        assert probe.predictions_by_cycle[2]["truck"] == 0.75


if __name__ == "__main__":
    pytest.main([__file__, "-v"])