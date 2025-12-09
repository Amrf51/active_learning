"""
Tests for Week 2 backend refactoring.

These tests verify:
1. Trainer epoch-level control
2. ActiveLearningLoop step-by-step execution
3. ALDataManager annotation support

Run with: pytest tests/test_backend_refactor.py -v
"""

import pytest
import tempfile
import shutil
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.state import EpochMetrics, CycleMetrics, QueriedImage
from src.config import Config


class DummyDataset(Dataset):
    """Simple dataset for testing."""
    
    def __init__(self, num_samples=100, num_classes=4, img_size=32):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.img_size = img_size
        
        self.samples = [
            (f"/fake/path/image_{i}.jpg", i % num_classes)
            for i in range(num_samples)
        ]
        
        torch.manual_seed(42)
        self.images = torch.randn(num_samples, 3, img_size, img_size)
        self.labels = torch.tensor([i % num_classes for i in range(num_samples)])
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class DummyModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, num_classes=4, img_size=32):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3 * img_size * img_size, num_classes)
    
    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)


class MockConfig:
    """Mock config for testing."""
    
    class Training:
        epochs = 5
        batch_size = 16
        learning_rate = 0.001
        optimizer = "adam"
        weight_decay = 0.0001
        early_stopping_patience = 3
        seed = 42
    
    class Data:
        num_workers = 0
        data_dir = "./data"
        val_split = 0.15
        test_split = 0.15
        augmentation = False
    
    class Model:
        name = "dummy"
        pretrained = False
        num_classes = 4
    
    class ActiveLearning:
        enabled = True
        num_cycles = 3
        sampling_strategy = "uncertainty"
        uncertainty_method = "least_confidence"
        initial_pool_size = 20
        batch_size_al = 10
        reset_mode = "none"
    
    class Checkpoint:
        save_every_n_epochs = 5
        save_best_model = True
        save_best_per_cycle = True
        log_every_n_batches = 10
    
    training = Training()
    data = Data()
    model = Model()
    active_learning = ActiveLearning()
    checkpoint = Checkpoint()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def dummy_dataset():
    """Create a dummy dataset."""
    return DummyDataset(num_samples=100, num_classes=4)


@pytest.fixture
def dummy_model():
    """Create a dummy model."""
    return DummyModel(num_classes=4)


@pytest.fixture
def mock_config():
    """Create mock config."""
    return MockConfig()


class TestTrainerEpochControl:
    """Tests for Trainer epoch-level control."""
    
    def test_train_single_epoch(self, temp_dir, dummy_dataset, dummy_model, mock_config):
        """Test training a single epoch."""
        from src.trainer import Trainer
        
        trainer = Trainer(
            model=dummy_model,
            config=mock_config,
            exp_dir=temp_dir,
            device="cpu"
        )
        
        train_loader = DataLoader(dummy_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(dummy_dataset, batch_size=16, shuffle=False)
        
        metrics = trainer.train_single_epoch(train_loader, val_loader, epoch_num=1)
        
        assert isinstance(metrics, EpochMetrics)
        assert metrics.epoch == 1
        assert 0 <= metrics.train_accuracy <= 1
        assert metrics.train_loss > 0
        assert metrics.val_accuracy is not None
    
    def test_train_multiple_epochs_incrementally(self, temp_dir, dummy_dataset, dummy_model, mock_config):
        """Test training multiple epochs one at a time."""
        from src.trainer import Trainer
        
        trainer = Trainer(
            model=dummy_model,
            config=mock_config,
            exp_dir=temp_dir,
            device="cpu"
        )
        
        train_loader = DataLoader(dummy_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(dummy_dataset, batch_size=16, shuffle=False)
        
        all_metrics = []
        for epoch in range(1, 4):
            metrics = trainer.train_single_epoch(train_loader, val_loader, epoch)
            all_metrics.append(metrics)
        
        assert len(all_metrics) == 3
        assert all_metrics[0].epoch == 1
        assert all_metrics[2].epoch == 3
        
        assert len(trainer.history["epoch"]) == 3
    
    def test_should_stop_early(self, temp_dir, dummy_dataset, dummy_model, mock_config):
        """Test early stopping detection."""
        from src.trainer import Trainer
        
        mock_config.training.early_stopping_patience = 2
        
        trainer = Trainer(
            model=dummy_model,
            config=mock_config,
            exp_dir=temp_dir,
            device="cpu"
        )
        
        assert not trainer.should_stop_early()
        
        trainer.patience_counter = 2
        assert trainer.should_stop_early()
    
    def test_get_predictions_for_indices(self, temp_dir, dummy_dataset, dummy_model, mock_config):
        """Test getting predictions for specific indices."""
        from src.trainer import Trainer
        
        trainer = Trainer(
            model=dummy_model,
            config=mock_config,
            exp_dir=temp_dir,
            device="cpu"
        )
        
        indices = [0, 5, 10, 15]
        class_names = ["bus", "car", "motorcycle", "truck"]
        
        predictions = trainer.get_predictions_for_indices(
            indices, dummy_dataset, class_names
        )
        
        assert len(predictions) == 4
        assert all("probabilities" in p for p in predictions)
        assert all("confidence" in p for p in predictions)
        assert all("predicted_class" in p for p in predictions)
        assert all(len(p["probabilities"]) == 4 for p in predictions)
    
    def test_compute_uncertainty_scores(self, temp_dir, dummy_dataset, dummy_model, mock_config):
        """Test uncertainty score computation."""
        from src.trainer import Trainer
        
        trainer = Trainer(
            model=dummy_model,
            config=mock_config,
            exp_dir=temp_dir,
            device="cpu"
        )
        
        prob_matrix = np.array([
            [0.9, 0.05, 0.03, 0.02],
            [0.25, 0.25, 0.25, 0.25],
            [0.5, 0.4, 0.05, 0.05],
        ])
        
        lc_scores = trainer.compute_uncertainty_scores(prob_matrix, "least_confidence")
        assert lc_scores[0] < lc_scores[1]
        
        entropy_scores = trainer.compute_uncertainty_scores(prob_matrix, "entropy")
        assert entropy_scores[0] < entropy_scores[1]
        
        margin_scores = trainer.compute_uncertainty_scores(prob_matrix, "margin")
        assert len(margin_scores) == 3
    
    def test_get_training_summary(self, temp_dir, dummy_dataset, dummy_model, mock_config):
        """Test getting training summary."""
        from src.trainer import Trainer
        
        trainer = Trainer(
            model=dummy_model,
            config=mock_config,
            exp_dir=temp_dir,
            device="cpu"
        )
        
        train_loader = DataLoader(dummy_dataset, batch_size=16)
        trainer.train_single_epoch(train_loader, None, 1)
        
        summary = trainer.get_training_summary()
        
        assert "best_val_accuracy" in summary
        assert "epochs_trained" in summary
        assert summary["epochs_trained"] == 1


class TestALDataManagerAnnotations:
    """Tests for ALDataManager annotation support."""
    
    def test_get_image_info(self, temp_dir, dummy_dataset):
        """Test getting image info."""
        from src.data_manager import ALDataManager
        
        manager = ALDataManager(
            dataset=dummy_dataset,
            initial_pool_size=20,
            seed=42,
            exp_dir=temp_dir
        )
        
        info = manager.get_image_info(0)
        
        assert "image_id" in info
        assert "path" in info
        assert "label" in info
    
    def test_get_ground_truth(self, temp_dir, dummy_dataset):
        """Test getting ground truth label."""
        from src.data_manager import ALDataManager
        
        manager = ALDataManager(
            dataset=dummy_dataset,
            initial_pool_size=20,
            seed=42,
            exp_dir=temp_dir
        )
        
        label = manager.get_ground_truth(0)
        assert isinstance(label, int)
        assert 0 <= label < 4
    
    def test_update_labeled_pool_with_annotations(self, temp_dir, dummy_dataset):
        """Test updating pool with user annotations."""
        from src.data_manager import ALDataManager
        
        manager = ALDataManager(
            dataset=dummy_dataset,
            initial_pool_size=20,
            seed=42,
            exp_dir=temp_dir
        )
        
        initial_labeled = manager.get_pool_info()["labeled"]
        
        unlabeled = manager.get_unlabeled_indices()[:5]
        
        annotations = []
        for img_id in unlabeled:
            gt = manager.get_ground_truth(img_id)
            annotations.append({
                "image_id": img_id,
                "user_label": gt
            })
        
        result = manager.update_labeled_pool_with_annotations(annotations)
        
        assert result["moved_count"] == 5
        assert result["annotation_accuracy"] == 1.0
        assert result["correct_count"] == 5
        
        new_labeled = manager.get_pool_info()["labeled"]
        assert new_labeled == initial_labeled + 5
    
    def test_annotation_with_wrong_labels(self, temp_dir, dummy_dataset):
        """Test annotations where user makes mistakes."""
        from src.data_manager import ALDataManager
        
        manager = ALDataManager(
            dataset=dummy_dataset,
            initial_pool_size=20,
            seed=42,
            exp_dir=temp_dir
        )
        
        unlabeled = manager.get_unlabeled_indices()[:4]
        
        annotations = []
        for i, img_id in enumerate(unlabeled):
            gt = manager.get_ground_truth(img_id)
            user_label = gt if i < 2 else (gt + 1) % 4
            annotations.append({
                "image_id": img_id,
                "user_label": user_label
            })
        
        result = manager.update_labeled_pool_with_annotations(annotations)
        
        assert result["moved_count"] == 4
        assert result["correct_count"] == 2
        assert result["annotation_accuracy"] == 0.5
    
    def test_get_annotation_summary(self, temp_dir, dummy_dataset):
        """Test annotation summary statistics."""
        from src.data_manager import ALDataManager
        
        manager = ALDataManager(
            dataset=dummy_dataset,
            initial_pool_size=20,
            seed=42,
            exp_dir=temp_dir
        )
        
        for _ in range(2):
            unlabeled = manager.get_unlabeled_indices()[:3]
            annotations = [
                {"image_id": img_id, "user_label": manager.get_ground_truth(img_id)}
                for img_id in unlabeled
            ]
            manager.update_labeled_pool_with_annotations(annotations)
        
        summary = manager.get_annotation_summary()
        
        assert summary["total_annotations"] == 6
        assert summary["cycles"] == 2
        assert summary["overall_accuracy"] == 1.0
    
    def test_get_class_distribution(self, temp_dir, dummy_dataset):
        """Test class distribution computation."""
        from src.data_manager import ALDataManager
        
        manager = ALDataManager(
            dataset=dummy_dataset,
            initial_pool_size=20,
            seed=42,
            exp_dir=temp_dir
        )
        
        labeled_dist = manager.get_class_distribution("labeled")
        unlabeled_dist = manager.get_class_distribution("unlabeled")
        
        assert sum(labeled_dist.values()) == 20
        assert sum(unlabeled_dist.values()) == 80


class TestActiveLearningLoopStepByStep:
    """Tests for ActiveLearningLoop step-by-step execution."""
    
    def test_prepare_cycle(self, temp_dir, dummy_dataset, dummy_model, mock_config):
        """Test cycle preparation."""
        from src.trainer import Trainer
        from src.data_manager import ALDataManager
        from src.active_loop import ActiveLearningLoop
        from src.strategies import random_sampling
        
        trainer = Trainer(dummy_model, mock_config, temp_dir, "cpu")
        data_manager = ALDataManager(dummy_dataset, initial_pool_size=20, exp_dir=temp_dir)
        
        val_loader = DataLoader(dummy_dataset, batch_size=16)
        test_loader = DataLoader(dummy_dataset, batch_size=16)
        
        al_loop = ActiveLearningLoop(
            trainer=trainer,
            data_manager=data_manager,
            strategy=random_sampling,
            val_loader=val_loader,
            test_loader=test_loader,
            exp_dir=temp_dir,
            config=mock_config,
            class_names=["bus", "car", "motorcycle", "truck"]
        )
        
        info = al_loop.prepare_cycle(1)
        
        assert info["cycle"] == 1
        assert info["labeled_count"] == 20
        assert info["unlabeled_count"] == 80
        assert al_loop.current_train_loader is not None
    
    def test_train_single_epoch_via_loop(self, temp_dir, dummy_dataset, dummy_model, mock_config):
        """Test training single epoch through AL loop."""
        from src.trainer import Trainer
        from src.data_manager import ALDataManager
        from src.active_loop import ActiveLearningLoop
        from src.strategies import random_sampling
        
        trainer = Trainer(dummy_model, mock_config, temp_dir, "cpu")
        data_manager = ALDataManager(dummy_dataset, initial_pool_size=20, exp_dir=temp_dir)
        
        val_loader = DataLoader(dummy_dataset, batch_size=16)
        test_loader = DataLoader(dummy_dataset, batch_size=16)
        
        al_loop = ActiveLearningLoop(
            trainer=trainer,
            data_manager=data_manager,
            strategy=random_sampling,
            val_loader=val_loader,
            test_loader=test_loader,
            exp_dir=temp_dir,
            config=mock_config,
            class_names=["bus", "car", "motorcycle", "truck"]
        )
        
        al_loop.prepare_cycle(1)
        
        metrics = al_loop.train_single_epoch(1)
        
        assert isinstance(metrics, EpochMetrics)
        assert metrics.epoch == 1
    
    def test_query_samples(self, temp_dir, dummy_dataset, dummy_model, mock_config):
        """Test querying samples for annotation."""
        from src.trainer import Trainer
        from src.data_manager import ALDataManager
        from src.active_loop import ActiveLearningLoop
        from src.strategies import random_sampling
        
        trainer = Trainer(dummy_model, mock_config, temp_dir, "cpu")
        data_manager = ALDataManager(dummy_dataset, initial_pool_size=20, exp_dir=temp_dir)
        
        val_loader = DataLoader(dummy_dataset, batch_size=16)
        test_loader = DataLoader(dummy_dataset, batch_size=16)
        
        al_loop = ActiveLearningLoop(
            trainer=trainer,
            data_manager=data_manager,
            strategy=random_sampling,
            val_loader=val_loader,
            test_loader=test_loader,
            exp_dir=temp_dir,
            config=mock_config,
            class_names=["bus", "car", "motorcycle", "truck"]
        )
        
        al_loop.prepare_cycle(1)
        al_loop.train_single_epoch(1)
        
        queried = al_loop.query_samples()
        
        assert len(queried) == mock_config.active_learning.batch_size_al
        assert all(isinstance(q, QueriedImage) for q in queried)
        assert all(q.image_id >= 0 for q in queried)
        assert all(q.predicted_confidence >= 0 for q in queried)
    
    def test_receive_annotations(self, temp_dir, dummy_dataset, dummy_model, mock_config):
        """Test receiving annotations from user."""
        from src.trainer import Trainer
        from src.data_manager import ALDataManager
        from src.active_loop import ActiveLearningLoop
        from src.strategies import random_sampling
        
        trainer = Trainer(dummy_model, mock_config, temp_dir, "cpu")
        data_manager = ALDataManager(dummy_dataset, initial_pool_size=20, exp_dir=temp_dir)
        
        val_loader = DataLoader(dummy_dataset, batch_size=16)
        test_loader = DataLoader(dummy_dataset, batch_size=16)
        
        al_loop = ActiveLearningLoop(
            trainer=trainer,
            data_manager=data_manager,
            strategy=random_sampling,
            val_loader=val_loader,
            test_loader=test_loader,
            exp_dir=temp_dir,
            config=mock_config,
            class_names=["bus", "car", "motorcycle", "truck"]
        )
        
        al_loop.prepare_cycle(1)
        al_loop.train_single_epoch(1)
        
        queried = al_loop.query_samples()
        
        annotations = [
            {"image_id": q.image_id, "user_label": q.ground_truth}
            for q in queried
        ]
        
        result = al_loop.receive_annotations(annotations)
        
        assert result["moved_count"] == len(queried)
        assert data_manager.get_pool_info()["labeled"] == 20 + len(queried)
    
    def test_finalize_cycle(self, temp_dir, dummy_dataset, dummy_model, mock_config):
        """Test finalizing a cycle."""
        from src.trainer import Trainer
        from src.data_manager import ALDataManager
        from src.active_loop import ActiveLearningLoop
        from src.strategies import random_sampling
        
        trainer = Trainer(dummy_model, mock_config, temp_dir, "cpu")
        data_manager = ALDataManager(dummy_dataset, initial_pool_size=20, exp_dir=temp_dir)
        
        val_loader = DataLoader(dummy_dataset, batch_size=16)
        test_loader = DataLoader(dummy_dataset, batch_size=16)
        
        al_loop = ActiveLearningLoop(
            trainer=trainer,
            data_manager=data_manager,
            strategy=random_sampling,
            val_loader=val_loader,
            test_loader=test_loader,
            exp_dir=temp_dir,
            config=mock_config,
            class_names=["bus", "car", "motorcycle", "truck"]
        )
        
        al_loop.prepare_cycle(1)
        al_loop.train_single_epoch(1)
        test_metrics = al_loop.run_evaluation()
        
        cycle_metrics = al_loop.finalize_cycle(test_metrics)
        
        assert isinstance(cycle_metrics, CycleMetrics)
        assert cycle_metrics.cycle == 1
        assert cycle_metrics.test_accuracy >= 0
        assert len(al_loop.cycle_results) == 1


class TestIntegration:
    """Integration tests for complete workflow."""
    
    def test_complete_cycle_step_by_step(self, temp_dir, dummy_dataset, dummy_model, mock_config):
        """Test running a complete cycle step by step."""
        from src.trainer import Trainer
        from src.data_manager import ALDataManager
        from src.active_loop import ActiveLearningLoop
        from src.strategies import random_sampling
        
        trainer = Trainer(dummy_model, mock_config, temp_dir, "cpu")
        data_manager = ALDataManager(dummy_dataset, initial_pool_size=20, exp_dir=temp_dir)
        
        val_loader = DataLoader(dummy_dataset, batch_size=16)
        test_loader = DataLoader(dummy_dataset, batch_size=16)
        
        al_loop = ActiveLearningLoop(
            trainer=trainer,
            data_manager=data_manager,
            strategy=random_sampling,
            val_loader=val_loader,
            test_loader=test_loader,
            exp_dir=temp_dir,
            config=mock_config,
            class_names=["bus", "car", "motorcycle", "truck"]
        )
        
        prep_info = al_loop.prepare_cycle(1)
        assert prep_info["labeled_count"] == 20
        
        for epoch in range(1, 3):
            metrics = al_loop.train_single_epoch(epoch)
            assert metrics.epoch == epoch
        
        test_metrics = al_loop.run_evaluation()
        assert "test_accuracy" in test_metrics
        
        queried = al_loop.query_samples()
        assert len(queried) > 0
        
        annotations = [
            {"image_id": q.image_id, "user_label": q.ground_truth}
            for q in queried
        ]
        result = al_loop.receive_annotations(annotations)
        assert result["moved_count"] == len(queried)
        
        cycle_metrics = al_loop.finalize_cycle(test_metrics)
        assert cycle_metrics.cycle == 1
        
        final_pool = data_manager.get_pool_info()
        assert final_pool["labeled"] == 20 + len(queried)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])