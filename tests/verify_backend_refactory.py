"""
Verification script for Week 2 backend refactoring.

This script tests the refactored components with real data (if available)
or dummy data, simulating the step-by-step execution pattern that the
worker process will use.

Run from project root:
    python scripts/verify_backend_refactor.py
    
With real data:
    python scripts/verify_backend_refactor.py --use-real-data
"""

import sys
import argparse
import shutil
import tempfile
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class DummyDataset(Dataset):
    """Dummy dataset for testing without real data."""
    
    def __init__(self, num_samples=200, num_classes=4):
        self.num_samples = num_samples
        self.num_classes = num_classes
        
        self.samples = [
            (f"/fake/path/class_{i % num_classes}/image_{i}.jpg", i % num_classes)
            for i in range(num_samples)
        ]
        
        torch.manual_seed(42)
        self.images = torch.randn(num_samples, 3, 64, 64)
        self.labels = torch.tensor([i % num_classes for i in range(num_samples)])
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class DummyModel(nn.Module):
    """Simple CNN for testing."""
    
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, num_classes)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class MockConfig:
    """Mock configuration matching the real Config structure."""
    
    class Training:
        epochs = 3
        batch_size = 16
        learning_rate = 0.001
        optimizer = "adam"
        weight_decay = 0.0001
        early_stopping_patience = 5
        seed = 42
    
    class Data:
        num_workers = 0
        data_dir = "./data/raw/kaggle-vehicle/"
        val_split = 0.15
        test_split = 0.15
        augmentation = False
    
    class Model:
        name = "dummy"
        pretrained = False
        num_classes = 4
    
    class ActiveLearning:
        enabled = True
        num_cycles = 2
        sampling_strategy = "uncertainty"
        uncertainty_method = "least_confidence"
        initial_pool_size = 30
        batch_size_al = 15
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


def run_verification(use_real_data: bool = False):
    """Run backend verification."""
    
    print("\n" + "="*60)
    print("BACKEND REFACTOR VERIFICATION")
    print("="*60)
    
    from src.trainer import Trainer
    from src.data_manager import ALDataManager
    from src.active_loop import ActiveLearningLoop
    from src.strategies import uncertainty_least_confidence, random_sampling
    from src.state import EpochMetrics, CycleMetrics, StateManager, ExperimentPhase
    
    temp_dir = Path(tempfile.mkdtemp())
    print(f"\nTest directory: {temp_dir}")
    
    try:
        config = MockConfig()
        class_names = ["bus", "car", "motorcycle", "truck"]
        
        print("\n[1] Setting up datasets...")
        
        if use_real_data:
            try:
                from src.dataloader import get_datasets
                datasets = get_datasets(
                    data_dir=config.data.data_dir,
                    val_split=config.data.val_split,
                    test_split=config.data.test_split,
                    augmentation=False,
                    seed=42
                )
                train_dataset = datasets["train_dataset"]
                val_dataset = datasets["val_dataset"]
                test_dataset = datasets["test_dataset"]
                class_names = datasets["class_names"]
                print(f"    Loaded real data: {len(train_dataset)} train samples")
            except Exception as e:
                print(f"    Could not load real data: {e}")
                print("    Falling back to dummy data")
                use_real_data = False
        
        if not use_real_data:
            full_dataset = DummyDataset(num_samples=200, num_classes=4)
            train_dataset = full_dataset
            val_dataset = DummyDataset(num_samples=40, num_classes=4)
            test_dataset = DummyDataset(num_samples=40, num_classes=4)
            print(f"    Using dummy data: {len(train_dataset)} train samples")
        
        print("\n[2] Initializing model...")
        if use_real_data:
            from src.models import get_model
            model = get_model(config.model, device="cpu")
        else:
            model = DummyModel(num_classes=4)
        print(f"    Model: {model.__class__.__name__}")
        
        print("\n[3] Initializing Trainer...")
        trainer = Trainer(
            model=model,
            config=config,
            exp_dir=temp_dir,
            device="cpu"
        )
        print("    Trainer ready")
        
        print("\n[4] Testing Trainer.train_single_epoch()...")
        simple_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        metrics = trainer.train_single_epoch(simple_loader, val_loader, epoch_num=1)
        print(f"    Epoch 1: train_loss={metrics.train_loss:.4f}, train_acc={metrics.train_accuracy:.4f}")
        assert isinstance(metrics, EpochMetrics)
        print("    train_single_epoch() works correctly")
        
        print("\n[5] Testing Trainer.get_predictions_for_indices()...")
        test_indices = [0, 1, 2, 3, 4]
        predictions = trainer.get_predictions_for_indices(test_indices, train_dataset, class_names)
        print(f"    Got predictions for {len(predictions)} samples")
        print(f"    Sample prediction: {predictions[0]['predicted_class']} ({predictions[0]['confidence']:.2%})")
        assert len(predictions) == 5
        assert all("probabilities" in p for p in predictions)
        print("    get_predictions_for_indices() works correctly")
        
        print("\n[6] Initializing ALDataManager...")
        trainer._reset_tracking()
        trainer.model = DummyModel(num_classes=4) if not use_real_data else get_model(config.model, device="cpu")
        trainer.optimizer = trainer._create_optimizer()
        
        data_manager = ALDataManager(
            dataset=train_dataset,
            initial_pool_size=config.active_learning.initial_pool_size,
            seed=42,
            exp_dir=temp_dir
        )
        pool_info = data_manager.get_pool_info()
        print(f"    Labeled: {pool_info['labeled']}, Unlabeled: {pool_info['unlabeled']}")
        
        print("\n[7] Testing ALDataManager.get_image_info()...")
        img_info = data_manager.get_image_info(0)
        print(f"    Image 0: path={img_info['path'][:50]}..., label={img_info['label']}")
        assert "image_id" in img_info
        print("    get_image_info() works correctly")
        
        print("\n[8] Initializing ActiveLearningLoop...")
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        al_loop = ActiveLearningLoop(
            trainer=trainer,
            data_manager=data_manager,
            strategy=uncertainty_least_confidence,
            val_loader=val_loader,
            test_loader=test_loader,
            exp_dir=temp_dir,
            config=config,
            class_names=class_names
        )
        print("    ActiveLearningLoop ready")
        
        print("\n[9] Testing step-by-step cycle execution...")
        
        print("\n    [9a] prepare_cycle(1)...")
        prep_info = al_loop.prepare_cycle(1)
        print(f"        Labeled: {prep_info['labeled_count']}, Unlabeled: {prep_info['unlabeled_count']}")
        
        print("\n    [9b] Training epochs...")
        for epoch in range(1, config.training.epochs + 1):
            metrics = al_loop.train_single_epoch(epoch)
            print(f"        Epoch {epoch}: loss={metrics.train_loss:.4f}, acc={metrics.train_accuracy:.4f}")
        
        print("\n    [9c] run_evaluation()...")
        test_metrics = al_loop.run_evaluation()
        print(f"        Test accuracy: {test_metrics['test_accuracy']:.4f}")
        
        print("\n    [9d] query_samples()...")
        queried_images = al_loop.query_samples()
        print(f"        Queried {len(queried_images)} images")
        if queried_images:
            q = queried_images[0]
            print(f"        Sample: id={q.image_id}, pred={q.predicted_class} ({q.predicted_confidence:.2%})")
            print(f"                reason: {q.selection_reason}")
        
        print("\n    [9e] receive_annotations() (simulated)...")
        annotations = [
            {"image_id": q.image_id, "user_label": q.ground_truth}
            for q in queried_images
        ]
        result = al_loop.receive_annotations(annotations)
        print(f"        Moved: {result['moved_count']}, Accuracy: {result['annotation_accuracy']:.1%}")
        
        print("\n    [9f] finalize_cycle()...")
        cycle_metrics = al_loop.finalize_cycle(test_metrics)
        print(f"        Cycle 1 complete: test_acc={cycle_metrics.test_accuracy:.4f}")
        
        print("\n[10] Verifying state integration...")
        state_manager = StateManager(temp_dir)
        state = state_manager.initialize_state("test_exp", "Backend Verification")
        
        state_manager.add_epoch_metrics(EpochMetrics(
            epoch=1, train_loss=0.5, train_accuracy=0.7, val_loss=0.6, val_accuracy=0.65
        ))
        
        state_manager.finalize_cycle(cycle_metrics)
        
        final_state = state_manager.read_state()
        print(f"    State has {len(final_state.cycle_results)} cycle(s)")
        assert len(final_state.cycle_results) == 1
        print("    State integration works correctly")
        
        print("\n[11] Testing second cycle preparation...")
        prep_info_2 = al_loop.prepare_cycle(2)
        print(f"    Cycle 2 - Labeled: {prep_info_2['labeled_count']}, Unlabeled: {prep_info_2['unlabeled_count']}")
        assert prep_info_2['labeled_count'] > prep_info['labeled_count']
        print("    Pool correctly updated between cycles")
        
        print("\n" + "="*60)
        print("ALL BACKEND VERIFICATIONS PASSED")
        print("="*60)
        
        print("\nSummary of verified functionality:")
        print("  - Trainer.train_single_epoch()")
        print("  - Trainer.get_predictions_for_indices()")
        print("  - Trainer.compute_uncertainty_scores()")
        print("  - ALDataManager.get_image_info()")
        print("  - ALDataManager.update_labeled_pool_with_annotations()")
        print("  - ActiveLearningLoop.prepare_cycle()")
        print("  - ActiveLearningLoop.train_single_epoch()")
        print("  - ActiveLearningLoop.run_evaluation()")
        print("  - ActiveLearningLoop.query_samples()")
        print("  - ActiveLearningLoop.receive_annotations()")
        print("  - ActiveLearningLoop.finalize_cycle()")
        print("  - StateManager integration")
        
        return True
        
    except Exception as e:
        print(f"\nVERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        print(f"\nCleaning up: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Verify backend refactoring")
    parser.add_argument(
        "--use-real-data",
        action="store_true",
        help="Use real dataset if available"
    )
    args = parser.parse_args()
    
    success = run_verification(use_real_data=args.use_real_data)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())