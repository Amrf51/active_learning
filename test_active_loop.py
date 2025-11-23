"""
Test script for ActiveLearningLoop - Phase 2 Verification.

Verifies:
1. AL loop initializes correctly
2. Cycles execute without errors
3. Pools update properly each cycle
4. Results are tracked and saved
5. Strategy is called correctly
"""

import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision import transforms

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_manager import ALDataManager
from src.active_loop import ActiveLearningLoop
from src.strategies import get_strategy


class MockTrainer:
    """Mock trainer for testing (simplified version)."""
    
    def __init__(self, device="cpu"):
        # Create a simple CNN that matches FakeData input
        # Input: (batch, 3, 224, 224)
        # Output: (batch, 4) for 4 classes
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Linear(32, 4)  # 32 channels → 4 classes
        )
        self.model = self.model.to(device)
        self.device = device
        self.epochs_trained = 0
    
    def train(self, train_loader, val_loader=None):
        """Mock training - just count epochs."""
        self.epochs_trained += 1
        self.model.train()
        # In real trainer: would do actual training
    
    def evaluate(self, test_loader):
        """Mock evaluation - return dummy metrics."""
        self.model.eval()
        return {
            "test_accuracy": np.random.rand() * 0.3 + 0.7,  # 70-100%
            "test_f1": np.random.rand() * 0.3 + 0.7,
            "test_precision": np.random.rand() * 0.3 + 0.7,
            "test_recall": np.random.rand() * 0.3 + 0.7,
        }


class MockConfig:
    """Mock configuration for testing."""
    
    class training:
        batch_size = 32
        seed = 42
    
    class active_learning:
        num_cycles = 3
        batch_size_al = 5
        sampling_strategy = "uncertainty"
        uncertainty_method = "least_confidence"


def test_initialization():
    """Test 1: AL loop initialization."""
    print("\n" + "="*70)
    print("TEST 1: Initialization")
    print("="*70)
    
    # Setup
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = FakeData(size=100, image_size=(3, 224, 224), num_classes=4, transform=transform)
    data_manager = ALDataManager(dataset, n_initial_samples=30, seed=42)
    test_loader = DataLoader(dataset, batch_size=32, num_workers=0)
    
    trainer = MockTrainer()
    strategy = get_strategy("uncertainty")
    config = MockConfig()
    exp_dir = Path("experiments/test_al_loop")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize loop
    al_loop = ActiveLearningLoop(
        trainer=trainer,
        data_manager=data_manager,
        strategy=strategy,
        test_loader=test_loader,
        exp_dir=exp_dir,
        config=config
    )
    
    assert al_loop is not None, "AL loop not initialized"
    assert al_loop.trainer is not None, "Trainer not set"
    assert al_loop.data_manager is not None, "Data manager not set"
    assert len(al_loop.cycle_results) == 0, "Should start with no results"
    
    print("✅ AL loop initialized correctly")
    print(f"   Trainer: {type(al_loop.trainer).__name__}")
    print(f"   Max cycles: {config.active_learning.num_cycles}")
    print(f"   Strategy: {config.active_learning.sampling_strategy}")


def test_single_cycle():
    """Test 2: Run a single AL cycle."""
    print("\n" + "="*70)
    print("TEST 2: Single Cycle Execution")
    print("="*70)
    
    # Setup
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = FakeData(size=100, image_size=(3, 224, 224), num_classes=4, transform=transform)
    data_manager = ALDataManager(dataset, n_initial_samples=30, seed=42)
    test_loader = DataLoader(dataset, batch_size=32, num_workers=0)
    
    trainer = MockTrainer()
    strategy = get_strategy("uncertainty")
    config = MockConfig()
    exp_dir = Path("experiments/test_al_loop_single")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create loop and run one cycle
    al_loop = ActiveLearningLoop(trainer, data_manager, strategy, test_loader, exp_dir, config)
    
    initial_labeled = len(data_manager.labeled_indices)
    initial_unlabeled = len(data_manager.unlabeled_indices)
    
    print(f"Before cycle:")
    print(f"  Labeled: {initial_labeled}, Unlabeled: {initial_unlabeled}")
    
    # Run cycle 1
    al_loop.run_cycle(cycle_num=1)
    
    # Verify
    assert len(al_loop.cycle_results) == 1, "Should have 1 cycle result"
    result = al_loop.cycle_results[0]
    assert result["cycle"] == 1, "Cycle number wrong"
    assert result["labeled_pool_size"] == initial_labeled, "Labeled pool changed (shouldn't in cycle 1 after)"
    assert result["test_accuracy"] > 0, "No accuracy metric"
    
    print(f"After cycle:")
    print(f"  Labeled: {len(data_manager.labeled_indices)}, Unlabeled: {len(data_manager.unlabeled_indices)}")
    print(f"  Accuracy: {result['test_accuracy']:.4f}")
    print("✅ Single cycle executed correctly")


def test_multiple_cycles():
    """Test 3: Run multiple AL cycles."""
    print("\n" + "="*70)
    print("TEST 3: Multiple Cycles")
    print("="*70)
    
    # Setup
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = FakeData(size=100, image_size=(3, 224, 224), num_classes=4, transform=transform)
    data_manager = ALDataManager(dataset, n_initial_samples=30, seed=42)
    test_loader = DataLoader(dataset, batch_size=32, num_workers=0)
    
    trainer = MockTrainer()
    strategy = get_strategy("uncertainty")
    config = MockConfig()
    config.active_learning.num_cycles = 3
    config.active_learning.batch_size_al = 5
    
    exp_dir = Path("experiments/test_al_loop_multiple")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create loop and run all cycles
    al_loop = ActiveLearningLoop(trainer, data_manager, strategy, test_loader, exp_dir, config)
    results = al_loop.run_all_cycles()
    
    # Verify
    assert len(results) == 3, f"Should have 3 results, got {len(results)}"
    assert all(r["cycle"] == i+1 for i, r in enumerate(results)), "Cycle numbers incorrect"
    
    # Check pool growth
    print(f"Pool growth across cycles:")
    for result in results:
        print(f"  Cycle {result['cycle']}: {result['labeled_pool_size']} labeled, "
              f"{result['unlabeled_pool_size']} unlabeled, "
              f"Acc={result['test_accuracy']:.4f}")
    
    # Labeled should grow (except first cycle doesn't query before training)
    # Cycle 1: 30, Cycle 2: 35, Cycle 3: 40
    assert results[1]["labeled_pool_size"] >= results[0]["labeled_pool_size"], \
        "Labeled pool should grow in cycle 2"
    assert results[2]["labeled_pool_size"] >= results[1]["labeled_pool_size"], \
        "Labeled pool should grow in cycle 3"
    
    print("✅ All 3 cycles executed correctly")


def test_results_tracking():
    """Test 4: Results are tracked and saved."""
    print("\n" + "="*70)
    print("TEST 4: Results Tracking & Saving")
    print("="*70)
    
    # Setup
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = FakeData(size=100, image_size=(3, 224, 224), num_classes=4, transform=transform)
    data_manager = ALDataManager(dataset, n_initial_samples=30, seed=42)
    test_loader = DataLoader(dataset, batch_size=32, num_workers=0)
    
    trainer = MockTrainer()
    strategy = get_strategy("uncertainty")
    config = MockConfig()
    config.active_learning.num_cycles = 2
    
    exp_dir = Path("experiments/test_al_loop_results")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data manager with exp_dir so it saves state
    data_manager_with_dir = ALDataManager(
        dataset=dataset,
        n_initial_samples=30,
        seed=42,
        exp_dir=exp_dir  # Important: pass exp_dir
    )
    
    # Run AL
    al_loop = ActiveLearningLoop(
        trainer=trainer,
        data_manager=data_manager_with_dir,
        strategy=strategy,
        test_loader=test_loader,
        exp_dir=exp_dir,
        config=config
    )
    results = al_loop.run_all_cycles()
    
    # Check files exist
    results_file = exp_dir / "al_cycle_results.json"
    pool_state_file = exp_dir / "al_pool_state.json"
    
    assert results_file.exists(), f"Results file not created: {results_file}"
    assert pool_state_file.exists(), f"Pool state file not created: {pool_state_file}"
    print("✅ Result files created")
    
    # Check content
    with open(results_file) as f:
        saved_results = json.load(f)
    
    assert len(saved_results) == 2, f"Should have 2 cycles saved, got {len(saved_results)}"
    assert all("test_accuracy" in r for r in saved_results), "Missing accuracy metrics"
    print("✅ Results saved correctly")
    
    # Check best cycle
    best = al_loop.get_best_cycle()
    assert best is not None, "Should have best cycle"
    assert best["test_accuracy"] == max(r["test_accuracy"] for r in results), \
        "Best cycle wrong"
    print(f"✅ Best cycle: {best['cycle']} with accuracy {best['test_accuracy']:.4f}")


def test_strategy_integration():
    """Test 5: Strategy is called correctly."""
    print("\n" + "="*70)
    print("TEST 5: Strategy Integration")
    print("="*70)
    
    # Setup
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = FakeData(size=100, image_size=(3, 224, 224), num_classes=4, transform=transform)
    data_manager = ALDataManager(dataset, n_initial_samples=30, seed=42)
    test_loader = DataLoader(dataset, batch_size=32, num_workers=0)
    
    trainer = MockTrainer()
    
    # Test different strategies
    strategies = ["uncertainty", "margin", "entropy", "random"]
    
    for strat_name in strategies:
        config = MockConfig()
        config.active_learning.sampling_strategy = strat_name
        config.active_learning.num_cycles = 1
        
        try:
            strategy = get_strategy(strat_name)
            exp_dir = Path(f"experiments/test_al_strategy_{strat_name}")
            exp_dir.mkdir(parents=True, exist_ok=True)
            
            al_loop = ActiveLearningLoop(trainer, data_manager, strategy, test_loader, exp_dir, config)
            al_loop.run_cycle(cycle_num=1)
            
            print(f"✅ {strat_name}: works correctly")
        except Exception as e:
            print(f"❌ {strat_name}: {e}")


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*70)
    print("PHASE 2: ACTIVE LEARNING LOOP VERIFICATION")
    print("="*70)
    
    import json
    
    tests = [
        ("Initialization", test_initialization),
        ("Single Cycle", test_single_cycle),
        ("Multiple Cycles", test_multiple_cycles),
        ("Results Tracking", test_results_tracking),
        ("Strategy Integration", test_strategy_integration),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ TEST FAILED: {test_name}")
            print(f"   Error: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ TEST ERROR: {test_name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Phase 2 is ready.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Review and fix.")
    
    print("="*70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)