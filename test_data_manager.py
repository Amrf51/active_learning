"""
Test script for ALDataManager - Phase 1 Verification.

Verifies:
1. Initial pool initialization
2. DataLoader creation
3. Pool updates (moving samples)
4. State persistence
"""

import sys
from pathlib import Path
import numpy as np
import torch
from torchvision.datasets import FakeData
from torchvision import transforms

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_manager import ALDataManager


def test_initialization():
    """Test 1: Initialize data manager with dummy dataset."""
    print("\n" + "="*70)
    print("TEST 1: Initialization")
    print("="*70)
    
    # Create dummy dataset (100 fake images, 4 classes)
    # No transforms needed for initialization test
    dummy_dataset = FakeData(
        size=100,
        image_size=(3, 224, 224),
        num_classes=4,
        random_offset=0
    )
    
    # Initialize manager
    manager = ALDataManager(
        dataset=dummy_dataset,
        n_initial_samples=30,
        seed=42
    )
    
    # Verify initialization
    assert len(manager.labeled_indices) == 30, \
        f"Expected 30 labeled, got {len(manager.labeled_indices)}"
    assert len(manager.unlabeled_indices) == 70, \
        f"Expected 70 unlabeled, got {len(manager.unlabeled_indices)}"
    assert len(manager.labeled_indices) + len(manager.unlabeled_indices) == 100, \
        "Labeled + Unlabeled != Total"
    
    # Verify no overlap
    labeled_set = set(manager.labeled_indices)
    unlabeled_set = set(manager.unlabeled_indices)
    assert len(labeled_set & unlabeled_set) == 0, \
        "Labeled and Unlabeled pools overlap!"
    
    print("✅ Pool initialization correct")
    print(f"   Labeled: {len(manager.labeled_indices)}")
    print(f"   Unlabeled: {len(manager.unlabeled_indices)}")
    print(f"   Total: {manager.total_samples}")


def test_dataloaders():
    """Test 2: DataLoader creation and iteration."""
    print("\n" + "="*70)
    print("TEST 2: DataLoaders")
    print("="*70)
    
    # Create transforms to convert PIL images to tensors
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dummy_dataset = FakeData(
        size=100,
        image_size=(3, 224, 224),
        num_classes=4,
        transform=transform  # Add transforms here
    )
    manager = ALDataManager(dummy_dataset, n_initial_samples=30, seed=42)
    
    # Create loaders with num_workers=0 to avoid multiprocessing issues
    labeled_loader = manager.get_labeled_loader(
        batch_size=16,
        shuffle=True,
        num_workers=0  # Set to 0 for testing
    )
    unlabeled_loader = manager.get_unlabeled_loader(
        batch_size=16,
        shuffle=False,
        num_workers=0  # Set to 0 for testing
    )
    
    # Verify loader properties
    assert len(labeled_loader) == np.ceil(30 / 16), \
        "Labeled loader has wrong number of batches"
    assert len(unlabeled_loader) == np.ceil(70 / 16), \
        "Unlabeled loader has wrong number of batches"
    
    print("✅ DataLoader creation correct")
    print(f"   Labeled batches: {len(labeled_loader)}")
    print(f"   Unlabeled batches: {len(unlabeled_loader)}")
    
    # Test iteration
    labeled_batch = next(iter(labeled_loader))
    unlabeled_batch = next(iter(unlabeled_loader))
    
    assert len(labeled_batch) == 2, "Batch should have (images, labels)"
    assert len(unlabeled_batch) == 2, "Batch should have (images, labels)"
    
    print("✅ DataLoaders iterate correctly")
    print(f"   Labeled batch size: {labeled_batch[0].shape}")
    print(f"   Unlabeled batch size: {unlabeled_batch[0].shape}")


def test_pool_update():
    """Test 3: Moving samples between pools."""
    print("\n" + "="*70)
    print("TEST 3: Pool Updates")
    print("="*70)
    
    dummy_dataset = FakeData(size=100, image_size=(3, 224, 224), num_classes=4)
    manager = ALDataManager(dummy_dataset, n_initial_samples=30, seed=42)
    
    initial_labeled = len(manager.labeled_indices)
    initial_unlabeled = len(manager.unlabeled_indices)
    
    print(f"Before query:")
    print(f"  Labeled: {initial_labeled}, Unlabeled: {initial_unlabeled}")
    
    # Query 10 samples from unlabeled pool
    query_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    manager.update_labeled_pool(query_indices)
    
    # Verify pools updated correctly
    assert len(manager.labeled_indices) == initial_labeled + 10, \
        f"Labeled pool should grow by 10, got {len(manager.labeled_indices)}"
    assert len(manager.unlabeled_indices) == initial_unlabeled - 10, \
        f"Unlabeled pool should shrink by 10, got {len(manager.unlabeled_indices)}"
    
    print(f"After query (10 samples):")
    print(f"  Labeled: {len(manager.labeled_indices)}, Unlabeled: {len(manager.unlabeled_indices)}")
    print("✅ Pool update correct")
    
    # Query again
    query_indices = np.array([0, 1, 2])
    manager.update_labeled_pool(query_indices)
    
    assert len(manager.labeled_indices) == initial_labeled + 13, \
        "Labeled pool size incorrect after second query"
    assert len(manager.unlabeled_indices) == initial_unlabeled - 13, \
        "Unlabeled pool size incorrect after second query"
    
    print(f"After second query (3 samples):")
    print(f"  Labeled: {len(manager.labeled_indices)}, Unlabeled: {len(manager.unlabeled_indices)}")
    print("✅ Multiple updates work correctly")
    
    # Verify history
    assert len(manager.query_history) == 2, "Should have 2 query events"
    print(f"✅ Query history tracked: {len(manager.query_history)} events")


def test_pool_info():
    """Test 4: Pool information retrieval."""
    print("\n" + "="*70)
    print("TEST 4: Pool Info")
    print("="*70)
    
    dummy_dataset = FakeData(size=100, image_size=(3, 224, 224), num_classes=4)
    manager = ALDataManager(dummy_dataset, n_initial_samples=30, seed=42)
    
    info = manager.get_pool_info()
    
    assert "labeled" in info, "Missing 'labeled' key"
    assert "unlabeled" in info, "Missing 'unlabeled' key"
    assert "total" in info, "Missing 'total' key"
    assert "labeled_percentage" in info, "Missing 'labeled_percentage' key"
    
    assert info["labeled"] == 30, "Labeled count wrong"
    assert info["unlabeled"] == 70, "Unlabeled count wrong"
    assert info["total"] == 100, "Total count wrong"
    assert info["labeled_percentage"] == 30.0, "Percentage wrong"
    
    print("✅ Pool info correct:")
    print(f"   {info}")


def test_state_persistence():
    """Test 5: Save and load state."""
    print("\n" + "="*70)
    print("TEST 5: State Persistence")
    print("="*70)
    
    # Create temporary directory
    exp_dir = Path("experiments/test_data_manager")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    dummy_dataset = FakeData(size=100, image_size=(3, 224, 224), num_classes=4)
    manager = ALDataManager(
        dataset=dummy_dataset,
        n_initial_samples=30,
        seed=42,
        exp_dir=exp_dir
    )
    
    # Query samples
    query_indices = np.array([0, 1, 2, 3, 4])
    manager.update_labeled_pool(query_indices)
    
    # Save state
    state = manager.save_state()
    
    # Verify saved file
    state_file = exp_dir / "al_pool_state.json"
    assert state_file.exists(), "State file not created"
    print(f"✅ State saved to {state_file}")
    
    # Create new manager and load state
    manager2 = ALDataManager(
        dataset=dummy_dataset,
        n_initial_samples=30,
        seed=42,
        exp_dir=exp_dir
    )
    
    # Load the saved state
    manager2.load_state(state)
    
    # Verify state matches
    assert manager2.labeled_indices == manager.labeled_indices, \
        "Labeled indices don't match after load"
    assert manager2.unlabeled_indices == manager.unlabeled_indices, \
        "Unlabeled indices don't match after load"
    assert len(manager2.query_history) == len(manager.query_history), \
        "Query history doesn't match after load"
    
    print("✅ State loaded correctly and matches original")


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*70)
    print("PHASE 1: DATA MANAGER VERIFICATION")
    print("="*70)
    
    tests = [
        ("Initialization", test_initialization),
        ("DataLoaders", test_dataloaders),
        ("Pool Updates", test_pool_update),
        ("Pool Info", test_pool_info),
        ("State Persistence", test_state_persistence),
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
            failed += 1
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Phase 1 is ready.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Review and fix.")
    
    print("="*70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)