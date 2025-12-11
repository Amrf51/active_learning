#!/usr/bin/env python3
"""
Simple test script to validate dashboard imports and basic functionality.
Run this to check if the dashboard components can be imported correctly.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing dashboard imports...")
    
    try:
        # Test state management imports
        from state import (
            ExperimentManager, 
            StateManager, 
            ExperimentConfig, 
            ExperimentState,
            ExperimentPhase,
            Command
        )
        print("✅ State management imports successful")
        
        # Test config imports
        from config import Config, ModelConfig, TrainingConfig
        print("✅ Configuration imports successful")
        
        # Test basic functionality
        experiments_dir = Path("test_experiments")
        manager = ExperimentManager(experiments_dir)
        print("✅ ExperimentManager creation successful")
        
        # Clean up test directory
        if experiments_dir.exists():
            import shutil
            shutil.rmtree(experiments_dir)
        
        print("\n🎉 All imports successful! Dashboard should work correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_config_creation():
    """Test configuration creation."""
    print("\nTesting configuration creation...")
    
    try:
        from state import ExperimentConfig, DatasetInfo
        
        # Create sample dataset info
        dataset_info = DatasetInfo(
            total_images=1000,
            num_classes=4,
            class_names=["car", "truck", "bus", "motorcycle"],
            class_counts={"car": 300, "truck": 250, "bus": 200, "motorcycle": 250},
            train_samples=800,
            val_samples=100,
            test_samples=100
        )
        print("✅ DatasetInfo creation successful")
        
        # Create sample experiment config
        config = ExperimentConfig(
            model_name="resnet18",
            pretrained=True,
            num_classes=4,
            class_names=["car", "truck", "bus", "motorcycle"],
            epochs_per_cycle=10,
            batch_size=32,
            learning_rate=0.001,
            optimizer="adam",
            weight_decay=1e-4,
            early_stopping_patience=5,
            num_cycles=5,
            sampling_strategy="uncertainty",
            uncertainty_method="entropy",
            initial_pool_size=50,
            batch_size_al=10,
            reset_mode="pretrained",
            seed=42,
            data_dir="data",
            val_split=0.1,
            test_split=0.1,
            augmentation=True
        )
        print("✅ ExperimentConfig creation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration creation error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Dashboard Import Test")
    print("=" * 50)
    
    success = test_imports()
    if success:
        success = test_config_creation()
    
    if success:
        print("\n🎯 Dashboard is ready to use!")
        print("Run: streamlit run dashboard.py")
    else:
        print("\n❌ Dashboard has issues. Check dependencies and imports.")
        sys.exit(1)