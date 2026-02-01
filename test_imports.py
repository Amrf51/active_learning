#!/usr/bin/env python3
"""
Test script to verify imports work correctly.
Run this from the al-car-classification directory.
"""

import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

print(f"Current directory: {current_dir}")
print(f"Python path: {sys.path[:3]}")

try:
    print("Testing imports...")
    
    # Test model imports
    print("1. Testing model imports...")
    from model.schemas import ExperimentPhase
    print("   ✅ model.schemas imported successfully")
    
    # Test controller imports
    print("2. Testing controller imports...")
    from controller.dispatcher import EventDispatcher
    print("   ✅ controller.dispatcher imported successfully")
    
    # Test views imports
    print("3. Testing views imports...")
    from views.controller_factory import get_controller
    print("   ✅ views.controller_factory imported successfully")
    
    print("\n🎉 All imports successful!")
    print("\nYou can now run: streamlit run dashboard.py")
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you're running from the al-car-classification directory")
    print("2. Check that all __init__.py files exist")
    print("3. Verify the file structure is correct")
    
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")