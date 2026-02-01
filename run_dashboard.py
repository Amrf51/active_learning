#!/usr/bin/env python3
"""
Dashboard launcher script.
This ensures the correct working directory and Python path setup.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Get the directory containing this script
    script_dir = Path(__file__).parent.absolute()
    
    # Change to the script directory
    os.chdir(script_dir)
    
    print(f"🚀 Starting Active Learning Dashboard...")
    print(f"📁 Working directory: {script_dir}")
    
    # Test imports first
    try:
        sys.path.insert(0, str(script_dir))
        from views.controller_factory import get_controller
        print("✅ Import test successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please check the file structure and __init__.py files")
        return 1
    
    # Run streamlit
    try:
        cmd = [sys.executable, "-m", "streamlit", "run", "dashboard.py"]
        print(f"🔧 Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start dashboard: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
        return 0

if __name__ == "__main__":
    sys.exit(main())