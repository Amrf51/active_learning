#!/usr/bin/env python3
"""
Multiprocessing Diagnostic Test
Tests different multiprocessing methods with torch import
"""
import multiprocessing as mp
import sys
import os

def worker_test():
    """Simple worker to test torch import."""
    print(f"  Worker PID: {os.getpid()}")
    try:
        import torch
        print(f"  ✓ Torch imported successfully: {torch.__version__}")
        print(f"  ✓ Torch location: {torch.__file__}")
        return True
    except Exception as e:
        print(f"  ✗ Torch import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_method(method_name):
    """Test specific multiprocessing method."""
    print(f"\n{'='*70}")
    print(f"Testing: {method_name.upper()}")
    print('='*70)

    try:
        ctx = mp.get_context(method_name)
        print(f"✓ Context created: {method_name}")

        proc = ctx.Process(target=worker_test, name=f"TestWorker-{method_name}")
        proc.start()
        proc.join(timeout=30)

        if proc.exitcode == 0:
            print(f"✓ {method_name}: SUCCESS")
            return True
        elif proc.exitcode is None:
            print(f"✗ {method_name}: TIMEOUT (process hung)")
            proc.terminate()
            proc.join()
            return False
        else:
            print(f"✗ {method_name}: FAILED (exitcode: {proc.exitcode})")
            return False

    except ValueError as e:
        print(f"✗ {method_name}: NOT AVAILABLE - {e}")
        return False
    except Exception as e:
        print(f"✗ {method_name}: ERROR - {e}")
        import traceback
        traceback.print_exc()
        return False

def check_environment():
    """Check environment details."""
    print(f"{'='*70}")
    print("ENVIRONMENT INFORMATION")
    print('='*70)
    print(f"Python Version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Main Process PID: {os.getpid()}")
    print(f"Working Directory: {os.getcwd()}")

    # Check if in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print(f"Virtual Environment: {sys.prefix}")
    else:
        print("Virtual Environment: Not detected")

    # Check filesystem
    try:
        import subprocess
        result = subprocess.run(['df', '-Th', sys.prefix],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"\nFilesystem Info:")
            print(result.stdout)
    except:
        pass

    # Check torch
    try:
        import torch
        print(f"\n✓ Torch available in main process: {torch.__version__}")
        print(f"  Location: {torch.__file__}")
    except ImportError as e:
        print(f"\n✗ Torch not available in main process: {e}")

if __name__ == '__main__':
    check_environment()

    # Determine available methods
    if sys.platform == 'win32':
        methods = ['spawn']
        print("\nNote: Windows only supports 'spawn' method")
    else:
        methods = ['spawn', 'forkserver', 'fork']

    print(f"\nTesting {len(methods)} multiprocessing method(s)...")

    results = {}
    for method in methods:
        results[method] = test_method(method)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)

    for method, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        recommendation = ""

        if success:
            if method == 'spawn':
                recommendation = " (RECOMMENDED - most compatible)"
            elif method == 'fork':
                recommendation = " (WARNING: Not safe with CUDA)"
            elif method == 'forkserver':
                recommendation = " (OK, but spawn preferred)"

        print(f"  {method:12s}: {status}{recommendation}")

    # Recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print('='*70)

    if results.get('spawn', False):
        print("✓ Use 'spawn' method - it works and is CUDA-safe")
        print("  Change app.py line ~52 to: mp.get_context('spawn')")
    elif results.get('fork', False):
        print("⚠ Only 'fork' works, but it's not CUDA-safe")
        print("  This may cause GPU issues. Investigate why spawn fails.")
    else:
        print("✗ No multiprocessing method works!")
        print("  Possible issues:")
        print("  - Virtual environment on NFS with permission issues")
        print("  - Torch installation corrupted")
        print("  - File system restrictions")
        print("\nNext steps:")
        print("  1. Check file permissions on torch package")
        print("  2. Reinstall torch in local (non-NFS) location")
        print("  3. Check system logs for security restrictions")

    print()
