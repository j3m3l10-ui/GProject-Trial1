#!/usr/bin/env python3
"""
Simple harvest test with improved camera offsets
Tests: Detection → Transform → IK → Motion → Visual Servoing
"""

import sys
sys.path.insert(0, '/media/metu2metu/ADATA UFD/GProject Trial1')

import os
import time

# Suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Run with minimal output first to see what happens
if __name__ == "__main__":
    print("\n" + "="*70)
    print("STARTING HARVEST TEST (with fixed camera offsets)")
    print("="*70)
    print("\nThis test will:")
    print("  1. Scan for tomatoes")
    print("  2. Move toward first detection")
    print("  3. Use visual servoing to center it")
    print("  4. NOT cut (--no-cut)")
    print("\nWatch the arm motion relative to detected tomato position!")
    print("="*70 + "\n")
    
    # Import and run main
    try:
        from main import run_harvesting
        run_harvesting(sim_mode=False)
    except KeyboardInterrupt:
        print("\n✓ Test interrupted by user")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
