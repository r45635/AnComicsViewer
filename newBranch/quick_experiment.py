#!/usr/bin/env python3
"""
Quick Experiment: Test Key Model+Config Combinations on Tintin Page 5
=====================================================================

This script runs a focused experiment testing the most promising
model and configuration combinations on Tintin page 5.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

# Configuration
PDF_PATH = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/Tintin - 161 - Le Lotus Bleu - .pdf"
MAIN_SCRIPT = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/newBranch/main.py"
RESULTS_DIR = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/quick_experiment"

# Key combinations to test (model_path, config_path, description)
TEST_COMBINATIONS = [
    # Best models with best configs
    ("/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/models/multibd_enhanced_v2.pt",
     "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/newBranch/config/detect.yaml",
     "Enhanced Model + Balanced Config"),

    ("/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/models/multibd_enhanced_v2.pt",
     "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/newBranch/config/detect_no_merge.yaml",
     "Enhanced Model + No Merge Config"),

    ("/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/anComicsViewer_v01.pt",
     "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/newBranch/config/detect.yaml",
     "Original Model + Balanced Config"),

    ("/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/anComicsViewer_v01.pt",
     "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/newBranch/config/detect_grid_gutters.yaml",
     "Original Model + Gutter Config"),

    ("/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/ancomics_improved.pt",
     "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/newBranch/config/detect.yaml",
     "Improved Model + Balanced Config"),

    # YOLO models for comparison
    ("/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/models/yolov8s.pt",
     "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/newBranch/config/detect.yaml",
     "YOLOv8 Small + Balanced Config"),
]

def run_test(model_path, config_path, description, test_id):
    """Run a single test"""
    print(f"\n🧪 Test {test_id}: {description}")
    print("-" * 50)

    output_dir = f"{RESULTS_DIR}/test_{test_id}"

    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None

    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        return None

    cmd = [
        sys.executable, MAIN_SCRIPT,
        "--config", config_path,
        "--debug-detect",
        "--save-debug-overlays", output_dir,
        PDF_PATH
    ]

    try:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        # Parse results from stdout
        stdout = result.stdout
        panels_raw = panels_merged = balloons_raw = balloons_merged = 0

        for line in stdout.split('\n'):
            if 'panels_raw=' in line:
                try:
                    panels_raw = int(line.split('panels_raw=')[1].split()[0])
                except:
                    pass
            if 'panels_merged=' in line:
                try:
                    panels_merged = int(line.split('panels_merged=')[1].split()[0])
                except:
                    pass
            if 'balloons_raw=' in line:
                try:
                    balloons_raw = int(line.split('balloons_raw=')[1].split()[0])
                except:
                    pass
            if 'balloons_merged=' in line:
                try:
                    balloons_merged = int(line.split('balloons_merged=')[1].split()[0])
                except:
                    pass

        success = result.returncode == 0

        if success:
            print("✅ SUCCESS")
            print(f"   📦 Panels: raw={panels_raw}, merged={panels_merged}")
            print(f"   💬 Balloons: raw={balloons_raw}, merged={balloons_merged}")
            print(f"   📁 Output: {output_dir}")
        else:
            print("❌ FAILED")
            print(f"   Error: {result.stderr[-500:]}")  # Last 500 chars of error

        return {
            'description': description,
            'model': os.path.basename(model_path),
            'config': os.path.basename(config_path),
            'success': success,
            'panels_raw': panels_raw,
            'panels_merged': panels_merged,
            'balloons_raw': balloons_raw,
            'balloons_merged': balloons_merged,
            'output_dir': output_dir,
            'stdout': stdout,
            'stderr': result.stderr
        }

    except subprocess.TimeoutExpired:
        print("❌ TIMEOUT (120s)")
        return {
            'description': description,
            'success': False,
            'error': 'Timeout'
        }
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return {
            'description': description,
            'success': False,
            'error': str(e)
        }

def main():
    """Run the quick experiment"""
    print("🚀 Quick Experiment: Tintin Page 5")
    print("=" * 50)
    print(f"PDF: {PDF_PATH}")
    print(f"Results: {RESULTS_DIR}")

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []

    for i, (model_path, config_path, description) in enumerate(TEST_COMBINATIONS, 1):
        result = run_test(model_path, config_path, description, i)
        if result:
            results.append(result)

    # Save results
    results_file = f"{RESULTS_DIR}/quick_experiment_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")

    # Summary
    successful = [r for r in results if r.get('success', False)]
    print("\n📊 SUMMARY:")
    print(f"   Total tests: {len(results)}")
    print(f"   Successful: {len(successful)}")
    print(f"   Failed: {len(results) - len(successful)}")

    if successful:
        print("\n✅ Successful tests:")
        for r in successful:
            print(f"   • {r['description']}: {r['panels_merged']} panels, {r['balloons_merged']} balloons")

    print("\n🎯 Check the debug overlay images in each test directory for visual analysis!")

if __name__ == "__main__":
    main()
