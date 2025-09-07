#!/usr/bin/env python3
"""
Experiment Runner: Test Model+Config Combinations on Tintin Page 5
==================================================================

This script runs detection on Tintin page 5 with different model and config combinations,
saving results for analysis.
"""

import argparse
import json
import sys
import os
from pathlib import Path
import yaml

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Import main module at the top level
import main as main_module

def load_config(path: str) -> dict:
    """Charge la configuration depuis un fichier YAML"""
    config = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"⚠️  Erreur lors du chargement de la config {path}: {e}")
    return config

def run_detection_test(pdf_path: str, page_num: int, model_path: str, config_path: str, output_dir: str, test_id: str):
    """
    Run detection test with specific model and config
    """
    # Import what we need
    import sys
    sys.path.insert(0, str(project_root))

    # Import the main module to access its functions and globals
    print(f"   Debug: main module already imported, has GLOBAL_CONFIG: {hasattr(main_module, 'GLOBAL_CONFIG')}")

    print(f"🧪 Running test {test_id}")
    print(f"   Model: {os.path.basename(model_path)}")
    print(f"   Config: {os.path.basename(config_path)}")
    print(f"   Output: {output_dir}")

    # Load config and set global variables
    if os.path.exists(config_path):
        config = load_config(config_path)
        # Set global config
        print(f"   Debug: About to access GLOBAL_CONFIG")
        global_config = getattr(main_module, 'GLOBAL_CONFIG')
        print(f"   Debug: GLOBAL_CONFIG accessed: {type(global_config)}")
        global_config.clear()
        global_config.update(config)
        print(f"   ✅ Config loaded: {len(config)} parameters")
    else:
        print(f"   ❌ Config not found: {config_path}")
        return None

    # Set debug mode by modifying the global variable directly
    setattr(main_module, 'DEBUG_DETECT', True)
    setattr(main_module, 'DEBUG_OVERLAY_DIR', output_dir)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize viewer
    viewer_class = getattr(main_module, 'PdfYoloViewer')
    viewer = viewer_class()

    # Load PDF
    try:
        import fitz
        viewer.pdf = fitz.open(pdf_path)
        viewer.page_index = page_num
        print(f"   ✅ PDF loaded: {len(viewer.pdf)} pages")
    except Exception as e:
        print(f"   ❌ PDF load failed: {e}")
        return None

    # Load model
    try:
        from ultralytics import YOLO
        viewer.model = YOLO(model_path)
        print(f"   ✅ Model loaded: {os.path.basename(model_path)}")
    except Exception as e:
        print(f"   ❌ Model load failed: {e}")
        return None

    # Load and process page
    try:
        viewer.load_page(page_num)
        print("   ✅ Page processed")
    except Exception as e:
        print(f"   ❌ Page processing failed: {e}")
        return None

    # Extract results from detections
    panels_count = 0
    balloons_count = 0
    
    if hasattr(viewer, 'dets') and viewer.dets:
        for det in viewer.dets:
            if det.cls == 0:  # panel
                panels_count += 1
            elif det.cls == 1:  # balloon
                balloons_count += 1

    results = {
        'test_id': test_id,
        'model': os.path.basename(model_path),
        'config': os.path.basename(config_path),
        'page': page_num,
        'panels_raw': panels_count,  # We'll use final count as approximation
        'panels_merged': panels_count,
        'balloons_raw': balloons_count,
        'balloons_merged': balloons_count,
        'output_dir': output_dir
    }

    print(f"   📊 Results: {results['panels_merged']} panels, {results['balloons_merged']} balloons")

    # Save results to JSON
    results_file = os.path.join(output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"   💾 Results saved to: {results_file}")

    return results

def main():
    """Run the experiment"""
    print("🚀 Experiment Runner: Tintin Page 5")
    print("=" * 50)

    # Test configurations
    test_configs = [
        {
            'model': '/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/models/multibd_enhanced_v2.pt',
            'config': '/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/newBranch/config/detect.yaml',
            'name': 'Enhanced_Balanced'
        },
        {
            'model': '/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/models/multibd_enhanced_v2.pt',
            'config': '/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/newBranch/config/detect_no_merge.yaml',
            'name': 'Enhanced_NoMerge'
        },
        {
            'model': '/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/anComicsViewer_v01.pt',
            'config': '/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/newBranch/config/detect.yaml',
            'name': 'Original_Balanced'
        },
        {
            'model': '/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/anComicsViewer_v01.pt',
            'config': '/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/newBranch/config/detect_grid_gutters.yaml',
            'name': 'Original_Gutter'
        },
        {
            'model': '/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/ancomics_improved.pt',
            'config': '/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/newBranch/config/detect.yaml',
            'name': 'Improved_Balanced'
        },
        {
            'model': '/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/models/yolov8s.pt',
            'config': '/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/newBranch/config/detect.yaml',
            'name': 'YOLOv8s_Balanced'
        }
    ]

    pdf_path = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/Tintin - 161 - Le Lotus Bleu - .pdf"
    page_num = 5  # 0-based index, so page 6 in PDF
    base_output_dir = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/experiment_results_page5"

    all_results = []

    for i, test_config in enumerate(test_configs, 1):
        test_id = f"test_{i:02d}_{test_config['name']}"
        output_dir = os.path.join(base_output_dir, test_id)

        print(f"\n{'='*50}")
        result = run_detection_test(
            pdf_path=pdf_path,
            page_num=page_num,
            model_path=test_config['model'],
            config_path=test_config['config'],
            output_dir=output_dir,
            test_id=test_id
        )

        if result:
            all_results.append(result)

    # Save summary
    summary_file = os.path.join(base_output_dir, 'experiment_summary.json')
    with open(summary_file, 'w') as f:
        json.dump({
            'experiment': 'Tintin Page 5 Model+Config Comparison',
            'pdf': pdf_path,
            'page': page_num,
            'ground_truth': {
                'panels': 6,
                'balloons': 13
            },
            'results': all_results
        }, f, indent=2)

    print(f"\n{'='*50}")
    print("🎯 EXPERIMENT COMPLETE")
    print(f"📊 Summary saved to: {summary_file}")
    print(f"📁 Results directory: {base_output_dir}")

    # Print ranking
    print("\n🏆 RANKING (by total detections):")
    sorted_results = sorted(all_results, key=lambda x: x['panels_merged'] + x['balloons_merged'], reverse=True)

    for i, result in enumerate(sorted_results, 1):
        total_dets = result['panels_merged'] + result['balloons_merged']
        print("2d")

if __name__ == "__main__":
    main()
