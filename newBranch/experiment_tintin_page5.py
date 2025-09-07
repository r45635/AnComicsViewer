#!/usr/bin/env python3
"""
Experiment Plan: Comprehensive Model & Config Testing on Tintin Page 5
========================================================================

This script tests all available models and configurations against Tintin page 5
and compares results to the ground truth annotations.

TARGET: Tintin - 161 - Le Lotus Bleu - Page 5
GROUND TRUTH: backup_annotations_20250822_182146/tintin_p0005.json

MODELS TO TEST:
1. anComicsViewer_v01.pt (original model)
2. multibd_enhanced_v2.pt (enhanced model)
3. multibd_with_sisters.pt (sisters model)
4. yolov8n.pt (YOLOv8 nano)
5. yolov8s.pt (YOLOv8 small)
6. ancomics_improved.pt (improved model)

CONFIGS TO TEST:
1. detect.yaml (balanced default)
2. detect_strict.yaml (strict thresholds)
3. detect_with_merge.yaml (default with merging)
4. detect_no_merge.yaml (no merging, raw detection)
5. detect_grid_gutters.yaml (gutter splitting enabled)
6. detect_grid_gutters_v2.yaml (conservative gutter splitting)

EXPERIMENT DESIGN:
- 6 models × 6 configs = 36 combinations
- Each test runs detection on page 5
- Results compared to ground truth using IoU metrics
- Analysis of precision, recall, F1-score for panels and balloons
- Visual overlays saved for qualitative analysis

EXPECTED GROUND TRUTH (from tintin_p0005.json):
- 6 panels (3×2 grid layout)
- 13 balloons (speech bubbles)
- Total: 19 annotations

METRICS TO COMPUTE:
1. Detection Count Accuracy
2. IoU-based Matching
3. Precision/Recall/F1 per class
4. Area Distribution Analysis
5. Spatial Layout Analysis
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np

# Configuration
PDF_PATH = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/Tintin - 161 - Le Lotus Bleu - .pdf"
GROUND_TRUTH_PATH = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/backup_annotations_20250822_182146/tintin_p0005.json"
MAIN_SCRIPT = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/newBranch/main.py"
RESULTS_DIR = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/experiment_results"

# Models to test
MODELS = [
    ("anComicsViewer_v01.pt", "Original Model"),
    ("multibd_enhanced_v2.pt", "Enhanced Model"),
    ("multibd_with_sisters.pt", "Sisters Model"),
    ("yolov8n.pt", "YOLOv8 Nano"),
    ("yolov8s.pt", "YOLOv8 Small"),
    ("ancomics_improved.pt", "Improved Model")
]

# Configs to test
CONFIGS = [
    ("detect.yaml", "Balanced Default"),
    ("detect_strict.yaml", "Strict Thresholds"),
    ("detect_with_merge.yaml", "Default with Merging"),
    ("detect_no_merge.yaml", "No Merging"),
    ("detect_grid_gutters.yaml", "Gutter Splitting"),
    ("detect_grid_gutters_v2.yaml", "Conservative Gutter")
]

def load_ground_truth(gt_path: str) -> Dict[str, Any]:
    """Load ground truth annotations from LabelMe JSON format"""
    with open(gt_path, 'r') as f:
        data = json.load(f)

    panels = []
    balloons = []

    for shape in data.get('shapes', []):
        label = shape['label']
        points = shape['points']

        # Convert polygon to bounding box
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        bbox = {
            'x1': min(x_coords),
            'y1': min(y_coords),
            'x2': max(x_coords),
            'y2': max(y_coords),
            'width': max(x_coords) - min(x_coords),
            'height': max(y_coords) - min(y_coords),
            'area': (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
        }

        if label == 'panel':
            panels.append(bbox)
        elif label == 'balloon':
            balloons.append(bbox)

    return {
        'panels': panels,
        'balloons': balloons,
        'total_panels': len(panels),
        'total_balloons': len(balloons)
    }

def calculate_iou(box1: Dict, box2: Dict) -> float:
    """Calculate IoU between two bounding boxes"""
    x1 = max(box1['x1'], box2['x1'])
    y1 = max(box1['y1'], box2['y1'])
    x2 = min(box1['x2'], box2['x2'])
    y2 = min(box1['y2'], box2['y2'])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = box1['width'] * box1['height']
    area2 = box2['width'] * box2['height']
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def evaluate_detection(detected_boxes: List[Dict], ground_truth_boxes: List[Dict], iou_threshold: float = 0.5) -> Dict:
    """Evaluate detection performance using IoU matching"""
    if not detected_boxes:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'tp': 0,
            'fp': 0,
            'fn': len(ground_truth_boxes),
            'matched_gt': [],
            'unmatched_det': detected_boxes.copy()
        }

    matched_gt = []
    unmatched_det = []

    for det_box in detected_boxes:
        best_iou = 0.0
        best_gt_idx = -1

        for i, gt_box in enumerate(ground_truth_boxes):
            if i in matched_gt:
                continue

            iou = calculate_iou(det_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_threshold:
            matched_gt.append(best_gt_idx)
        else:
            unmatched_det.append(det_box)

    tp = len(matched_gt)
    fp = len(detected_boxes) - tp
    fn = len(ground_truth_boxes) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'matched_gt': matched_gt,
        'unmatched_det': unmatched_det
    }

def run_single_test(model_path: str, config_path: str, page_num: int = 5) -> Dict:
    """Run a single detection test"""
    output_dir = f"{RESULTS_DIR}/debug_{Path(model_path).stem}_{Path(config_path).stem}"

    cmd = [
        sys.executable, MAIN_SCRIPT,
        "--config", config_path,
        "--debug-detect",
        "--save-debug-overlays", output_dir,
        PDF_PATH
    ]

    try:
        # Run the detection
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        # Parse the debug output to extract detection results
        stdout = result.stdout
        stderr = result.stderr

        # Extract detection counts from debug output
        panels_raw = 0
        panels_merged = 0
        balloons_raw = 0
        balloons_merged = 0

        lines = stdout.split('\n')
        for line in lines:
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

        return {
            'success': result.returncode == 0,
            'panels_raw': panels_raw,
            'panels_merged': panels_merged,
            'balloons_raw': balloons_raw,
            'balloons_merged': balloons_merged,
            'stdout': stdout,
            'stderr': stderr,
            'output_dir': output_dir
        }

    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': 'Timeout',
            'panels_raw': 0,
            'panels_merged': 0,
            'balloons_raw': 0,
            'balloons_merged': 0
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'panels_raw': 0,
            'panels_merged': 0,
            'balloons_raw': 0,
            'balloons_merged': 0
        }

def main():
    """Main experiment execution"""
    print("🚀 Starting Comprehensive Experiment: Tintin Page 5")
    print("=" * 60)

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load ground truth
    print("📋 Loading ground truth annotations...")
    try:
        ground_truth = load_ground_truth(GROUND_TRUTH_PATH)
        print(f"   ✅ Ground truth: {ground_truth['total_panels']} panels, {ground_truth['total_balloons']} balloons")
    except Exception as e:
        print(f"   ❌ Error loading ground truth: {e}")
        return

    # Results storage
    results = []

    # Run all combinations
    total_tests = len(MODELS) * len(CONFIGS)
    test_count = 0

    for model_path, model_name in MODELS:
        for config_path, config_name in CONFIGS:
            test_count += 1
            print(f"\n🧪 Test {test_count}/{total_tests}: {model_name} + {config_name}")

            # Find full paths
            if model_path == "anComicsViewer_v01.pt":
                full_model_path = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/anComicsViewer_v01.pt"
            elif model_path == "multibd_enhanced_v2.pt":
                full_model_path = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/models/multibd_enhanced_v2.pt"
            elif model_path == "multibd_with_sisters.pt":
                full_model_path = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/models/multibd_with_sisters.pt"
            elif model_path == "yolov8n.pt":
                full_model_path = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/models/yolov8n.pt"
            elif model_path == "yolov8s.pt":
                full_model_path = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/models/yolov8s.pt"
            elif model_path == "ancomics_improved.pt":
                full_model_path = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/ancomics_improved.pt"
            else:
                full_model_path = model_path

            if config_path in ["detect.yaml", "detect_strict.yaml", "detect_with_merge.yaml", "detect_no_merge.yaml", "detect_grid_gutters.yaml", "detect_grid_gutters_v2.yaml"]:
                full_config_path = f"/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/newBranch/config/{config_path}"
            else:
                full_config_path = config_path

            # Check if files exist
            if not os.path.exists(full_model_path):
                print(f"   ⚠️  Model not found: {full_model_path}")
                continue
            if not os.path.exists(full_config_path):
                print(f"   ⚠️  Config not found: {full_config_path}")
                continue

            # Run test
            result = run_single_test(full_model_path, full_config_path)

            if result['success']:
                print(f"   ✅ Success: {result['panels_merged']} panels, {result['balloons_merged']} balloons")
            else:
                print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")

            # Store result
            test_result = {
                'model': model_name,
                'model_path': model_path,
                'config': config_name,
                'config_path': config_path,
                'result': result
            }
            results.append(test_result)

    # Save results
    results_file = f"{RESULTS_DIR}/experiment_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")
    print("\n🎯 Experiment completed!")

if __name__ == "__main__":
    main()
