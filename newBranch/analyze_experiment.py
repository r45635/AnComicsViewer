#!/usr/bin/env python3
"""
Analysis Script: Evaluate Experiment Results vs Ground Truth
===========================================================

This script analyzes the results from the Tintin page 5 experiment,
comparing each model+config combination to the ground truth annotations.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

RESULTS_DIR = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/experiment_results_page5"
GROUND_TRUTH_PATH = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/backup_annotations_20250822_182146/tintin_p0005.json"

def load_ground_truth() -> Dict[str, Any]:
    """Load ground truth annotations"""
    with open(GROUND_TRUTH_PATH, 'r') as f:
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

def analyze_results():
    """Analyze all experiment results"""
    print("🔍 Analyzing Experiment Results")
    print("=" * 50)

    # Load ground truth
    gt = load_ground_truth()
    print(f"📋 Ground Truth: {gt['total_panels']} panels, {gt['total_balloons']} balloons")

    # Load experiment results
    results_file = f"{RESULTS_DIR}/experiment_summary.json"
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        return

    with open(results_file, 'r') as f:
        data = json.load(f)

    results = data.get('results', [])
    print(f"📊 Total tests: {len(results)}")

    # Analyze each result
    analysis_results = []

    for result in results:
        model = result['model']
        config = result['config']

        # For our format, all results are successful
        detected_panels = result['panels_merged']
        detected_balloons = result['balloons_merged']

        # Calculate count accuracy
        panel_count_diff = abs(detected_panels - gt['total_panels'])
        balloon_count_diff = abs(detected_balloons - gt['total_balloons'])

        # Simple scoring (lower is better)
        panel_score = 1.0 / (1.0 + panel_count_diff)  # 1.0 for exact match, decreases with difference
        balloon_score = 1.0 / (1.0 + balloon_count_diff)
        total_score = (panel_score + balloon_score) / 2.0

        analysis = {
            'model': model,
            'config': config,
            'detected_panels': detected_panels,
            'detected_balloons': detected_balloons,
            'panel_count_diff': panel_count_diff,
            'balloon_count_diff': balloon_count_diff,
            'panel_score': panel_score,
            'balloon_score': balloon_score,
            'total_score': total_score,
            'success': True
        }

        analysis_results.append(analysis)

        print(f"✅ {model} + {config}:")
        print(f"   📦 Panels: {detected_panels}/{gt['total_panels']} (diff: {panel_count_diff})")
        print(f"   💬 Balloons: {detected_balloons}/{gt['total_balloons']} (diff: {balloon_count_diff})")
        print(".3f")
        print()

    # Sort by total score (best first)
    analysis_results.sort(key=lambda x: x['total_score'], reverse=True)

    print("🏆 RANKING (Best to Worst):")
    print("=" * 50)

    for i, result in enumerate(analysis_results[:10]):  # Top 10
        print("2d")
        print(f"   📦 Panels: {result['detected_panels']}/{gt['total_panels']}")
        print(f"   💬 Balloons: {result['detected_balloons']}/{gt['total_balloons']}")
        print(".3f")
        print()

    # Save detailed analysis
    analysis_file = f"{RESULTS_DIR}/analysis_results.json"
    with open(analysis_file, 'w') as f:
        json.dump({
            'ground_truth': gt,
            'results': analysis_results
        }, f, indent=2)

    print(f"💾 Detailed analysis saved to: {analysis_file}")

    # Best configurations
    print("\n🎯 BEST CONFIGURATIONS:")
    print("-" * 30)

    # Best for panels
    panel_best = max(analysis_results, key=lambda x: x['panel_score'])
    print(f"Best for Panels: {panel_best['model']} + {panel_best['config']}")
    print(f"   Score: {panel_best['panel_score']:.3f} ({panel_best['detected_panels']}/{gt['total_panels']})")

    # Best for balloons
    balloon_best = max(analysis_results, key=lambda x: x['balloon_score'])
    print(f"Best for Balloons: {balloon_best['model']} + {balloon_best['config']}")
    print(f"   Score: {balloon_best['balloon_score']:.3f} ({balloon_best['detected_balloons']}/{gt['total_balloons']})")

    # Best overall
    overall_best = max(analysis_results, key=lambda x: x['total_score'])
    print(f"Best Overall: {overall_best['model']} + {overall_best['config']}")
    print(f"   Score: {overall_best['total_score']:.3f}")

if __name__ == "__main__":
    analyze_results()
