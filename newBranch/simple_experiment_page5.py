#!/usr/bin/env python3
"""
Simple Experiment Runner: Test Model+Config Combinations on Tintin Page 5
=========================================================================

This script runs detection on Tintin page 5 with different model and config combinations,
saving results for analysis. Uses a standalone approach without GUI dependencies.
"""

import argparse
import json
import sys
import os
from pathlib import Path
import yaml

def load_config(path: str) -> dict:
    """Charge la configuration depuis un fichier YAML"""
    config = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"⚠️  Erreur lors du chargement de la config {path}: {e}")
    return config

def run_standalone_detection(pdf_path: str, page_num: int, model_path: str, config_path: str, output_dir: str, test_id: str):
    """
    Run detection test using ultralytics directly
    """
    try:
        import fitz  # PyMuPDF
        from ultralytics import YOLO
        import cv2
        import numpy as np
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        return None

    print(f"🧪 Running test {test_id}")
    print(f"   Model: {os.path.basename(model_path)}")
    print(f"   Config: {os.path.basename(config_path)}")
    print(f"   Output: {output_dir}")

    # Load config
    config = load_config(config_path)
    print(f"   ✅ Config loaded: {len(config)} parameters")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load PDF and extract page
    try:
        pdf = fitz.open(pdf_path)
        page = pdf[page_num]
        dpi = 300
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        print(f"   ✅ PDF loaded: {pix.width}x{pix.height}")
    except Exception as e:
        print(f"   ❌ PDF processing failed: {e}")
        return None

    # Load model
    try:
        model = YOLO(model_path)
        print(f"   ✅ Model loaded: {os.path.basename(model_path)}")
    except Exception as e:
        print(f"   ❌ Model load failed: {e}")
        return None

    # Run detection
    try:
        # Use config parameters for detection
        conf = config.get('panel_conf', 0.18)
        iou = 0.6
        max_det = config.get('max_det', 400)

        results = model.predict(
            source=img,
            conf=conf,
            iou=iou,
            max_det=max_det,
            verbose=False
        )[0]

        # Extract detections
        panels = []
        balloons = []

        if results and hasattr(results, 'boxes') and results.boxes is not None:
            boxes = results.boxes
            for i in range(len(boxes)):
                cls = int(boxes.cls[i])
                conf_val = float(boxes.conf[i])
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()

                if cls == 0:  # panel
                    panels.append([x1, y1, x2, y2, conf_val])
                elif cls == 1:  # balloon
                    balloons.append([x1, y1, x2, y2, conf_val])

        print(f"   📊 Results: {len(panels)} panels, {len(balloons)} balloons")

        # Save debug overlay
        if len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img.copy()

        # Draw panels in green
        for bbox in panels:
            x1, y1, x2, y2, conf = bbox
            cv2.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Draw balloons in blue
        for bbox in balloons:
            x1, y1, x2, y2, conf = bbox
            cv2.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        overlay_path = os.path.join(output_dir, 'detection_overlay.png')
        cv2.imwrite(overlay_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        print(f"   💾 Overlay saved: {overlay_path}")

    except Exception as e:
        print(f"   ❌ Detection failed: {e}")
        return None

    # Prepare results
    results = {
        'test_id': test_id,
        'model': os.path.basename(model_path),
        'config': os.path.basename(config_path),
        'page': page_num,
        'panels_raw': len(panels),
        'panels_merged': len(panels),  # No merging in this simple version
        'balloons_raw': len(balloons),
        'balloons_merged': len(balloons),
        'output_dir': output_dir
    }

    # Save results
    results_file = os.path.join(output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   💾 Results saved: {results_file}")

    return results

def main():
    """Run the experiment"""
    print("🚀 Simple Experiment Runner: Tintin Page 5")
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
        result = run_standalone_detection(
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
