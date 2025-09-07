#!/usr/bin/env python3
"""
Quick validation script to test model performance on the reviewed dataset
"""

import os
import subprocess
import yaml

def validate_current_model():
    """Validate the current best model"""

    print("🔍 Validating Current Model Performance")
    print("=" * 40)

    # Check available models
    models_dir = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/models"
    available_models = [f for f in os.listdir(models_dir) if f.endswith('.pt')]

    print("Available models:")
    for i, model in enumerate(available_models, 1):
        print(f"   {i}. {model}")

    # Use the most recent enhanced model
    best_model = os.path.join(models_dir, "multibd_enhanced_v2.pt")

    if not os.path.exists(best_model):
        print(f"❌ Model not found: {best_model}")
        return

    print(f"\n🎯 Testing model: {os.path.basename(best_model)}")

    # Validation command
    val_cmd = [
        "yolo", "val",
        f"model={best_model}",
        "data=/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/dataset/multibd_enhanced.yaml",
        "imgsz=1280",
        "batch=8",
        "device=mps",
        "save_json=True",
        "plots=True",
        "verbose=True"
    ]

    print("Running validation...")
    try:
        result = subprocess.run(val_cmd, capture_output=True, text=True, cwd="/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer")
        print("Validation Results:")
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    except Exception as e:
        print(f"❌ Validation failed: {e}")

def analyze_annotation_quality():
    """Analyze the quality of annotations after manual review"""

    print("\n📊 Analyzing Annotation Quality")
    print("=" * 35)

    import json

    # Check a few manually reviewed files
    reviewed_files = [
        "sisters_p012.json",
        "sisters_p018.json",
        "sisters_p025.json",
        "sisters_p055.json",
        "sisters_p065.json"
    ]

    labels_dir = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/dataset/labels/train"

    total_panels = 0
    total_balloons = 0
    files_analyzed = 0

    for filename in reviewed_files:
        json_path = os.path.join(labels_dir, filename)
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                shapes = data.get('shapes', [])
                panels = [s for s in shapes if s.get('label') == 'panel']
                balloons = [s for s in shapes if s.get('label') == 'balloon']

                total_panels += len(panels)
                total_balloons += len(balloons)
                files_analyzed += 1

                print(f"   {filename}: {len(panels)} panels, {len(balloons)} balloons")

            except Exception as e:
                print(f"   ❌ Error reading {filename}: {e}")

    if files_analyzed > 0:
        avg_panels = total_panels / files_analyzed
        avg_balloons = total_balloons / files_analyzed

        print("\n📈 Averages:")
        print(f"   Average panels per page: {avg_panels:.1f}")
        print(f"   Average balloons per page: {avg_balloons:.1f}")
def main():
    validate_current_model()
    analyze_annotation_quality()

    print("\n🎯 Recommendations for Optimized Training:")
    print("   1. ✅ Dataset quality improved through manual review")
    print("   2. 🔄 Use YOLOv8m for better capacity")
    print("   3. 📈 Increase epochs to 150 for better convergence")
    print("   4. 🎛️  Use AdamW optimizer for stability")
    print("   5. 🔧 Fine-tune loss weights for panels/balloons")
    print("   6. 🚀 Enable AMP for faster training")
    print("   7. 💾 Regular checkpoints every 25 epochs")

if __name__ == "__main__":
    main()
