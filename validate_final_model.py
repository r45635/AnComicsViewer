#!/usr/bin/env python3
"""
Post-training validation and analysis script
"""

import os
import subprocess
import yaml
import json
from pathlib import Path

def validate_trained_model(model_path, data_yaml):
    """Validate the trained model"""

    print("🔍 Validating Trained Model")
    print("=" * 30)

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False

    print(f"🎯 Testing model: {os.path.basename(model_path)}")

    # Validation command
    val_cmd = [
        "yolo", "val",
        f"model={model_path}",
        f"data={data_yaml}",
        "imgsz=1280",
        "batch=4",  # Smaller batch for validation
        "device=mps",
        "save_json=True",
        "plots=True",
        "verbose=True",
        "conf=0.15",  # Lower confidence for evaluation
        "iou=0.5"     # Standard IoU threshold
    ]

    print("Running validation...")
    try:
        result = subprocess.run(val_cmd, capture_output=True, text=True, cwd="/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer")
        print("Validation Results:")
        print(result.stdout)

        if result.stderr:
            print("Errors/Warnings:")
            print(result.stderr)

        # Check for results file
        results_dir = os.path.dirname(model_path)
        results_file = os.path.join(results_dir, "results.csv")

        if os.path.exists(results_file):
            print(f"\n📊 Detailed results saved to: {results_file}")

            # Read final results
            with open(results_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    last_line = lines[-1].strip()
                    print(f"📈 Final metrics: {last_line}")

        return True

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def analyze_training_results(run_dir):
    """Analyze the training results"""

    print("\n📊 Analyzing Training Results")
    print("=" * 32)

    if not os.path.exists(run_dir):
        print(f"❌ Training directory not found: {run_dir}")
        return

    # Check for results.csv
    results_file = os.path.join(run_dir, "results.csv")
    if os.path.exists(results_file):
        print("✅ Training results found")

        # Read training metrics
        with open(results_file, 'r') as f:
            lines = f.readlines()

        if len(lines) > 1:
            header = lines[0].strip()
            last_line = lines[-1].strip()

            print(f"📈 Final training metrics:")
            headers = header.split(',')
            values = last_line.split(',')

            for h, v in zip(headers, values):
                if h in ['epoch', 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']:
                    try:
                        print(f"   {h}: {float(v):.3f}")
                    except ValueError:
                        print(f"   {h}: {v}")
                else:
                    print(f"   {h}: {v}")
        print("⚠️  No training results found")

    # Check for best model
    best_model = os.path.join(run_dir, "weights", "best.pt")
    if os.path.exists(best_model):
        model_size = os.path.getsize(best_model) / (1024 * 1024)  # MB
        print(f"✅ Best model found: {model_size:.1f} MB")
    else:
        print("⚠️  Best model not found")

def main():
    print("🎯 AnComicsViewer - Post-Training Analysis")
    print("=" * 45)

    # Configuration
    run_name = "ancomics_final_optimized"
    run_dir = f"/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/runs/train/{run_name}"
    best_model = os.path.join(run_dir, "weights", "best.pt")
    data_yaml = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/dataset/multibd_enhanced.yaml"

    # Analyze training results
    analyze_training_results(run_dir)

    # Validate best model
    if os.path.exists(best_model):
        print("\n" + "="*50)
        success = validate_trained_model(best_model, data_yaml)

        if success:
            print("\n✅ Model validation completed successfully!")
            print("🎉 Ready for deployment in AnComicsViewer")
        else:
            print("\n⚠️  Model validation encountered issues")
    else:
        print("❌ Best model not found. Training may have failed.")

    print("\n📋 Next Steps:")
    print("   1. Review validation results above")
    print("   2. If satisfied, copy best.pt to data/models/")
    print("   3. Update main.py to use the new model")
    print("   4. Test with actual comic pages")

if __name__ == "__main__":
    main()
