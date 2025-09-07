#!/usr/bin/env python3
"""
Optimized YOLO Training Script for AnComicsViewer
Based on manual annotation review and dataset improvements
"""

import os
import yaml
import subprocess
from pathlib import Path

def create_optimized_training_config():
    """Create optimized training configuration based on dataset analysis"""

    # Dataset analysis
    dataset_path = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/dataset"

    # Count images and labels
    train_images = len([f for f in os.listdir(f"{dataset_path}/images/train") if f.endswith(('.jpg', '.png'))])
    val_images = len([f for f in os.listdir(f"{dataset_path}/images/val") if f.endswith(('.jpg', '.png'))])
    train_labels = len([f for f in os.listdir(f"{dataset_path}/labels/train") if f.endswith('.txt')])
    val_labels = len([f for f in os.listdir(f"{dataset_path}/labels/val") if f.endswith('.txt')])

    print("📊 Dataset Analysis:")
    print(f"   Train: {train_images} images, {train_labels} labels")
    print(f"   Val: {val_images} images, {val_labels} labels")
    print(f"   Total: {train_images + val_images} images, {train_labels + val_labels} labels")

    # Optimized training configuration
    config = {
        'model': 'yolov8m.pt',  # Medium model for better capacity
        'data': f'{dataset_path}/multibd_enhanced.yaml',
        'epochs': 150,  # Increased for better convergence
        'imgsz': 1280,  # Keep high resolution for detail
        'batch': 8,  # Smaller batch for stability
        'name': 'ancomics_optimized_post_review',
        'save': True,
        'save_period': 25,  # Save checkpoints every 25 epochs
        'cache': True,
        'workers': 4,
        'device': 'mps',  # Apple Silicon GPU
        'optimizer': 'AdamW',  # Better optimizer
        'lr0': 0.001,  # Learning rate
        'lrf': 0.01,  # Final learning rate factor
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,  # Box loss gain
        'cls': 0.5,  # Classification loss gain
        'dfl': 1.5,  # Distribution focal loss gain
        'patience': 30,  # Early stopping patience
        'seed': 42,
        'deterministic': True,
        'close_mosaic': 15,  # Close mosaic augmentation later
        'mixup': 0.1,  # Light mixup
        'copy_paste': 0.1,  # Light copy-paste
        'amp': True,  # Automatic mixed precision
        'plots': True,
        'verbose': True
    }

    return config

def create_training_script(config):
    """Create the training command"""

    cmd_parts = ['yolo', 'train']
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd_parts.append(f'{key}={value}')
        else:
            cmd_parts.append(f'{key}={value}')

    return ' '.join(cmd_parts)

def main():
    print("🚀 AnComicsViewer - Optimized Training Setup")
    print("=" * 50)

    # Create optimized configuration
    config = create_optimized_training_config()

    # Display configuration
    print("\n🔧 Optimized Training Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")

    # Create training command
    train_cmd = create_training_script(config)

    print("\n📝 Training Command:")
    print(f"   {train_cmd}")

    # Save configuration to file
    config_file = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/train_optimized_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\n💾 Configuration saved to: {config_file}")

    # Create shell script
    script_content = f'''#!/bin/bash
# Optimized training script for AnComicsViewer
# Generated after manual annotation review

echo "🚀 Starting Optimized YOLO Training"
echo "==================================="

# Check YOLO installation
if ! command -v yolo &> /dev/null; then
    echo "❌ YOLO CLI not found. Install with: pip install ultralytics"
    exit 1
fi

# Run training
{train_cmd}

echo ""
echo "✅ Training completed!"
echo "📁 Results: runs/train/ancomics_optimized_post_review/"
'''

    script_path = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/train_optimized.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)

    os.chmod(script_path, 0o755)
    print(f"📜 Training script created: {script_path}")

    print("\n🎯 Key Optimizations:")
    print("   • YOLOv8m (medium) for better capacity than v8s")
    print("   • AdamW optimizer for stable convergence")
    print("   • Smaller batch size (8) for stability")
    print("   • Balanced loss weights for panels/balloons")
    print("   • Light data augmentation to prevent overfitting")
    print("   • Early stopping with patience=30")
    print("   • Automatic mixed precision for speed")

    print("\n▶️  To start training, run:")
    print(f"   ./{os.path.basename(script_path)}")

if __name__ == "__main__":
    main()
