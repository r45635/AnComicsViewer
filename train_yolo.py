#!/usr/bin/env python3
"""
YOLO Training Script with PyTorch 2.6 Compatibility Fix
Handles the weights_only loading issue with ultralytics models.
"""

import os
import sys
import torch

# Set environment variables for MPS
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Patch torch.load to use weights_only=False for ultralytics compatibility
original_torch_load = torch.load

def patched_torch_load(f, map_location=None, pickle_module=None, weights_only=None, **kwargs):
    """Patched torch.load that sets weights_only=False for ultralytics compatibility."""
    if weights_only is None:
        weights_only = False
    return original_torch_load(f, map_location=map_location, pickle_module=pickle_module, 
                              weights_only=weights_only, **kwargs)

# Apply the patch
torch.load = patched_torch_load

# Import YOLO after setting up compatibility
from ultralytics import YOLO

def train_yolo():
    """Train YOLO model with the specified configuration."""
    
    print("🚀 Starting YOLO training with PyTorch 2.6 compatibility...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print()
    
    # Training configuration
    config = {
        'data': 'dataset/yolo/data.yaml',
        'model': 'yolov8n.pt',
        'imgsz': 1024,
        'epochs': 80,
        'batch': 4,
        'workers': 0,
        'device': 'mps',
        'name': 'overfit_small',
        'cache': False,
        'mosaic': 0.0,
        'mixup': 0.0,
        'hsv_h': 0.0,
        'hsv_s': 0.0,
        'hsv_v': 0.0,
        'close_mosaic': 0
    }
    
    print("📋 Training Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()
    
    try:
        # Load model
        print("📦 Loading YOLOv8n model...")
        model = YOLO(config['model'])
        
        # Start training
        print("🎯 Starting training...")
        results = model.train(**{k: v for k, v in config.items() if k != 'model'})
        
        print("✅ Training completed successfully!")
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

if __name__ == "__main__":
    results = train_yolo()
    if results:
        print("🎉 Training finished!")
    else:
        sys.exit(1)
