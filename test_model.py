#!/usr/bin/env python3
"""
Test Trained YOLO Model
Tests the trained panel detection model on validation images.
"""

import os
import torch
from pathlib import Path

# Apply PyTorch compatibility fix
original_torch_load = torch.load
def patched_torch_load(f, map_location=None, pickle_module=None, weights_only=None, **kwargs):
    if weights_only is None:
        weights_only = False
    return original_torch_load(f, map_location=map_location, pickle_module=pickle_module, 
                              weights_only=weights_only, **kwargs)
torch.load = patched_torch_load

from ultralytics import YOLO


def test_trained_model():
    """Test the trained model on validation images."""
    
    # Model paths
    best_model = "runs/detect/overfit_small/weights/best.pt"
    val_images = "dataset/yolo/images/val"
    
    print("🧪 Testing Trained YOLO Model")
    print("=" * 40)
    print(f"Model: {best_model}")
    print(f"Test images: {val_images}")
    print()
    
    if not Path(best_model).exists():
        print(f"❌ Model not found: {best_model}")
        return False
    
    if not Path(val_images).exists():
        print(f"❌ Validation images not found: {val_images}")
        return False
    
    try:
        # Load trained model
        print("📦 Loading trained model...")
        model = YOLO(best_model)
        
        # Run inference on validation set
        print("🔍 Running inference on validation images...")
        results = model.predict(
            source=val_images,
            save=True,
            save_txt=True,
            conf=0.25,
            iou=0.7,
            device='mps',
            project='runs/test',
            name='validation_test'
        )
        
        print(f"✅ Inference completed!")
        print(f"📁 Results saved to: runs/test/validation_test/")
        print(f"🖼️  Processed {len(results)} images")
        
        # Show detection summary
        total_detections = 0
        for i, result in enumerate(results):
            detections = len(result.boxes) if result.boxes is not None else 0
            total_detections += detections
            img_name = Path(result.path).name
            print(f"   {img_name}: {detections} panels detected")
        
        print(f"\n📊 Total detections: {total_detections}")
        print(f"📈 Average per image: {total_detections/len(results):.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return False


if __name__ == "__main__":
    success = test_trained_model()
    if success:
        print("\n🎉 Model testing completed successfully!")
        print("📋 Next steps:")
        print("   1. Check results in runs/test/validation_test/")
        print("   2. Integrate model into AnComicsViewer")
        print("   3. Update ML detector to use trained weights")
    else:
        print("❌ Model testing failed!")
