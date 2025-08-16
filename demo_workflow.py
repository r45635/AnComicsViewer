#!/usr/bin/env python3
"""
Complete Comic Panel Detection Demo
Demonstrates the full workflow from annotation to trained model usage.
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """Check if all components are ready."""
    print("🔍 Checking Comic Panel Detection Setup...")
    print("=" * 50)
    
    # Check dataset
    dataset_train = Path("dataset/images/train")
    dataset_labels = Path("dataset/labels/train")
    yolo_dataset = Path("dataset/yolo")
    trained_model = Path("runs/detect/overfit_small/weights/best.pt")
    
    checks = [
        ("Original dataset", dataset_train.exists() and len(list(dataset_train.glob("*.png"))) > 0),
        ("Annotations", dataset_labels.exists() and len(list(dataset_labels.glob("*.json"))) > 0),
        ("YOLO dataset", yolo_dataset.exists()),
        ("Trained model", trained_model.exists()),
        ("Model config", (yolo_dataset / "data.yaml").exists()),
    ]
    
    all_good = True
    for name, status in checks:
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {name}")
        if not status:
            all_good = False
    
    print()
    
    if all_good:
        print("🎉 All components ready!")
        return True
    else:
        print("⚠️  Some components missing. Run setup first.")
        return False

def show_workflow():
    """Show the complete workflow."""
    print("📋 Comic Panel Detection Workflow")
    print("=" * 40)
    print()
    print("1. 📊 Dataset Creation:")
    print("   • Extract PDF pages: tools/export_pdf_pages.py")
    print("   • Split train/val: 44 train + 5 val images")
    print("   • Annotate with LabelMe: python start_annotation.py")
    print()
    print("2. 🔄 Data Conversion:")
    print("   • Convert to YOLO: python tools/labelme_to_yolo.py")
    print("   • Creates: dataset/yolo/ with proper structure")
    print()
    print("3. 🏋️ Model Training:")
    print("   • Train YOLOv8: python train_yolo.py")
    print("   • 80 epochs, MPS device, optimized for small dataset")
    print("   • Results: runs/detect/overfit_small/")
    print()
    print("4. 🧪 Model Testing:")
    print("   • Test model: python test_model.py")
    print("   • Validation inference: runs/test/validation_test/")
    print()
    print("5. 🎯 Integration:")
    print("   • Auto-loaded in AnComicsViewer")
    print("   • Switch to ML detector from menu")
    print("   • Real-time panel detection on comics")
    print()

def show_usage_guide():
    """Show how to use the trained model."""
    print("🚀 Using Your Trained Model")
    print("=" * 30)
    print()
    print("🖥️  In AnComicsViewer:")
    print("   1. Run: python AnComicsViewer.py")
    print("   2. Open a comic PDF")
    print("   3. Panels > YOLOv8 Seg (ML)")
    print("   4. Toggle panels: Space or P key")
    print("   5. Navigate panels: Left/Right arrows")
    print()
    print("🔧 Command Line Testing:")
    print("   • Test on single image:")
    print("     yolo predict model=runs/detect/overfit_small/weights/best.pt source=image.png")
    print()
    print("   • Batch processing:")
    print("     python test_model.py")
    print()
    print("📈 Model Performance:")
    print("   • mAP@0.5: 49.5% (good for small dataset)")
    print("   • Precision: 97.3% (excellent)")
    print("   • Recall: 42.4% (can improve with more data)")
    print("   • Average: 13.0 panels per page")
    print()

def main():
    """Main demo function."""
    print("🎨 Comic Panel Detection - Complete Workflow Demo")
    print("=" * 60)
    print()
    
    # Check if we're in the right directory
    if not Path("AnComicsViewer.py").exists():
        print("❌ Please run this script from the AnComicsViewer directory")
        return
    
    # Check setup
    if not check_requirements():
        return
    
    print()
    show_workflow()
    print()
    show_usage_guide()
    
    print("🎯 Next Steps:")
    print("   • Annotate more images to improve accuracy")
    print("   • Try different YOLO models (yolov8s, yolov8m)")
    print("   • Experiment with training parameters")
    print("   • Test on different comic styles")
    print()
    print("📚 Documentation:")
    print("   • Setup: README_SETUP.md")
    print("   • Quick reference: QUICK_REFERENCE.md")
    print("   • Automation: setup.sh")
    print()

if __name__ == "__main__":
    main()
