#!/usr/bin/env python3
"""
Golden City Dataset Annotation Guide
Instructions for using LabelMe to annotate comic panels and text regions.
"""

def create_annotation_instructions():
    print("🎯 GOLDEN CITY DATASET - ANNOTATION GUIDE")
    print("=" * 50)
    print()
    print("📁 Dataset Structure:")
    print("  dataset/")
    print("  ├── images/")
    print("  │   ├── train/        (44 images for training)")
    print("  │   └── val/          (5 images for validation)")
    print("  ├── labels/")
    print("  │   ├── train/        (YOLO labels will go here)")
    print("  │   └── val/          (YOLO labels will go here)")
    print("  └── predefined_classes.txt  (panel, text)")
    print()
    print("🏷️ ANNOTATION CLASSES:")
    print("  1. 'panel' - Comic panel boundaries")
    print("  2. 'text'  - Text regions (speech bubbles, titles, etc.)")
    print()
    print("🖱️ LABELME ANNOTATION STEPS:")
    print("  1. LabelMe will open with the first image")
    print("  2. Use polygon tool to draw around panels:")
    print("     - Click to create polygon points")
    print("     - Right-click to finish polygon")
    print("     - Label as 'panel' or 'text'")
    print("  3. Save (Ctrl+S) - creates .json file")
    print("  4. Next image (Ctrl+D)")
    print("  5. Repeat for all training images")
    print()
    print("💡 ANNOTATION TIPS:")
    print("  - Focus on main story panels (ignore page borders)")
    print("  - Include speech bubbles as 'text' class")
    print("  - Be precise with panel boundaries")
    print("  - Skip validation set for now (use for testing)")
    print()
    print("🔄 YOLO CONVERSION:")
    print("  After annotation, run:")
    print("  python -c \"")
    print("  import labelme")
    print("  # Convert LabelMe JSON to YOLO format")
    print("  labelme_export_yolo dataset/images/train dataset/labels/train")
    print("  \"")
    print()
    print("🚀 Starting LabelMe annotation tool...")
    print("   Close the annotation window when done.")

def start_labelme():
    import subprocess
    import os
    
    create_annotation_instructions()
    print()
    
    # Launch LabelMe
    train_images = "dataset/images/train"
    train_labels = "dataset/labels/train"
    
    cmd = [
        ".venv/bin/labelme",
        train_images,
        "--output", train_labels,
        "--nodata"  # Don't save image data in JSON
    ]
    
    print(f"Launching: {' '.join(cmd)}")
    print("Note: LabelMe will open in a new window")
    
    try:
        subprocess.run(cmd, cwd=os.getcwd())
    except KeyboardInterrupt:
        print("\nAnnotation session interrupted.")
    except Exception as e:
        print(f"Error launching LabelMe: {e}")
        print("\nAlternative: Run manually with:")
        print(f"  {' '.join(cmd)}")

if __name__ == "__main__":
    start_labelme()
