#!/usr/bin/env python3
"""
Golden City Dataset Annotation Guide
Instructions for using LabelMe to annotate comic panels and text regions.
"""

def create_annotation_instructions():
    print("🎯 MULTI-BD DATASET - ANNOTATION GUIDE")
    print("=" * 50)
    print()
    print("📁 Dataset Structure:")
    print("  dataset/")
    print("  ├── images/")
    print("  │   ├── train/        (Images from multiple comics)")
    print("  │   └── val/          (Validation images)")
    print("  ├── labels/")
    print("  │   ├── train/        (LabelMe JSON annotations)")
    print("  │   └── val/          (Validation annotations)")
    print("  └── predefined_classes.txt  (panel classes)")
    print()
    print("🎨 SUPPORTED COMIC STYLES:")
    print("  • Golden City (p*.png)     - Modern BD with complex layouts")
    print("  • Tintin (tintin_*.png)    - Classic simple panels")
    print("  • Pin-up du B24 (pinup_*.png) - Aviation/war theme, mixed style")
    print("  • [Ready for more series]")
    print()
    print("🏷️ ANNOTATION CLASSES:")
    print("  1. 'panel' - All comic panel boundaries (simple + complex)")
    print("     ├── Traditional rectangular panels (Tintin style)")
    print("     ├── Irregular shaped panels (modern BD)")
    print("     └── Inset panels (panels within panels)")
    print("  2. 'text'  - Text regions (speech bubbles, captions)")
    print()
    print("🖱️ LABELME ANNOTATION STEPS:")
    print("  1. LabelMe will open with images from all series")
    print("  2. Use polygon tool to draw around ALL panels:")
    print("     - Click to create polygon points")
    print("     - Right-click to finish polygon")
    print("     - Label as 'panel' (unified class)")
    print("  3. Annotate text regions as 'text'")
    print("  4. Save (Ctrl+S) - creates .json file")
    print("  5. Next image (Ctrl+D)")
    print("  6. Repeat for all series")
    print()
    print("💡 ANNOTATION STRATEGY:")
    print("  - Treat ALL panels the same regardless of style")
    print("  - Include inset panels in Golden City")
    print("  - Include simple panels in Tintin")
    print("  - This creates a unified 'panel' detector")
    print("  - Mix different styles for robust training")
    print()
    print("� Current Dataset Status:")
    import os
    from pathlib import Path
    
    train_dir = Path("dataset/images/train")
    if train_dir.exists():
        golden_count = len(list(train_dir.glob("p*.png")))
        tintin_count = len(list(train_dir.glob("tintin_*.png")))
        pinup_count = len(list(train_dir.glob("pinup_*.png")))
        other_count = len(list(train_dir.glob("*.png"))) - golden_count - tintin_count - pinup_count
        print(f"  • Golden City: {golden_count} images")
        print(f"  • Tintin: {tintin_count} images") 
        print(f"  • Pin-up du B24: {pinup_count} images")
        if other_count > 0:
            print(f"  • Other series: {other_count} images")
        print(f"  • Total: {golden_count + tintin_count + pinup_count + other_count} images")
    
    labels_dir = Path("dataset/labels/train")
    if labels_dir.exists():
        annotated_count = len(list(labels_dir.glob("*.json")))
        print(f"  • Annotated: {annotated_count} images")
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
