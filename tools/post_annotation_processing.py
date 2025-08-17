#!/usr/bin/env python3
"""
Post-Annotation Processing for Golden City Dataset
Converts LabelMe JSON files to YOLO format and prepares for training.
"""

import os
import json
import glob
from pathlib import Path

def convert_labelme_to_yolo():
    """Convert LabelMe JSON annotations to YOLO format."""
    
    print("🔄 Converting LabelMe annotations to YOLO format...")
    
    # Define class mapping
    class_names = ["panel", "text"]
    class_to_id = {name: i for i, name in enumerate(class_names)}
    
    train_images = Path("dataset/images/train")
    train_labels = Path("dataset/labels/train")
    
    json_files = list(train_labels.glob("*.json"))
    
    if not json_files:
        print("❌ No JSON annotation files found in dataset/labels/train/")
        print("   Make sure you've saved annotations in LabelMe")
        return False
    
    print(f"📄 Found {len(json_files)} annotation files")
    
    converted = 0
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Get image dimensions
            img_height = data.get('imageHeight', 0)
            img_width = data.get('imageWidth', 0)
            
            if img_height == 0 or img_width == 0:
                print(f"⚠️  Warning: Missing image dimensions in {json_file.name}")
                continue
            
            # Convert to YOLO format
            yolo_lines = []
            
            for shape in data.get('shapes', []):
                if shape['shape_type'] != 'polygon':
                    continue
                
                label = shape['label']
                if label not in class_to_id:
                    print(f"⚠️  Warning: Unknown class '{label}' in {json_file.name}")
                    continue
                
                class_id = class_to_id[label]
                points = shape['points']
                
                # Normalize coordinates
                normalized_points = []
                for x, y in points:
                    norm_x = x / img_width
                    norm_y = y / img_height
                    normalized_points.extend([norm_x, norm_y])
                
                # Create YOLO segmentation line
                line = f"{class_id} " + " ".join(f"{coord:.6f}" for coord in normalized_points)
                yolo_lines.append(line)
            
            # Save YOLO format file
            yolo_file = train_labels / (json_file.stem + ".txt")
            with open(yolo_file, 'w') as f:
                f.write("\\n".join(yolo_lines))
            
            converted += 1
            print(f"✓ Converted {json_file.name} -> {yolo_file.name} ({len(yolo_lines)} annotations)")
            
        except Exception as e:
            print(f"❌ Error converting {json_file.name}: {e}")
    
    print(f"\\n🎉 Conversion complete: {converted}/{len(json_files)} files converted")
    return converted > 0

def update_dataset_yaml():
    """Update the dataset.yaml configuration."""
    
    dataset_yaml = Path("ml/dataset.yaml")
    
    yaml_content = f"""# Golden City Dataset Configuration
path: ../dataset
train: images/train
val: images/val
test: images/val  # Use val as test for now

nc: 2  # Number of classes
names:
  0: panel
  1: text

# Dataset info
source: Golden City - T01 - Pilleurs d'épaves (Franco-Belgian comic)
author: Daniel Pecqueur & Nicolas Malfin
annotated_by: Manual annotation with LabelMe
format: YOLO segmentation (polygon)
"""
    
    with open(dataset_yaml, 'w') as f:
        f.write(yaml_content)
    
    print(f"✓ Updated {dataset_yaml}")

def check_annotation_status():
    """Check the current annotation status."""
    
    print("📊 GOLDEN CITY DATASET STATUS")
    print("=" * 40)
    
    train_images = list(Path("dataset/images/train").glob("*.png"))
    val_images = list(Path("dataset/images/val").glob("*.png"))
    json_files = list(Path("dataset/labels/train").glob("*.json"))
    yolo_files = list(Path("dataset/labels/train").glob("*.txt"))
    
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    print(f"JSON annotations: {len(json_files)}")
    print(f"YOLO labels: {len(yolo_files)}")
    
    if json_files:
        print(f"\\nAnnotation progress: {len(json_files)}/{len(train_images)} images ({len(json_files)/len(train_images)*100:.1f}%)")
        
        if len(json_files) == len(train_images):
            print("🎉 All training images annotated!")
        else:
            remaining = len(train_images) - len(json_files)
            print(f"📝 {remaining} images still need annotation")
    else:
        print("\\n📝 No annotations found - start with LabelMe annotation")
    
    print(f"\\nNext steps:")
    if json_files and len(yolo_files) < len(json_files):
        print("1. Run conversion: python post_annotation_processing.py")
    elif yolo_files:
        print("1. ✓ Annotations converted to YOLO format")
        print("2. Ready for training: yolo segment train model=yolov8n-seg.pt data=ml/dataset.yaml")
    else:
        print("1. Continue annotation with LabelMe")
        print("2. Convert annotations when done")

def main():
    check_annotation_status()
    
    json_files = list(Path("dataset/labels/train").glob("*.json"))
    if json_files:
        print("\\n" + "=" * 40)
        convert_to_yolo = input("Convert annotations to YOLO format? (y/n): ").lower().startswith('y')
        
        if convert_to_yolo:
            if convert_labelme_to_yolo():
                update_dataset_yaml()
                print("\\n✅ Dataset ready for YOLO training!")
            else:
                print("\\n❌ Conversion failed")

if __name__ == "__main__":
    main()
