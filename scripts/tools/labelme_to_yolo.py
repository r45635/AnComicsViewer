#!/usr/bin/env python3
"""
LabelMe to YOLO Converter
Converts LabelMe JSON annotations to YOLO format with automatic train/val split.

Features:
- Supports both rectangle and polygon shapes
- Converts panel and panel_inset classes
- 80/20 train/validation split
- Automatic coordinate normalization and clamping
- Creates complete YOLO dataset structure
"""

import os
import json
import shutil
import random
from pathlib import Path
import argparse


def convert_polygon_to_bbox(points):
    """Convert polygon points to bounding box."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def clamp_bbox(x1, y1, x2, y2, img_width, img_height):
    """Clamp bounding box coordinates to image bounds."""
    x1 = max(0, min(x1, img_width))
    y1 = max(0, min(y1, img_height))
    x2 = max(0, min(x2, img_width))
    y2 = max(0, min(y2, img_height))
    return x1, y1, x2, y2


def bbox_to_yolo(x1, y1, x2, y2, img_width, img_height):
    """Convert bounding box to YOLO format (normalized center, width, height)."""
    # Calculate center and dimensions
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1
    
    # Normalize to [0, 1]
    center_x /= img_width
    center_y /= img_height
    width /= img_width
    height /= img_height
    
    return center_x, center_y, width, height


def convert_labelme_to_yolo():
    """Main conversion function."""
    
    # Configuration
    source_dir = Path("dataset/images/train")
    labels_dir = Path("dataset/labels/train")
    output_dir = Path("dataset/yolo")
    
    # Class mapping
    class_names = ["panel", "panel_inset"]
    class_to_id = {name: i for i, name in enumerate(class_names)}
    
    print(f"🔄 Converting LabelMe annotations to YOLO format...")
    print(f"Source images: {source_dir}")
    print(f"Source labels: {labels_dir}")
    print(f"Output: {output_dir}")
    print(f"Classes: {class_names}")
    print()
    
    # Find all image files
    image_files = list(source_dir.glob("*.png")) + list(source_dir.glob("*.jpg"))
    print(f"Found {len(image_files)} images")
    
    # Find corresponding JSON files
    json_files = []
    images_with_labels = []
    
    for img_file in image_files:
        json_file = labels_dir / (img_file.stem + ".json")
        if json_file.exists():
            json_files.append(json_file)
            images_with_labels.append(img_file)
    
    print(f"Found {len(json_files)} JSON annotation files")
    
    if not json_files:
        print("❌ No JSON annotation files found!")
        print(f"Expected location: {labels_dir}")
        return False
    
    # Process annotations
    converted_annotations = []
    total_shapes = 0
    
    for json_file, img_file in zip(json_files, images_with_labels):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Get image dimensions
            img_width = data.get('imageWidth', 0)
            img_height = data.get('imageHeight', 0)
            
            if img_width == 0 or img_height == 0:
                print(f"⚠️  Warning: Missing image dimensions in {json_file.name}")
                continue
            
            # Process shapes
            yolo_annotations = []
            file_shapes = 0
            
            for shape in data.get('shapes', []):
                label = shape.get('label', '').lower()
                shape_type = shape.get('shape_type', '')
                points = shape.get('points', [])
                
                # Check if this is a valid class
                if label not in class_to_id:
                    continue
                
                class_id = class_to_id[label]
                
                # Convert shape to bounding box
                if shape_type == 'rectangle' and len(points) == 2:
                    # Rectangle: two corner points
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                elif shape_type == 'polygon' and len(points) >= 3:
                    # Polygon: convert to bounding box
                    x1, y1, x2, y2 = convert_polygon_to_bbox(points)
                else:
                    print(f"⚠️  Warning: Unsupported shape type '{shape_type}' in {json_file.name}")
                    continue
                
                # Ensure proper order (x1 < x2, y1 < y2)
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                
                # Clamp to image bounds
                x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, img_width, img_height)
                
                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    print(f"⚠️  Warning: Invalid box in {json_file.name}")
                    continue
                
                # Convert to YOLO format
                center_x, center_y, width, height = bbox_to_yolo(x1, y1, x2, y2, img_width, img_height)
                
                yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
                file_shapes += 1
                total_shapes += 1
            
            if yolo_annotations:
                converted_annotations.append({
                    'image_file': img_file,
                    'json_file': json_file,
                    'annotations': yolo_annotations,
                    'shape_count': file_shapes
                })
        
        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {e}")
    
    print(f"✓ Converted {len(converted_annotations)} files with {total_shapes} annotations")
    
    if not converted_annotations:
        print("❌ No valid annotations found!")
        return False
    
    # Split into train/validation (80/20)
    random.seed(42)  # For reproducible splits
    random.shuffle(converted_annotations)
    
    split_idx = int(len(converted_annotations) * 0.8)
    train_data = converted_annotations[:split_idx]
    val_data = converted_annotations[split_idx:]
    
    print(f"📊 Dataset split: {len(train_data)} train, {len(val_data)} validation")
    
    # Create output directory structure
    output_dir.mkdir(exist_ok=True)
    (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    # Copy images and create label files
    def copy_dataset_split(data_split, split_name):
        for item in data_split:
            img_file = item['image_file']
            annotations = item['annotations']
            
            # Copy image
            dst_img = output_dir / "images" / split_name / img_file.name
            shutil.copy2(img_file, dst_img)
            
            # Write YOLO label file
            label_file = output_dir / "labels" / split_name / (img_file.stem + ".txt")
            with open(label_file, 'w') as f:
                f.write('\n'.join(annotations))
    
    print("📁 Copying train split...")
    copy_dataset_split(train_data, "train")
    
    print("📁 Copying validation split...")
    copy_dataset_split(val_data, "val")
    
    # Create data.yaml
    data_yaml_content = f"""# YOLO Dataset Configuration
# Generated from LabelMe annotations

path: {output_dir.absolute()}
train: images/train
val: images/val
test: images/val  # Use validation as test for now

# Classes
nc: {len(class_names)}
names: {class_names}

# Dataset info
source: LabelMe annotations
classes: {class_to_id}
total_images: {len(converted_annotations)}
train_images: {len(train_data)}
val_images: {len(val_data)}
total_annotations: {total_shapes}
"""
    
    data_yaml_path = output_dir / "data.yaml"
    with open(data_yaml_path, 'w') as f:
        f.write(data_yaml_content)
    
    print(f"✓ Created dataset configuration: {data_yaml_path}")
    
    # Print summary
    print()
    print("🎉 CONVERSION COMPLETE!")
    print("=" * 40)
    print(f"📊 Dataset Statistics:")
    print(f"   Total images: {len(converted_annotations)}")
    print(f"   Train: {len(train_data)} images")
    print(f"   Validation: {len(val_data)} images")
    print(f"   Total annotations: {total_shapes}")
    print()
    print(f"📁 Output structure:")
    print(f"   {output_dir}/")
    print(f"   ├── images/train/     ({len(train_data)} images)")
    print(f"   ├── images/val/       ({len(val_data)} images)")
    print(f"   ├── labels/train/     ({len(train_data)} label files)")
    print(f"   ├── labels/val/       ({len(val_data)} label files)")
    print(f"   └── data.yaml         (YOLO config)")
    print()
    print(f"🚀 Ready for training:")
    print(f"   yolo detect train data={data_yaml_path} model=yolov8n.pt epochs=100")
    print()
    
    # Class distribution
    class_counts = {name: 0 for name in class_names}
    for item in converted_annotations:
        for annotation in item['annotations']:
            class_id = int(annotation.split()[0])
            class_counts[class_names[class_id]] += 1
    
    print(f"📈 Class distribution:")
    for class_name, count in class_counts.items():
        percentage = (count / total_shapes * 100) if total_shapes > 0 else 0
        print(f"   {class_name}: {count} ({percentage:.1f}%)")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert LabelMe annotations to YOLO format")
    parser.add_argument("--source", default="dataset/images/train", help="Source image directory")
    parser.add_argument("--labels", default="dataset/labels/train", help="LabelMe JSON directory")
    parser.add_argument("--output", default="dataset/yolo", help="Output YOLO dataset directory")
    parser.add_argument("--classes", nargs="+", default=["panel", "panel_inset"], help="Class names")
    
    args = parser.parse_args()
    
    # Update global paths if provided
    if args.source != "dataset/images/train":
        globals()['source_dir'] = Path(args.source)
    if args.labels != "dataset/labels/train":
        globals()['labels_dir'] = Path(args.labels)
    if args.output != "dataset/yolo":
        globals()['output_dir'] = Path(args.output)
    
    success = convert_labelme_to_yolo()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
