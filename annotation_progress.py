#!/usr/bin/env python3
"""
Batch Annotation Progress Tracker
Tracks annotation progress and manages batch annotation workflow.
"""

import os
import json
from pathlib import Path

def check_annotation_progress():
    """Check current annotation progress."""
    
    train_images = list(Path("dataset/images/train").glob("*.png"))
    train_labels = list(Path("dataset/labels/train").glob("*.json"))
    
    print("📊 PROGRESSION DES ANNOTATIONS")
    print("=" * 40)
    print(f"Images totales: {len(train_images)}")
    print(f"Images annotées: {len(train_labels)}")
    print(f"Images restantes: {len(train_images) - len(train_labels)}")
    print(f"Progression: {len(train_labels)/len(train_images)*100:.1f}%")
    print()
    
    # Show unannotated images
    annotated_stems = {f.stem for f in train_labels}
    unannotated = [img for img in train_images if img.stem not in annotated_stems]
    
    if unannotated:
        print("🎯 Images à annoter:")
        for i, img in enumerate(unannotated[:10]):  # Show first 10
            print(f"   {i+1:2d}. {img.name}")
        if len(unannotated) > 10:
            print(f"   ... et {len(unannotated)-10} autres")
        print()
    
    return len(train_labels), len(train_images)

def analyze_annotations():
    """Analyze existing annotations."""
    
    labels_dir = Path("dataset/labels/train")
    json_files = list(labels_dir.glob("*.json"))
    
    if not json_files:
        print("❌ Aucune annotation trouvée")
        return
    
    total_panels = 0
    total_text = 0
    files_with_panels = 0
    
    print("📈 ANALYSE DES ANNOTATIONS")
    print("=" * 30)
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            file_panels = 0
            file_text = 0
            
            for shape in data.get('shapes', []):
                label = shape.get('label', '').lower()
                if label == 'panel':
                    file_panels += 1
                    total_panels += 1
                elif label == 'text':
                    file_text += 1
                    total_text += 1
            
            if file_panels > 0:
                files_with_panels += 1
                
        except Exception as e:
            print(f"⚠️  Erreur lecture {json_file.name}: {e}")
    
    print(f"Total panels: {total_panels}")
    print(f"Total text: {total_text}")
    print(f"Fichiers avec panels: {files_with_panels}/{len(json_files)}")
    print(f"Moyenne panels/image: {total_panels/len(json_files):.1f}")
    print()

def suggest_next_images():
    """Suggest which images to annotate next."""
    
    train_images = list(Path("dataset/images/train").glob("*.png"))
    train_labels = list(Path("dataset/labels/train").glob("*.json"))
    
    annotated_stems = {f.stem for f in train_labels}
    unannotated = [img for img in train_images if img.stem not in annotated_stems]
    
    if not unannotated:
        print("🎉 Toutes les images sont annotées!")
        return
    
    # Sort by filename for consistent ordering
    unannotated.sort(key=lambda x: x.name)
    
    print("🎯 IMAGES RECOMMANDÉES (5 suivantes):")
    print("=" * 40)
    for i, img in enumerate(unannotated[:5]):
        print(f"   {i+1}. {img.name}")
    print()
    print("💡 Pour annoter ces images:")
    print("   python start_annotation.py")
    print()

def estimate_training_improvement():
    """Estimate how much more data would help."""
    
    annotated_count, total_count = check_annotation_progress()
    
    if annotated_count == 0:
        return
    
    print("📈 ESTIMATION D'AMÉLIORATION")
    print("=" * 35)
    
    # Current performance (from last training)
    current_map = 0.4954  # mAP@0.5 from your training
    current_recall = 0.42438
    
    # Rough estimates based on data scaling laws
    potential_data_multiplier = total_count / annotated_count
    estimated_improvement = min(0.15, 0.05 * (potential_data_multiplier - 1))
    
    estimated_new_map = min(0.85, current_map + estimated_improvement)
    estimated_new_recall = min(0.80, current_recall + estimated_improvement * 1.5)
    
    print(f"Performance actuelle:")
    print(f"   mAP@0.5: {current_map:.1%}")
    print(f"   Recall: {current_recall:.1%}")
    print()
    print(f"Performance estimée avec toutes les données:")
    print(f"   mAP@0.5: {estimated_new_map:.1%} (+{estimated_improvement:.1%})")
    print(f"   Recall: {estimated_new_recall:.1%} (+{estimated_improvement*1.5:.1%})")
    print()
    print(f"Images supplémentaires nécessaires: {total_count - annotated_count}")
    print()

def main():
    """Main progress tracking function."""
    
    print("🎯 GOLDEN CITY DATASET - SUIVI DES ANNOTATIONS")
    print("=" * 55)
    print()
    
    # Check progress
    check_annotation_progress()
    
    # Analyze existing annotations
    analyze_annotations()
    
    # Suggest next steps
    suggest_next_images()
    
    # Estimate improvements
    estimate_training_improvement()
    
    print("🚀 PROCHAINES ÉTAPES:")
    print("=" * 20)
    print("1. 🏷️  Annoter plus d'images: python start_annotation.py")
    print("2. 🔄 Régénérer dataset: python tools/labelme_to_yolo.py") 
    print("3. 🏋️  Continuer training: python continue_training.py")
    print("4. 🧪 Tester modèle: python test_model.py")

if __name__ == "__main__":
    main()
