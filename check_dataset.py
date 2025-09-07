#!/usr/bin/env python3
"""
Script pour vérifier la cohérence entre images et annotations
"""

import os
import glob

def check_dataset_consistency():
    """Vérifie que chaque image a son annotation correspondante"""
    
    # Dossiers train
    train_images = glob.glob("/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/dataset/images/train/*.png")
    train_labels = glob.glob("/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/dataset/labels/train/*.txt")
    
    # Dossiers val
    val_images = glob.glob("/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/dataset/images/val/*.png")
    val_labels = glob.glob("/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/dataset/labels/val/*.txt")
    
    print(f"Train: {len(train_images)} images, {len(train_labels)} labels")
    print(f"Val: {len(val_images)} images, {len(val_labels)} labels")
    
    # Vérifier train
    train_img_basenames = {os.path.splitext(os.path.basename(img))[0] for img in train_images}
    train_lbl_basenames = {os.path.splitext(os.path.basename(lbl))[0] for lbl in train_labels}
    
    orphan_train_images = train_img_basenames - train_lbl_basenames
    orphan_train_labels = train_lbl_basenames - train_img_basenames
    
    if orphan_train_images:
        print(f"Images train sans annotations ({len(orphan_train_images)}): {list(orphan_train_images)[:5]}")
    if orphan_train_labels:
        print(f"Annotations train sans images ({len(orphan_train_labels)}): {list(orphan_train_labels)[:5]}")
    
    # Vérifier val
    val_img_basenames = {os.path.splitext(os.path.basename(img))[0] for img in val_images}
    val_lbl_basenames = {os.path.splitext(os.path.basename(lbl))[0] for lbl in val_labels}
    
    orphan_val_images = val_img_basenames - val_lbl_basenames
    orphan_val_labels = val_lbl_basenames - val_img_basenames
    
    if orphan_val_images:
        print(f"Images val sans annotations ({len(orphan_val_images)}): {list(orphan_val_images)[:5]}")
    if orphan_val_labels:
        print(f"Annotations val sans images ({len(orphan_val_labels)}): {list(orphan_val_labels)[:5]}")
    
    return len(orphan_train_images) + len(orphan_val_images) == 0

if __name__ == "__main__":
    is_consistent = check_dataset_consistency()
    print(f"Dataset cohérent: {is_consistent}")
