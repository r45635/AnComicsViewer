#!/usr/bin/env python3
"""
Script pour supprimer les images sans annotations correspondantes
"""

import os
import glob

def clean_orphan_images():
    """Supprime les images qui n'ont pas d'annotations correspondantes"""
    
    # Train
    train_images = glob.glob("/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/dataset/images/train/*.png")
    train_labels = glob.glob("/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/dataset/labels/train/*.txt")
    
    train_img_basenames = {os.path.splitext(os.path.basename(img))[0]: img for img in train_images}
    train_lbl_basenames = {os.path.splitext(os.path.basename(lbl))[0] for lbl in train_labels}
    
    orphan_train_images = set(train_img_basenames.keys()) - train_lbl_basenames
    
    print(f"Suppression de {len(orphan_train_images)} images train orphelines...")
    for basename in orphan_train_images:
        img_path = train_img_basenames[basename]
        print(f"Suppression: {img_path}")
        os.remove(img_path)
    
    # Val
    val_images = glob.glob("/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/dataset/images/val/*.png")
    val_labels = glob.glob("/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/dataset/labels/val/*.txt")
    
    val_img_basenames = {os.path.splitext(os.path.basename(img))[0]: img for img in val_images}
    val_lbl_basenames = {os.path.splitext(os.path.basename(lbl))[0] for lbl in val_labels}
    
    orphan_val_images = set(val_img_basenames.keys()) - val_lbl_basenames
    
    print(f"Suppression de {len(orphan_val_images)} images val orphelines...")
    for basename in orphan_val_images:
        img_path = val_img_basenames[basename]
        print(f"Suppression: {img_path}")
        os.remove(img_path)
    
    print("Nettoyage des images orphelines terminé!")

if __name__ == "__main__":
    clean_orphan_images()
