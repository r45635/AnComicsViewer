#!/usr/bin/env python3
"""
Script pour nettoyer les fichiers d'annotations vides qui peuvent causer des erreurs pendant l'entraînement YOLO
"""

import os
import glob

def clean_empty_labels():
    """Supprime les fichiers d'annotations vides et leurs images correspondantes"""
    
    # Chemins des dossiers
    labels_dirs = [
        "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/dataset/labels/train",
        "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/dataset/labels/val"
    ]
    
    images_dirs = [
        "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/dataset/images/train",
        "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/dataset/images/val"
    ]
    
    for labels_dir, images_dir in zip(labels_dirs, images_dirs):
        print(f"Nettoyage de {labels_dir}...")
        
        # Trouver tous les fichiers .txt vides
        empty_files = []
        for txt_file in glob.glob(os.path.join(labels_dir, "*.txt")):
            if os.path.getsize(txt_file) == 0:
                empty_files.append(txt_file)
        
        print(f"Trouvé {len(empty_files)} fichiers vides")
        
        # Supprimer les fichiers vides et leurs images correspondantes
        for txt_file in empty_files:
            basename = os.path.splitext(os.path.basename(txt_file))[0]
            
            # Supprimer le fichier d'annotation vide
            print(f"Suppression de {txt_file}")
            os.remove(txt_file)
            
            # Supprimer l'image correspondante (chercher différentes extensions)
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                img_file = os.path.join(images_dir, basename + ext)
                if os.path.exists(img_file):
                    print(f"Suppression de {img_file}")
                    os.remove(img_file)
                    break

if __name__ == "__main__":
    clean_empty_labels()
    print("Nettoyage terminé!")
