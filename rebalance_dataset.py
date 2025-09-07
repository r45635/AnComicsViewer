#!/usr/bin/env python3
"""
Script pour rééquilibrer le dataset train/val après le nettoyage
Objectif: avoir au moins 15-20% du dataset en validation
"""

import os
import shutil
import random
import glob

def rebalance_dataset():
    """Rééquilibrer le dataset en déplaçant des échantillons de train vers val"""
    
    # Compter les échantillons actuels
    train_images = glob.glob("/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/dataset/images/train/*.png")
    val_images = glob.glob("/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/dataset/images/val/*.png")
    
    total = len(train_images) + len(val_images)
    current_val_pct = len(val_images) / total * 100 if total > 0 else 0
    
    print(f"📊 État actuel du dataset:")
    print(f"   Train: {len(train_images)} images")
    print(f"   Val: {len(val_images)} images")
    print(f"   Total: {total} images")
    print(f"   Validation: {current_val_pct:.1f}%")
    
    # Objectif: 20% en validation (minimum 25 échantillons)
    target_val_count = max(25, int(total * 0.20))
    need_to_move = target_val_count - len(val_images)
    
    if need_to_move <= 0:
        print("✅ Le dataset est déjà bien équilibré")
        return
    
    print(f"\n🔄 Rééquilibrage nécessaire:")
    print(f"   Objectif validation: {target_val_count} images")
    print(f"   À déplacer: {need_to_move} images de train vers val")
    
    if need_to_move > len(train_images):
        print("❌ Impossible: pas assez d'échantillons en train")
        return
    
    # Sélectionner aléatoirement les échantillons à déplacer
    random.seed(42)  # Pour la reproductibilité
    random.shuffle(train_images)
    to_move = train_images[:need_to_move]
    
    print(f"\n📁 Déplacement de {len(to_move)} échantillons...")
    
    moved_count = 0
    for img_path in to_move:
        # Obtenir le nom de base sans extension
        basename = os.path.splitext(os.path.basename(img_path))[0]
        
        # Chemins source
        src_img = img_path
        src_label = f"/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/dataset/labels/train/{basename}.txt"
        
        # Chemins destination
        dst_img = f"/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/dataset/images/val/{basename}.png"
        dst_label = f"/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/dataset/labels/val/{basename}.txt"
        
        # Vérifier que le label existe
        if not os.path.exists(src_label):
            print(f"⚠️  Label manquant pour {basename}, ignoré")
            continue
        
        try:
            # Déplacer l'image
            shutil.move(src_img, dst_img)
            # Déplacer le label
            shutil.move(src_label, dst_label)
            moved_count += 1
            print(f"   ✅ {basename}")
        except Exception as e:
            print(f"   ❌ Erreur pour {basename}: {e}")
    
    print(f"\n🎯 Rééquilibrage terminé:")
    print(f"   {moved_count} échantillons déplacés avec succès")
    
    # Vérifier le nouvel état
    train_images_new = glob.glob("/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/dataset/images/train/*.png")
    val_images_new = glob.glob("/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/dataset/images/val/*.png")
    total_new = len(train_images_new) + len(val_images_new)
    new_val_pct = len(val_images_new) / total_new * 100 if total_new > 0 else 0
    
    print(f"\n📊 Nouvel état:")
    print(f"   Train: {len(train_images_new)} images")
    print(f"   Val: {len(val_images_new)} images")
    print(f"   Total: {total_new} images")
    print(f"   Validation: {new_val_pct:.1f}%")

if __name__ == "__main__":
    rebalance_dataset()
