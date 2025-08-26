#!/usr/bin/env python3
"""
Script de sanity-check pour tester le modèle YOLO sur une image PNG.
Usage: python3 quickcheck.py <image.png>
"""

import sys
import os
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent))

def quickcheck_model(image_path: str):
    """Test rapide du modèle YOLO sur une image."""
    try:
        from ultralytics import YOLO
        import cv2
        import numpy as np
        from collections import Counter
        
        pt = "data/models/multibd_enhanced_v2.pt"
        
        print(f"🔍 Testing model: {pt}")
        print(f"📄 On image: {image_path}")
        
        if not os.path.exists(pt):
            print(f"❌ Model file not found: {pt}")
            return False
            
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return False
        
        # Charger l'image
        img = cv2.imread(image_path)  # BGR format
        if img is None:
            print(f"❌ Failed to load image: {image_path}")
            return False
            
        print(f"✅ Image loaded: {img.shape}")
        
        # Charger le modèle
        model = YOLO(pt)
        print(f"✅ Model loaded")
        print(f"📋 Model names: {model.names}")
        
        # Test de prédiction
        print(f"🔬 Running prediction...")
        r = model.predict(
            img, 
            conf=0.25, 
            iou=0.6, 
            imgsz=1280, 
            verbose=False, 
            classes=None  # ⚠️ Pas de filtre
        )
        
        b = r[0].boxes
        if b is None or b.cls is None or b.cls.numel() == 0:
            print("❌ AUCUNE box brute détectée")
            
            # Test avec seuil plus bas
            print("🔄 Trying with lower confidence...")
            r2 = model.predict(
                img, 
                conf=0.05, 
                iou=0.6, 
                imgsz=1280, 
                verbose=False, 
                classes=None
            )
            b2 = r2[0].boxes
            if b2 is None or b2.cls is None or b2.cls.numel() == 0:
                print("❌ AUCUNE box même avec conf=0.05")
                return False
            else:
                b = b2
                print(f"✅ {b.cls.numel()} boxes avec conf=0.05")
        else:
            print(f"✅ {b.cls.numel()} boxes détectées")
        
        # Analyser les classes détectées
        cls = b.cls.cpu().numpy().astype(int)
        scores = b.conf.cpu().numpy()
        
        print(f"📊 Classes détectées (IDs): {Counter(cls)}")
        
        class_names = [str(model.names[int(i)]).strip().lower() for i in cls]
        print(f"📊 Classes détectées (noms): {Counter(class_names)}")
        
        # Afficher quelques détails
        print(f"📈 Scores min/max: {scores.min():.3f} / {scores.max():.3f}")
        
        # Vérifier si on a des panels
        panel_count = sum(1 for name in class_names if 'panel' in name)
        if panel_count > 0:
            print(f"🎯 {panel_count} panels détectés - MODÈLE OK!")
            return True
        else:
            print(f"⚠️ Aucun panel détecté. Classes trouvées: {set(class_names)}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur durant le test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Point d'entrée principal."""
    if len(sys.argv) != 2:
        print("Usage: python3 quickcheck.py <image.png>")
        print("Example: python3 quickcheck.py test_page.png")
        return 1
    
    image_path = sys.argv[1]
    
    print("🚀 QuickCheck - Test modèle YOLO")
    print("=" * 50)
    
    success = quickcheck_model(image_path)
    
    print("=" * 50)
    if success:
        print("🎉 QuickCheck réussi - Modèle opérationnel!")
        return 0
    else:
        print("⚠️ Problème détecté - Vérifiez le modèle ou l'image")
        return 1

if __name__ == "__main__":
    sys.exit(main())
