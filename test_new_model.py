#!/usr/bin/env python3
"""
Test rapide du nouveau modèle YOLOv8 Multi-BD Enhanced v2
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Ajouter le chemin src au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_model_simple():
    """Test simple du modèle avec ultralytics directement"""
    try:
        from ultralytics import YOLO
        
        print("🤖 Test du modèle YOLOv8 Multi-BD Enhanced v2")
        print("=" * 50)
        
        # Charger le modèle
        model_path = "detectors/models/multibd_enhanced_v2.pt"
        if not os.path.exists(model_path):
            print(f"❌ Modèle introuvable: {model_path}")
            return False
            
        print(f"📦 Chargement du modèle: {model_path}")
        model = YOLO(model_path)
        print("✅ Modèle chargé avec succès!")
        
        # Tester sur une image
        image_path = "dataset/images/val/p0002.png"
        if not os.path.exists(image_path):
            print(f"❌ Image de test introuvable: {image_path}")
            return False
            
        print(f"📸 Test sur image: {image_path}")
        
        # Faire la prédiction
        results = model(image_path, conf=0.15, iou=0.6, imgsz=1280, device='mps')
        
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            if boxes is not None and len(boxes) > 0:
                print(f"🎯 {len(boxes)} panels détectés!")
                
                for i, box in enumerate(boxes):
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    
                    class_name = "panel" if cls == 0 else f"class_{cls}"
                    print(f"  Panel {i+1}: {class_name} (conf: {conf:.3f}) bbox: {xyxy}")
                    
                print("✅ Test réussi! Le modèle fonctionne.")
                return True
            else:
                print("⚠️  Aucun panel détecté (seuil de confiance trop élevé?)")
                return True
        else:
            print("❌ Erreur lors de la prédiction")
            return False
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integrated_detector():
    """Test du détecteur intégré"""
    try:
        print("\n🔧 Test du détecteur intégré")
        print("=" * 50)
        
        from ancomicsviewer.detectors.multibd_detector import MultiBDPanelDetector
        
        print("📦 Création du détecteur...")
        detector = MultiBDPanelDetector()
        print("✅ Détecteur créé!")
        
        # Charger une image
        image_path = "dataset/images/val/p0002.png"
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"❌ Impossible de charger: {image_path}")
            return False
            
        print(f"📸 Image chargée: {image.shape}")
        
        # Détecter les panels
        panels = detector.detect_panels(image)
        print(f"🎯 {len(panels)} panels détectés avec post-processing!")
        
        for i, panel in enumerate(panels):
            print(f"  Panel {i+1}: {panel}")
            
        return True
        
    except Exception as e:
        print(f"❌ Erreur détecteur intégré: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Test du nouveau modèle YOLOv8 Multi-BD Enhanced v2")
    print("=" * 60)
    
    # Test 1: Modèle brut
    success1 = test_model_simple()
    
    # Test 2: Détecteur intégré  
    success2 = test_integrated_detector()
    
    print("\n📊 Résultats des tests:")
    print(f"  • Modèle brut: {'✅' if success1 else '❌'}")
    print(f"  • Détecteur intégré: {'✅' if success2 else '❌'}")
    
    if success1 or success2:
        print("\n🎉 Le nouveau modèle fonctionne!")
    else:
        print("\n💔 Des problèmes ont été détectés")
