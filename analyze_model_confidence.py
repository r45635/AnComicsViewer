#!/usr/bin/env python3
"""
Analyse des seuils de confiance pour comprendre la différence de détection
"""

import sys
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import torch

# Fix pour PyTorch 2.8.0 - charger avec weights_only=False
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def load_model_safe(model_path):
    """Charge un modèle YOLO en gérant les problèmes de compatibilité."""
    try:
        return YOLO(model_path)
    except Exception as e:
        print(f"⚠️  Erreur chargement modèle: {e}")
        return None

def analyze_confidence_thresholds():
    """Analyse la détection avec différents seuils de confiance."""
    
    print("🔍 ANALYSE DES SEUILS DE CONFIANCE")
    print("=" * 50)
    
    # Modèles
    old_model_path = "runs/detect/overfit_small/weights/best.pt"
    new_model_path = "runs/detect/mixed_golden_tintin/weights/best.pt"
    
    if not Path(old_model_path).exists():
        print(f"❌ Ancien modèle non trouvé: {old_model_path}")
        return
        
    if not Path(new_model_path).exists():
        print(f"❌ Nouveau modèle non trouvé: {new_model_path}")
        return
    
    print("✅ Chargement des modèles...")
    old_model = load_model_safe(old_model_path)
    new_model = load_model_safe(new_model_path)
    
    if old_model is None or new_model is None:
        print("❌ Impossible de charger les modèles")
        return
    
    # Images de test
    test_images = [
        ("dataset/images/train/p0003.png", "Golden City"),
        ("dataset/images/train/tintin_p0001.png", "Tintin")
    ]
    
    # Seuils à tester
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    
    for img_path, style in test_images:
        if not Path(img_path).exists():
            print(f"⚠️  Image non trouvée: {img_path}")
            continue
            
        print(f"\n📸 {style} - {Path(img_path).name}")
        print("-" * 40)
        print("Seuil  | Ancien | Nouveau | Diff")
        print("-" * 40)
        
        for threshold in thresholds:
            # Ancien modèle
            old_results = old_model.predict(img_path, conf=threshold, verbose=False)
            old_panels = len(old_results[0].boxes) if old_results[0].boxes is not None else 0
            
            # Nouveau modèle  
            new_results = new_model.predict(img_path, conf=threshold, verbose=False)
            new_panels = len(new_results[0].boxes) if new_results[0].boxes is not None else 0
            
            diff = new_panels - old_panels
            diff_str = f"+{diff}" if diff > 0 else str(diff)
            
            print(f"{threshold:5.2f}  |   {old_panels:2d}   |    {new_panels:2d}    |  {diff_str:>3}")

def analyze_detection_scores():
    """Analyse les scores de confiance des détections."""
    
    print("\n🎯 ANALYSE DES SCORES DE CONFIANCE")
    print("=" * 50)
    
    # Modèles
    old_model_path = "runs/detect/overfit_small/weights/best.pt"
    new_model_path = "runs/detect/mixed_golden_tintin/weights/best.pt"
    
    old_model = load_model_safe(old_model_path)
    new_model = load_model_safe(new_model_path)
    
    if old_model is None or new_model is None:
        print("❌ Impossible de charger les modèles")
        return
    
    # Image de test
    test_image = "dataset/images/train/p0003.png"
    if not Path(test_image).exists():
        print(f"❌ Image test non trouvée: {test_image}")
        return
    
    print(f"📸 Analyse détaillée: {Path(test_image).name}")
    
    # Prédictions avec seuil très bas
    old_results = old_model.predict(test_image, conf=0.01, verbose=False)
    new_results = new_model.predict(test_image, conf=0.01, verbose=False)
    
    print("\n=== ANCIEN MODÈLE ===")
    if old_results[0].boxes is not None:
        confidences = old_results[0].boxes.conf.cpu().numpy()
        classes = old_results[0].boxes.cls.cpu().numpy()
        
        print(f"Détections totales: {len(confidences)}")
        for i, (conf, cls) in enumerate(zip(confidences, classes)):
            class_name = "panel" if cls == 0 else "panel_inset"
            print(f"  {i+1:2d}. {class_name:11} - conf: {conf:.3f}")
        
        print(f"Score moyen: {np.mean(confidences):.3f}")
        print(f"Score médian: {np.median(confidences):.3f}")
        print(f"Score min: {np.min(confidences):.3f}")
        print(f"Score max: {np.max(confidences):.3f}")
    
    print("\n=== NOUVEAU MODÈLE ===")
    if new_results[0].boxes is not None:
        confidences = new_results[0].boxes.conf.cpu().numpy()
        classes = new_results[0].boxes.cls.cpu().numpy()
        
        print(f"Détections totales: {len(confidences)}")
        for i, (conf, cls) in enumerate(zip(confidences, classes)):
            class_name = "panel" if cls == 0 else "panel_inset"
            print(f"  {i+1:2d}. {class_name:11} - conf: {conf:.3f}")
            
        print(f"Score moyen: {np.mean(confidences):.3f}")
        print(f"Score médian: {np.median(confidences):.3f}")
        print(f"Score min: {np.min(confidences):.3f}")
        print(f"Score max: {np.max(confidences):.3f}")

if __name__ == "__main__":
    analyze_confidence_thresholds()
    analyze_detection_scores()
