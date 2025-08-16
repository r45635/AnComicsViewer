#!/usr/bin/env python3
"""
Test Complet du Dataset Mixte
Teste le modèle sur Golden City et Tintin, puis lance un nouvel entraînement.
"""

import os
import torch
from pathlib import Path
import subprocess

# Apply PyTorch compatibility fix
original_torch_load = torch.load
def patched_torch_load(f, map_location=None, pickle_module=None, weights_only=None, **kwargs):
    if weights_only is None:
        weights_only = False
    return original_torch_load(f, map_location=map_location, pickle_module=pickle_module, 
                              weights_only=weights_only, **kwargs)
torch.load = patched_torch_load

from ultralytics import YOLO

def test_current_model():
    """Teste le modèle actuel sur les deux styles."""
    
    print("🧪 TEST DU MODÈLE ACTUEL")
    print("=" * 30)
    
    model_path = "runs/detect/overfit_small/weights/best.pt"
    if not Path(model_path).exists():
        print("❌ Modèle non trouvé")
        return False
    
    model = YOLO(model_path)
    
    # Test Golden City
    gc_images = list(Path("dataset/images/train").glob("p*.png"))[:3]
    tintin_images = list(Path("dataset/images/train").glob("tintin_*.png"))[:3]
    
    print("🏛️ Golden City (3 échantillons):")
    gc_total = 0
    for img in gc_images:
        results = model.predict(str(img), conf=0.1, verbose=False)
        panels = len(results[0].boxes) if results[0].boxes is not None else 0
        gc_total += panels
        print(f"   {img.name}: {panels} panels")
    
    print(f"🎨 Tintin (3 échantillons):")
    tintin_total = 0
    for img in tintin_images:
        results = model.predict(str(img), conf=0.1, verbose=False)
        panels = len(results[0].boxes) if results[0].boxes is not None else 0
        tintin_total += panels
        print(f"   {img.name}: {panels} panels")
    
    print(f"\n📊 Moyenne:")
    print(f"   Golden City: {gc_total/3:.1f} panels/page")
    print(f"   Tintin: {tintin_total/3:.1f} panels/page")
    
    return True

def train_mixed_dataset():
    """Lance un nouvel entraînement avec le dataset mixte."""
    
    print("\n🏋️ ENTRAÎNEMENT DATASET MIXTE")
    print("=" * 35)
    
    # Set environment for MPS
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    try:
        # Load base model (ou reprendre depuis le meilleur)
        model = YOLO("runs/detect/overfit_small/weights/best.pt")  # Continuer depuis le modèle existant
        
        print("📊 Configuration d'entraînement:")
        print("   - Dataset: Golden City + Tintin (32 images)")
        print("   - Epochs: 60 (entraînement adaptatif)")
        print("   - Batch: 4")
        print("   - Learning rate: Réduit pour fine-tuning")
        print()
        
        # Training with mixed dataset
        results = model.train(
            data="dataset/yolo/data.yaml",
            epochs=60,
            imgsz=1024,
            batch=4,
            workers=0,
            device='mps',
            name='mixed_golden_tintin',
            cache=False,
            resume=False,
            # Fine-tuning parameters
            lr0=0.003,  # Lower learning rate
            warmup_epochs=2,
            patience=15,
            # Light augmentation for mixed styles
            mosaic=0.2,
            mixup=0.1,
            hsv_h=0.01,
            hsv_s=0.5,
            hsv_v=0.3,
            degrees=0.0,
            translate=0.05,
            scale=0.3,
            fliplr=0.5
        )
        
        print("✅ Entraînement terminé!")
        print(f"📁 Résultats: runs/detect/mixed_golden_tintin/")
        return True
        
    except Exception as e:
        print(f"❌ Erreur entraînement: {e}")
        return False

def compare_models():
    """Compare l'ancien et le nouveau modèle."""
    
    print("\n🔍 COMPARAISON DES MODÈLES")
    print("=" * 30)
    
    old_model_path = "runs/detect/overfit_small/weights/best.pt"
    new_model_path = "runs/detect/mixed_golden_tintin/weights/best.pt"
    
    if not Path(new_model_path).exists():
        print("❌ Nouveau modèle non trouvé")
        return
    
    old_model = YOLO(old_model_path)
    new_model = YOLO(new_model_path)
    
    # Test sur une page de chaque style
    test_images = [
        "dataset/images/train/p0003.png",  # Golden City
        "dataset/images/train/tintin_p0001.png"  # Tintin
    ]
    
    for img_path in test_images:
        if not Path(img_path).exists():
            continue
            
        print(f"\n📸 Test sur {Path(img_path).name}:")
        
        # Ancien modèle
        old_results = old_model.predict(img_path, conf=0.1, verbose=False)
        old_panels = len(old_results[0].boxes) if old_results[0].boxes is not None else 0
        
        # Nouveau modèle
        new_results = new_model.predict(img_path, conf=0.1, verbose=False)
        new_panels = len(new_results[0].boxes) if new_results[0].boxes is not None else 0
        
        print(f"   Ancien modèle: {old_panels} panels")
        print(f"   Nouveau modèle: {new_panels} panels")
        
        if new_panels > old_panels:
            print(f"   📈 Amélioration: +{new_panels - old_panels}")
        elif new_panels < old_panels:
            print(f"   📉 Dégradation: {new_panels - old_panels}")
        else:
            print(f"   ➡️  Identique")

def main():
    """Fonction principale."""
    
    print("🎯 TEST COMPLET - DATASET MIXTE GOLDEN CITY + TINTIN")
    print("=" * 60)
    print()
    
    # 1. Test du modèle actuel
    if not test_current_model():
        return
    
    # 2. Demander confirmation pour l'entraînement
    print("\n🚀 Lancer l'entraînement avec le dataset mixte ?")
    print("   - 32 images annotées (Golden City + Tintin)")
    print("   - 217 annotations de panels")
    print("   - Entraînement de 60 epochs")
    
    response = input("\nContinuer ? (y/N): ").strip().lower()
    
    if response == 'y':
        # 3. Entraîner le nouveau modèle
        if train_mixed_dataset():
            # 4. Comparer les modèles
            compare_models()
            
            print("\n🎉 RÉSUMÉ FINAL:")
            print("=" * 20)
            print("✅ Modèle mixte entraîné")
            print("📊 Performances comparées")
            print("🎯 Prêt pour intégration dans AnComicsViewer")
            print()
            print("💡 Pour utiliser le nouveau modèle:")
            print("   Modifier AnComicsViewer.py ligne 986:")
            print("   trained_model = 'runs/detect/mixed_golden_tintin/weights/best.pt'")
    else:
        print("⏸️ Entraînement reporté")

if __name__ == "__main__":
    main()
