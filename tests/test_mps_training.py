#!/usr/bin/env python3
"""
Script de test pour l'entraînement MPS - Version DEBUG
"""

import os
import sys
import torch
from ultralytics import YOLO, settings

def main():
    print("🚀 Multi-BD Enhanced v2 - Test Entraînement MPS")
    print("=" * 60)
    
    # Test 1: MPS
    print("🍎 Test MPS...")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    
    if torch.backends.mps.is_available():
        device = "mps"
        print("   ✅ MPS activé")
    else:
        device = "cpu"
        print("   ⚠️  MPS non disponible")
    
    # Test 2: Settings Ultralytics
    print("⚙️  Test settings...")
    settings.update({"runs_dir": "runs", "datasets_dir": "dataset"})
    print("   ✅ Settings configurés")
    
    # Test 3: Dataset
    print("📊 Test dataset...")
    dataset_path = "dataset/multibd_enhanced.yaml"
    if os.path.exists(dataset_path):
        print(f"   ✅ Dataset trouvé: {dataset_path}")
    else:
        print(f"   ❌ Dataset manquant: {dataset_path}")
        return False
    
    # Test 4: Modèle YOLO avec patch PyTorch
    print("🤖 Test chargement YOLO...")
    
    # Patch PyTorch pour résoudre weights_only issue
    orig_load = torch.load
    def patched_load(f, map_location=None, pickle_module=None, weights_only=None, **kw):
        if weights_only is None:
            weights_only = False
        return orig_load(f, map_location=map_location, pickle_module=pickle_module,
                         weights_only=weights_only, **kw)
    torch.load = patched_load
    
    try:
        model = YOLO("yolov8n.pt")  # Plus petit pour test
        print("   ✅ YOLO chargé")
    except Exception as e:
        print(f"   ❌ Erreur YOLO: {e}")
        return False
    finally:
        torch.load = orig_load
    
    # Test 5: Mini entraînement
    print("🎯 Test mini-entraînement (1 epoch)...")
    try:
        results = model.train(
            data=dataset_path,
            device=device,
            imgsz=640,
            epochs=1,
            batch=1,
            workers=0,
            cache=False,
            project="runs/test_mps",
            name="test",
            exist_ok=True,
            verbose=False
        )
        print("   ✅ Mini-entraînement réussi!")
        return True
    except Exception as e:
        print(f"   ❌ Erreur entraînement: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}")
    sys.exit(0 if success else 1)
