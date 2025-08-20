#!/usr/bin/env python3
"""
Script d'entraînement Multi-BD Enhanced v2 - Version Optimisée MPS
Utilise Apple Silicon MPS et optimisations avancées pour éviter les timeouts NMS
"""

import os
import sys
import torch
from ultralytics import YOLO, settings

def main():
    """Entraînement optimisé avec MPS pour éviter les timeouts NMS."""
    print("🚀 Multi-BD Enhanced v2 - Entraînement Optimisé MPS")
    print("=" * 60)
    
    # 1) Configuration MPS / Apple Silicon pour éviter timeouts
    print("🍎 Configuration Apple Silicon MPS...")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    
    # Vérification MPS
    if torch.backends.mps.is_available():
        device = "mps"
        print("   ✅ Apple Silicon MPS activé")
        print(f"   🔧 MPS Fallback: {os.getenv('PYTORCH_ENABLE_MPS_FALLBACK')}")
    else:
        device = "cpu"
        print("   ⚠️  MPS non disponible, utilisation CPU")
    
    # 2) Configuration répertoires Ultralytics
    settings.update({
        "runs_dir": "runs", 
        "datasets_dir": "dataset"
    })
    
    # 3) Vérification dataset
    dataset_path = "dataset/multibd_enhanced.yaml"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset non trouvé: {dataset_path}")
        return False
    
    print(f"📊 Dataset: {dataset_path}")
    
    # 4) Modèle de base avec patch PyTorch pour YOLO
    print("🤖 Chargement modèle YOLOv8s...")
    
    # Patch PyTorch pour résoudre weights_only issue avec YOLO
    orig_load = torch.load
    def patched_load(f, map_location=None, pickle_module=None, weights_only=None, **kw):
        if weights_only is None:
            weights_only = False
        return orig_load(f, map_location=map_location, pickle_module=pickle_module,
                         weights_only=weights_only, **kw)
    torch.load = patched_load
    
    try:
        model = YOLO("yolov8s.pt")
        print("   ✅ YOLOv8s chargé avec succès")
    finally:
        torch.load = orig_load
    
    # 5) Entraînement optimisé - Paramètres pour bordures nettes + NMS rapide
    print("🎯 Démarrage entraînement avec optimisations MPS...")
    
    try:
        results = model.train(
            # Dataset et device
            data=dataset_path,
            device=device,
            
            # Résolution et batch optimisés pour MPS
            imgsz=1280,               # Plus grand = bordures plus précises
            epochs=200,
            batch=16,                 # Équilibré pour RAM unifiée Apple Silicon
            workers=0,                # macOS: évite surcharge dataloaders
            
            # Performance et cache
            cache=True,
            seed=42,
            patience=50,
            
            # Répertoire résultats
            project="runs/multibd_enhanced_v2",
            name="yolov8s-mps-1280",
            exist_ok=True,
            
            # NMS optimisé pour éviter timeouts
            conf=0.15,                # Seuil conf plus haut (au lieu de ~0.001)
            iou=0.60,                 # IoU plus strict pour NMS
            max_det=200,              # Borne le nb de boxes après NMS
            
            # Augmentations douces (préserve lignes droites des BD)
            mosaic=0.10,              # Réduit pour garder structure panels
            mixup=0.0,                # Évite artifacts sur cases
            close_mosaic=10,          # Ferme mosaic tôt
            
            # Transformations géométriques limitées
            degrees=0.0,              # Pas de rotation (BD = lignes droites)
            shear=0.0,                # Pas de cisaillement
            perspective=0.0,          # Pas de perspective
            translate=0.02,           # Translation très légère
            scale=0.50,               # Échelle modérée
            
            # Flips et couleur
            flipud=0.0,               # Pas de flip vertical (BD sens lecture)
            fliplr=0.5,               # Flip horizontal OK
            hsv_h=0.015,              # Variation teinte légère
            hsv_s=0.40,               # Saturation modérée
            hsv_v=0.40,               # Luminosité modérée
            
            # Génération de résultats
            plots=True,               # Génère graphiques résultats
            verbose=True
        )
        
        print("✅ Entraînement terminé avec succès!")
        
        # 6) Validation explicite sur même device pour éviter NMS CPU
        print("🧪 Validation finale sur MPS...")
        val_results = model.val(
            device=device,
            imgsz=1280,
            workers=0,
            conf=0.15,                # Cohérent avec l'entraînement
            iou=0.60,                 # Cohérent avec l'entraînement
            max_det=200,              # Borne les boxes après NMS
            verbose=False
        )
        
        # 7) Copie du meilleur modèle
        best_weights = "runs/multibd_enhanced_v2/yolov8s-mps-1280/weights/best.pt"
        if os.path.exists(best_weights):
            import shutil
            target_path = "detectors/models/multibd_enhanced_v2.pt"
            os.makedirs("detectors/models", exist_ok=True)
            shutil.copy2(best_weights, target_path)
            print(f"🎯 Modèle copié vers: {target_path}")
        
        # 8) Affichage résultats
        if hasattr(val_results, 'box'):
            mAP50 = val_results.box.map50
            mAP = val_results.box.map
            print(f"📈 mAP50: {mAP50:.3f}")
            print(f"📈 mAP50-95: {mAP:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur pendant l'entraînement: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
