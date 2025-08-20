#!/usr/bin/env python3
"""
Multi-BD Enhanced v2 - Script d'Entraînement Final Optimisé
===========================================================
Entraînement optimisé avec support MPS et prévention des timeouts NMS.
Version finale après validation du pipeline complet.
"""

import os
import sys
import torch
import warnings
from pathlib import Path

def setup_environment():
    """Configuration optimale de l'environnement d'entraînement"""
    print("🚀 Multi-BD Enhanced v2 - Entraînement Final Optimisé")
    print("=" * 60)
    
    # Configuration Apple Silicon MPS optimisée
    print("🍎 Configuration Apple Silicon MPS...")
    if torch.backends.mps.is_available():
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        print("   ✅ Apple Silicon MPS activé")
        print(f"   🔧 MPS Fallback: {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')}")
    else:
        print("   ⚠️  MPS non disponible, utilisation CPU/CUDA")
    
    # Optimisations PyTorch
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Suppression des warnings non critiques
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    return True

def patch_ultralytics():
    """Patch pour résoudre le problème weights_only avec YOLO"""
    try:
        import ultralytics.nn.tasks
        original_torch_load = torch.load
        
        def patched_torch_load(*args, **kwargs):
            kwargs.pop('weights_only', None)  # Supprime weights_only si présent
            return original_torch_load(*args, **kwargs)
        
        torch.load = patched_torch_load
        print("   🔧 Patch PyTorch load appliqué")
        return True
    except Exception as e:
        print(f"   ⚠️  Patch PyTorch: {e}")
        return False

def main():
    """Fonction principale d'entraînement optimisée"""
    try:
        # Configuration environnement
        setup_environment()
        
        # Configuration dataset
        dataset_path = "dataset/multibd_enhanced.yaml"
        print(f"📊 Dataset: {dataset_path}")
        
        # Patch Ultralytics
        patch_ultralytics()
        
        # Import YOLO après configuration
        from ultralytics import YOLO
        
        # Chargement modèle
        print("🤖 Chargement modèle YOLOv8s...")
        model = YOLO('yolov8s.pt')
        print("   ✅ YOLOv8s chargé avec succès")
        
        # Détection device optimal
        device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🎯 Device sélectionné: {device}")
        
        # Configuration d'entraînement optimisée
        print("🎯 Démarrage entraînement avec optimisations complètes...")
        
        # Entraînement avec paramètres optimisés
        results = model.train(
            # Dataset et projet
            data=dataset_path,
            project='runs/multibd_enhanced_v2',
            name='yolov8s-final-optimized',
            
            # Paramètres d'entraînement
            epochs=200,
            batch=16,
            imgsz=1280,
            device=device,
            
            # Optimisations apprentissage
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            
            # Optimisations NMS (éviter timeouts)
            conf=0.15,        # Seuil de confiance optimisé
            iou=0.60,         # IoU threshold pour NMS
            max_det=200,      # Limite détections par image
            
            # Augmentations données
            hsv_h=0.015,
            hsv_s=0.4,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.02,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=0.1,       # Mosaic réduit pour éviter artefacts
            mixup=0.0,
            copy_paste=0.0,
            
            # Configuration entraînement
            cache='ram',
            workers=0,        # Optimisé pour MPS
            patience=50,
            save_period=-1,
            seed=42,
            deterministic=True,
            
            # Options de sortie
            plots=True,
            val=True,
            save=True,
            exist_ok=True,
            verbose=True
        )
        
        print("\n🎉 Entraînement terminé avec succès!")
        if hasattr(results, 'save_dir'):
            print(f"📁 Résultats sauvegardés dans: {results.save_dir}")
        
        # Affichage métriques finales
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print("\n📊 Métriques finales:")
            print(f"   📈 mAP50: {metrics.get('metrics/mAP50(B)', 'N/A')}")
            print(f"   📈 mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
        
        return results
        
    except KeyboardInterrupt:
        print("\n⏹️  Entraînement interrompu par l'utilisateur")
        return None
    except Exception as e:
        print(f"\n❌ Erreur pendant l'entraînement: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Vérification prérequis
    if not Path("dataset/multibd_enhanced.yaml").exists():
        print("❌ Dataset non trouvé: dataset/multibd_enhanced.yaml")
        sys.exit(1)
    
    # Lancement entraînement
    results = main()
    
    if results:
        print("\n✅ Script d'entraînement terminé avec succès")
        sys.exit(0)
    else:
        print("\n❌ Script d'entraînement terminé avec erreurs")
        sys.exit(1)
