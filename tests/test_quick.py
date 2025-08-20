#!/usr/bin/env python3
"""
Multi-BD Enhanced v2 - Test Simple du Modèle Entraîné
====================================================
Script pour tester rapidement le détecteur optimisé.
"""

import os
import torch
from pathlib import Path

def main():
    """Test rapide du modèle entraîné"""
    print("🧪 Test Multi-BD Enhanced v2")
    print("=" * 40)
    
    try:
        # Configuration PyTorch 2.8
        import torch.serialization
        import ultralytics.nn.tasks
        torch.serialization.add_safe_globals([
            ultralytics.nn.tasks.DetectionModel,
            ultralytics.nn.tasks.SegmentationModel,
            ultralytics.nn.tasks.ClassificationModel,
            ultralytics.nn.tasks.PoseModel,
            ultralytics.nn.tasks.OBBModel
        ])
        
        from ultralytics import YOLO
        
        # Configuration MPS
        if torch.backends.mps.is_available():
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            device = 'mps'
            print("🍎 Device: MPS (Apple Silicon)")
        else:
            device = 'cpu'
            print("💻 Device: CPU")
        
        # Chargement du meilleur modèle
        best_model = "runs/multibd_enhanced_v2/yolov8s-final-optimized/weights/best.pt"
        
        if not Path(best_model).exists():
            print(f"❌ Modèle non trouvé: {best_model}")
            print("📋 Modèles disponibles:")
            for pt_file in Path("runs").rglob("*.pt"):
                print(f"   • {pt_file}")
            return False
        
        print(f"📦 Chargement: {best_model}")
        model = YOLO(best_model)
        
        # Test de validation rapide
        print("🔍 Validation sur dataset...")
        results = model.val(
            data='dataset/multibd_enhanced.yaml',
            device=device,
            imgsz=640,  # Taille réduite pour test rapide
            conf=0.15,
            iou=0.60,
            max_det=200,
            verbose=False
        )
        
        # Affichage des résultats
        print("\n📊 Résultats:")
        if hasattr(results, 'box') and results.box is not None:
            print(f"   📈 mAP50: {results.box.map50:.3f} ({results.box.map50*100:.1f}%)")
            print(f"   📈 mAP50-95: {results.box.map:.3f} ({results.box.map*100:.1f}%)")
            print(f"   📈 Précision: {results.box.mp:.3f}")
            print(f"   📈 Rappel: {results.box.mr:.3f}")
        else:
            print("   ⚠️  Métriques non disponibles")
        
        # Test sur une image
        val_images = list(Path("dataset/yolo/images/val").glob("*.jpg"))
        if val_images:
            test_image = val_images[0]
            print(f"\n🖼️  Test image: {test_image.name}")
            
            pred_results = model.predict(
                source=str(test_image),
                device=device,
                imgsz=1280,
                conf=0.15,
                iou=0.60,
                max_det=200,
                save=True,
                save_dir="runs/test_predictions",
                verbose=False
            )
            
            if pred_results and len(pred_results) > 0:
                result = pred_results[0]
                if result.boxes is not None:
                    classes = result.boxes.cls
                    if hasattr(classes, 'cpu'):
                        classes = classes.cpu().numpy()
                    
                    num_panels = sum(classes == 0)
                    num_insets = sum(classes == 1)
                    print(f"   📋 Panneaux détectés: {num_panels}")
                    print(f"   📋 Inserts détectés: {num_insets}")
                    print(f"   💾 Image sauvée: runs/test_predictions/")
                else:
                    print("   ⚠️  Aucune détection")
        
        print("\n✅ Test terminé avec succès!")
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
