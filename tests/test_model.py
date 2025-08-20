#!/usr/bin/env python3
"""
Multi-BD Enhanced v2 - Test du Modèle Entraîné
==============================================
Script pour tester le détecteur optimisé avec post-processing.
"""

import os
import sys
import torch
from pathlib import Path
import cv2
import numpy as np

def setup_environment():
    """Configuration de l'environnement de test"""
    print("🧪 Multi-BD Enhanced v2 - Test du Modèle Entraîné")
    print("=" * 55)
    
    # Configuration MPS
    if torch.backends.mps.is_available():
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        print("🍎 Device: MPS (Apple Silicon)")
    else:
        print("💻 Device: CPU/CUDA")

def test_model():
    """Test du modèle entraîné"""
    try:
        # Configuration des globals sûrs pour PyTorch 2.8
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
        
        # Chemins des modèles
        best_model = "runs/multibd_enhanced_v2/yolov8s-final-optimized/weights/best.pt"
        last_model = "runs/multibd_enhanced_v2/yolov8s-final-optimized/weights/last.pt"
        
        # Utiliser le meilleur modèle si disponible
        model_path = best_model if Path(best_model).exists() else last_model
        
        if not Path(model_path).exists():
            print(f"❌ Modèle non trouvé: {model_path}")
            return False
        
        print(f"📦 Chargement modèle: {model_path}")
        model = YOLO(model_path)
        
        # Device
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        # Test sur dataset de validation
        print("🔍 Test sur dataset de validation...")
        results = model.val(
            data='dataset/multibd_enhanced.yaml',
            device=device,
            imgsz=1280,
            conf=0.15,
            iou=0.60,
            max_det=200,
            verbose=True
        )
        
        print("\n📊 Résultats de validation:")
        if hasattr(results, 'box'):
            print(f"   📈 mAP50: {results.box.map50:.3f}")
            print(f"   📈 mAP50-95: {results.box.map50_95:.3f}")
            print(f"   📈 Précision: {results.box.mp:.3f}")
            print(f"   📈 Rappel: {results.box.mr:.3f}")
        
        return model, results
        
    except Exception as e:
        print(f"❌ Erreur test modèle: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_single_image(model, image_path):
    """Test sur une image unique"""
    try:
        print(f"\n🖼️  Test sur image: {image_path}")
        
        # Prédiction
        results = model.predict(
            source=image_path,
            device='mps' if torch.backends.mps.is_available() else 'cpu',
            imgsz=1280,
            conf=0.15,
            iou=0.60,
            max_det=200,
            save=True,
            save_dir="runs/test_predictions"
        )
        
        if results and len(results) > 0:
            result = results[0]
            num_detections = len(result.boxes) if result.boxes is not None else 0
            print(f"   ✅ {num_detections} détections trouvées")
            
            if result.boxes is not None:
                # Statistiques par classe
                classes = result.boxes.cls.cpu().numpy()
                class_names = ['panel', 'panel_inset']
                
                for class_id in [0, 1]:
                    count = np.sum(classes == class_id)
                    if count > 0:
                        print(f"   📋 {class_names[class_id]}: {count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur test image: {e}")
        return False

def test_with_postprocessing(model):
    """Test avec post-processing intégré"""
    try:
        print("\n🔧 Test avec post-processing...")
        
        # Import des modules de post-processing
        sys.path.append('detectors')
        from postproc import enhance_panel_borders
        from reading_order import sort_reading_order
        
        # Test sur une image de validation
        val_images = list(Path("dataset/yolo/images/val").glob("*.jpg"))
        if not val_images:
            print("❌ Aucune image de validation trouvée")
            return False
        
        test_image = val_images[0]
        print(f"🖼️  Image test: {test_image.name}")
        
        # Prédiction
        results = model.predict(
            source=str(test_image),
            device='mps' if torch.backends.mps.is_available() else 'cpu',
            imgsz=1280,
            conf=0.15,
            iou=0.60,
            max_det=200,
            verbose=False
        )
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None:
                # Conversion vers format numpy
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                print(f"   📋 Détections brutes: {len(boxes)}")
                
                # Post-processing: amélioration des bordures
                enhanced_boxes = enhance_panel_borders(boxes, result.orig_img)
                print(f"   🔧 Après amélioration bordures: {len(enhanced_boxes)}")
                
                # Post-processing: ordre de lecture
                if len(enhanced_boxes) > 1:
                    sorted_indices = sort_reading_order(enhanced_boxes)
                    sorted_boxes = enhanced_boxes[sorted_indices]
                    print(f"   📖 Ordre de lecture appliqué: {len(sorted_boxes)} panneaux")
                
                print("   ✅ Post-processing terminé avec succès")
                return True
        
        print("   ⚠️  Aucune détection pour le post-processing")
        return True
        
    except Exception as e:
        print(f"❌ Erreur post-processing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Fonction principale de test"""
    setup_environment()
    
    # Test du modèle
    model, val_results = test_model()
    if model is None:
        print("\n❌ Échec du test de validation")
        return False
    
    # Test sur image unique
    val_images = list(Path("dataset/yolo/images/val").glob("*.jpg"))
    if val_images:
        success = test_single_image(model, val_images[0])
        if not success:
            print("\n⚠️  Échec du test image unique")
    
    # Test avec post-processing
    success = test_with_postprocessing(model)
    if not success:
        print("\n⚠️  Échec du test post-processing")
    
    print("\n🎉 Tests terminés!")
    print("\n📁 Résultats sauvegardés dans:")
    print("   • runs/test_predictions/ (prédictions)")
    print("   • runs/multibd_enhanced_v2/ (modèles)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
