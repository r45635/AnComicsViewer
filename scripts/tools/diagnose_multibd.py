#!/usr/bin/env python3
"""
Diagnostic et amélioration du modèle Multi-BD
============================================

Ce script analyse les problèmes du modèle Multi-BD et propose des améliorations :
1. Diagnostic des classes mal prédites
2. Analyse de la précision des contours 
3. Amélioration des paramètres de détection
4. Test de différents modèles disponibles
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import json

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from detectors.multibd_detector import MultiBDPanelDetector
    from ultralytics import YOLO
    import torch
except ImportError as e:
    print(f"❌ Dépendances manquantes: {e}")
    print("💡 Installez avec: pip install ultralytics torch")
    sys.exit(1)

class MultiBDDiagnostic:
    """Outil de diagnostic et amélioration du modèle Multi-BD."""
    
    def __init__(self):
        self.models_dir = Path("runs/detect")
        self.dataset_dir = Path("dataset")
        self.available_models = self._find_available_models()
        
    def _find_available_models(self) -> Dict[str, Path]:
        """Trouve tous les modèles disponibles."""
        models = {}
        for model_dir in self.models_dir.glob("*/weights/best.pt"):
            name = model_dir.parent.parent.name
            models[name] = model_dir
        return models
    
    def list_available_models(self):
        """Affiche tous les modèles disponibles."""
        print("🔍 Modèles disponibles:")
        for name, path in self.available_models.items():
            size = path.stat().st_size / (1024*1024)  # MB
            print(f"  📦 {name}: {path} ({size:.1f} MB)")
        print()
    
    def analyze_model_performance(self, model_name: str = "multibd_mixed_model"):
        """Analyse les performances d'un modèle."""
        if model_name not in self.available_models:
            print(f"❌ Modèle '{model_name}' non trouvé")
            return
            
        model_path = self.available_models[model_name]
        print(f"🔬 Analyse du modèle: {model_name}")
        print(f"📂 Chemin: {model_path}")
        
        try:
            # Charger le modèle
            detector = MultiBDPanelDetector(weights=str(model_path))
            
            # Afficher les infos du modèle
            info = detector.get_model_info()
            print(f"📊 Performance rapportée:")
            for key, value in info['performance'].items():
                print(f"  {key}: {value}")
            
            print(f"🎯 Classes: {info['classes']}")
            print(f"🔧 Confidence: {info['confidence']}")
            print(f"🔧 IoU: {info['iou_threshold']}")
            
        except Exception as e:
            print(f"❌ Erreur chargement modèle: {e}")
    
    def test_detection_parameters(self, model_name: str = "multibd_mixed_model"):
        """Test différents paramètres de détection."""
        if model_name not in self.available_models:
            print(f"❌ Modèle '{model_name}' non trouvé")
            return
            
        model_path = self.available_models[model_name]
        print(f"🧪 Test des paramètres de détection")
        
        # Paramètres à tester
        conf_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        iou_values = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        # Chercher une image de test
        test_images = list(self.dataset_dir.glob("**/train/*.png"))[:3]  # 3 premières images
        
        if not test_images:
            print("❌ Aucune image de test trouvée dans dataset/")
            return
            
        results = []
        
        for conf in conf_values:
            for iou in iou_values:
                try:
                    detector = MultiBDPanelDetector(
                        weights=str(model_path),
                        conf=conf,
                        iou=iou
                    )
                    
                    total_detections = 0
                    for img_path in test_images:
                        # Charger l'image avec cv2
                        img = cv2.imread(str(img_path))
                        if img is None:
                            continue
                            
                        # Convertir en QImage (simulation)
                        h, w = img.shape[:2]
                        
                        # Simulation simple - on teste juste le modèle YOLO directement
                        yolo_results = detector.model.predict(
                            source=img,
                            conf=conf,
                            iou=iou,
                            verbose=False
                        )[0]
                        
                        if yolo_results.boxes is not None:
                            total_detections += len(yolo_results.boxes)
                    
                    avg_detections = total_detections / len(test_images)
                    results.append({
                        'conf': conf,
                        'iou': iou,
                        'avg_detections': avg_detections
                    })
                    
                    print(f"  conf={conf:.1f}, iou={iou:.1f} → {avg_detections:.1f} détections/image")
                    
                except Exception as e:
                    print(f"  ❌ Erreur conf={conf:.1f}, iou={iou:.1f}: {e}")
        
        # Trouver les meilleurs paramètres
        if results:
            # Trier par nombre de détections (on veut un équilibre)
            results.sort(key=lambda x: abs(x['avg_detections'] - 5.0))  # Idéal ~5 panels/page
            best = results[0]
            print(f"\n🎯 Paramètres optimaux suggérés:")
            print(f"  Confidence: {best['conf']}")
            print(f"  IoU: {best['iou']}")
            print(f"  Détections moyennes: {best['avg_detections']:.1f}")
    
    def suggest_improvements(self):
        """Propose des améliorations pour le modèle."""
        print("💡 Suggestions d'amélioration du modèle Multi-BD:\n")
        
        print("🎯 1. PROBLÈME: Mauvaise différenciation titre/cases")
        print("   Solutions:")
        print("   - Augmenter les données d'entraînement avec plus d'exemples de titres")
        print("   - Ajouter une classe 'title' ou 'text' au modèle")
        print("   - Utiliser des seuils de confiance différents par classe")
        print("   - Post-traitement: filtrer les détections en haut de page (zone titre)")
        print()
        
        print("🎯 2. PROBLÈME: Précision des contours")
        print("   Solutions:")
        print("   - Utiliser YOLO segmentation (YOLOv8-seg) au lieu de detection")
        print("   - Augmenter la résolution d'entraînement (640→832 ou 1024)")
        print("   - Post-traitement: raffiner les contours avec OpenCV")
        print("   - Utiliser plusieurs échelles de détection")
        print()
        
        print("🎯 3. AMÉLIORATIONS IMMÉDIATES")
        print("   A. Ajuster les paramètres:")
        print("      - Baisser confidence pour détecter plus (0.2 → 0.15)")
        print("      - Ajuster IoU selon le style (0.5 → 0.4 pour plus de chevauchement)")
        print()
        
        print("   B. Post-traitement intelligent:")
        print("      - Filtrer par position (éviter zone titre en haut)")
        print("      - Filtrer par ratio aspect (éviter les lignes de texte)")
        print("      - Merger les détections proches")
        print()
        
        print("   C. Modèle hybride:")
        print("      - Combiner Multi-BD + Heuristique OpenCV")
        print("      - Utiliser Multi-BD pour détecter, OpenCV pour raffiner")
        print()
        
    def create_improved_detector(self) -> str:
        """Crée un détecteur Multi-BD amélioré."""
        improved_code = '''
class ImprovedMultiBDDetector(MultiBDPanelDetector):
    """Détecteur Multi-BD avec améliorations pour titre/contours."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Paramètres optimisés
        self.conf = 0.15  # Plus sensible
        self.iou = 0.4    # Plus de chevauchement autorisé
        
    def detect_panels(self, qimage, page_point_size):
        """Détection avec post-traitement amélioré."""
        # Détection de base
        raw_panels = super().detect_panels(qimage, page_point_size)
        
        # Post-traitement intelligent
        filtered_panels = []
        page_height = page_point_size.height()
        
        for panel in raw_panels:
            # Filtrer zone titre (20% du haut de la page)
            if panel.y() < page_height * 0.2:
                # Vérifier le ratio aspect pour différencier titre/case
                aspect_ratio = panel.width() / panel.height()
                if aspect_ratio > 4.0:  # Ligne de texte probable
                    continue
                    
            # Filtrer les détections trop petites
            min_area = page_height * page_point_size.width() * 0.01  # 1% de la page
            if panel.width() * panel.height() < min_area:
                continue
                
            # Filtrer les ratios aspect anormaux 
            aspect_ratio = panel.width() / panel.height()
            if aspect_ratio > 6.0 or aspect_ratio < 0.15:  # Trop large ou trop étroit
                continue
                
            filtered_panels.append(panel)
        
        return filtered_panels
        '''
        
        # Sauvegarder le code amélioré
        output_path = Path("tools/improved_multibd_detector.py")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(improved_code)
            
        print(f"💾 Détecteur amélioré sauvé: {output_path}")
        return str(output_path)

def main():
    """Point d'entrée principal."""
    print("🔬 AnComicsViewer - Diagnostic Multi-BD")
    print("=" * 50)
    
    diagnostic = MultiBDDiagnostic()
    
    # 1. Lister les modèles disponibles
    diagnostic.list_available_models()
    
    # 2. Analyser le modèle principal
    diagnostic.analyze_model_performance("multibd_mixed_model")
    print()
    
    # 3. Tester différents paramètres
    print("🧪 Test des paramètres de détection:")
    diagnostic.test_detection_parameters("multibd_mixed_model")
    print()
    
    # 4. Proposer des améliorations
    diagnostic.suggest_improvements()
    
    # 5. Créer un détecteur amélioré
    diagnostic.create_improved_detector()

if __name__ == "__main__":
    main()
