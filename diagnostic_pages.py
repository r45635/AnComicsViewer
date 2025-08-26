#!/usr/bin/env python3
"""
Diagnostic précis du problème de détection
"""

import sys
import os
sys.path.insert(0, '.')

def diagnostic_pages_specifiques():
    """Teste les pages problématiques spécifiques"""
    
    print("🔍 DIAGNOSTIC PAGES SPÉCIFIQUES")
    print("=" * 50)
    
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QImage
        from PySide6.QtCore import QSizeF
        from src.ancomicsviewer.detectors.robust_yolo_detector import RobustYoloDetector
        
        app = QApplication(sys.argv)
        detector = RobustYoloDetector()
        
        # Test avec différentes confidence pour voir la sensibilité
        print("📋 TEST CONFIDENCE EXTREMES:")
        
        qimg = QImage('realistic_page.png')
        qimg_app = qimg.scaled(1239, 1754)  # Taille app réelle
        pt = QSizeF(595.0, 842.0)
        
        # Test 1: Confidence très basse (pour voir TOUT ce que détecte YOLO)
        detector._cfg.conf_panel = 0.01
        detector._cfg.drop_fullpage_ratio = 0.99  # Presque rien supprimé
        panels_ultra_low = detector.detect_panels(qimg_app, pt)
        print(f"Conf=0.01: {len(panels_ultra_low)} panels")
        
        # Test 2: Confidence haute (pour voir seulement les très sûrs)  
        detector._cfg.conf_panel = 0.50
        detector._cfg.drop_fullpage_ratio = 0.99
        panels_high = detector.detect_panels(qimg_app, pt)
        print(f"Conf=0.50: {len(panels_high)} panels")
        
        # Test 3: Sans aucun filtre post-processing
        detector._cfg.conf_panel = 0.20
        detector._cfg.drop_fullpage_ratio = 0.99  # Pas de suppression
        detector._cfg.min_area_ratio = 0.0001     # Pas de suppression
        panels_no_filter = detector.detect_panels(qimg_app, pt)
        print(f"Sans filtre: {len(panels_no_filter)} panels")
        
        print(f"\n📊 ANALYSE:")
        print(f"  Ultra-low conf: {len(panels_ultra_low)}")
        print(f"  High conf: {len(panels_high)}")  
        print(f"  Sans filtre: {len(panels_no_filter)}")
        
        if len(panels_ultra_low) == 0:
            print("❌ PROBLÈME: Même en confidence ultra-basse, rien détecté!")
            print("   → Le modèle ne voit rien sur cette image")
            
        if len(panels_high) > 0:
            print("✅ Le modèle détecte avec haute confidence")
        else:
            print("⚠️ Rien en haute confidence → détections peu fiables")
            
        # Test sur différentes tailles
        print(f"\n📏 TEST DIFFÉRENTES TAILLES:")
        
        sizes = [
            (400, 600, "Petite"),
            (800, 1200, "Originale"), 
            (1239, 1754, "App"),
            (1600, 2400, "Grande")
        ]
        
        detector._cfg.conf_panel = 0.15
        detector._cfg.drop_fullpage_ratio = 0.90
        
        for w, h, desc in sizes:
            qimg_test = qimg.scaled(w, h)
            panels_test = detector.detect_panels(qimg_test)
            print(f"  {desc} ({w}x{h}): {len(panels_test)} panels")
            
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnostic_pages_specifiques()
