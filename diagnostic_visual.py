#!/usr/bin/env python3
"""
Diagnostic VISUEL des détections de panels
==========================================
Analyse pourquoi les panels ne sont pas détectés aux bons endroits
"""

import sys
import os
sys.path.insert(0, '.')

import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage, QPainter, QPen, QColor, QFont
from PySide6.QtCore import Qt, QRectF

def diagnostic_visual_complet():
    """Diagnostic visuel complet avec sauvegarde d'image annotée"""
    
    # Créer QApplication si nécessaire
    if not QApplication.instance():
        app = QApplication(sys.argv)
    
    print("🔍 DIAGNOSTIC VISUEL DÉTECTION PANELS")
    print("=" * 60)
    
    try:
        from src.ancomicsviewer.detectors.robust_yolo_detector import RobustYoloDetector
        
        # 1. Créer détecteur avec logs maximaux
        print("🔥 Création détecteur avec logs maximaux...")
        detector = RobustYoloDetector()
        
        # 2. Charger image test
        test_images = ['realistic_page.png', 'test_page.png', 'sample.png']
        qimg = None
        img_path = None
        
        for test_img in test_images:
            if os.path.exists(test_img):
                qimg = QImage(test_img)
                if not qimg.isNull():
                    img_path = test_img
                    break
        
        if qimg is None or qimg.isNull():
            print("❌ Aucune image test trouvée!")
            return
            
        print(f"🖼️ Image chargée: {img_path}")
        print(f"📏 Dimensions: {qimg.width()}x{qimg.height()}")
        print(f"🎨 Format: {qimg.format()}")
        
        # 3. Détection avec logs complets
        print("\n🎯 DÉTECTION AVEC LOGS COMPLETS:")
        print("-" * 40)
        panels = detector.detect_panels(qimg)
        
        print(f"\n📊 RÉSULTATS:")
        print(f"✅ Panels détectés: {len(panels)}")
        
        # 4. Analyse détaillée de chaque panel
        print(f"\n📋 ANALYSE DÉTAILLÉE DES PANELS:")
        print("-" * 40)
        
        for i, panel in enumerate(panels):
            x, y, w, h = panel.x(), panel.y(), panel.width(), panel.height()
            print(f"Panel {i+1}:")
            print(f"  Position: ({x:.1f}, {y:.1f})")
            print(f"  Taille: {w:.1f} x {h:.1f}")
            print(f"  Ratio position: ({x/qimg.width():.3f}, {y/qimg.height():.3f})")
            print(f"  Ratio taille: ({w/qimg.width():.3f}, {h/qimg.height():.3f})")
            
            # Vérifications de cohérence
            if x < 0 or y < 0:
                print(f"  ⚠️ PROBLÈME: Position négative!")
            if x + w > qimg.width() or y + h > qimg.height():
                print(f"  ⚠️ PROBLÈME: Panel dépasse de l'image!")
            if w < 10 or h < 10:
                print(f"  ⚠️ PROBLÈME: Panel trop petit!")
            if w > qimg.width() * 0.9 or h > qimg.height() * 0.9:
                print(f"  ⚠️ PROBLÈME: Panel trop grand (probable faux positif)!")
            print()
        
        # 5. Créer image annotée pour visualisation
        print("🎨 Création image annotée pour debug...")
        annotated_img = qimg.copy()
        painter = QPainter(annotated_img)
        
        # Styles pour annotation
        pen_panel = QPen(QColor(255, 0, 0), 3)  # Rouge épais pour panels
        pen_text = QPen(QColor(255, 255, 0), 2)  # Jaune pour texte
        font = QFont("Arial", 16, QFont.Weight.Bold)
        painter.setFont(font)
        
        # Dessiner chaque panel
        for i, panel in enumerate(panels):
            # Rectangles des panels
            painter.setPen(pen_panel)
            painter.drawRect(panel)
            
            # Numéros des panels
            painter.setPen(pen_text)
            text_x = panel.x() + 5
            text_y = panel.y() + 20
            painter.drawText(int(text_x), int(text_y), f"P{i+1}")
            
            # Coordonnées détaillées
            coord_text = f"({panel.x():.0f},{panel.y():.0f})"
            painter.drawText(int(text_x), int(text_y + 25), coord_text)
        
        # Informations générales sur l'image
        painter.setPen(QPen(QColor(0, 255, 0), 2))
        info_text = f"IMG: {qimg.width()}x{qimg.height()} | PANELS: {len(panels)}"
        painter.drawText(10, qimg.height() - 20, info_text)
        
        painter.end()
        
        # Sauvegarder image annotée
        base_name = os.path.splitext(img_path)[0] if img_path else "test"
        debug_path = f"DEBUG_PANELS_{base_name}_annotated.png"
        annotated_img.save(debug_path)
        print(f"💾 Image debug sauvée: {debug_path}")
        
        # 6. Test avec différents paramètres
        print(f"\n🔬 TEST AVEC PARAMÈTRES ALTERNATIFS:")
        print("-" * 40)
        
        # Test avec confidence plus basse
        print("Test confidence plus basse...")
        detector._cfg.conf_panel = 0.10
        detector._cfg.conf_inset = 0.10
        panels_low_conf = detector.detect_panels(qimg)
        print(f"✅ Panels (conf=0.10): {len(panels_low_conf)}")
        
        # Test avec confidence plus haute
        print("Test confidence plus haute...")
        detector._cfg.conf_panel = 0.40
        detector._cfg.conf_inset = 0.40
        panels_high_conf = detector.detect_panels(qimg)
        print(f"✅ Panels (conf=0.40): {len(panels_high_conf)}")
        
        # Comparaison
        print(f"\n📈 COMPARAISON PARAMÈTRES:")
        print(f"  Conf normale (0.20): {len(panels)} panels")
        print(f"  Conf basse (0.10): {len(panels_low_conf)} panels")
        print(f"  Conf haute (0.40): {len(panels_high_conf)} panels")
        
        print(f"\n🎯 DIAGNOSTIC TERMINÉ!")
        print(f"📁 Vérifiez l'image: {debug_path}")
        
    except Exception as e:
        print(f"❌ ERREUR DIAGNOSTIC: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnostic_visual_complet()
