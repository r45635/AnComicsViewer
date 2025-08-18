#!/usr/bin/env python3
"""
Test rapide du détecteur Multi-BD amélioré intégré.
"""

import sys
import os
sys.path.append('.')

def test_improved_multibd():
    """Test du détecteur amélioré."""
    try:
        # Import du détecteur de base
        from detectors.multibd_detector import MultiBDPanelDetector
        print("✅ Import MultiBDPanelDetector réussi")
        
        # Classe améliorée (comme dans AnComicsViewer.py)
        class ImprovedMultiBDDetector(MultiBDPanelDetector):
            def __init__(self, **kwargs):
                super().__init__(conf=0.15, iou=0.4, **kwargs)
                print(f"✅ ImprovedMultiBDDetector initialisé (conf={self.conf}, iou={self.iou})")
                
            def detect_panels(self, qimage, page_point_size):
                raw_panels = super().detect_panels(qimage, page_point_size)
                if not raw_panels:
                    return []
                    
                print(f"🔍 Détections brutes: {len(raw_panels)}")
                
                filtered = []
                page_height = page_point_size.height()
                page_width = page_point_size.width()
                page_area = page_width * page_height
                
                for panel in raw_panels:
                    # Filtrer zone titre (25% du haut)
                    if panel.y() < page_height * 0.25:
                        aspect_ratio = panel.width() / panel.height()
                        if aspect_ratio > 4.0:  # Ligne de texte probable
                            continue
                        if panel.width() < page_width * 0.3:  # Trop étroit pour titre zone
                            continue
                            
                    # Filtrer par taille (0.8% minimum de la page)
                    if panel.width() * panel.height() < page_area * 0.008:
                        continue
                        
                    # Filtrer par ratio aspect anormal
                    aspect_ratio = panel.width() / panel.height()
                    if aspect_ratio > 4.0 or aspect_ratio < 0.2:
                        continue
                        
                    filtered.append(panel)
                
                print(f"✅ Détections filtrées: {len(filtered)}")
                return filtered
        
        # Test d'instanciation
        detector = ImprovedMultiBDDetector()
        print("✅ Détecteur Multi-BD amélioré créé avec succès")
        
        # Test des paramètres
        print(f"🎯 Paramètres: conf={detector.conf}, iou={detector.iou}")
        
        return True
        
    except ImportError as e:
        if "matplotlib" in str(e):
            print("❌ matplotlib manquant")
            print("💡 Solution: .venv/bin/pip install matplotlib")
        else:
            print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Test détecteur Multi-BD amélioré")
    print("=" * 40)
    
    success = test_improved_multibd()
    
    if success:
        print("\n✅ Tous les tests passés!")
        print("🚀 Le détecteur Multi-BD amélioré est prêt")
    else:
        print("\n❌ Échec des tests")
        
    sys.exit(0 if success else 1)
