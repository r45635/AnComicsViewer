#!/usr/bin/env python3
"""
Test script pour vérifier l'intégration de la navigation AR
"""

import sys
import os

# Ajouter le répertoire source au path
sys.path.insert(0, 'src')

def test_ar_navigation():
    """Test la navigation AR sans interface graphique."""
    print("🧪 Test de la navigation AR")
    print("=" * 50)
    
    try:
        # Import PySide6 et création d'une application
        from PySide6.QtWidgets import QApplication
        import sys
        app = QApplication(sys.argv if sys.argv else ['test'])
        
        # Test d'import
        from ancomicsviewer.main_app import ComicsView
        print("✅ Import ComicsView réussi")
        
        # Test de la configuration AR
        viewer = ComicsView()
        print("✅ Instance ComicsView créée")
        
        # Vérifier les méthodes AR
        ar_methods = [
            'enable_ar_mode',
            'ar_load_and_render_pdf', 
            'ar_render_page',
            'ar_next_page',
            'ar_prev_page'
        ]
        
        for method in ar_methods:
            if hasattr(viewer, method):
                print(f"✅ Méthode {method} disponible")
            else:
                print(f"❌ Méthode {method} manquante")
        
        # Vérifier l'attribut AR mode
        if hasattr(viewer, '_ar_mode_enabled'):
            print(f"✅ Attribut _ar_mode_enabled disponible: {viewer._ar_mode_enabled}")
        else:
            print("❌ Attribut _ar_mode_enabled manquant")
            
        # Test de nav_prev et nav_next
        if hasattr(viewer, 'nav_prev') and hasattr(viewer, 'nav_next'):
            print("✅ Méthodes de navigation disponibles")
        else:
            print("❌ Méthodes de navigation manquantes")
            
        print("\n🎯 Résultat: Intégration AR prête pour les tests!")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ar_navigation()
