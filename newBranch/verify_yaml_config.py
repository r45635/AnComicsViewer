#!/usr/bin/env python3
"""
Vérification finale des paramètres YAML ajustés
"""

import os
import sys
import yaml

def check_yaml_parameters():
    """Vérifie que tous les paramètres requis sont dans le YAML avec les bonnes valeurs"""
    
    config_path = "config/detect_with_merge.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Fichier YAML non trouvé: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Paramètres requis avec leurs valeurs attendues
    expected_params = {
        'panel_conf': 0.30,
        'balloon_conf': 0.38,
        'panel_nms_iou': 0.30,
        'balloon_nms_iou': 0.25,
        'panel_area_min_pct': 0.03,
        'panel_area_max_pct': 0.90,
        'balloon_area_min_pct': 0.0020,
        'balloon_area_max_pct': 0.30,
        'min_box_w_px': 32,
        'min_box_h_px': 28,
        'page_margin_inset_pct': 0.015,
        'balloon_min_overlap_panel': 0.06,
        'max_panels': 12,
        'max_balloons': 24
    }
    
    print("🔍 VÉRIFICATION DES PARAMÈTRES YAML")
    print("=" * 50)
    
    all_correct = True
    
    for param, expected in expected_params.items():
        actual = config.get(param)
        
        if actual is None:
            print(f"❌ {param:25} = MANQUANT (attendu: {expected})")
            all_correct = False
        elif actual == expected:
            print(f"✅ {param:25} = {actual}")
        else:
            print(f"⚠️  {param:25} = {actual} (attendu: {expected})")
            # Tolérance pour les flottants
            if isinstance(expected, float) and isinstance(actual, (int, float)):
                if abs(float(actual) - expected) < 0.001:
                    print(f"   └─ ✅ Valeur acceptable (différence négligeable)")
                else:
                    all_correct = False
            else:
                all_correct = False
    
    print("\n" + "=" * 50)
    
    if all_correct:
        print("🎉 TOUS LES PARAMÈTRES SONT CORRECTEMENT CONFIGURÉS!")
        return True
    else:
        print("❌ Certains paramètres nécessitent des corrections")
        return False

def test_parameters_in_code():
    """Test que les paramètres sont bien utilisés dans le code"""
    
    print("\n🧪 TEST D'UTILISATION DES PARAMÈTRES")
    print("=" * 40)
    
    try:
        # Import avec gestion des erreurs
        sys.path.insert(0, os.path.dirname(__file__))
        from main import PdfYoloViewer, GLOBAL_CONFIG
        from PySide6.QtWidgets import QApplication
        import yaml
        
        # Charger la config
        with open('config/detect_with_merge.yaml', 'r') as f:
            GLOBAL_CONFIG.clear()
            GLOBAL_CONFIG.update(yaml.safe_load(f))
        
        # Créer l'application
        app = QApplication.instance() or QApplication([])
        viewer = PdfYoloViewer()
        
        # Tester l'accès aux paramètres
        test_params = {
            'panel_conf': 0.30,
            'balloon_conf': 0.38,
            'max_panels': 12,
            'max_balloons': 24
        }
        
        for param, expected in test_params.items():
            actual = viewer._cfg(param, 999)  # default différent pour vérifier
            if actual == expected:
                print(f"✅ {param}: {actual}")
            else:
                print(f"❌ {param}: {actual} (attendu: {expected})")
                return False
        
        print("✅ Les paramètres sont correctement lus par le code!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        return False

if __name__ == "__main__":
    print("🎯 VÉRIFICATION COMPLÈTE DES PARAMÈTRES YAML")
    print("=" * 60)
    
    success1 = check_yaml_parameters()
    success2 = test_parameters_in_code()
    
    print("\n" + "=" * 60)
    print("📋 RÉSUMÉ:")
    
    if success1 and success2:
        print("🎉 TOUS LES TESTS RÉUSSIS!")
        print("✅ Paramètres YAML correctement configurés")
        print("✅ Paramètres correctement utilisés par le code")
        print("✅ Le système de post-traitement raffiné est opérationnel")
    else:
        print("❌ Des problèmes ont été détectés")
        if not success1:
            print("  - Configuration YAML à corriger")
        if not success2:
            print("  - Utilisation des paramètres à vérifier")
    
    print("\n🔧 Fichier de configuration: config/detect_with_merge.yaml")
    print("🚀 Prêt pour utilisation avec: python3 main.py")
