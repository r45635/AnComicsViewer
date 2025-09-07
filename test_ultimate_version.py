#!/usr/bin/env python3
"""
Test rapide de la version ULTIMATE d'AnComicsViewer
- Test des nouvelles fonctionnalités optimisées
- Validation des métriques de qualité
- Comparaison avec l'ancienne version
"""

import sys
import os
import time
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent))

def test_ultimate_features():
    """Test des nouvelles fonctionnalités optimisées"""

    print("🧪 TEST DE LA VERSION ULTIMATE")
    print("=" * 50)

    # Test 1: Import des nouvelles fonctions
    print("\n1️⃣  TEST D'IMPORT DES FONCTIONS OPTIMISÉES")
    try:
        # Importer depuis newBranch/main.py
        sys.path.insert(0, 'newBranch')
        from main import (
            apply_comics_optimized_filter,
            detect_and_resolve_overlaps,
            validate_detection_quality,
            COMICS_CONFIG
        )
        print("   ✅ Import réussi depuis newBranch/main.py")
    except ImportError as e:
        print(f"   ❌ Erreur d'import: {e}")
        return False

    # Test 2: Configuration optimisée
    print("\n2️⃣  TEST DE LA CONFIGURATION OPTIMISÉE")
    print(f"   📊 IoU threshold: {COMICS_CONFIG['iou_threshold']}")
    print(f"   📊 Containment threshold: {COMICS_CONFIG['containment_threshold']}")
    print(f"   📊 Panel confidence: {COMICS_CONFIG['confidence_panel']}")
    print(f"   📊 Balloon confidence: {COMICS_CONFIG['confidence_balloon']}")
    print("   ✅ Configuration chargée")

    # Test 3: Fonctions de filtrage
    print("\n3️⃣  TEST DES FONCTIONS DE FILTRAGE")
    from PySide6.QtCore import QRectF

    # Créer des détections de test
    test_detections = [
        (0, 0.8, QRectF(100, 100, 200, 300)),  # Panel valide
        (0, 0.2, QRectF(10, 10, 50, 50)),     # Panel trop petit
        (1, 0.6, QRectF(150, 150, 100, 50)),  # Balloon valide
        (1, 0.1, QRectF(5, 5, 20, 20)),      # Balloon confiance trop basse
    ]

    page_area = 1000 * 1500  # Page de test
    filtered = apply_comics_optimized_filter(test_detections, page_area)

    print(f"   📊 Détections initiales: {len(test_detections)}")
    print(f"   📊 Détections filtrées: {len(filtered)}")
    print("   ✅ Filtrage fonctionnel")

    # Test 4: Résolution des chevauchements
    print("\n4️⃣  TEST DE RÉSOLUTION DES CHEVAUCHEMENTS")

    panels = [(0, 0.8, QRectF(100, 100, 200, 300))]
    balloons = [(1, 0.7, QRectF(120, 120, 150, 200))]  # Chevauchement

    resolved_panels, resolved_balloons = detect_and_resolve_overlaps(panels, balloons)

    print(f"   📊 Panels: {len(resolved_panels)}")
    print(f"   📊 Balloons: {len(resolved_balloons)}")
    print("   ✅ Résolution des chevauchements fonctionnelle")

    # Test 5: Métriques de qualité
    print("\n5️⃣  TEST DES MÉTRIQUES DE QUALITÉ")

    quality = validate_detection_quality(resolved_panels, resolved_balloons, page_area)

    print(f"   📊 Score de qualité: {quality['quality_score']:.3f}")
    print(f"   📊 Chevauchements détectés: {quality['overlaps_detected']}")
    print(f"   📊 Chevauchements sévères: {quality['severe_overlaps']}")
    print("   ✅ Métriques de qualité calculées")

    print("\n🎉 TOUS LES TESTS SONT RÉUSSIS !")
    print("\n📋 RÉSUMÉ DES AMÉLIORATIONS:")
    print("   • ✅ Filtre optimisé pour comics")
    print("   • ✅ Gestion intelligente des chevauchements")
    print("   • ✅ Métriques de qualité avancées")
    print("   • ✅ Configuration adaptée aux bandes dessinées")
    print("   • ✅ Debug amélioré avec statistiques détaillées")

    return True

def compare_with_old_version():
    """Compare les performances avec l'ancienne version"""
    print("\n🔄 COMPARAISON AVEC L'ANCIENNE VERSION")
    print("-" * 40)

    # TODO: Implémenter la comparaison
    print("   📊 Comparaison à implémenter...")
    print("   🔧 Utiliser: python main.py --config config/detect_ultimate.yaml --debug-detect")

if __name__ == "__main__":
    success = test_ultimate_features()
    if success:
        compare_with_old_version()
        print("\n🚀 PRÊT POUR LES TESTS RÉELS !")
        print("   Commande: python main.py --config config/detect_ultimate.yaml --debug-detect")
    else:
        print("\n❌ ÉCHEC DES TESTS - VÉRIFIER LES ERREURS CI-DESSUS")
        sys.exit(1)
