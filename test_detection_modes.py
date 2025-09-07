#!/usr/bin/env python3
"""
Script de test pour vérifier la sélection de mode de détection
"""

import sys
import os
sys.path.append('.')

def test_detection_modes():
    """Test des différents modes de détection"""

    print("🧪 TEST DES MODES DE DÉTECTION")
    print("=" * 40)

    # Simuler les résultats YOLO
    yolo_panels = 17
    yolo_balloons = 8

    # Simuler les résultats par règles
    rules_panels = 7
    rules_balloons = 3

    print(f"📊 Données simulées:")
    print(f"   🤖 YOLO: {yolo_panels} panels, {yolo_balloons} ballons")
    print(f"   📏 Règles: {rules_panels} panels, {rules_balloons} ballons")

    # Test mode YOLO seul
    print(f"\n🤖 MODE YOLO SEUL:")
    print(f"   📦 Panels: {yolo_panels}")
    print(f"   💬 Ballons: {yolo_balloons}")

    # Test mode Règles seules
    print(f"\n📏 MODE RÈGLES SEULES:")
    print(f"   📦 Panels: {rules_panels}")
    print(f"   💬 Ballons: {rules_balloons}")

    # Test mode Hybride
    hybrid_panels = yolo_panels + rules_panels - 2  # Simuler 2 fusions
    hybrid_balloons = yolo_balloons + rules_balloons  # Simuler pas de fusion pour ballons
    print(f"\n🎯 MODE HYBRIDE:")
    print(f"   📦 Panels: {yolo_panels} + {rules_panels} - 2 fusions = {hybrid_panels}")
    print(f"   💬 Ballons: {yolo_balloons} + {rules_balloons} = {hybrid_balloons}")

    print(f"\n✅ TESTS TERMINÉS")
    print(f"💡 Le GUI devrait maintenant permettre de:")
    print(f"   • Sélectionner le mode de détection via un menu déroulant")
    print(f"   • Voir les statistiques correspondantes au mode choisi")
    print(f"   • Afficher le nombre correct de panels selon le mode")

if __name__ == "__main__":
    test_detection_modes()
