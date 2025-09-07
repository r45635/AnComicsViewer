#!/usr/bin/env python3
"""
Test rapide des paramètres de détection pour la page 5
"""

import yaml
import sys
sys.path.append('.')

def test_config_parsing():
    """Tester le parsing de la configuration"""
    print("=== TEST DU PARSING DE CONFIGURATION ===")

    # Charger le fichier YAML
    with open('config/detect.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("📄 Valeurs dans le fichier YAML :")
    print(f"   panel_conf: {config.get('panel_conf')}")
    print(f"   balloon_conf: {config.get('balloon_conf')}")
    print(f"   panel_area_min_pct: {config.get('panel_area_min_pct')}")
    print(f"   balloon_area_min_pct: {config.get('balloon_area_min_pct')}")
    print(f"   balloon_min_w: {config.get('balloon_min_w')}")
    print(f"   balloon_min_h: {config.get('balloon_min_h')}")
    print()

    # Simuler le calcul des seuils
    img_area = 2400 * 3634  # Dimensions de la page 5
    balloon_area_min_px = config.get('balloon_area_min_pct', 0.0006) * img_area
    panel_area_min_px = config.get('panel_area_min_pct', 0.02) * img_area

    print("🔢 Calcul des seuils en pixels :")
    print(f"   Image area: {img_area:,} pixels")
    print(f"   Balloon area min: {balloon_area_min_px:.0f} pixels ({config.get('balloon_area_min_pct')*100:.4f}%)")
    print(f"   Panel area min: {panel_area_min_px:.0f} pixels ({config.get('panel_area_min_pct')*100:.1f}%)")
    print()

    # Comparer avec les annotations de référence
    print("📊 COMPARAISON AVEC LES ANNOTATIONS :")
    print("   Balloon le plus petit: ~2000 pixels (devrait être détecté)")
    print("   Balloon le plus grand: ~250000 pixels (devrait être détecté)")
    print("   Panel le plus petit: ~200000 pixels (devrait être détecté)")
    print("   Panel le plus grand: ~800000 pixels (devrait être détecté)")
    print()

    if balloon_area_min_px > 2000:
        print("⚠️  PROBLÈME: Le seuil balloon_area_min est trop élevé!")
        print("   Les petits ballons ne seront pas détectés.")
    else:
        print("✅ Balloon area threshold semble correct.")

if __name__ == "__main__":
    test_config_parsing()
