#!/usr/bin/env python3
"""
Script de debug pour tracer les panels perdus dans le pipeline de post-processing
"""

import sys
import os
sys.path.append('.')

def debug_panel_pipeline():
    """Simule le pipeline de post-processing pour identifier où les panels disparaissent"""

    print("🔍 DEBUG PANEL PIPELINE")
    print("=" * 50)

    # Simulation des données d'entrée (basé sur les résultats réels)
    initial_panels = 24  # Résultat de la fusion hybride
    print(f"📥 Entrée: {initial_panels} panels après fusion hybride")

    # Étape 1: Filtrage par confiance et taille
    # Supposons que tous passent (conf > 0.15, area > 1.5%)
    after_conf_filter = initial_panels
    print(f"✅ Après filtre confiance/taille: {after_conf_filter} panels")

    # Étape 2: Merging IoU (panel_merge_iou: 0.50)
    # Supposons 4 fusions (réduction de 4 panels)
    after_iou_merge = after_conf_filter - 4
    print(f"✅ Après merging IoU (0.50): {after_iou_merge} panels (4 fusions)")

    # Étape 3: Merging par rangées (enable_row_merge: true)
    # Supposons 2 fusions supplémentaires
    after_row_merge = after_iou_merge - 2
    print(f"✅ Après merging rangées: {after_row_merge} panels (2 fusions)")

    # Étape 4: Filtrage par containment (panel_containment_merge: 0.85)
    # Supposons 5 panels supprimés car contenus dans d'autres
    after_containment = after_row_merge - 5
    print(f"✅ Après filtrage containment (0.85): {after_containment} panels (5 supprimés)")

    # Étape 5: Gutter splitting (gutter_split_enable: false)
    # Désactivé, pas de changement
    after_gutter = after_containment
    print(f"✅ Après gutter splitting (désactivé): {after_gutter} panels")

    # Étape 6: Détection page complète (full_page_panel_pct: 0.99)
    # Vérifier si le plus grand panel couvre >99% de la page
    largest_panel_pct = 0.85  # Supposons 85% (ne déclenche pas)
    full_page_triggered = largest_panel_pct >= 0.99
    if full_page_triggered:
        after_full_page = 1
        print(f"✅ Après détection page complète: {after_full_page} panel (DÉCLENCHÉ - {largest_panel_pct:.1%})")
    else:
        after_full_page = after_gutter
        print(f"✅ Après détection page complète: {after_full_page} panels (non déclenché - {largest_panel_pct:.1%})")

    # Étape 7: Limite max_panels (max_panels: 20)
    max_panels = 20
    if after_full_page > max_panels:
        after_limit = max_panels
        print(f"✅ Après limite max_panels: {after_limit} panels (limité à {max_panels})")
    else:
        after_limit = after_full_page
        print(f"✅ Après limite max_panels: {after_limit} panels (pas de limite)")

    print(f"\n🎯 RÉSULTAT FINAL: {after_limit} panels")
    print(f"📊 Panels perdus: {initial_panels - after_limit}")

    # Analyse des causes probables
    print(f"\n🔍 ANALYSE DES CAUSES:")
    if after_limit <= 3:
        print(f"   🚨 PROBLÈME CRITIQUE: Seulement {after_limit} panels restants!")
        print(f"   💡 CAUSES POSSIBLES:")
        print(f"      • Merging containment trop agressif (0.85)")
        print(f"      • Détection page complète déclenchée")
        print(f"      • Merging IoU trop permissif (0.50)")
        print(f"      • Limite max_panels trop basse")

    # Recommandations
    print(f"\n💡 RECOMMANDATIONS:")
    print(f"   1. Augmenter panel_containment_merge à 0.95")
    print(f"   2. Augmenter panel_merge_iou à 0.70")
    print(f"   3. Augmenter full_page_panel_pct à 0.999")
    print(f"   4. Augmenter max_panels à 30")
    print(f"   5. Désactiver enable_row_merge temporairement")

if __name__ == "__main__":
    debug_panel_pipeline()
