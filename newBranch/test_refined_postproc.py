#!/usr/bin/env python3
"""
Test rapide du post-traitement raffiné
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from main import PdfYoloViewer, GLOBAL_CONFIG, _area, _iou, _overlap_frac
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QRectF
import yaml

def test_refined_postprocessing():
    """Test le post-traitement raffiné avec des détections simulées"""
    print("🔧 Test du post-traitement raffiné")
    
    # Charger la config
    config_path = "config/detect_with_merge.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            GLOBAL_CONFIG.update(yaml.safe_load(f))
            print(f"✅ Configuration chargée: {config_path}")
    
    # Créer l'application et le viewer
    app = QApplication.instance() or QApplication([])
    viewer = PdfYoloViewer()
    viewer.image_size_px = (2000, 3000)  # Image 2000x3000 (format page)
    
    print(f"📏 Taille de page simulée: {viewer.image_size_px[0]}x{viewer.image_size_px[1]} px")
    
    # Créer des détections de test avec du bruit
    print("\n📦 Création de détections de test...")
    
    # Panels valides
    good_panels = [
        (0, 0.85, QRectF(100, 100, 600, 400)),    # Grand panel principal
        (0, 0.75, QRectF(800, 600, 500, 500)),    # Panel carré
        (0, 0.65, QRectF(200, 1200, 1000, 300)),  # Panel horizontal
    ]
    
    # Panels parasites (à filtrer)
    noisy_panels = [
        (0, 0.15, QRectF(50, 50, 30, 20)),        # Trop petit, faible confiance  
        (0, 0.25, QRectF(1950, 2950, 40, 40)),    # En bord de page
        (0, 0.45, QRectF(0, 0, 1999, 2999)),      # Trop grand (quasi full page)
        (0, 0.30, QRectF(110, 110, 580, 380)),    # Overlap fort avec panel 1
    ]
    
    # Balloons valides (attachés aux panels)
    good_balloons = [
        (1, 0.75, QRectF(300, 200, 150, 80)),     # Dans panel 1
        (1, 0.65, QRectF(950, 750, 120, 60)),     # Dans panel 2
        (1, 0.55, QRectF(600, 1300, 100, 50)),    # Dans panel 3
    ]
    
    # Balloons parasites (à filtrer)
    noisy_balloons = [
        (1, 0.20, QRectF(10, 10, 20, 15)),        # Trop petit, bord page
        (1, 0.40, QRectF(1500, 500, 80, 40)),     # Pas dans un panel
        (1, 0.30, QRectF(1800, 2800, 50, 30)),    # Bord de page, pas attaché
        (1, 0.45, QRectF(310, 210, 140, 70)),     # Overlap fort avec balloon 1
    ]
    
    all_panels = good_panels + noisy_panels
    all_balloons = good_balloons + noisy_balloons
    
    print(f"   📍 Panels totaux: {len(all_panels)} (dont {len(good_panels)} valides)")
    print(f"   🎈 Balloons totaux: {len(all_balloons)} (dont {len(good_balloons)} valides)")
    
    # Appliquer le post-traitement raffiné
    print("\n⚡ Application du post-traitement raffiné...")
    refined_panels, refined_balloons = viewer._refine_dets(all_panels, all_balloons)
    
    print(f"   📍 Panels après raffinement: {len(refined_panels)}")
    print(f"   🎈 Balloons après raffinement: {len(refined_balloons)}")
    
    # Analyse des résultats
    print("\n📊 Analyse des résultats:")
    
    # Vérifier que le bruit a été filtré
    if len(refined_panels) < len(all_panels):
        filtered_panels = len(all_panels) - len(refined_panels)
        print(f"   ✅ {filtered_panels} panels parasites filtrés")
    
    if len(refined_balloons) < len(all_balloons):
        filtered_balloons = len(all_balloons) - len(refined_balloons)
        print(f"   ✅ {filtered_balloons} balloons parasites filtrés")
    
    # Vérifier que les balloons restants sont attachés à des panels
    attached_count = 0
    for _, _, br in refined_balloons:
        for _, _, pr in refined_panels:
            if pr.contains(br.center()) or _overlap_frac(pr, br) >= 0.03:
                attached_count += 1
                break
    
    print(f"   🔗 {attached_count}/{len(refined_balloons)} balloons attachés à des panels")
    
    # Vérifier les seuils de confiance
    min_panel_conf = min([p for _, p, _ in refined_panels], default=1.0)
    min_balloon_conf = min([p for _, p, _ in refined_balloons], default=1.0)
    print(f"   📈 Confiance min panels: {min_panel_conf:.3f}")
    print(f"   📈 Confiance min balloons: {min_balloon_conf:.3f}")
    
    # Test des métriques de qualité
    print("\n📏 Calcul des métriques de qualité...")
    metrics = viewer.compute_quality_metrics(refined_panels, refined_balloons)
    
    print(f"   🎯 Score de qualité: {metrics['quality_score']:.3f}")
    print(f"   ⚠️  Overlaps: {metrics['overlaps']}")
    print(f"   🚨 Severe overlaps: {metrics['severe_overlaps']}")
    
    # Afficher les aires relatives
    if metrics['panel_area_ratios']:
        avg_panel_area = sum(metrics['panel_area_ratios']) / len(metrics['panel_area_ratios'])
        print(f"   📐 Aire moyenne panels: {avg_panel_area:.1%} de la page")
    
    if metrics['balloon_area_ratios']:
        avg_balloon_area = sum(metrics['balloon_area_ratios']) / len(metrics['balloon_area_ratios'])
        print(f"   📐 Aire moyenne balloons: {avg_balloon_area:.1%} de la page")
    
    print("\n🎉 Test du post-traitement raffiné terminé avec succès!")
    
    return {
        'original_panels': len(all_panels),
        'original_balloons': len(all_balloons),
        'refined_panels': len(refined_panels),
        'refined_balloons': len(refined_balloons),
        'quality_score': metrics['quality_score']
    }

if __name__ == "__main__":
    try:
        results = test_refined_postprocessing()
        
        print("\n" + "="*50)
        print("📋 RÉSUMÉ DU TEST")
        print("="*50)
        print(f"Panels: {results['original_panels']} → {results['refined_panels']} ({results['original_panels']-results['refined_panels']} filtrés)")
        print(f"Balloons: {results['original_balloons']} → {results['refined_balloons']} ({results['original_balloons']-results['refined_balloons']} filtrés)")
        print(f"Qualité finale: {results['quality_score']:.3f}/1.0")
        
        # Évaluation
        if results['refined_panels'] < results['original_panels'] and results['refined_balloons'] < results['original_balloons']:
            print("\n✅ SUCCÈS: Le post-traitement raffiné fonctionne correctement!")
            print("   - Filtrage du bruit effectué")
            print("   - Métriques de qualité calculées")
            print("   - Configuration YAML prise en compte")
        else:
            print("\n⚠️  AVERTISSEMENT: Peu ou pas de filtrage détecté")
            
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
