#!/usr/bin/env python3
"""
Test rapide des nouveaux paramètres de détection
"""

import sys
import os
sys.path.insert(0, '/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/newBranch')

from PySide6.QtCore import QRectF
from main import debug_detection_stats, enable_detection_debug

def test_new_parameters():
    """Test les nouveaux paramètres avec des données simulées"""

    print("🧪 TEST DES NOUVEAUX PARAMÈTRES DE DÉTECTION")
    print("=" * 60)

    # Activer le debug
    enable_detection_debug(True)

    # Simuler des données de détection plus réalistes
    print("\n📊 SIMULATION AVEC DONNÉES RÉALISTES:")

    # Simulation de détection brute avec plus de panels et ballons
    panels_bruts = [
        (0, 0.15, QRectF(100, 100, 200, 300)),   # Panel conf 0.15
        (0, 0.08, QRectF(350, 100, 180, 280)),   # Panel conf 0.08
        (0, 0.25, QRectF(100, 450, 220, 320)),   # Panel conf 0.25
        (0, 0.05, QRectF(350, 450, 190, 290)),   # Panel conf 0.05 (sera filtré avec anciens seuils)
        (0, 0.12, QRectF(600, 100, 150, 200)),   # Panel conf 0.12
        (0, 0.03, QRectF(600, 350, 160, 180)),   # Panel conf 0.03 (nouveau seuil)
    ]

    ballons_bruts = [
        (1, 0.10, QRectF(120, 120, 40, 25)),    # Ballon conf 0.10
        (1, 0.05, QRectF(380, 130, 35, 20)),    # Ballon conf 0.05
        (1, 0.08, QRectF(150, 480, 45, 30)),    # Ballon conf 0.08
        (1, 0.02, QRectF(380, 480, 30, 18)),    # Ballon conf 0.02 (nouveau seuil)
    ]

    page_area = 1000000  # Simulation d'une page de 1000x1000px

    debug_detection_stats("DÉTECTION BRUTE (SIMULÉE)", panels_bruts, ballons_bruts, page_area)

    # Simuler le filtrage avec les nouveaux seuils
    panel_conf_new = 0.01   # Nouveau seuil panels
    balloon_conf_new = 0.03 # Nouveau seuil ballons
    panel_area_min = 0.0005 # 0.05% de la page
    balloon_area_min = 0.0001 # 0.01% de la page

    panels_filtres = []
    for c, p, r in panels_bruts:
        area_pct = (r.width() * r.height()) / page_area * 100
        if p >= panel_conf_new and area_pct >= (panel_area_min * 100):
            panels_filtres.append((c, p, r))

    ballons_filtres = []
    for c, p, r in ballons_bruts:
        area_pct = (r.width() * r.height()) / page_area * 100
        if p >= balloon_conf_new and area_pct >= (balloon_area_min * 100) and r.width() >= 10 and r.height() >= 8:
            ballons_filtres.append((c, p, r))

    debug_detection_stats("APRÈS FILTRE AVEC NOUVEAUX SEUILS", panels_filtres, ballons_filtres, page_area)

    print("\n✅ Test terminé !")
    print(f"📊 Résultat: {len(panels_filtres)} panels, {len(ballons_filtres)} ballons")
    print("💡 Les nouveaux seuils devraient détecter plus d'éléments")

if __name__ == "__main__":
    test_new_parameters()
