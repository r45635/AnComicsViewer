#!/usr/bin/env python3
"""
Détecteur Multi-BD Ultra-Avancé pour AnComicsViewer
Résout les problèmes de :
1. Prise en compte du champ texte complet d'un chapitre
2. Ordre des cases (séquence de lecture)
"""

import sys
import os
sys.path.append('.')

from detectors.multibd_detector import MultiBDPanelDetector
from PySide6.QtCore import QRectF

class UltraMultiBDDetector(MultiBDPanelDetector):
    """Détecteur Multi-BD avec intelligence avancée pour BD."""
    
    def __init__(self, **kwargs):
        super().__init__(conf=0.25, iou=0.35, **kwargs)
        print(f"🚀 UltraMultiBDDetector initialisé (conf={self.conf}, iou={self.iou})")
    
    def detect_panels(self, qimage, page_point_size):
        """Détection avec intelligence BD complète."""
        raw_panels = super().detect_panels(qimage, page_point_size)
        if not raw_panels:
            return []
        
        print(f"🔍 Détections brutes: {len(raw_panels)}")
        
        page_height = page_point_size.height()
        page_width = page_point_size.width()
        page_area = page_width * page_height
        
        # ÉTAPE 1: Classification intelligente des zones
        title_zones = []
        panel_candidates = []
        
        for panel in raw_panels:
            # Analyser la position et les caractéristiques
            y_pos = panel.y() / page_height  # Position relative
            aspect_ratio = panel.width() / panel.height()
            area_ratio = (panel.width() * panel.height()) / page_area
            
            # Classification des types de zones
            is_title_zone = self._is_title_zone(panel, page_height, page_width, aspect_ratio)
            is_too_small = area_ratio < 0.015  # Moins de 1.5% de la page
            is_noise = aspect_ratio > 5.0 or aspect_ratio < 0.15
            
            if is_title_zone:
                title_zones.append(panel)
                print(f"📝 Zone titre détectée: y={y_pos:.2f}, ratio={aspect_ratio:.2f}")
            elif not is_too_small and not is_noise:
                panel_candidates.append(panel)
        
        # ÉTAPE 2: Fusion intelligente des zones de titre
        merged_title_zones = self._merge_title_zones(title_zones, page_width)
        
        # ÉTAPE 3: Filtrage et nettoyage des cases
        filtered_panels = self._filter_and_clean_panels(panel_candidates, page_area)
        
        # ÉTAPE 4: Tri par ordre de lecture BD
        ordered_panels = self._sort_reading_order(filtered_panels, page_height, page_width)
        
        # ÉTAPE 5: Combinaison finale (titres + cases ordonnées)
        final_panels = merged_title_zones + ordered_panels
        
        print(f"✅ Résultat final: {len(merged_title_zones)} titres + {len(ordered_panels)} cases = {len(final_panels)} zones")
        
        return final_panels
    
    def _is_title_zone(self, panel, page_height, page_width, aspect_ratio):
        """Détermine si une zone est un titre de chapitre."""
        y_relative = panel.y() / page_height
        width_relative = panel.width() / page_width
        
        # Zone en haut de page (30% supérieur)
        if y_relative < 0.3:
            # Large et relativement plat (titre de chapitre)
            if width_relative > 0.4 and aspect_ratio > 2.0:
                return True
            # Centré horizontalement (titre)
            x_center = (panel.x() + panel.width()/2) / page_width
            if 0.2 < x_center < 0.8 and aspect_ratio > 1.5:
                return True
        
        # Zone très large sur toute la largeur (bandeau titre)
        if width_relative > 0.8 and aspect_ratio > 3.0:
            return True
            
        return False
    
    def _merge_title_zones(self, title_zones, page_width):
        """Fusionne les zones de titre fragmentées."""
        if not title_zones:
            return []
        
        # Grouper les titres par proximité verticale
        title_groups = []
        tolerance_y = 50  # pixels
        
        for title in title_zones:
            placed = False
            for group in title_groups:
                # Vérifier si proche verticalement d'un titre existant
                if any(abs(title.y() - existing.y()) < tolerance_y for existing in group):
                    group.append(title)
                    placed = True
                    break
            
            if not placed:
                title_groups.append([title])
        
        # Fusionner chaque groupe en une zone titre complète
        merged_titles = []
        for group in title_groups:
            if len(group) == 1:
                merged_titles.append(group[0])
            else:
                # Créer une boîte englobante
                min_x = min(t.x() for t in group)
                min_y = min(t.y() for t in group)
                max_x = max(t.x() + t.width() for t in group)
                max_y = max(t.y() + t.height() for t in group)
                
                merged_title = QRectF(min_x, min_y, max_x - min_x, max_y - min_y)
                merged_titles.append(merged_title)
                print(f"🔗 Fusion titre: {len(group)} fragments → 1 zone complète")
        
        return merged_titles
    
    def _filter_and_clean_panels(self, panels, page_area):
        """Filtre et nettoie les candidates de cases."""
        filtered = []
        
        for panel in panels:
            area_ratio = (panel.width() * panel.height()) / page_area
            aspect_ratio = panel.width() / panel.height()
            
            # Filtres de qualité
            if area_ratio < 0.02:  # Trop petit (2% minimum)
                continue
            if aspect_ratio > 4.0:  # Trop plat (probablement du texte)
                continue
            if aspect_ratio < 0.2:  # Trop vertical (probablement du bruit)
                continue
            
            filtered.append(panel)
        
        # Supprimer les doublons/chevauchements
        return self._remove_overlapping_panels(filtered)
    
    def _remove_overlapping_panels(self, panels):
        """Supprime les panels qui se chevauchent trop."""
        if len(panels) <= 1:
            return panels
        
        # Trier par aire décroissante (garder les plus grands)
        panels_sorted = sorted(panels, key=lambda p: p.width() * p.height(), reverse=True)
        
        filtered = []
        for panel in panels_sorted:
            # Vérifier le chevauchement avec les panels déjà acceptés
            overlaps = False
            for existing in filtered:
                overlap_area = self._intersection_area(panel, existing)
                panel_area = panel.width() * panel.height()
                existing_area = existing.width() * existing.height()
                
                # Si plus de 70% de chevauchement, c'est un doublon
                if overlap_area > 0.7 * min(panel_area, existing_area):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered.append(panel)
        
        return filtered
    
    def _intersection_area(self, rect1, rect2):
        """Calcule l'aire d'intersection entre deux rectangles."""
        x_overlap = max(0, min(rect1.x() + rect1.width(), rect2.x() + rect2.width()) - 
                          max(rect1.x(), rect2.x()))
        y_overlap = max(0, min(rect1.y() + rect1.height(), rect2.y() + rect2.height()) - 
                          max(rect1.y(), rect2.y()))
        return x_overlap * y_overlap
    
    def _sort_reading_order(self, panels, page_height, page_width):
        """Trie les cases selon l'ordre de lecture BD (occidental: gauche→droite, haut→bas)."""
        if not panels:
            return panels
        
        # Grouper les panels par "lignes" (même hauteur approximative)
        rows = []
        row_tolerance = page_height * 0.15  # 15% de la hauteur de page
        
        for panel in panels:
            panel_center_y = panel.y() + panel.height() / 2
            placed_in_row = False
            
            for row in rows:
                # Vérifier si le panel appartient à cette ligne
                row_center_y = sum(p.y() + p.height()/2 for p in row) / len(row)
                if abs(panel_center_y - row_center_y) < row_tolerance:
                    row.append(panel)
                    placed_in_row = True
                    break
            
            if not placed_in_row:
                rows.append([panel])
        
        # Trier les lignes par position verticale (haut → bas)
        rows.sort(key=lambda row: min(p.y() for p in row))
        
        # Dans chaque ligne, trier par position horizontale (gauche → droite)
        ordered_panels = []
        for row in rows:
            row_sorted = sorted(row, key=lambda p: p.x())
            ordered_panels.extend(row_sorted)
            print(f"📚 Ligne de {len(row_sorted)} cases triée")
        
        return ordered_panels

def test_ultra_detector():
    """Test du détecteur ultra-avancé."""
    print("🧪 Test UltraMultiBDDetector")
    print("=" * 40)
    
    try:
        detector = UltraMultiBDDetector()
        print("✅ UltraMultiBDDetector créé avec succès")
        
        # Test des méthodes
        from PySide6.QtCore import QRectF, QSizeF
        
        # Simuler quelques détections
        fake_panels = [
            QRectF(100, 50, 400, 80),   # Titre potentiel
            QRectF(50, 200, 200, 150),  # Case 1
            QRectF(300, 200, 200, 150), # Case 2  
            QRectF(50, 400, 450, 120),  # Case large
        ]
        
        page_size = QSizeF(600, 800)
        
        # Test du tri par ordre de lecture
        ordered = detector._sort_reading_order(fake_panels[1:], page_size.height(), page_size.width())
        print(f"✅ Tri ordre lecture: {len(ordered)} cases")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

if __name__ == "__main__":
    success = test_ultra_detector()
    print(f"\n{'✅ Succès!' if success else '❌ Échec!'}")
