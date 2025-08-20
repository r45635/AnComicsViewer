
class ImprovedMultiBDDetector(MultiBDPanelDetector):
    """Détecteur Multi-BD avec améliorations pour titre/contours."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Paramètres optimisés
        self.conf = 0.15  # Plus sensible
        self.iou = 0.4    # Plus de chevauchement autorisé
        
    def detect_panels(self, qimage, page_point_size):
        """Détection avec post-traitement amélioré."""
        # Détection de base
        raw_panels = super().detect_panels(qimage, page_point_size)
        
        # Post-traitement intelligent
        filtered_panels = []
        page_height = page_point_size.height()
        
        for panel in raw_panels:
            # Filtrer zone titre (20% du haut de la page)
            if panel.y() < page_height * 0.2:
                # Vérifier le ratio aspect pour différencier titre/case
                aspect_ratio = panel.width() / panel.height()
                if aspect_ratio > 4.0:  # Ligne de texte probable
                    continue
                    
            # Filtrer les détections trop petites
            min_area = page_height * page_point_size.width() * 0.01  # 1% de la page
            if panel.width() * panel.height() < min_area:
                continue
                
            # Filtrer les ratios aspect anormaux 
            aspect_ratio = panel.width() / panel.height()
            if aspect_ratio > 6.0 or aspect_ratio < 0.15:  # Trop large ou trop étroit
                continue
                
            filtered_panels.append(panel)
        
        return filtered_panels
        