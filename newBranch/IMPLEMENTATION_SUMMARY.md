# AnComicsViewer - Améliorations Implémentées

## 🎯 Objectifs atteints

Ce document résume toutes les améliorations implémentées selon les spécifications demandées.

## ✅ 1. Imports & État global

### Imports ajoutés
- `QPointF` ajouté aux imports PySide6 (même si pas encore utilisé)
- Tous les imports nécessaires sont présents

### Variables globales ajoutées/vérifiées
```python
GLOBAL_CONFIG: Dict[str, Any] = {}
DEBUG_DETECT: bool = False
DEBUG_OVERLAY_DIR: Optional[str] = None
METRICS_OUT: Optional[str] = None  # Nouveau: export JSON optionnel
```

### Champs de calibration dans PdfYoloViewer.__init__()
```python
self.render_dpi: float = 300.0
self.page_size_pts: Tuple[float, float] = (0.0, 0.0)   # (w,h) en points PDF
self.image_size_px: Tuple[int, int] = (0, 0)          # (W,H) en pixels de l'image rendue
```

## ✅ 2. Calibration pixel↔PDF

### Rendu à DPI fixe dans load_page()
```python
# Store PDF page size in points (1 pt = 1/72 inch)
self.page_size_pts = (float(page.rect.width), float(page.rect.height))

# Render at fixed DPI for stable calibration
dpi = self.render_dpi
zoom = dpi / 72.0
pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
qimg = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format.Format_RGB888).copy()

# Store image size in pixels for calibration
self.image_size_px = (qimg.width(), qimg.height())
```

### Méthodes utilitaires de conversion
```python
def pixel_to_pdf_rect(self, r: QRectF) -> QRectF:
    """Convert a rectangle in image pixels to PDF points (72 dpi units)."""
    
def pdf_to_pixel_rect(self, r: QRectF) -> QRectF:
    """Convert a rectangle in PDF points to image pixels."""
```

## ✅ 3. Métriques de qualité

### Helpers de calcul ajoutés
```python
def _area(r: QRectF) -> float:
    return max(0.0, r.width() * r.height())

def _iou(a: QRectF, b: QRectF) -> float:
    # Intersection over Union

def _overlap_frac(a: QRectF, b: QRectF) -> float:
    """Return the fraction of rectangle b that is contained within rectangle a."""
```

### Méthode compute_quality_metrics()
Calcule pour chaque page :
- Nombre de panels et balloons
- Ratios d'aire par rapport à la page
- Nombre d'overlaps (IoU > 0.10 ou containment > 0.60)
- Nombre de severe overlaps (IoU > 0.50 ou containment > 0.90)
- Score de qualité : `1.0 - severe_ratio - 0.5*overlap_ratio`

### Export JSON optionnel
- Argument CLI `--metrics-out filename.json`
- Sauvegarde incrémentale (append en tant que liste)
- Gestion des erreurs avec messages informatifs

## ✅ 4. Post-traitement raffiné

### NMS class-aware amélioré
```python
def apply_nms_class_aware(dets: List[Tuple[int, float, QRectF]], iou_thr: float):
    """Apply NMS within each class separately (class-aware NMS)."""
```

### Méthode _refine_dets() complète
Filtrages appliqués dans l'ordre :
1. **Seuils de confiance par classe** (`panel_conf`, `balloon_conf`)
2. **Filtres de taille** (`min_box_w_px`, `min_box_h_px`)
3. **Filtres d'aire** (`*_area_min_pct`, `*_area_max_pct`)
4. **Filtre de marge** (`page_margin_inset_pct`)
5. **NMS par classe** (`panel_nms_iou`, `balloon_nms_iou`)
6. **Règle d'attachement balloon→panel** (`balloon_min_overlap_panel`)
7. **Limitation de quantité** (`max_panels`, `max_balloons`)

### Intégration dans la pipeline
- Appliqué après la NMS grossière
- Respecte les toggles UI (panels/balloons on/off)
- Compatible avec tous les modes de détection

## ✅ 5. Paramètres YAML étendus

### Nouvelles clés de configuration
```yaml
# Seuils de confiance
panel_conf: 0.30
balloon_conf: 0.38

# NMS par classe
panel_nms_iou: 0.30
balloon_nms_iou: 0.25

# Filtres de taille et marge
panel_area_min_pct: 0.03
panel_area_max_pct: 0.90
balloon_area_min_pct: 0.0020
balloon_area_max_pct: 0.30
min_box_w_px: 32
min_box_h_px: 28
page_margin_inset_pct: 0.015

# Règle d'assignation
balloon_min_overlap_panel: 0.06

# Limites de sortie
max_panels: 12
max_balloons: 24
```

## ✅ 6. CLI et utilisation

### Nouvel argument
```bash
--metrics-out outputs/metrics.json
```

### Exemples d'utilisation
```bash
# Usage basique avec métriques
python main.py --pdf comic.pdf --page 4 --metrics-out outputs/metrics.json

# Usage avancé avec debug
python main.py --pdf comic.pdf --page 4 \
  --metrics-out outputs/metrics.json \
  --debug-detect --save-debug-overlays debug \
  --config config/detect_with_merge.yaml

# Traitement multi-pages
for i in {0..10}; do
  python main.py --pdf comic.pdf --page $i --metrics-out batch_metrics.json
done
```

## 🧪 Tests et validation

### Script de test automatisé
- `test_implementation.py` : Tests unitaires de toutes les fonctions
- Vérification des helpers (_area, _iou, _overlap_frac)
- Test de la NMS class-aware
- Test de la calibration pixel↔PDF
- Test des métriques de qualité
- Test du chargement de config

### Script de démonstration
- `demo.sh` : Démonstration complète des nouvelles fonctionnalités
- Tests avec différentes configurations
- Traitement multi-pages
- Exemples d'usage pratique

## 🎯 Résultats attendus

### Améliorations de qualité
- **Réduction du bruit** : Moins de fausses détections grâce aux filtres raffinés
- **NMS class-aware** : Évite la suppression incorrecte entre classes différentes
- **Règle d'attachement** : Les balloons orphelins sont filtrés
- **Filtres de marge** : Ignore les détections en bordure de page

### Métriques de qualité
- **Score objectif** : Quality score entre 0 et 1
- **Détection des problèmes** : Overlaps et severe overlaps comptabilisés
- **Export structuré** : JSON avec toutes les métriques par page

### Calibration précise
- **Rendu stable** : DPI fixe (300) pour tous les PDF
- **Conversion exacte** : Pixel↔PDF basée sur les vraies dimensions
- **Réversibilité** : Round-trip pixel→PDF→pixel sans perte

### Flexibilité
- **Configuration YAML** : Tous les paramètres ajustables
- **Rétrocompatibilité** : Fonctionne avec les anciens fichiers YAML
- **Mode debug** : Overlays visuels pour validation

## 🚀 État final

✅ **Toutes les spécifications sont implémentées**  
✅ **Interface utilisateur préservée**  
✅ **Navigation fonctionnelle maintenue**  
✅ **Configuration YAML étendue**  
✅ **Tests de validation passés**  
✅ **Documentation et exemples fournis**

Le viewer est maintenant équipé de :
- Calibration pixel↔PDF précise
- Métriques de qualité automatiques  
- Post-traitement raffiné anti-bruit
- Export JSON optionnel
- Configuration flexible via YAML
- Rendu à DPI fixe stable
