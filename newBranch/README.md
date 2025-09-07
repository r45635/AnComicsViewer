# AnComicsViewer MINI - Enhanced Version

Viewer PDF minimal et robuste pour bandes dessinées avec overlays YOLO et post-traitement intelligent.
- **Aucun décalage**: l'inférence se fait sur **la même** QImage que celle affichée.
- **Overlays qui suivent**: rectangles **enfants du pixmap** → zoom/scroll/resize ne posent plus de problème.
- **HiDPI neutralisé**: `setDevicePixelRatio(1.0)` sur le pixmap.
- **Post-traitement intelligent**: fusion intelligente, détection full-page, nettoyage automatique.

## 🚀 Nouvelles fonctionnalités (Version améliorée)

### 1. Nettoyage automatique
- Suppression des petites prédictions (< 2% de la surface de page)
- Filtrage par confiance (panel_conf, balloon_conf)
- Suppression des panels/balloons trop petits

### 2. Fusion intelligente
- Fusion IoU avec seuil configurable (`panel_merge_iou`)
- Fusion par proximité (`panel_merge_dist`)
- Fusion par rangées (`panel_row_overlap`, `panel_row_gap_pct`)
- Filtrage par containment hiérarchique

### 3. Détection full-page
- Détection automatique des pages entières (`full_page_panel_pct`)
- Gestion intelligente des ballons sur pages complètes
- Suppression des faux positifs

### 4. Gutter splitting
- Découpage automatique des panels fusionnés incorrectement
- Détection des gouttières blanches
- Reconstruction des grilles régulières

### 5. Configuration complète
- Tous les paramètres dans `config/detect.yaml`
- Support pour 25+ paramètres de configuration
- Documentation complète des options

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# si PyTorch n'est pas installé, installe une version compatible Apple Silicon (MPS) :
# pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

## Utilisation

```bash
python main.py
```
1. Menu **Ouvrir PDF…** pour charger un album.
2. Menu **Charger modèle…** pour charger `anComicsViewer_v01.pt`.
3. Défile, zoome : les boîtes restent parfaitement alignées.

## Configuration

Le fichier `config/detect.yaml` contient tous les paramètres :

```yaml
# Détection et filtrage
panel_conf: 0.18
panel_area_min_pct: 0.02
balloon_conf: 0.22
max_panels: 20
max_balloons: 12

# Fusion intelligente
panel_merge_iou: 0.25
panel_merge_dist: 0.02
panel_containment_merge: 0.55
enable_panel_merge: true

# Gutter splitting
gutter_split_enable: true
gutter_min_gap_px: 6
gutter_min_contrast: 20
gutter_min_coverage: 0.75

# Full-page detection
full_page_panel_pct: 0.93
full_page_keep_balloons: true
```

## Notes techniques
- Les classes par défaut sont `["panel", "balloon"]`. Si ton modèle expose d'autres noms, ils seront utilisés automatiquement.
- Les boîtes sont **cosmétiques** (1px) et les labels **ignorent les transformations** pour rester lisibles.
- Support du cache de détection pour les performances
- Tiling intelligent pour les grandes images

## Pourquoi ça corrige tes soucis ?
- **Une seule source de vérité** (QImage unique) ⇒ pas de remap ni d'approximation.
- **Parentage correct** des overlays ⇒ Qt applique la même transform au pixmap et aux boîtes.
- **DPI fixe** côté rendu PDF ⇒ le zoom se fait dans la vue, pas en re-rendant l'image.
- **Post-traitement intelligent** ⇒ élimination des faux positifs et fusion correcte des panels adjacents.
- **Configuration complète** ⇒ adaptation facile à différents types de BD.
