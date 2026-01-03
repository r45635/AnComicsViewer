# Freeform Panel Detection - Technical Documentation

## Vue d'ensemble

Le système de détection **freeform** est un fallback robuste pour les planches BD complexes où la détection classique (basée sur les bords noirs et les gouttières blanches) échoue. Il est conçu pour gérer :

- 📐 Panneaux en forme de **parallélogrammes** ou rectangles orientés
- 🎨 Fonds **colorés** ou **teintés** (vert pâle, bleu, etc.) où les gouttières ne sont pas blanches
- 🔀 Layouts **complexes** avec formes libres
- 🖼️ Pages avec un seul grand panneau englobant

## Architecture

### 1. Déclenchement du fallback

Le fallback freeform s'active automatiquement si :
- **Trop peu de panels** : `len(panels) < 2`
- **Un panel trop grand** : Un panel couvre > 50% de la page
- Paramètre `use_freeform_fallback=True` dans `DetectorConfig`

### 2. Pipeline de détection

```
Image BGR → Estimation fond → Masque fond → Watershed → Extraction régions → Merge → Tri lecture → QRectF
```

#### Étape 1 : Estimation de la couleur de fond
**Fonction** : `estimate_background_color_lab()`

- Échantillonne les **bords** de l'image (3% par défaut)
- Convertit en espace **Lab** (perceptuel)
- Calcule la **médiane** (robuste aux pixels d'encre)
- Retourne `(L, a, b)` médian

**Paramètre clé** : `sample_pct=0.03`

#### Étape 2 : Création du masque de fond
**Fonction** : `make_background_mask()`

- Convertit l'image en Lab
- Calcule la **distance euclidienne** pixel-par-pixel à `bg_lab`
- Seuillage : `dist < bg_delta` → fond
- Nettoyage morphologique :
  - **OPEN** (kernel 3x3) : enlever le bruit
  - **CLOSE** (kernel 7x7) : combler les micro-trous

**Paramètres clés** :
- `bg_delta` (défaut: 15.0) : tolérance Lab pour la distance de fond
- Plus élevé → plus de pixels considérés comme fond

**Sortie debug** : `freeform_bg_mask.png`

#### Étape 3 : Segmentation Watershed
**Fonction** : `segment_panels_watershed()`

##### 3.1 Préparation
- **Foreground mask** : `mask_fg = NOT(mask_bg)`
- Nettoyage : OPEN + CLOSE (kernel 5x5)

##### 3.2 Marqueurs sûrs
- **Sure background** : dilate `mask_bg` (2 itérations)
- **Sure foreground** :
  - Distance transform sur `mask_fg`
  - Seuillage : `dist > sure_fg_ratio * dist_max`
  - Crée des "graines" au centre des objets

**Paramètre clé** : `sure_fg_ratio` (défaut: 0.35)
- Plus bas → plus de graines → plus de régions détectées
- Plus haut → moins de graines → régions fusionnées

##### 3.3 Région inconnue
- `unknown = sure_bg - sure_fg`
- Pixels à assigner par watershed

##### 3.4 Markers et Watershed
- `connectedComponents(sure_fg)` → labels initiaux
- `markers[unknown] = 0`
- `cv2.watershed(img_bgr, markers)` → labels finaux

**Sorties debug** :
- `freeform_mask_fg.png`
- `freeform_sure_fg.png`
- `freeform_markers.png`

#### Étape 4 : Extraction des régions
**Fonction** : `extract_panel_regions()`

Pour chaque label watershed (> 1) :

1. **Extraction contour**
   - Masque binaire pour ce label
   - `findContours()` → prendre le plus grand

2. **Calcul propriétés**
   - `area = contourArea()`
   - `bbox = boundingRect()` → (x, y, w, h)
   - `fill_ratio = area / (w*h)`
   - `obb = minAreaRect() → boxPoints()` (4 points orientés)
   - `poly = approxPolyDP()` (simplification)
   - `centroid` via moments

3. **Filtrage**
   - `area >= min_area_ratio * img_area` (défaut: 0.005 = 0.5%)
   - `area <= max_area_ratio * img_area` (défaut: 0.95)
   - `fill_ratio >= min_fill_ratio` (défaut: 0.15)

**Paramètres clés** :
- `min_area_ratio_freeform` : 0.005 (panels très petits acceptés)
- `min_fill_ratio_freeform` : 0.15 (formes assez remplies)
- `approx_eps_ratio` : 0.01 (précision polygone)

**Classe** : `PanelRegion`
```python
@dataclass
class PanelRegion:
    contour: NDArray          # Nx1x2
    poly: NDArray             # Simplifié
    bbox: (x, y, w, h)       # Aligné axes
    obb: NDArray             # 4 points orientés
    area: float
    fill_ratio: float
    touches_border: bool
    centroid: (cx, cy)
```

#### Étape 5 : Merge des chevauchements
**Fonction** : `merge_overlapping_regions()`

- Calcule IoU (Intersection over Union) sur bboxes
- Si `IoU > iou_merge_thr` → fusionner via `convexHull()`
- Recalcule propriétés de la région fusionnée

**Paramètre clé** : `iou_merge_thr` (défaut: 0.20)

#### Étape 6 : Tri en ordre de lecture
**Fonction** : `sort_reading_order()`

1. Trier par `centroid_y`
2. Grouper en "lignes" : même ligne si `|cy - cy_ref| < 0.5 * median_height`
3. Trier chaque ligne par `centroid_x` (ou inverse si RTL)

**Paramètre** : `reading_rtl` (sens de lecture)

#### Étape 7 : Conversion finale
- `PanelRegion.to_qrectf(scale)` → utilise `bbox` pour QRectF
- Compatible avec le reste du pipeline

**Sortie debug** : `freeform_regions_contours.png`
- Vert : contours
- Bleu : bbox
- Rouge : obb
- Jaune : numéros d'ordre

## Configuration

### Paramètres dans `DetectorConfig`

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `use_freeform_fallback` | `True` | Active/désactive le fallback |
| `bg_delta` | 15.0 | Distance Lab pour fond (plus haut = plus tolérant) |
| `sure_fg_ratio` | 0.35 | Ratio distance transform (plus bas = plus de seeds) |
| `min_area_ratio_freeform` | 0.005 | Surface min panel (% page) |
| `min_fill_ratio_freeform` | 0.15 | Taux remplissage min |
| `iou_merge_thr` | 0.20 | Seuil IoU pour merge |
| `approx_eps_ratio` | 0.01 | Précision polygone |

### Tuning recommandé par cas

**Cas 1 : Fond coloré uniforme (vert/bleu pâle)**
```python
config = DetectorConfig(
    bg_delta=18.0,           # Plus tolérant
    sure_fg_ratio=0.30,      # Plus de seeds
    min_fill_ratio_freeform=0.12  # Formes + variées
)
```

**Cas 2 : Parallélogrammes / formes orientées**
```python
config = DetectorConfig(
    min_fill_ratio_freeform=0.20,  # Formes bien définies
    approx_eps_ratio=0.02          # Polygones plus simples
)
```

**Cas 3 : Beaucoup de petits panels**
```python
config = DetectorConfig(
    min_area_ratio_freeform=0.003,  # Panels très petits OK
    sure_fg_ratio=0.25,             # Beaucoup de seeds
    iou_merge_thr=0.15              # Merge moins agressif
)
```

## Debugging

### Images générées (si `debug=True`)

Dans `debug_output/` :

1. **freeform_bg_mask.png** : Masque de fond (blanc = fond)
2. **freeform_mask_fg.png** : Masque premier plan (blanc = contenu)
3. **freeform_sure_fg.png** : Seeds foreground (après distance transform)
4. **freeform_markers.png** : Labels watershed normalisés
5. **freeform_regions_contours.png** : Visualisation finale

### Logs clés

```
[Freeform] Triggering fallback: single large panel covering 51.9% of page
[Freeform] Background Lab: L=255.0, a=128.0, b=128.0
[Freeform] Background mask: 47.9% of image, delta=15.0
[Freeform] Distance transform max=422.2, threshold=147.8
[Freeform] Connected components found: 5
[Freeform] Watershed produced 5 regions
[Freeform] Extracted 2 regions after filtering
[Freeform] After merge: 2 regions
[Freeform] Sorted 2 regions into 2 rows
```

### Script de test

```bash
python tests/scripts/test_freeform.py "samples_PDF/mycomic.pdf" 6
```

## Limitations actuelles

1. **Panels très fins** : Les bandes verticales étroites peuvent être filtrées si `fill_ratio` trop bas
2. **Texte hors cases** : Peut créer des fausses régions si le texte est dense
3. **Fond dégradé** : Marche moins bien si le fond n'est pas uniforme
4. **Overlap complexe** : Le merge par IoU peut rater certains cas de panels imbriqués

## Améliorations futures

- [ ] Split automatique des grandes régions (détection de multi-panels collés)
- [ ] Détection adaptative de `bg_delta` basée sur variance locale
- [ ] Support des fonds en dégradé via clustering k-means
- [ ] Filtrage spécifique des bulles de texte (forme circulaire/ovale)
- [ ] Export des `poly` et `obb` pour crop perspective (futur)

## Auteur & Date

Implémenté le 3 janvier 2026 pour AnComicsViewer.
