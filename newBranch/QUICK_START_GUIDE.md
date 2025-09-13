# 🚀 Guide d'utilisation rapide - AnComicsViewer Enhanced

## Nouvelles fonctionnalités

### 📏 Calibration pixel↔PDF
- Rendu à DPI fixe (300) pour stabilité
- Conversion précise entre coordonnées pixel et PDF
- Méthodes `pixel_to_pdf_rect()` et `pdf_to_pixel_rect()`

### 📊 Métriques de qualité
- Score de qualité automatique (0-1)
- Détection des overlaps et severe overlaps
- Export JSON optionnel des métriques

### 🎯 Post-traitement raffiné
- NMS class-aware (panels et balloons séparément)
- Filtres de taille, aire, et marge
- Règle d'attachement balloon→panel
- Limitations de quantité configurables

## 🖥️ Utilisation CLI

### Usage basique
```bash
python3 main.py --pdf comic.pdf --page 5
```

### Avec export de métriques
```bash
python3 main.py --pdf comic.pdf --page 5 --metrics-out metrics.json
```

### Avec configuration personnalisée
```bash
python3 main.py --pdf comic.pdf --config my_config.yaml --metrics-out metrics.json
```

### Mode debug complet
```bash
python3 main.py --pdf comic.pdf --page 5 \
    --debug-detect \
    --save-debug-overlays debug \
    --metrics-out metrics.json \
    --config config/detect_with_merge.yaml
```

## ⚙️ Configuration YAML

### Créer un fichier de config personnalisé
```yaml
# Seuils de confiance
panel_conf: 0.30          # Seuil pour les panels
balloon_conf: 0.38        # Seuil pour les balloons

# NMS par classe  
panel_nms_iou: 0.30       # NMS IoU pour panels
balloon_nms_iou: 0.25     # NMS IoU pour balloons

# Filtres de taille
panel_area_min_pct: 0.03  # Min 3% de la page pour un panel
panel_area_max_pct: 0.90  # Max 90% de la page pour un panel
balloon_area_min_pct: 0.0020  # Min 0.2% pour un balloon
balloon_area_max_pct: 0.30    # Max 30% pour un balloon

# Filtres physiques
min_box_w_px: 32          # Largeur minimum en pixels
min_box_h_px: 28          # Hauteur minimum en pixels
page_margin_inset_pct: 0.015  # Marge de page (1.5%)

# Règle d'attachement
balloon_min_overlap_panel: 0.06  # Min 6% overlap pour attacher balloon→panel

# Limites de sortie
max_panels: 12            # Max panels à garder
max_balloons: 24          # Max balloons à garder
```

## 📊 Format des métriques JSON

```json
[
  {
    "page_index": 0,
    "panels": 3,
    "balloons": 8,
    "panel_area_ratios": [0.15, 0.12, 0.18],
    "balloon_area_ratios": [0.008, 0.012, ...],
    "overlaps": 2,
    "severe_overlaps": 0,
    "quality_score": 0.876
  }
]
```

## 🎛️ Interface utilisateur

### Contrôles existants conservés
- **Open PDF** : Charger un PDF
- **◀ Prev / Next ▶** : Navigation
- **Panels** : Toggle affichage panels  
- **Balloons** : Toggle affichage balloons
- **Fit Window** : Ajuster à la fenêtre
- **Mode combo** : HYBRID/YOLO/RULES

### Nouvelles informations affichées
- Status bar étendu : `Page X: panels=N, balloons=M | quality=0.XXX`
- Métriques en temps réel
- Indication de la qualité de détection

## 🔧 Scripts utiles

### Test de l'implémentation
```bash
python3 test_implementation.py
```

### Démonstration complète
```bash
./demo.sh
```

### Traitement batch
```bash
# Traiter les pages 0 à 10
for i in {0..10}; do
  python3 main.py --pdf comic.pdf --page $i --metrics-out batch_metrics.json
done
```

## 🐛 Debug et diagnostic

### Overlays de debug
```bash
python3 main.py --pdf comic.pdf --debug-detect --save-debug-overlays debug_dir
```
Génère des images avec les détections visualisées.

### Vérification des métriques
```bash
python3 -c "
import json
with open('metrics.json', 'r') as f:
    data = json.load(f)
for item in data:
    print(f'Page {item[\"page_index\"]}: quality={item[\"quality_score\"]:.3f}')
"
```

## ⚡ Conseils de performance

### Pour des PDFs volumineux
- Utilisez `imgsz_max: 1536` dans le YAML
- Réduisez `max_det: 300` si trop lent
- Activez le tiling automatique

### Pour une qualité maximale
- Augmentez `imgsz_max: 2048`
- Réduisez les seuils de confiance
- Utilisez `tile_target: 1024`

### Pour réduire le bruit
- Augmentez `panel_conf` et `balloon_conf`
- Réduisez `max_panels` et `max_balloons`
- Augmentez `*_area_min_pct`

## 🎯 Résultats attendus

### Avant les améliorations
- Détections bruyantes
- Fausses détections en bordure
- Balloons orphelins conservés
- Overlaps non contrôlés

### Après les améliorations
- Détections plus propres
- Filtrage intelligent du bruit
- Balloons attachés aux panels
- Métriques de qualité objectives
- Calibration pixel↔PDF précise

## 📞 Support

Pour toute question ou problème :
1. Vérifiez `IMPLEMENTATION_SUMMARY.md`
2. Lancez `test_implementation.py`
3. Consultez les logs de debug
4. Vérifiez la configuration YAML
