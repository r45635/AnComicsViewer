# Fast & Stable Inference Optimizations

## ✅ Optimisations Implémentées

### 1. **YOLO Predict Optimizations**
- ✅ Remplacé `augment=True` par `augment=False` pour plus de rapidité
- ✅ `imgsz = min(IMGSZ_MAX, max(W, H))` - pas de demi-tailles dynamiques
- ✅ Single pass quand `max(W,H) <= TILE_TGT * 1.15 and not force_tiling`
- ✅ Tiling limité à maximum 3x3 (jamais plus de 9 tuiles)
- ✅ Remplacé `max_det=500` par `max_det=MAX_DET` (configurable, défaut: 400)

### 2. **Detection Cache System**
- ✅ Cache par page PDF avec clé SHA1 basée sur:
  - `pdf_path:page_index:page_widthxpage_height:model_name:IMGSZ_MAX:TILE_TGT`
- ✅ Cache en mémoire + cache disque (fichiers `.pkl`)
- ✅ Évite la recomputation lors de la revisite des pages

### 3. **Postprocessing Optimizations**
- ✅ Filtrage strict après NMS:
  - Panels: `conf >= PANEL_CONF` AND `area >= 1.2% page`
  - Balloons: `area >= BAL_MIN_PCT` AND `w >= BAL_MIN_W` AND `h >= BAL_MIN_H`
- ✅ Limitation des ballons à `MAX_BAL` (défaut: 12)

### 4. **Status Bar Information**
- ✅ Affichage détaillé: `Page N: panels=X, balloons=Y (imgsz=..., tiles=nxn, cache=hit/miss)`
- ✅ Indication des hits/miss du cache
- ✅ Information sur la stratégie de tiling utilisée

## 🚀 Améliorations de Performance

### **Stratégie de Tiling Intelligente**
```python
if max_side <= TILE_TGT * 1.15 and not FORCE_TILING:
    # Single pass - pas de tiling
    tiles_info = "tiles=1x1"
else:
    # Tiling limité 2x2 ou 3x3 maximum
    if max_side <= TILE_TGT * 2: nx, ny = 2, 2
    elif max_side <= TILE_TGT * 3: nx, ny = 3, 3
    else: nx, ny = 3, 3  # Cap à 3x3 même pour très grandes images
```

### **Cache Performance**
- **Memory Cache**: Accès instantané aux résultats récents
- **Disk Cache**: Persistance entre sessions
- **Cache Key**: Détection intelligente des changements de paramètres

### **Detection Filtering**
- Suppression précoce des détections faibles
- Limitation proactive du nombre de ballons
- Évite le traitement inutile des éléments non valides

## ⚙️ Configuration

### **Paramètres de Performance** (config/detect.yaml)
```yaml
# Performance optimisée
imgsz_max: 1280          # Réduit de 1536 → 1280
tile_target: 896         # Réduit de 1024 → 896  
tile_overlap: 0.20       # Réduit de 0.25 → 0.20
max_det: 400             # Réduit de 500 → 400
max_balloons: 12         # Réduit de 20 → 12

# Seuils plus stricts
panel_conf: 0.18         # Augmenté de 0.08 → 0.18
force_tiling: false      # Évite le tiling forcé
```

## 📊 Bénéfices Attendus

1. **Vitesse d'Inférence**: 40-60% plus rapide grâce à:
   - `augment=False`
   - Single pass quand possible
   - Tiling limité à 9 tuiles max
   - Paramètres optimisés

2. **Navigation Fluide**: Cache évite la recomputation:
   - Pages déjà visitées = accès instantané
   - Pas de délai lors des allers-retours

3. **Utilisation Mémoire**: Optimisée via:
   - Limitation du nombre de détections
   - Cache avec gestion automatique
   - Filtrage précoce

4. **Stabilité**: Paramètres conservateurs évitent:
   - Over-fitting des tuiles
   - Surcharge de détections
   - Problèmes de mémoire

## 🔧 Debug & Monitoring

### **Status Bar Informatif**
```
Page 5: panels=8, balloons=3 (imgsz=1280, tiles=2x2, cache=miss)
Page 6: panels=6, balloons=5 (imgsz=1280, cache=hit)
```

### **Cache Location**
- Répertoire: `.detection_cache/`
- Fichiers: `{sha1_hash}.pkl`
- Nettoyage: Manuel (les fichiers sont petits)

## 🎯 Usage Recommandé

### **Pour BD Classiques**
- Configuration par défaut optimale
- Cache très efficace (pages similaires)
- Single pass souvent suffisant

### **Pour Très Grandes Images**
- Tiling automatique 2x2 ou 3x3
- Limitation stricte évite l'explosion
- Cache sauvegarde les résultats longs

### **Pour Performance Maximum**
```yaml
imgsz_max: 1024          # Encore plus rapide
tile_target: 768         # Tuiles plus petites
max_det: 300             # Moins de détections
```
