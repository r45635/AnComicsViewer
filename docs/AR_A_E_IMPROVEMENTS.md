# 🎯 AR-A à AR-E : Améliorations Ordre de Lecture et Gestion des Titres

## ✅ Améliorations implémentées

### AR-A : Paramètre row_band_frac
- ✅ Ajouté `self.row_band_frac = 0.06` dans `PanelDetector.__init__`
- ✅ Inclus dans le log `params:` pour debugging
- 🎯 **Objectif** : Contrôler la tolérance de groupement en rangées (6% de la hauteur de page)

### AR-B : Méthode de tri par rangées  
- ✅ Implémenté `_sort_reading_order()` dans `PanelDetector`
- 🔧 **Fonctionnalités** :
  - Groupement des cases par rangées avec tolérance `row_band_frac`
  - Support lecture droite→gauche (manga) et gauche→droite (BD)
  - Tri horizontal dans chaque rangée
  - Tri vertical entre rangées

### AR-C : Utilisation du nouveau tri
- ✅ Remplacé le tri simple par `self._sort_reading_order(rects, page_point_size)`
- 🎯 **Amélioration** : Ordre de lecture plus intelligent et cohérent

### AR-D : Interface de réglage  
- ✅ Ajouté contrôle `QDoubleSpinBox` pour `row_band_frac` dans le dialog de tuning
- ✅ Plage : 0.01 à 0.20 (1% à 20% de la page)
- ✅ Application en temps réel via `_apply()`
- 🎛️ **Interface** : "Row band frac" dans les paramètres avancés

### AR-E : Preset Franco-Belge amélioré
- ✅ Configuration plus agressive pour la détection de titres :
  - `title_row_top_frac = 0.28` (28% au lieu de 20%)
  - `title_row_max_h_frac = 0.18` (18% au lieu de 12%)
  - `title_row_min_boxes = 2` (2 au lieu de 4)
  - `title_row_min_meanL = 0.88` (88% au lieu de 80%)
  - `title_row_median_w_frac_max = 0.30` (30% max)
- ✅ Paramètres optimisés :
  - `row_band_frac = 0.06`
  - `proj_smooth_k = 29` (lissage anti-glyph)

## 🎮 Comment tester

### 1. **Ordre de lecture amélioré**
```bash
.venv/bin/python main.py
# Ouvrir une BD → Detection → voir l'ordre des cases
```

### 2. **Réglage row_band_frac**
```bash
# Dans l'application :
# Detection → Panel Tuning → chercher "Row band frac"
# Ajuster entre 0.01 (strict) et 0.20 (tolérant)
```

### 3. **Preset Franco-Belge optimisé**
```bash
# Dans l'application :
# Detection → Panel Tuning → Presets → Franco-Belge
# Voir les paramètres de titre plus agressifs
```

## 🎯 Bénéfices attendus

### ✅ Problème 1 : "Champ texte complet d'un chapitre"
- **Avant** : Titres fragmentés ou mal détectés
- **Après** : 
  - Zone de titre élargie (28% au lieu de 20%)
  - Seuils plus permissifs pour les gros titres
  - Fusion intelligente des fragments de titre

### ✅ Problème 2 : "Ordre des cases"
- **Avant** : Tri simple par (top, left) sans groupement
- **Après** :
  - Groupement intelligent par rangées avec tolérance
  - Respect de l'ordre de lecture naturel
  - Support manga (droite→gauche) et BD occidentale

## 🔧 Paramètres techniques

| Paramètre | Valeur | Impact |
|-----------|--------|--------|
| `row_band_frac` | 0.06 | Tolérance groupement rangées (6% page) |
| `title_row_top_frac` | 0.28 | Zone titre étendue (28% haut page) |
| `title_row_max_h_frac` | 0.18 | Hauteur max titre (18% page) |
| `title_row_min_boxes` | 2 | Min 2 boîtes pour titre (vs 4) |
| `title_row_min_meanL` | 0.88 | Luminosité min titre (88%) |
| `proj_smooth_k` | 29 | Anti-fragmentation texte |

## 🧪 Tests de validation

```python
# Test du tri par rangées
from AnComicsViewer import PanelDetector
detector = PanelDetector()

# Vérification paramètre
assert detector.row_band_frac == 0.06

# Test tri 4 cases (2x2)
rects = [QRectF(200,100,50,50), QRectF(100,100,50,50), 
         QRectF(200,200,50,50), QRectF(100,200,50,50)]
sorted_rects = detector._sort_reading_order(rects, QSizeF(400,400))

# Ordre attendu : TL, TR, BL, BR
assert sorted_rects[0].x() == 100 and sorted_rects[0].y() == 100  # Top-Left
assert sorted_rects[1].x() == 200 and sorted_rects[1].y() == 100  # Top-Right
assert sorted_rects[2].x() == 100 and sorted_rects[2].y() == 200  # Bottom-Left
assert sorted_rects[3].x() == 200 and sorted_rects[3].y() == 200  # Bottom-Right
```

Les améliorations AR-A à AR-E sont maintenant **entièrement implémentées** et devraient considérablement améliorer la gestion des titres de chapitre et l'ordre de lecture des cases ! 🚀
