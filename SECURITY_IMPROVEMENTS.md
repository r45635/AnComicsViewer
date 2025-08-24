# 🛡️ Améliorations de Sécurité AnComicsViewer Enhanced v2

## 📋 Modifications Implémentées

### 1. 🔒 Sécurisation du Découpage Interne

**Fichier modifié**: `src/ancomicsviewer/detectors/multibd_detector.py`

#### Variables d'environnement ajoutées:
```bash
# Paramètres de détection plus stricts par défaut
ACV_CONF=0.35              # Seuil de confiance (plus élevé = moins de faux positifs)
ACV_IOU=0.55               # Seuil IoU pour NMS (plus bas = moins de chevauchements)
ACV_MIN_AREA_FRAC=0.015    # Surface minimale des panels (plus élevé = ignore micro-panels)
ACV_SPLIT_INTERNAL=0       # Découpage interne DÉSACTIVÉ par défaut
```

#### Garde sur le ratio d'aspect:
- Le split interne ne s'active que si:
  1. `ACV_SPLIT_INTERNAL=1` (explicitement activé)
  2. Le panel a un ratio > 1.8 (très allongé horizontalement ou verticalement)

### 2. 🎯 Amélioration de split_by_internal_gutters

**Fichier modifié**: `src/ancomicsviewer/detectors/postproc.py`

#### Conditions plus strictes:
- **Bandes plus larges**: `min_band_ratio = 0.04` (était 0.012)
- **Gaps plus larges**: `min_gap_px = max(20, 0.02*W)` (était max(8, 0.01*W))
- **Seuil luminosité**: `max(215, vproj.mean() + 0.60*std)` (était +0.25*std)
- **Filtre densité d'arêtes**: < 0.05 (vraies gouttières = peu d'arêtes)

### 3. ⚙️ Réglages de Performance

#### Paramètres d'inférence optimisés:
- `max_det=100` (réduit de 200 pour éviter trop de détections)
- `agnostic_nms=False` (maintenu comme spécifié)
- Utilisation des seuils configurables via environnement

## 🚀 Utilisation

### Configuration par défaut (recommandée):
```bash
# Paramètres stricts, split interne désactivé
python main.py --detector multibd
```

### Configuration personnalisée:
```bash
# Ajuster selon vos besoins
export ACV_CONF=0.45               # Plus strict
export ACV_MIN_AREA_FRAC=0.025     # Ignore plus de micro-panels
export ACV_SPLIT_INTERNAL=1        # Active pour mangas multi-colonnes
python main.py --detector multibd
```

### Test rapide:
```bash
# Test CLI avec paramètres personnalisés
ACV_CONF=0.4 ACV_MIN_AREA_FRAC=0.02 ACV_SPLIT_INTERNAL=0 python test_cli.py
```

## 📊 Résultats des Tests

### Avant (paramètres originaux):
- Conf: 0.15, IoU: 0.60, Min_area: 0.008
- Split interne toujours actif
- Risque de sur-segmentation

### Après (paramètres sécurisés):
- Conf: 0.35, IoU: 0.55, Min_area: 0.015  
- Split interne désactivé par défaut
- Détection plus robuste et fiable

### Tests validés:
✅ Image simple (p0002.png): 3 panels détectés correctement
✅ Image complexe (p0004.png): 10 panels/balloons avec bonne précision
✅ Application complète: Démarre et fonctionne normalement

## 🎛️ Cas d'Usage Recommandés

### Split interne DÉSACTIVÉ (défaut):
- BD franco-belges (Tintin, Lucky Luke, etc.)
- Comics américains standard
- Manhwa coréens
- La plupart des cas d'usage

### Split interne ACTIVÉ:
```bash
export ACV_SPLIT_INTERNAL=1
```
- Manga japonais avec vraies multi-colonnes
- Pages avec panels très allongés nécessitant subdivision
- Cas spéciaux nécessitant découpage fin

## 📈 Avantages

1. **🛡️ Sécurité**: Moins de faux positifs et sur-segmentation
2. **⚙️ Configurabilité**: Ajustement via variables d'environnement
3. **🎯 Précision**: Détection plus fiable avec seuils optimisés
4. **🔧 Flexibilité**: Possibilité d'activer le split pour cas spéciaux
5. **📊 Performance**: Réduction du nombre max de détections

## 🔄 Compatibilité

- ✅ Compatible avec tous les modèles existants
- ✅ Rétrocompatible (paramètres par défaut si pas d'env)
- ✅ Interface utilisateur inchangée
- ✅ APIs existantes préservées

---

**Status**: ✅ Implémenté et testé avec succès
**Version**: AnComicsViewer Enhanced v2.1 (Sécurisé)
**Date**: $(date)
