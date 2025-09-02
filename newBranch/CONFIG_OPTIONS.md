# Configuration des Options de Détection

Ce fichier décrit les nouvelles options de configuration disponibles dans `config/detect.yaml` pour contrôler le comportement de la détection et fusion des panels.

## Options de Fusion des Panels

### Fusion Intelligente (`enable_panel_merge`)
- **Valeur par défaut**: `false`
- **Description**: Active/désactive la fusion intelligente des panels adjacents
- **Utilisation**: Mettre à `false` si les panels sont incorrectement fusionnés

```yaml
enable_panel_merge: false  # Désactive la fusion intelligente
```

### Fusion par Rangées (`enable_row_merge`)
- **Valeur par défaut**: `true`
- **Description**: Active/désactive la fusion des panels sur la même rangée horizontale
- **Utilisation**: Mettre à `false` si les panels horizontaux sont trop fusionnés

```yaml
enable_row_merge: false  # Désactive la fusion par rangées
```

### Anti-Grille (`enable_antigrille`)
- **Valeur par défaut**: `true`
- **Description**: Active/désactive la détection et suppression des artefacts de grille de tuiles
- **Utilisation**: Mettre à `false` si la page entière est incorrectement détectée comme un artefact

```yaml
enable_antigrille: false  # Désactive l'anti-grille
```

## Scénarios d'Utilisation

### Problème: Trop de panels fusionnés
Si vous constatez que des panels séparés sont incorrectement fusionnés :

```yaml
enable_panel_merge: false    # Désactive la fusion intelligente
enable_row_merge: false      # Désactive la fusion par rangées
panel_merge_iou: 0.5         # Ou augmentez ce seuil (plus strict)
panel_row_overlap: 0.8       # Ou augmentez ce seuil (plus strict)
```

### Problème: Performance lente
Pour améliorer les performances :

```yaml
imgsz_max: 1024              # Réduire la taille d'image
tile_target: 768             # Réduire la taille des tuiles
max_det: 300                 # Réduire le nombre de détections
enable_row_merge: false      # Désactiver les fusions coûteuses
```

### Problème: Page entière détectée comme artefact
Si une page de BD normale est détectée comme grille d'artefacts :

```yaml
enable_antigrille: false     # Désactiver complètement
# OU ajuster les seuils :
antigrille_min_count: 12     # Augmenter le nombre min de panels
antigrille_grid_fill: 0.7    # Augmenter le taux de remplissage requis
```

## Configuration Recommandée pour BD Classiques

```yaml
# Configuration conservative pour BD traditionnelles
enable_panel_merge: true
enable_row_merge: false      # Souvent pas nécessaire
enable_antigrille: true
panel_merge_iou: 0.4
panel_row_overlap: 0.7
imgsz_max: 1280
tile_target: 896
```

## Configuration Recommandée pour Webtoons

```yaml
# Configuration pour webtoons (panels verticaux)
enable_panel_merge: true
enable_row_merge: false      # Pas adapté aux webtoons
enable_antigrille: false     # Peut interférer avec les longs scrolls
panel_merge_iou: 0.3
```
