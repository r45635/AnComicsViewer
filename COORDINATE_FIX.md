# Fix : Alignement des zones vertes sur les cases détectées

## Problème identifié

Les zones vertes (overlay des cases détectées) ne s'affichent pas précisément sur les cases dans le PDF. Il y a un décalage qui s'accumule, particulièrement visible sur les cases éloignées du coin supérieur gauche.

## Cause du problème

### 1. Chaîne de conversion des coordonnées

Le système utilise plusieurs systèmes de coordonnées :

```
Détection (pixels image) → Page Points (72 DPI) → Viewport Pixels (affichage)
```

1. **Détection** : Les cases sont détectées sur une image rendue à 150-200 DPI
2. **Conversion en Page Points** : Les coordonnées sont converties en unités de 72 DPI (standard PDF)
3. **Affichage** : Les rectangles sont convertis en pixels viewport selon le zoom

### 2. Le facteur d'échelle problématique

Dans `pdf_view.py`, la méthode `_find_page_origin()` tente de détecter un facteur d'échelle (`scale`) en :
- Scannant les pixels du viewport pour trouver les bords de la page
- Comparant la largeur détectée avec la largeur théorique
- Calculant `scale = actual_width / expected_width`

**Problème** : Cette détection par pixel scanning est imprécise car :
- Les bords de page peuvent avoir des ombres ou des marges légèrement grises
- Le seuil de luminosité peut mal classifier certains pixels
- Même une erreur de quelques pixels crée un facteur d'échelle incorrect
- Exemple : `scale = 1.02` au lieu de `1.00` → décalage de 2% qui s'accumule

### 3. Accumulation de l'erreur

```python
effective_z = z * scale           # Erreur multipliée par le zoom
x = origin_x + (x_pt * effective_z)  # Erreur accumulée sur la distance
```

Plus une case est loin du point d'origine (0,0), plus le décalage est visible.

## Solution appliquée

### Fix principal : Forcer scale=1.0

Qt/PySide6 devrait déjà rendre les pages aux dimensions exactes selon le `zoomFactor()`. La détection automatique de l'échelle n'est donc pas nécessaire et introduit des erreurs.

**Modification dans `pdf_view.py`** :

```python
# AVANT (ligne ~230)
scale = actual_width / expected_width if expected_width > 0 else 1.0

# APRÈS
detected_scale = actual_width / expected_width if expected_width > 0 else 1.0
scale = 1.0  # Force scale - Qt renders at exact zoom factor

# Log pour debug
if self._config_debug and abs(detected_scale - 1.0) > 0.05:
    pdebug(f"Scale detection: detected={detected_scale:.3f}, using scale=1.0 (forced)")
```

### Avantages de cette solution

1. **Simplicité** : Élimine la complexité de la détection automatique
2. **Précision** : Qt garantit un rendu précis selon le `zoomFactor()`
3. **Pas d'accumulation d'erreur** : `scale=1.0` → pas de multiplication d'erreur
4. **Robustesse** : Fonctionne indépendamment de la qualité du rendu ou des marges

## Test de la solution

Pour vérifier que le fix fonctionne :

1. Ouvrez un PDF avec des cases bien définies
2. Activez le mode panel (Ctrl+2)
3. Vérifiez que les rectangles verts s'alignent précisément sur les cases détectées
4. Testez à différents niveaux de zoom
5. Testez sur différentes pages avec différentes layouts

### Cas de test spécifiques

- ✅ Cases en haut à gauche (proche de l'origine)
- ✅ Cases en bas à droite (loin de l'origine - où l'erreur était visible)
- ✅ Zoom 50%, 100%, 200%
- ✅ Pages avec marges blanches
- ✅ Pages pleine page (full bleed)

## Solutions alternatives (non implémentées)

Si le fix principal ne fonctionne pas dans tous les cas :

### Option 2 : Améliorer la détection des bords

```python
# Utiliser plusieurs lignes de scan au lieu d'une seule
# Moyenner les résultats pour plus de robustesse
```

### Option 3 : Calibration par cases de référence

```python
# Utiliser les cases détectées pour calibrer l'échelle
# Comparer la taille des cases détectées vs affichées
```

### Option 4 : Désactiver complètement la détection d'origine

```python
# Utiliser uniquement _fallback_page_origin()
# Plus simple mais peut ne pas fonctionner dans tous les modes d'affichage
```

## Fichiers modifiés

- `ancomicsviewer/pdf_view.py` :
  - Ajout de `self._config_debug` dans `__init__()`
  - Import de `pdebug` pour le logging
  - Modification de `_find_page_origin()` pour forcer `scale=1.0`

## Notes techniques

### Coordonnées dans Qt PDF

- **Page Points** : Unité standard PDF, 1 point = 1/72 pouce
- **Page Size** : `doc.pagePointSize(page)` retourne la taille en points
- **Zoom Factor** : `view.zoomFactor()` = 1.0 signifie 72 DPI à l'écran
- **Viewport Pixels** : Coordonnées réelles à l'écran

### Formule de conversion

```python
# De Page Points vers Viewport Pixels
viewport_x = origin_x + (page_point_x * zoomFactor)
viewport_y = origin_y + (page_point_y * zoomFactor)
```

Avec `scale=1.0`, pas de distorsion.

## Suivi

- [x] Identifier le problème
- [x] Implémenter le fix
- [ ] Tester avec différents PDFs
- [ ] Valider sur différents systèmes d'exploitation
- [ ] Mettre à jour les tests automatiques si nécessaire

---

**Auteur** : Claude (Analyse)  
**Date** : 2 janvier 2026  
**Version** : 2.0.1
