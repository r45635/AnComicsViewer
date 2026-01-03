# Test Scripts and Debugging

Ce dossier contient les scripts de test, de diagnostic et de tuning pour le détecteur de panneaux.

## Scripts disponibles

### `smoke_test.py`
Test de régression basique - vérifie que le détecteur fonctionne sur une image synthétique simple (3 panneaux).

**Utilisation:**
```bash
python tests/scripts/smoke_test.py
```

**Résultat attendu:**
- Détecte 3 panneaux sur l'image de test synthétique
- Affiche les rectangles détectés

### `diagnose_page.py`
Diagnostic détaillé d'une page PDF spécifique. Montre les étapes intermédiaires du détecteur (masks, contours, rectangles avant/après merge).

**Utilisation:**
```bash
python tests/scripts/diagnose_page.py <pdf_path> <page_number>
# Exemple:
python tests/scripts/diagnose_page.py samples_PDF/Gremillets.pdf 6
```

**Sortie:**
- Images de diagnostic sauvegardées dans `debug_output/`
- Affiche le nombre de panneaux détectés et leurs coordonnées

### `tune_detector.py`
Tuning itératif des paramètres de détection. Permet d'ajuster les seuils pour optimiser la détection sur une page donnée.

**Utilisation:**
```bash
python tests/scripts/tune_detector.py <pdf_path> <page_number>
# Exemple:
python tests/scripts/tune_detector.py samples_PDF/Gremillets.pdf 6
```

**Paramètres ajustables:**
- `adaptive_block`: Taille du bloc pour le seuillage adaptatif (défaut: 51)
- `adaptive_C`: Constante soustraite (défaut: 5)
- `morph_kernel`: Kernel pour la morphologie (défaut: 3)
- `min_area_pct`: Pourcentage minimum de surface (défaut: 0.008)

### `detect_panels.py`
Script standalone pour tester la détection sur des images ou PDFs. Supporte plusieurs modes de détection.

**Utilisation:**
```bash
python tests/scripts/detect_panels.py <image_path>
```

## Fichiers de référence

- `page4_panels.json`: Résultats de détection pour la page 4 (20 panneaux détectés)
- `debug_output/`: Images de diagnostic de la session précédente

## Notes importantes

1. La détection n'est pas parfaite sur toutes les pages (ex: page 6 peut avoir des imprécisions)
2. Les paramètres de DetectorConfig dans `ancomicsviewer/config.py` peuvent être ajustés pour améliorer la détection
3. Les scripts de test utilisent la même logique que l'application principale (imports depuis `ancomicsviewer`)
