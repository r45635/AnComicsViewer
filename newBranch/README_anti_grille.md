# AnComicsViewer - Patch Anti-Grille + Profil Qualité

## 🎯 Objectif
Ce patch résout le problème des **faux panneaux "en damier"** dus au tuilage en implémentant :
- ✅ **Détection d'artefacts de tuiles** avec suppression automatique des grilles
- ✅ **Profil qualité** avec paramètres améliorés pour une meilleure précision
- ✅ **Conservation du patch "full-page"** déjà en place

## 📋 Changements Implémentés

### 1. Configuration YAML - Profil Qualité Restauré

**Fichier :** `config/detect.yaml`

```yaml
# --- inference quality profile ---
imgsz_max: 1536          # (était 960) - Résolution max augmentée
tile_target: 1024        # (était 768) - Taille de tuile augmentée  
tile_overlap: 0.25       # (était 0.15) - Chevauchement augmenté
panel_conf: 0.18         # (était 0.15) - Seuil panneaux plus strict
max_det: 600             # (était 200) - Plus de détections max
```

### 2. Heuristique Anti-Grille

**Fichier :** `main.py` - Méthode `_run_detection()`

**Emplacement :** Après les merges IoU/containment, avant la détection full-page

**Logique de détection :**
- 🔍 **Analyse des dimensions** : Vérifie si la médiane des panels correspond aux dimensions de tuiles (±35%)
- 📊 **Comptage de grille** : Détecte si ≥8 panels forment une grille uniforme
- 🎯 **Répartition spatiale** : Vérifie la distribution homogène dans les colonnes/lignes
- ⚡ **Action corrective** : Remplace la grille par un panel pleine page + filtrage ballons

### 3. Intégration avec Full-Page

Les deux heuristiques se complètent :
1. **Anti-grille** → Détecte et corrige les artefacts de tuilage
2. **Full-page** → Gère les couvertures de comics naturally grandes

## 🚀 Utilisation

```bash
cd newBranch
python main.py
```

L'application détectera automatiquement :
- **Pages grillées** → `panels=1` au lieu de 20+ artefacts
- **Couvertures** → Panel unique avec bulles filtrées si configuré  
- **Pages normales** → Détection précise avec le profil qualité

## 📊 Résultats Attendus

### Avant le patch
- Status: `panels=25, balloons=3` (artefacts de grille)
- Nombreux rectangles de taille similaire alignés

### Après le patch  
- Status: `panels=1, balloons=2` (grille supprimée)
- Un seul panel couvrant la page entière

## ⚙️ Configuration

Les paramètres peuvent être ajustés dans `config/detect.yaml` :

```yaml
# Seuil de détection de grille (minimum 8 panels)
# Tolérance dimensionnelle: ±35% de la taille de tuile
# Répartition: ≥50% des colonnes ET lignes remplies

# Si détection positive -> collapse vers panel pleine page
full_page_keep_balloons: true          # Garder les bulles chevauchantes
full_page_balloon_overlap_pct: 0.15    # Seuil de chevauchement
```

## 🔧 Débogage

Pour tester sur une page spécifique :
```bash
cd tools
python eval_one_page.py ../path/to/comic.pdf page_number
```

## 📦 Contenu du Package

- `main.py` - Application avec patch anti-grille intégré
- `config/detect.yaml` - Profil qualité optimisé
- `anComicsViewer_v01.pt` - Modèle YOLO
- `tools/eval_one_page.py` - Outil de test CLI
- `requirements.txt` - Dépendances

---

**Status :** ✅ Implémenté et testé  
**Compatibilité :** Patch rétrocompatible avec les fonctionnalités existantes
