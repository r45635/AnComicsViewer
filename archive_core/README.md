# AnComicsViewer - Archive Core
=============================

Ce répertoire contient les outils pour créer des archives de distribution d'AnComicsViewer.

## 📦 Archives Disponibles

### `AnComicsViewer_Core_App_v2.0.0.zip` (58 KB)
Archive légère contenant uniquement les fichiers essentiels pour faire fonctionner l'application :

- ✅ Code source principal (`main.py`, `src/ancomicsviewer/`)
- ✅ Configuration (`setup.py`, `requirements.txt`)
- ✅ Assets de base (icônes)
- ✅ Scripts ML utilitaires essentiels
- ❌ Pas de datasets volumineux
- ❌ Pas de modèles ML (18+ MB)
- ❌ Pas de documentation développeur

## 🛠️ Scripts de Génération

### `create_simple_archive.py`
Script simplifié et fiable pour créer l'archive de distribution :
```bash
cd archive_core/
python create_simple_archive.py
```

### `create_app_archive.py`
Version complète avec scan automatique des fichiers (peut être lente) :
```bash
cd archive_core/
python create_app_archive.py
```

## 🎯 Utilisation de l'Archive

```bash
# Extraction
unzip AnComicsViewer_Core_App_v2.0.0.zip
cd AnComicsViewer/

# Installation
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# Lancement
python main.py
```

## 📋 Contenu de l'Archive

**Fichiers principaux :**
- `main.py` - Point d'entrée CLI avec arguments
- `src/ancomicsviewer/` - Package Python principal
- `setup.py` + `pyproject.toml` - Installation pip
- `requirements.txt` - Dépendances de base

**Fonctionnalités incluses :**
- ✅ Lecteur PDF interactif
- ✅ Détection heuristique de cases (sans ML)
- ✅ Interface PySide6 moderne
- ✅ Cache intelligent
- ✅ Scripts ML pour entraînement (optionnel)

**Modèles ML (téléchargement séparé) :**
- YOLO pré-entraînés (téléchargés automatiquement)
- Multi-BD Enhanced v2.0 (disponible séparément)

## 🔧 Maintenance

Pour mettre à jour l'archive :
1. Modifier `create_simple_archive.py` si nécessaire
2. Exécuter le script
3. Tester l'archive extraite
4. Commit et tag git pour la version

---
**Dernière mise à jour** : 19 août 2025  
**Taille archive** : ~58 KB  
**Fichiers inclus** : 26 fichiers essentiels
