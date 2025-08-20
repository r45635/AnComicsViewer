# AnComicsViewer - Archive Application Core
========================================

Cette archive contient uniquement les fichiers essentiels pour faire fonctionner AnComicsViewer.

## 📦 Contenu de l'Archive

- **main.py** - Point d'entrée principal (RECOMMANDÉ)
- **src/ancomicsviewer/** - Package Python principal
- **requirements.txt** - Dépendances de base
- **requirements-ml.txt** - Dépendances ML additionnelles
- **setup.py** & **pyproject.toml** - Configuration d'installation
- **assets/** - Icônes de l'application
- **scripts/ml/** - Scripts ML utilitaires

## 🚀 Installation Rapide

```bash
# 1. Extraire l'archive
unzip AnComicsViewer_v*.zip
cd AnComicsViewer/

# 2. Créer un environnement virtuel (recommandé)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou .venv\Scripts\activate  # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
python main.py
```

## 💾 Installation Package (Optionnel)

```bash
# Installation en mode développement
pip install -e .

# Lancement via commande système
ancomicsviewer
```

## 🎯 Utilisation

```bash
# Interface graphique normale
python main.py

# Ouvrir un fichier spécifique
python main.py mon_comic.pdf

# Avec preset Franco-Belge
python main.py --preset fb --detector multibd comics.pdf

# Aide complète
python main.py --help
```

## 📋 Dépendances Principales

- Python 3.8+
- PySide6 (interface graphique)
- OpenCV (traitement d'image)
- NumPy (calculs numériques)
- Pillow (manipulation d'images)

## ⚡ Fonctionnalités

✅ Lecteur PDF interactif  
✅ Détection automatique de cases  
✅ Détecteur Multi-BD Enhanced (94.2% mAP50)  
✅ Navigation intelligente  
✅ Cache intelligent  
✅ Interface moderne PySide6  

---
**Version**: 2.0.0+  
**Taille archive**: ~60 KB  
**Repository**: https://github.com/r45635/AnComicsViewer
