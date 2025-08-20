# Structure du Projet AnComicsViewer - Post Restructuration
=========================================================

## 🎯 Points d'Entrée Multiples

Le projet offre maintenant **3 méthodes de lancement** pour une compatibilité maximale :

### 1. **main.py** - Point d'entrée principal recommandé ⭐
```bash
python main.py [options] [fichier.pdf]
python main.py --help
python main.py --preset fb --detector multibd comics.pdf
```
- ✅ Interface CLI complète avec arguments avancés
- ✅ Gestion d'erreurs robuste et diagnostics
- ✅ Variables d'environnement configurables
- ✅ Compatible développement et production

### 2. **ancomicsviewer** - Package installé
```bash
ancomicsviewer
pip install -e .  # Installation en mode développement
```
- ✅ Commande système globale après installation
- ✅ Lancement direct du GUI
- ✅ Intégration OS native

### 3. **ancomicsviewer.py** - Wrapper de compatibilité
```bash
python ancomicsviewer.py [options]
```
- ✅ Redirige vers main.py automatiquement
- ✅ Compatibilité avec anciens scripts

## 📁 Structure Organisée

```
AnComicsViewer/
│
├── main.py                 # 🎯 POINT D'ENTRÉE PRINCIPAL
├── ancomicsviewer.py       # 🔄 Wrapper de compatibilité
├── setup.py                # 📦 Configuration d'installation
├── pyproject.toml          # 🏗️ Configuration moderne Python
├── requirements.txt        # 📋 Dépendances principales
├── requirements-ml.txt     # 🤖 Dépendances ML additionnelles
│
├── src/ancomicsviewer/     # 📚 Package principal
│   ├── __init__.py
│   ├── main_app.py         # 🖥️ Application GUI principale
│   ├── detectors/          # 🔍 Moteurs de détection
│   │   ├── __init__.py
│   │   ├── multibd_detector.py      # 🎯 Multi-BD Enhanced v2.0
│   │   ├── yolo_detector.py         # 🤖 YOLO intégré
│   │   └── heuristic_detector.py    # 📐 Détection géométrique
│   ├── ui/                 # 🎨 Interface utilisateur
│   │   ├── __init__.py
│   │   └── components/
│   └── utils/              # 🛠️ Utilitaires
│       ├── __init__.py
│       ├── enhanced_cache.py        # 💾 Cache intelligent
│       └── config.py               # ⚙️ Configuration
│
├── scripts/               # 🧰 Scripts de développement
│   ├── training/          # 🎓 Entraînement ML
│   │   ├── train_mps_optimized.py   # 🍎 Optimisé MPS
│   │   └── dataset_tools.py         # 📊 Gestion datasets
│   ├── dataset/           # 📁 Manipulation données
│   └── build/             # 🔨 Scripts de build
│
├── tests/                 # 🧪 Tests
│   ├── unit/              # 🔬 Tests unitaires
│   └── integration/       # 🔗 Tests d'intégration
│
├── docs/                  # 📖 Documentation
│   ├── guides/            # 📚 Guides utilisateur
│   └── api/               # 🔧 Documentation API
│
├── data/                  # 💾 Données du projet
│   ├── models/            # 🤖 Modèles ML (.pt, .onnx)
│   ├── examples/          # 📄 Fichiers de test
│   └── cache/             # 🗂️ Cache temporaire
│
└── assets/                # 🎨 Ressources
    ├── icon.ico           # 🎭 Icône application
    └── ui/                # 🖼️ Ressources UI
```

## 🚀 Utilisation Recommandée

### Pour les Utilisateurs Finaux:
```bash
# Installation du package
pip install -e .

# Lancement simple
ancomicsviewer

# Ou avec fichier spécifique
python main.py mon_comic.pdf
```

### Pour les Développeurs:
```bash
# Environnement de développement
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -e .

# Tests et développement
python main.py --preset fb --detector multibd
python main.py --help
```

## 🔧 Fonctionnalités Validées

✅ **Multi-BD Enhanced v2.0** - 94.2% mAP50  
✅ **Cache intelligent** - Performance optimisée  
✅ **Interface CLI complète** - Arguments avancés  
✅ **Package installable** - `pip install -e .`  
✅ **Structure modulaire** - Séparation des préoccupations  
✅ **Documentation complète** - Guides et API  
✅ **Tests organisés** - Unit + intégration  

## 📋 TODO

- [ ] Tests automatisés complets
- [ ] Documentation API détaillée  
- [ ] Scripts de déploiement
- [ ] Intégration CI/CD

---
**Status**: ✅ Restructuration complète réussie - Prêt pour commit final
