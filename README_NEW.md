# 📖 AnComicsViewer

**Lecteur PDF intelligent pour bandes dessinées avec détection automatique de cases**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/r45635/AnComicsViewer/releases)

AnComicsViewer est un lecteur PDF moderne conçu spécifiquement pour les bandes dessinées, combinant une interface utilisateur intuitive avec des algorithmes avancés de détection de cases utilisant l'apprentissage automatique.

## ✨ Caractéristiques

### 🎯 **Détection Intelligente de Cases**
- **Multi-BD Enhanced v2.0** : Modèle ML optimisé (94.2% mAP50)
- **Support Apple Silicon** : Optimisé MPS pour M1/M2 Mac
- **Algorithmes multiples** : Heuristique, YOLO, Multi-styles
- **Post-processing avancé** : Alignement sur gouttières, ordre de lecture

### 🖥️ **Interface Moderne**
- **PySide6** : Interface native et responsive
- **Navigation intelligente** : Entre pages et cases
- **Overlay interactif** : Visualisation des détections
- **Presets** : Franco-Belge, Manga, Comics US

### ⚡ **Performance**
- **Cache intelligent** : Détections persistantes
- **Multi-threading** : Traitement en arrière-plan
- **Formats supportés** : PDF haute résolution
- **Cross-platform** : Windows, macOS, Linux

## 🚀 Installation

### Installation Simple
```bash
# Cloner le repository
git clone https://github.com/r45635/AnComicsViewer.git
cd AnComicsViewer

# Installation avec pip
pip install -e .

# Avec support ML
pip install -e ".[ml]"
```

### Installation Développeur
```bash
# Environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou .venv\\Scripts\\activate  # Windows

# Installation complète
pip install -e ".[ml,dev]"
```

### Utilisation Rapide
```bash
# Script automatisé (recommandé)
./run.sh  # Linux/Mac
# ou run.ps1  # Windows

# Lancement direct
python main.py
```

## 📖 Utilisation

### Interface Graphique
```bash
# Lancement simple
ancomicsviewer

# Avec options
ancomicsviewer --preset manga --detector multibd comics.pdf
```

### Ligne de Commande
```bash
python main.py --help

# Exemples
python main.py --preset fb mycomics.pdf        # Franco-Belge
python main.py --detector multibd --page 5     # Multi-BD, page 5
python main.py --dpi 300 manga.pdf             # Haute résolution
```

### Variables d'Environnement
```bash
export ANCOMICS_PRESET=manga
export ANCOMICS_DETECTOR=multibd
export ANCOMICS_DPI=300
```

## 🏗️ Structure du Projet

```
AnComicsViewer/
├── 📁 src/ancomicsviewer/          # Code source principal
│   ├── 🐍 __init__.py              # Package AnComicsViewer
│   ├── 🖥️ main_app.py              # Application principale
│   ├── 📁 detectors/               # Algorithmes de détection
│   │   ├── 🧠 multibd_detector.py  # Multi-BD Enhanced v2.0
│   │   ├── 🔍 yolo_seg.py          # YOLO segmentation
│   │   ├── 📊 base.py              # Interface de base
│   │   └── 🔧 postproc.py          # Post-processing
│   ├── 📁 ui/                      # Interface utilisateur
│   └── 📁 utils/                   # Utilitaires
│       └── 💾 enhanced_cache.py    # Cache amélioré
├── 📁 scripts/                     # Scripts utilitaires
│   ├── 📁 training/                # Entraînement ML
│   ├── 📁 dataset/                 # Gestion données
│   └── 📁 build/                   # Compilation
├── 📁 tests/                       # Tests
│   ├── 📁 unit/                    # Tests unitaires
│   └── 📁 integration/             # Tests d'intégration
├── 📁 data/                        # Données
│   ├── 📁 models/                  # Modèles ML
│   └── 📁 examples/                # PDFs d'exemple
├── 📁 docs/                        # Documentation
│   ├── 📁 guides/                  # Guides utilisateur
│   └── 📁 api/                     # Documentation API
├── 🐍 main.py                      # Point d'entrée
├── ⚙️ setup.py                     # Installation
├── 📋 pyproject.toml               # Configuration moderne
└── 📖 README.md                    # Ce fichier
```

## 🤖 Détecteurs Disponibles

### 1. **Heuristique (OpenCV)**
- Algorithme rapide basé sur les contours
- Idéal pour : BD classiques, contours nets
- Paramètres ajustables en temps réel

### 2. **YOLO Segmentation**
- Segmentation par apprentissage automatique
- Idéal pour : Styles variés, layouts complexes
- Pré-entraîné sur COCO

### 3. **Multi-BD Enhanced v2.0** ⭐
- **Modèle recommandé** optimisé pour BD
- Entraîné sur : Golden City, Tintin, Pin-up
- Performance : 94.2% mAP50
- Support Apple Silicon MPS

## 📊 Performance

| Détecteur | mAP50 | Vitesse | Styles |
|-----------|-------|---------|--------|
| Heuristique | ~80% | Très rapide | Classique |
| YOLO | ~85% | Rapide | Universel |
| **Multi-BD v2** | **94.2%** | Optimisé | Multi-styles |

## 🛠️ Développement

### Tests
```bash
# Tests unitaires
pytest tests/unit/

# Tests d'intégration
pytest tests/integration/

# Coverage
pytest --cov=src/ancomicsviewer
```

### Entraînement ML
```bash
# Dataset preparation
python scripts/dataset/prepare_enhanced_dataset.py

# Training (Apple Silicon optimized)
python scripts/training/train_mps_optimized.py

# Validation
python scripts/training/test_quick.py
```

### Build Standalone
```bash
# Executable
python scripts/build/build_standalone.py

# Avec PyInstaller
python scripts/build/pyinstaller_config.py
```

## 📚 Documentation

- 📖 [Guide Utilisateur](docs/guides/USER_GUIDE.md)
- 🔧 [Guide Développeur](docs/guides/DEVELOPER_GUIDE.md)
- 🤖 [Documentation ML](docs/guides/ML_GUIDE.md)
- 🏗️ [Guide de Build](docs/guides/BUILD_GUIDE.md)
- 📊 [API Reference](docs/api/)

## 🤝 Contribution

Les contributions sont les bienvenues ! Consultez [CONTRIBUTING.md](docs/CONTRIBUTING.md).

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/amazing-feature`)
3. Commit les changements (`git commit -m 'Add amazing feature'`)
4. Push vers la branche (`git push origin feature/amazing-feature`)
5. Ouvrir une Pull Request

## 📝 License

Ce projet est sous licence MIT. Voir [LICENSE](LICENSE) pour plus de détails.

## 🙏 Remerciements

- **Ultralytics** pour YOLOv8
- **PySide6** pour l'interface utilisateur
- **OpenCV** pour le traitement d'image
- La communauté **BD/Comics** pour les retours

## 📞 Support

- 🐛 [Issues GitHub](https://github.com/r45635/AnComicsViewer/issues)
- 💬 [Discussions](https://github.com/r45635/AnComicsViewer/discussions)
- 📧 [Email](mailto:vincent@example.com)

---

⭐ **Si vous aimez ce projet, n'hésitez pas à lui donner une étoile !**
