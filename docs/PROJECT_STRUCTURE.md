# 🏗️ Structure du Projet AnComicsViewer

## 📋 Vue d'Ensemble

AnComicsViewer v2.0 utilise une structure de projet moderne et organisée suivant les meilleures pratiques Python. Cette restructuration améliore la maintenabilité, la testabilité et la facilité de déploiement.

## 📁 Arborescence Complète

```
AnComicsViewer/
├── 📄 main.py                           # Point d'entrée principal
├── ⚙️ setup.py                          # Configuration installation setuptools
├── 📋 pyproject.toml                    # Configuration moderne (PEP 518)
├── 📄 MANIFEST.in                       # Manifeste package
├── 📖 README.md                         # Documentation principale
├── 📝 LICENSE                           # Licence MIT
├── 🚫 .gitignore                        # Fichiers à ignorer Git
├── 📦 requirements.txt                  # Dépendances base
├── 🤖 requirements-ml.txt               # Dépendances ML
│
├── 📁 src/                              # 🎯 CODE SOURCE PRINCIPAL
│   └── 📁 ancomicsviewer/               # Package Python principal
│       ├── 🐍 __init__.py               # Init package + API publique
│       ├── 🖥️ main_app.py               # Application PySide6 principale
│       ├── 📁 detectors/                # 🤖 DÉTECTEURS DE PANELS
│       │   ├── 🐍 __init__.py           # API détecteurs
│       │   ├── 📊 base.py               # Interface de base
│       │   ├── 🧠 multibd_detector.py   # Multi-BD Enhanced v2.0
│       │   ├── 🔍 yolo_seg.py           # YOLO segmentation
│       │   ├── 🔧 postproc.py           # Post-processing
│       │   └── 📖 reading_order.py      # Ordre de lecture
│       ├── 📁 ui/                       # 🖥️ INTERFACE UTILISATEUR
│       │   └── 🐍 __init__.py           # Composants UI
│       └── 📁 utils/                    # 🛠️ UTILITAIRES
│           ├── 🐍 __init__.py           # API utilitaires
│           └── 💾 enhanced_cache.py     # Cache amélioré
│
├── 📁 scripts/                          # 🔧 SCRIPTS UTILITAIRES
│   ├── 📁 training/                     # 🎓 Entraînement ML
│   │   ├── 🚀 train_mps_optimized.py   # Entraînement Apple Silicon
│   │   ├── 📊 train_enhanced_v2.py     # Script d'entraînement v2
│   │   └── 🧪 test_quick.py             # Test rapide modèle
│   ├── 📁 dataset/                      # 📊 Gestion datasets
│   │   ├── 📋 dataset_analyzer.py       # Analyse dataset
│   │   └── 🏗️ dataset_manager.py        # Gestion dataset
│   ├── 📁 build/                        # 🏗️ Compilation
│   │   ├── 🔨 build_standalone.py       # Build exécutable
│   │   └── ⚙️ pyinstaller_config.py     # Config PyInstaller
│   ├── 📁 ml/                           # 🤖 Scripts ML avancés
│   └── 📁 tools/                        # 🛠️ Outils divers
│
├── 📁 tests/                            # 🧪 TESTS
│   ├── 📁 unit/                         # Tests unitaires
│   └── 📁 integration/                  # Tests d'intégration
│
├── 📁 data/                             # 📊 DONNÉES
│   ├── 📁 models/                       # 🤖 Modèles ML
│   │   ├── 🧠 yolov8s.pt               # Modèle YOLO de base
│   │   └── 🏆 best.pt                   # Multi-BD Enhanced v2.0
│   └── 📁 examples/                     # 📚 PDFs d'exemple
│       ├── 📖 Golden City - T01.pdf     # BD Franco-Belge
│       ├── 📖 Tintin - Le Lotus Bleu.pdf
│       └── 📖 La Pin-up du B24.pdf
│
├── 📁 docs/                             # 📚 DOCUMENTATION
│   ├── 📖 README_OLD.md                 # Ancien README (archive)
│   ├── 📁 guides/                       # 📖 Guides utilisateur
│   │   ├── 👤 USER_GUIDE.md             # Guide utilisateur
│   │   ├── 💻 DEVELOPER_GUIDE.md        # Guide développeur
│   │   ├── 🤖 ML_GUIDE.md               # Guide ML/IA
│   │   └── 🏗️ BUILD_GUIDE.md            # Guide compilation
│   └── 📁 api/                          # 📊 Documentation API
│       └── 🔗 docs.md                   # Documentation API
│
├── 📁 assets/                           # 🎨 RESSOURCES
│   └── 🖼️ icon.ico                      # Icône application
│
├── 📁 .venv/                            # 🐍 Environnement virtuel
├── 📁 runs/                             # 🏃 Résultats entraînement
├── 📁 dataset/                          # 📊 Dataset YOLO format
└── 📁 .git/                             # 📝 Contrôle version Git
```

## 🎯 Principes d'Organisation

### 1. **Séparation des Responsabilités**
- **`src/`** : Code source de l'application
- **`scripts/`** : Outils et utilitaires
- **`tests/`** : Tests automatisés
- **`docs/`** : Documentation
- **`data/`** : Données et modèles

### 2. **Structure Package Python Standard**
- **Package installable** avec `pip install -e .`
- **Imports relatifs** pour la modularité
- **API publique** définie dans `__init__.py`
- **Points d'entrée** via `setup.py` et `pyproject.toml`

### 3. **Compatibilité et Maintenance**
- **Cross-platform** : Windows, macOS, Linux
- **Python 3.8+** : Support versions récentes
- **Dépendances modulaires** : Base vs ML vs Dev
- **Tests automatisés** : CI/CD ready

## 📦 Installation et Utilisation

### Installation Développeur
```bash
# Clone + setup environnement
git clone https://github.com/r45635/AnComicsViewer.git
cd AnComicsViewer

# Installation package editable avec ML
pip install -e ".[ml,dev]"
```

### Utilisation
```bash
# Via point d'entrée installé
ancomicsviewer

# Via script direct
python main.py

# Avec options
ancomicsviewer --preset manga --detector multibd comics.pdf
```

## 🔄 Migration depuis l'Ancienne Structure

### Changements d'Imports

**Avant :**
```python
from AnComicsViewer import PanelDetector
from detectors.multibd_detector import MultiBDPanelDetector
from enhanced_cache import PanelCacheManager
```

**Maintenant :**
```python
from ancomicsviewer.main_app import PanelDetector
from ancomicsviewer.detectors import MultiBDPanelDetector
from ancomicsviewer.utils import PanelCacheManager
```

### Nouveaux Chemins

**Fichiers déplacés :**
- `AnComicsViewer.py` → `src/ancomicsviewer/main_app.py`
- `detectors/` → `src/ancomicsviewer/detectors/`
- `enhanced_cache.py` → `src/ancomicsviewer/utils/`
- Scripts → `scripts/` (organisés par catégorie)
- Tests → `tests/` (unit + integration)
- Documentation → `docs/` (guides + api)

## 🧪 Tests

### Structure des Tests
```bash
tests/
├── unit/                    # Tests unitaires
│   ├── test_detectors.py    # Tests détecteurs
│   ├── test_cache.py        # Tests cache
│   └── test_ui.py           # Tests interface
└── integration/             # Tests d'intégration
    ├── test_full_pipeline.py
    └── test_ml_integration.py
```

### Exécution
```bash
# Tous les tests
pytest

# Tests unitaires seulement
pytest tests/unit/

# Avec coverage
pytest --cov=src/ancomicsviewer
```

## 🚀 Déploiement

### Build Standalone
```bash
# Exécutable
python scripts/build/build_standalone.py

# Distribution
python setup.py sdist bdist_wheel
```

### CI/CD Integration
La nouvelle structure est optimisée pour :
- **GitHub Actions**
- **Pre-commit hooks**
- **Automated testing**
- **Package publishing**

## 🔧 Configuration

### Variables d'Environnement
```bash
# Presets et détecteurs
export ANCOMICS_PRESET=manga
export ANCOMICS_DETECTOR=multibd
export ANCOMICS_DPI=300

# Chemins (automatiques avec nouvelle structure)
export ANCOMICS_MODELS_PATH=data/models/
export ANCOMICS_CACHE_PATH=~/.ancomicsviewer/
```

### Fichiers de Configuration
- **`pyproject.toml`** : Configuration projet moderne
- **`setup.py`** : Installation setuptools
- **`requirements*.txt`** : Dépendances modulaires
- **`MANIFEST.in`** : Contenu package

## 📈 Avantages de la Nouvelle Structure

### ✅ **Maintenabilité**
- Code organisé par responsabilité
- Imports clairs et logiques
- API publique bien définie

### ✅ **Testabilité**
- Tests isolés par module
- Coverage facilité
- CI/CD integration

### ✅ **Déployabilité**
- Package installable standard
- Distribution PyPI ready
- Build standalone optimisé

### ✅ **Développement**
- Structure familière Python
- Outils de développement intégrés
- Documentation centralisée

## 🛡️ Compatibilité

La nouvelle structure maintient la **compatibilité fonctionnelle** avec l'ancienne version tout en améliorant significativement l'organisation et la maintenabilité du code.

Les **points d'entrée** (`main.py`, scripts de lancement) restent identiques pour l'utilisateur final.
