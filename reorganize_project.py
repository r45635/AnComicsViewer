#!/usr/bin/env python3
"""
🗂️ Réorganisation du projet AnComicsViewer
Nettoie et organise les fichiers pour garder seulement l'essentiel
"""

import os
import shutil
from pathlib import Path

def reorganize_project():
    """Réorganise le projet en gardant l'essentiel."""
    
    print("🗂️ RÉORGANISATION DU PROJET ANCOMICSVIEWER")
    print("=" * 50)
    
    # Structure cible
    structure = {
        "core": [
            "AnComicsViewer.py",
            "detectors/",
            "patch_pytorch.py"
        ],
        "tools": [
            # Scripts d'entraînement et dataset
            "train_multibd_model.py",
            "dataset_analyzer.py", 
            "start_annotation.py",
            "integrate_pinup_system.py",
            "integrate_tintin.py",
            "tools/labelme_to_yolo.py",
            
            # Tests importants
            "test_multibd_integration.py",
            "smoke_test.py",
            
            # Utilitaires
            "post_annotation_processing.py",
            "integration_summary.py",
            "release_summary.py"
        ],
        "docs": [
            "README.md",
            "MULTIBD_GUIDE.md", 
            "RELEASE_NOTES_v2.0.md",
            "QUICK_REFERENCE.md"
        ],
        "config": [
            "requirements.txt",
            "requirements-ml.txt",
            ".gitignore",
            "LICENSE"
        ],
        "assets": [
            "favicon.png",
            "icon.png",
            "logo.png"
        ],
        "to_remove": [
            # Scripts de développement temporaires
            "add_tintin.py",
            "analyze_model_confidence.py",
            "annotate_tintin.py", 
            "annotation_progress.py",
            "auto_pipeline.py",
            "continue_training.py",
            "dataset_manager.py",
            "demo_multibd.py",
            "demo_workflow.py",
            "extract_tintin.py",
            "integrate_pinup.py",
            "manage_dataset.py",
            "monitor_annotations.py",
            
            # Tests de développement
            "test_integration.py",
            "test_mixed_dataset.py",
            "test_ml_detector.py", 
            "test_ml_inference.py",
            "test_model.py",
            "test_single_class.py",
            "train_yolo.py",
            
            # Fichiers temporaires/obsolètes
            "temp_tintin/",
            "golden_city_samples/",
            "analyze_golden_city.py",
            "golden_city_test_guide.py",
            "generate_logo.py",
            "docs.md",
            "run.sh",
            "run_win.ps1",
            "setup.sh",
            "ML_ROADMAP.md",
            "README_SETUP.md",
            
            # Modèles de base (gardés dans runs/)
            "yolov8n-seg.pt",
            "yolov8n.pt"
        ]
    }
    
    # Créer les dossiers de destination
    tools_dir = Path("tools")
    tools_dir.mkdir(exist_ok=True)
    
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    
    print("📁 Dossiers créés : tools/, docs/, assets/")
    
    # Déplacer les outils
    moved_count = 0
    for tool_file in structure["tools"]:
        src = Path(tool_file)
        if src.exists():
            if "/" in tool_file:  # Déjà dans un sous-dossier
                continue
            dest = tools_dir / src.name
            try:
                shutil.move(str(src), str(dest))
                print(f"🔧 Déplacé : {src} → tools/{src.name}")
                moved_count += 1
            except Exception as e:
                print(f"❌ Erreur déplacement {src} : {e}")
    
    # Déplacer la documentation
    for doc_file in structure["docs"]:
        src = Path(doc_file)
        if src.exists() and src != Path("README.md"):  # Garder README à la racine
            dest = docs_dir / src.name
            try:
                shutil.move(str(src), str(dest))
                print(f"📖 Déplacé : {src} → docs/{src.name}")
                moved_count += 1
            except Exception as e:
                print(f"❌ Erreur déplacement {src} : {e}")
    
    # Déplacer les assets
    for asset_file in structure["assets"]:
        src = Path(asset_file)
        if src.exists():
            dest = assets_dir / src.name
            try:
                shutil.move(str(src), str(dest))
                print(f"🎨 Déplacé : {src} → assets/{src.name}")
                moved_count += 1
            except Exception as e:
                print(f"❌ Erreur déplacement {src} : {e}")
    
    # Supprimer les fichiers obsolètes
    removed_count = 0
    for obsolete_file in structure["to_remove"]:
        path = Path(obsolete_file)
        if path.exists():
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                    print(f"🗑️  Dossier supprimé : {path}")
                else:
                    path.unlink()
                    print(f"🗑️  Fichier supprimé : {path}")
                removed_count += 1
            except Exception as e:
                print(f"❌ Erreur suppression {path} : {e}")
    
    print(f"\n✅ Réorganisation terminée !")
    print(f"📦 {moved_count} fichiers déplacés")
    print(f"🗑️  {removed_count} fichiers/dossiers supprimés")
    
    return True

def create_new_readme():
    """Crée un README mis à jour pour la structure réorganisée."""
    
    readme_content = """# 🎯 AnComicsViewer v2.0 - Multi-BD Revolution

**Lecteur de BD/Comics avec détection IA de panels multi-styles**

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/r45635/AnComicsViewer/releases/tag/v2.0.0)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)

## 🚀 **Fonctionnalités**

### 🤖 **Détection IA Multi-Styles**
- **Détecteur YOLO spécialisé BD** : 91.1% mAP50 precision
- **3 styles supportés** : Golden City, Tintin, Pin-up du B24
- **Temps réel** : ~35ms par page
- **Pas de configuration** : Paramètres automatiques

### 📖 **Lecture Avancée**
- **Navigation panel par panel** précise
- **Zoom intelligent** sur les cases
- **Interface native** PySide6
- **Support PDF** complet

### 🛠️ **Architecture Modulaire**
- **Détecteurs interchangeables** : Heuristique, YOLO, Multi-BD
- **Pipeline ML complet** : Annotation → Entraînement → Intégration
- **Outils inclus** : Dataset, tests, documentation

## ⚡ **Installation & Utilisation**

### 📋 **Prérequis**
```bash
# Python 3.8+ requis
python --version

# Installation dépendances
pip install -r requirements.txt
```

### 🚀 **Lancement**
```bash
# Démarrage simple
python AnComicsViewer.py

# Activer le détecteur Multi-BD
# Menu ⚙️ → Detector → Multi-BD (Trained)
```

### 🎯 **Utilisation**
1. **Ouvrir PDF** : File → Open
2. **Activer Multi-BD** : ⚙️ → Detector → Multi-BD (Trained)  
3. **Naviguer** : Flèches ou clic sur panels
4. **Zoomer** : Molette ou raccourcis

## 📁 **Structure du Projet**

```
AnComicsViewer/
├── AnComicsViewer.py          # Application principale
├── detectors/                 # Détecteurs de panels
│   ├── base.py               # Interface base
│   ├── multibd_detector.py   # Détecteur Multi-BD ⭐
│   └── yolo_seg.py          # Détecteur YOLO générique
├── tools/                     # Outils de développement
│   ├── train_multibd_model.py    # Entraînement modèle
│   ├── dataset_analyzer.py      # Analyse dataset
│   ├── labelme_to_yolo.py       # Conversion annotations
│   └── test_multibd_integration.py # Tests
├── docs/                      # Documentation
│   ├── MULTIBD_GUIDE.md      # Guide utilisateur complet
│   ├── RELEASE_NOTES_v2.0.md # Notes de version
│   └── QUICK_REFERENCE.md    # Référence rapide
├── assets/                    # Ressources
├── dataset/                   # Dataset d'entraînement
├── runs/                      # Modèles entraînés
└── requirements.txt           # Dépendances
```

## 🧪 **Tests & Validation**

### ✅ **Tests Intégrés**
```bash
# Test d'intégration complet
python tools/test_multibd_integration.py

# Test de base du viewer
python tools/smoke_test.py

# Analyse du dataset
python tools/dataset_analyzer.py coverage
```

### 🔧 **Développement**
```bash
# Ré-entraînement du modèle (si nécessaire)
python tools/train_multibd_model.py

# Annotation nouveau dataset
python tools/start_annotation.py

# Analyse performance
python tools/integration_summary.py
```

## 🎯 **Performance**

### 📊 **Métriques du Modèle Multi-BD**
- **mAP50** : 91.1% (précision excellente)
- **mAP50-95** : 88.3% (robustesse multi-échelles)  
- **Précision** : 84.0% (peu de faux positifs)
- **Rappel** : 88.7% (détection complète)

### ⚡ **Performance Temps Réel**
- **Inférence** : ~32ms par image
- **Preprocessing** : ~0.6ms  
- **Total** : ~35ms par page BD

### 🎨 **Styles Supportés**
- **🟡 Golden City** : Style moderne complexe
- **🔵 Tintin** : Style classique ligne claire
- **🔴 Pin-up du B24** : Style aviation/guerre

## 📚 **Documentation**

- **📖 [Guide Utilisateur](docs/MULTIBD_GUIDE.md)** - Documentation complète
- **📋 [Notes de Release](docs/RELEASE_NOTES_v2.0.md)** - Nouveautés v2.0
- **⚡ [Référence Rapide](docs/QUICK_REFERENCE.md)** - Raccourcis et astuces

## 🔧 **Configuration Avancée**

### 🎛️ **Réglages Détecteur**
```python
# Ajuster la confiance (interface graphique)
Menu ⚙️ → Detector → Multi-BD (Trained)

# Ou par code (développement)
detector.set_confidence(0.2)  # 0.05-0.95
detector.set_iou_threshold(0.5)  # 0.1-0.9
```

### 📦 **Modèle Personnalisé**
```python
from detectors.multibd_detector import MultiBDPanelDetector
detector = MultiBDPanelDetector(weights="custom_model.pt")
```

## 🌟 **Nouveautés v2.0**

- **🆕 Détecteur Multi-BD** spécialisé (91.1% mAP50)
- **🆕 Pipeline ML complet** (PDF → annotation → entraînement)
- **🆕 Interface native** avec basculement détecteurs
- **🆕 Documentation exhaustive** et outils développement
- **🆕 Architecture modulaire** extensible

## 🤝 **Contribution**

### 🚀 **Développement**
1. Fork le projet
2. Créer une branche feature
3. Tester avec `python tools/test_multibd_integration.py`
4. Soumettre une Pull Request

### 📊 **Améliorer le Dataset**
1. Ajouter nouvelles BD avec `tools/integrate_*_system.py`
2. Annoter avec `tools/start_annotation.py`
3. Ré-entraîner avec `tools/train_multibd_model.py`

## 📄 **Licence**

MIT License - voir [LICENSE](LICENSE) pour détails.

## 🏆 **Crédits**

- **YOLOv8** (Ultralytics) - Détection objets
- **PySide6** - Interface utilisateur  
- **OpenCV** - Traitement d'image
- **PyTorch** - Backend IA

---

**🎯 AnComicsViewer v2.0 - La révolution de la lecture BD avec IA ! 🚀**
"""
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print("📖 README.md mis à jour")

def update_gitignore():
    """Met à jour .gitignore pour la nouvelle structure."""
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific
dataset/images/
dataset/labels/
runs/
ml/runs/
*.pt
*.pdf
temp_*/

# Large files (should use Git LFS)
*.pdf
La Pin-up du B24 - T01.pdf
Tintin - 161 - Le Lotus Bleu - .pdf
Golden City - T01 - Pilleurs d'épaves.pdf
"""
    
    with open(".gitignore", "w", encoding="utf-8") as f:
        f.write(gitignore_content)
    
    print("🚫 .gitignore mis à jour")

def main():
    """Fonction principale de réorganisation."""
    
    print("🚀 Démarrage de la réorganisation...")
    
    if reorganize_project():
        create_new_readme()
        update_gitignore()
        
        print(f"\n🎉 RÉORGANISATION TERMINÉE AVEC SUCCÈS !")
        print("=" * 50)
        print("📁 Structure finale :")
        print("   • Core : AnComicsViewer.py, detectors/")
        print("   • Tools : tools/ (entraînement, tests, utilitaires)")
        print("   • Docs : docs/ (documentation)")
        print("   • Assets : assets/ (icônes, images)")
        print("   • Config : requirements.txt, .gitignore, LICENSE")
        print()
        print("🔧 Prochaines étapes :")
        print("   1. Tester : python tools/test_multibd_integration.py")
        print("   2. Commiter : git add . && git commit")
        print("   3. Lancer : python AnComicsViewer.py")
        
        return True
    else:
        print("❌ Échec de la réorganisation")
        return False

if __name__ == "__main__":
    main()
