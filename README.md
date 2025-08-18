# 🎯 AnComicsViewer v2.0 - Multi-BD Revolution

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

# Création environnement virtuel (recommandé)
python -m venv .venv

# Installation dépendances
pip install -r requirements.txt -r requirements-ml.txt
```

### 🚀 **Lancement**

#### 🎯 **Méthode recommandée - Script automatique**
```bash
# Linux/macOS
./run.sh

# Windows PowerShell
.\run.ps1

# Ou directement avec Python
python main.py
```

#### 🔧 **Lancement manuel**
```bash
# Avec environnement virtuel
.venv/bin/python main.py        # Linux/macOS
.venv\Scripts\python main.py    # Windows

# Ou directement (non recommandé)
python AnComicsViewer.py
```

### 🎯 **Utilisation**
1. **Ouvrir PDF** : File → Open ou glisser-déposer
2. **Activer Multi-BD** : ⚙️ → Detector → Multi-BD (Trained)
3. **Navigation panels** : Touches `N` et `Shift+N`  
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
