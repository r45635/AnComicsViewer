# 📦 Archive Core - AnComicsViewer

## 🎯 Archive la plus récente (RECOMMANDÉE)

**`AnComicsViewer_Complete_v5.0.0_BDStabilized_20250824_1217.zip`** ⭐

### ✅ Fonctionnalités BD Stabilized Detector v5.0

- **Détection de panels robuste** avec configuration automatique
- **Cache Enhanced v5** avec invalidation intelligente  
- **Seuils de confiance optimisés** (CONF_BASE=0.05, CONF_MIN=0.01)
- **Pipeline post-processing complet** avec filtres adaptatifs
- **Interface graphique complète** Qt6/PySide6
- **Scripts CLI inclus** pour validation et tests
- **Build standalone** prêt pour distribution

### 🚀 Installation rapide

```bash
# Extraire l'archive
unzip AnComicsViewer_Complete_v5.0.0_BDStabilized_20250824_1217.zip
cd AnComicsViewer/

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
python3 main.py
```

### � Tests et validation

```bash
# Test minimal (imports et détecteur)
python3 scripts/cli_minimal.py --skip-detection

# Test de détection basique
python3 scripts/test_basic_detection.py

# Tests diagnostiques complets
python3 debug_predict_raw.py
python3 diagnostic_detection.py
```

### 📊 Contenu de l'archive

- **56 fichiers** essentiels (48.4 MB)
- Code source complet dans `src/`
- Modèles YOLO dans `data/models/`
- Documentation complète
- Scripts de build et test
- Configuration optimisée

---

## 📋 Archives précédentes

### v4.0.0 (Legacy)
- `AnComicsViewer_Core_App_v4.0.0_BDStabilized_20250824_1138.zip`
- `AnComicsViewer_Core_App_v3.0.0_GenericTTA_20250824_0848.zip`

### v2.0.0 (Historique)
- `AnComicsViewer_Core_App_v2.0.0_12_gf599b17.zip`
- `AnComicsViewer_Core_App_v2.0.0_11_g932a4df_dirty.zip`
- `AnComicsViewer_Core_App_v2.0.0.zip`

---

## 🛠️ Scripts d'archivage

- `create_complete_archive.py` - Archive complète avec tous les fichiers
- `create_app_archive.py` - Archive application seule
- `create_simple_archive.py` - Archive basique

---

**📅 Dernière mise à jour :** 24 août 2025  
**🎯 Version recommandée :** v5.0.0 BD Stabilized  
**✅ Statut :** Production Ready
