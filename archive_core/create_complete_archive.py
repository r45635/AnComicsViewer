#!/usr/bin/env python3
"""
Création d'archive complète AnComicsViewer avec BD Stabilized Detector v5.0
Inclut tous les fichiers essentiels pour un fonctionnement complet.
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path
from datetime import datetime

def get_version():
    """Obtient la version du projet."""
    try:
        # Essayer de lire depuis pyproject.toml
        with open("../pyproject.toml", "r") as f:
            for line in f:
                if line.startswith("version"):
                    version = line.split("=")[1].strip().strip('"')
                    return version
    except:
        pass
    return "5.0.0"

def create_archive():
    """Crée l'archive complète."""
    
    # Configuration
    version = get_version()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    archive_name = f"AnComicsViewer_Complete_v{version}_BDStabilized_{timestamp}.zip"
    
    print(f"🗃️ Création de l'archive: {archive_name}")
    print(f"📦 Version: {version}")
    print(f"🕐 Timestamp: {timestamp}")
    
    # Aller au répertoire parent
    os.chdir("..")
    
    # Fichiers et dossiers essentiels à inclure
    essential_items = [
        # Code source principal
        "src/",
        "main.py",
        "ancomicsviewer.py",
        
        # Configuration et métadonnées
        "pyproject.toml",
        "requirements.txt",
        "setup.py",
        "MANIFEST.in",
        
        # Ressources
        "assets/",
        "data/models/",  # Modèles YOLO
        
        # Documentation
        "README.md",
        "LICENSE",
        "BUILD_GUIDE.md",
        "CLI_USAGE.md",
        "PROJECT_STATUS.md",
        "ENHANCED_V2_SUMMARY.md",
        "MULTIBD_ENHANCED.md",
        
        # Scripts utiles
        "build_standalone.py",
        "run.sh",
        "run.ps1",
        "examples_cli.sh",
        
        # Configuration de build
        "AnComicsViewer.spec",
        "release_config.json",
        
        # Tests essentiels
        "test_predict_raw.py",
        "debug_predict_raw.py",
        "diagnostic_detection.py",
        
        # Scripts CLI
        "scripts/cli_minimal.py",
        "scripts/test_basic_detection.py",
    ]
    
    # Créer l'archive
    with zipfile.ZipFile(f"archive_core/{archive_name}", 'w', zipfile.ZIP_DEFLATED) as zipf:
        files_added = 0
        
        for item in essential_items:
            if os.path.exists(item):
                if os.path.isfile(item):
                    print(f"📄 Ajout fichier: {item}")
                    zipf.write(item)
                    files_added += 1
                elif os.path.isdir(item):
                    print(f"📁 Ajout dossier: {item}")
                    for root, dirs, files in os.walk(item):
                        # Ignorer les dossiers de cache et temporaires
                        dirs[:] = [d for d in dirs if not d.startswith(('.', '__pycache__', 'runs', 'wandb'))]
                        
                        for file in files:
                            if not file.startswith('.') and not file.endswith('.pyc'):
                                file_path = os.path.join(root, file)
                                zipf.write(file_path)
                                files_added += 1
            else:
                print(f"⚠️ Non trouvé: {item}")
        
        # Ajouter un README spécifique à l'archive
        readme_content = f"""# AnComicsViewer Complete Archive v{version}

## 🎯 Archive BD Stabilized Detector v5.0

Cette archive contient la version complète d'AnComicsViewer avec:

### ✅ Fonctionnalités incluses
- **BD Stabilized Detector v5.0** avec détection de panels optimisée
- **Cache Enhanced v5** avec gestion robuste des configurations
- **Pipeline post-processing complet** avec filtres adaptatifs
- **Configuration automatique** des seuils de confiance
- **Support CPU/GPU** avec fallback automatique
- **Interface graphique complète** Qt6/PySide6
- **CLI tools** pour tests et validation

### 🚀 Installation rapide

1. **Extraire l'archive:**
```bash
unzip {archive_name}
cd AnComicsViewer/
```

2. **Installation des dépendances:**
```bash
pip install -r requirements.txt
```

3. **Lancement:**
```bash
# Interface graphique
python3 main.py

# ou via module
python3 -m ancomicsviewer

# CLI
python3 scripts/cli_minimal.py --help
```

### 🔧 Configuration

Le système utilise maintenant:
- `CONF_BASE = 0.05` (seuil de confiance optimisé)
- `CONF_MIN = 0.01` (seuil minimum pour fallback)
- Cache v5 avec invalidation automatique
- Filtrage par nom de classe (robuste aux changements d'index)

### 📊 Tests de validation

```bash
# Test de base
python3 test_predict_raw.py

# Diagnostic complet
python3 diagnostic_detection.py

# Debug étape par étape
python3 debug_predict_raw.py
```

### 🏗️ Build standalone

```bash
python3 build_standalone.py
```

### 📝 Logs et Debug

- Logs détaillés avec `[Panels]` prefix
- Mode debug disponible
- Traceback complet des erreurs
- Monitoring des performances

---
**Généré le:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Version:** {version}
**Statut:** Production Ready ✅
"""
        
        # Écrire le README dans l'archive
        zipf.writestr("README_ARCHIVE.md", readme_content)
        files_added += 1
        
        print(f"\n✅ Archive créée avec succès!")
        print(f"📦 Fichier: archive_core/{archive_name}")
        print(f"📊 Fichiers inclus: {files_added}")
        
    return archive_name

def main():
    """Point d'entrée principal."""
    try:
        archive_name = create_archive()
        
        # Vérification de l'archive
        archive_path = f"archive_core/{archive_name}"
        if os.path.exists(archive_path):
            size_mb = os.path.getsize(archive_path) / (1024 * 1024)
            print(f"📏 Taille: {size_mb:.1f} MB")
            
            # Test de l'archive
            with zipfile.ZipFile(archive_path, 'r') as zipf:
                files_in_archive = len(zipf.namelist())
                print(f"🗂️ Fichiers dans l'archive: {files_in_archive}")
                
                # Vérifier quelques fichiers clés
                key_files = ["main.py", "src/ancomicsviewer/main_app.py", "README_ARCHIVE.md"]
                for key_file in key_files:
                    if key_file in zipf.namelist():
                        print(f"✅ {key_file}")
                    else:
                        print(f"❌ Manquant: {key_file}")
            
            print(f"\n🎉 Archive complète prête: {archive_name}")
            return True
        else:
            print(f"❌ Erreur: Archive non créée")
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors de la création: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
