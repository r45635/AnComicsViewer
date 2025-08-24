#!/usr/bin/env python3
"""
AnComicsViewer - Créateur d'Archive Simplifié
===========================================
"""

import zipfile
import sys
from pathlib import Path
from datetime import datetime

def get_version():
    """Récupère la version."""
    try:
        project_dir = Path(__file__).parent.parent.absolute()
        import subprocess
        result = subprocess.run(
            ["git", "describe", "--tags", "--long", "--dirty"],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip().replace("v", "").replace("-", "_").replace("+", "_")
            return version
    except Exception:
        pass
    return f"2.0.0_dev_{datetime.now().strftime('%Y%m%d')}"

def create_simple_archive():
    """Crée l'archive avec une liste de fichiers prédéfinie."""
    script_dir = Path(__file__).parent.absolute()
    base_dir = script_dir.parent.absolute()
    
    version = get_version()
    archive_name = f"AnComicsViewer_Core_App_v{version}.zip"
    archive_path = script_dir / archive_name
    
    print(f"🎯 Création de l'archive AnComicsViewer Core Application")
    print(f"📦 Version: {version}")
    print(f"📁 Source: {base_dir}")
    print(f"🗜️ Archive: {archive_name}")
    print("-" * 60)
    
    # Liste prédéfinie des fichiers essentiels
    essential_files = [
        "main.py",
        "setup.py", 
        "pyproject.toml",
        "requirements.txt",
        "requirements-ml.txt",
        "ARCHIVE_README.md",
        "assets/icon.ico",
        "assets/icon.png",
        "src/ancomicsviewer/__init__.py",
        "src/ancomicsviewer/main_app.py",
        "src/ancomicsviewer/ui/__init__.py",
        "src/ancomicsviewer/utils/__init__.py",
        "src/ancomicsviewer/utils/enhanced_cache.py",
        "src/ancomicsviewer/detectors/__init__.py",
        "src/ancomicsviewer/detectors/base.py",
        "src/ancomicsviewer/detectors/multibd_detector.py",
        "src/ancomicsviewer/detectors/postproc.py",
        "src/ancomicsviewer/detectors/reading_order.py",
        "src/ancomicsviewer/detectors/yolo_seg.py",
        "src/ancomicsviewer/detectors/models/multibd_enhanced_v2.pt",
        "data/models/multibd_enhanced_v2.pt",
        "detectors/models/multibd_enhanced_v2.pt",
        "scripts/ml/models/__init__.py",
        "scripts/ml/models/yolo_detector.py",
        "scripts/ml/benchmark.py",
        "scripts/ml/dataset.yaml",
        "scripts/tools/train_multibd_model.py",
        "scripts/tools/labelme_to_yolo.py",
        "scripts/tools/ultra_multibd_detector.py",
        "test_cli.py",
        "simple_web_viewer.py",
    ]
    
    total_size = 0
    files_included = []
    
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        for rel_path in essential_files:
            file_path = base_dir / rel_path
            if file_path.exists():
                zipf.write(file_path, rel_path)
                file_size = file_path.stat().st_size
                total_size += file_size
                files_included.append(rel_path)
                print(f"✅ {rel_path} ({file_size:,} bytes)")
            else:
                print(f"⚠️ Fichier manquant: {rel_path}")
    
    # Statistiques finales
    print("\n" + "=" * 60)
    print(f"📊 STATISTIQUES DE L'ARCHIVE")
    print("=" * 60)
    print(f"✅ Fichiers inclus: {len(files_included)}")
    print(f"📏 Taille totale: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
    
    if archive_path.exists():
        archive_size = archive_path.stat().st_size
        print(f"🗜️ Archive: {archive_path}")
        print(f"📦 Taille archive: {archive_size:,} bytes ({archive_size/1024/1024:.2f} MB)")
    
    print(f"\n🎯 UTILISATION:")
    print(f"   1. Extraire: unzip {archive_name}")
    print(f"   2. Installer: pip install -r requirements.txt")
    print(f"   3. Lancer: python main.py")
    
    return archive_path

def main():
    try:
        archive_path = create_simple_archive()
        print(f"\n✅ Archive créée: {archive_path}")
        return 0
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
