#!/usr/bin/env python3
"""
AnComicsViewer - Créateur d'Archive Application
==============================================

Script pour créer une archi            # Ignorer certains dossiers dès le départ pour optimiser
            dirs[:] = [d for d in dirs if d.lower() not in {
                '.git', '__pycache__', '.venv', 'venv', 'docs', 'tests',
                'build', 'dist', 'cache', '.pytest_cache', '.mypy_cache',
                'wandb', 'notebooks', 'dataset', 'datasets', 'data',
                '.github'  # Garder runs/ pour les modèles
            }]avec uniquement les fichiers essentiels
pour le fonctionnement de l'application principale AnComicsViewer.

Inclut:
- Fichiers Python du core (.py)
- Fichiers de configuration (.ini, .yaml, .toml)
- Requirements et setup
- Assets essentiels
- Point d'entrée principal

Exclut:
- Documentation (docs/, README, etc.)
- Tests (tests/, test_*.py)
- Scripts d'entraînement (scripts/training/)
- Cache et fichiers temporaires
- Modèles ML volumineux (optionnel)

Auteur: Vincent Cruvellier
"""

import os
import zipfile
import sys
from pathlib import Path
from datetime import datetime

def get_version():
    """Récupère la version pour nommer l'archive."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "describe", "--tags", "--long", "--dirty"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            # Nettoyer la version pour le nom de fichier
            version = version.replace("v", "").replace("-", "_").replace("+", "_")
            return version
    except Exception:
        pass
    return f"2.0.0_dev_{datetime.now().strftime('%Y%m%d')}"

def should_include_file(file_path, base_dir):
    """Détermine si un fichier doit être inclus dans l'archive."""
    relative_path = file_path.relative_to(base_dir)
    path_str = str(relative_path).lower()
    
    # Extensions autorisées
    allowed_extensions = {'.py', '.ini', '.yaml', '.yml', '.toml', '.txt', '.ico', '.png', '.jpg', '.svg'}
    
    # Vérifier l'extension
    if file_path.suffix.lower() not in allowed_extensions:
        return False
    
    # Fichiers spécifiques à inclure (configuration et setup)
    core_files = {'requirements.txt', 'requirements-ml.txt', 'setup.py', 'pyproject.toml', 'MANIFEST.in', 'main.py', 'ARCHIVE_README.md'}
    if file_path.name in core_files:
        return True
    
    # Exclure le script de création d'archive lui-même
    if file_path.name in {'create_app_archive.py', 'ancomicsviewer.py'}:
        return False
    
    # Dossiers à exclure complètement
    excluded_dirs = {
        'docs', 'tests', '__pycache__', '.git', '.venv', 'venv',
        'build', 'dist', 'cache', '.pytest_cache', '.mypy_cache',
        'wandb', 'notebooks', 'examples', 'data', '.github',
        'dataset', 'datasets'  # Exclure les datasets mais garder runs/ pour les modèles
    }
    
    # Vérifier si le fichier est dans un dossier exclu
    for part in relative_path.parts:
        if part.lower() in excluded_dirs:
            return False
    
    # Scripts: ne garder que les scripts dans scripts/ml/ et certains utilitaires
    if relative_path.parts[0] == 'scripts':
        # Exclure les scripts vides (taille 0)
        if file_path.stat().st_size == 0:
            return False
        # Ne garder que les scripts utiles
        useful_script_patterns = ['ml/', 'tools/train_', 'tools/ultra_', 'tools/labelme', 'tools/benchmark']
        if not any(pattern in path_str for pattern in useful_script_patterns):
            return False
    
    # Assets: ne garder que l'icône principal
    if relative_path.parts[0] == 'assets':
        if file_path.name not in {'icon.ico', 'icon.png'}:
            return False
    
    # Fichiers de test à exclure
    if any(pattern in path_str for pattern in ['test_', '_test', 'tests']):
        return False
    
    # Documentation à exclure
    if any(pattern in path_str for pattern in ['readme', 'license', 'changelog', 'contributing']):
        return False
    
    # Fichiers temporaires et cache
    if any(pattern in path_str for pattern in ['.pyc', '.pyo', '.tmp', '.log', '.cache']):
        return False
    
    # Exclure les fichiers de métadonnées générés
    if '.egg-info' in path_str:
        return False
    
    # Modèles ML volumineux (optionnel - décommenter pour exclure)
    # if file_path.suffix.lower() in {'.pt', '.pth', '.onnx', '.weights'}:
    #     return False
    
    # Pour les modèles dans runs/ : ne garder que les meilleurs modèles
    if relative_path.parts[0] == 'runs':
        # Ne garder que les modèles finaux optimisés
        if file_path.suffix.lower() in {'.pt', '.pth', '.onnx', '.weights'}:
            # Ne garder que best.pt des modèles multibd_enhanced_v2
            if (file_path.name == 'best.pt' and 
                ('multibd_enhanced_v2' in str(relative_path) or 'multibd_enhanced_v2_stable' in str(relative_path))):
                return True
            else:
                return False
        # Garder les fichiers de config des runs
        if file_path.suffix.lower() in {'.yaml', '.yml', '.txt', '.json'}:
            # Ne garder que les configs des modèles multibd
            if 'multibd_enhanced_v2' in str(relative_path) or 'multibd_enhanced_v2_stable' in str(relative_path):
                return True
            return False
        # Exclure les autres fichiers de runs (logs, etc.)
        return False
    
    # Ne garder que les fichiers dans src/ancomicsviewer/ (package principal)
    if relative_path.parts[0] == 'src':
        if len(relative_path.parts) < 2 or relative_path.parts[1] != 'ancomicsviewer':
            return False
    
    return True

def create_app_archive():
    """Crée l'archive ZIP de l'application."""
    base_dir = Path(__file__).parent.absolute()
    version = get_version()
    archive_name = f"AnComicsViewer_v{version}_app_only.zip"
    archive_path = base_dir / archive_name
    
    print(f"🎯 Création de l'archive AnComicsViewer Core Application")
    print(f"📦 Version: {version}")
    print(f"📁 Répertoire source: {base_dir}")
    print(f"🗜️ Archive: {archive_name}")
    print("-" * 60)
    
    files_included = []
    files_excluded = []
    total_size = 0
    
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        
        # Parcourir tous les fichiers du projet
        for root, dirs, files in os.walk(base_dir):
            root_path = Path(root)
            
            # Ignorer certains dossiers dès le départ pour optimiser
            dirs[:] = [d for d in dirs if d.lower() not in {
                '.git', '__pycache__', '.venv', 'venv', 'docs', 'tests',
                'build', 'dist', 'cache', '.pytest_cache', '.mypy_cache',
                'runs', 'wandb', 'notebooks', 'dataset', 'datasets', 'data',
                '.github'
            }]
            
            for file in files:
                file_path = root_path / file
                
                try:
                    if should_include_file(file_path, base_dir):
                        # Calculer le chemin relatif dans l'archive
                        arc_path = file_path.relative_to(base_dir)
                        
                        # Ajouter à l'archive
                        zipf.write(file_path, arc_path)
                        
                        # Statistiques
                        file_size = file_path.stat().st_size
                        total_size += file_size
                        files_included.append((str(arc_path), file_size))
                        
                        print(f"✅ {arc_path} ({file_size:,} bytes)")
                    else:
                        files_excluded.append(str(file_path.relative_to(base_dir)))
                        
                except Exception as e:
                    print(f"⚠️ Erreur avec {file_path}: {e}")
                    files_excluded.append(str(file_path.relative_to(base_dir)))
    
    # Afficher les statistiques
    print("\n" + "=" * 60)
    print(f"📊 STATISTIQUES DE L'ARCHIVE")
    print("=" * 60)
    print(f"✅ Fichiers inclus: {len(files_included)}")
    print(f"❌ Fichiers exclus: {len(files_excluded)}")
    print(f"📏 Taille totale: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
    print(f"🗜️ Archive créée: {archive_path}")
    print(f"📦 Taille archive: {archive_path.stat().st_size:,} bytes ({archive_path.stat().st_size/1024/1024:.2f} MB)")
    
    # Afficher les fichiers principaux inclus
    print(f"\n📋 FICHIERS CORE INCLUS:")
    core_files = [f for f, s in files_included if any(pattern in f.lower() for pattern in [
        'main.py', 'setup.py', '__init__.py', 'requirements', 'pyproject'
    ])]
    for file in sorted(core_files):
        print(f"   • {file}")
    
    # Afficher la structure des packages
    print(f"\n📚 PACKAGES PYTHON INCLUS:")
    src_files = [f for f, s in files_included if f.startswith('src/')]
    if src_files:
        packages = set()
        for file in src_files:
            parts = Path(file).parts
            if len(parts) >= 3:  # src/package/...
                packages.add(f"src/{parts[1]}")
        for package in sorted(packages):
            count = len([f for f, s in files_included if f.startswith(package)])
            print(f"   • {package}/ ({count} fichiers)")
    
    print(f"\n🎯 POUR UTILISER L'ARCHIVE:")
    print(f"   1. Extraire: unzip {archive_name}")
    print(f"   2. Installer: cd AnComicsViewer && pip install -r requirements.txt")
    print(f"   3. Lancer: python main.py")
    print(f"   4. Ou installer package: pip install -e .")
    
    return archive_path

def main():
    """Point d'entrée principal."""
    try:
        archive_path = create_app_archive()
        print(f"\n✅ Archive créée avec succès: {archive_path}")
        return 0
    except Exception as e:
        print(f"\n❌ Erreur lors de la création de l'archive: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
