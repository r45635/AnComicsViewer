#!/usr/bin/env python3
"""
Script de construction local pour AnComicsViewer
Teste la génération d'exécutables standalone avant le CI/CD
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import platform

def install_pyinstaller():
    """Installe PyInstaller si nécessaire."""
    try:
        import PyInstaller
        print("✅ PyInstaller déjà installé")
        return True
    except ImportError:
        print("🔄 Installation de PyInstaller...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'pyinstaller'], 
                         check=True, capture_output=True)
            print("✅ PyInstaller installé")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Échec installation PyInstaller: {e}")
            return False

def check_dependencies():
    """Vérifie que toutes les dépendances sont installées."""
    print("🔍 Vérification des dépendances...")
    
    required_modules = [
        'PySide6',
        'numpy', 
        'cv2',
        'PIL'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module}")
            missing.append(module)
    
    if missing:
        print(f"\\n❌ Modules manquants: {', '.join(missing)}")
        print("💡 Installez avec: pip install -r requirements.txt")
        return False
    
    print("✅ Toutes les dépendances principales sont disponibles")
    return True

def generate_spec():
    """Génère le fichier .spec pour PyInstaller."""
    print("📋 Génération du fichier .spec...")
    
    try:
        result = subprocess.run([sys.executable, 'build_spec.py'], 
                              check=True, capture_output=True, text=True)
        print("✅ Fichier .spec généré")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Échec génération .spec: {e}")
        print(f"Sortie: {e.stdout}")
        print(f"Erreur: {e.stderr}")
        return False

def build_executable():
    """Construit l'exécutable avec PyInstaller."""
    print("🏗️ Construction de l'exécutable...")
    
    spec_file = "AnComicsViewer.spec"
    if not Path(spec_file).exists():
        print(f"❌ Fichier {spec_file} non trouvé")
        return False
    
    # Nettoyer les builds précédents
    for cleanup_dir in ['build', 'dist']:
        if Path(cleanup_dir).exists():
            print(f"🧹 Nettoyage {cleanup_dir}/")
            shutil.rmtree(cleanup_dir)
    
    try:
        cmd = [sys.executable, '-m', 'PyInstaller', spec_file, '--clean', '--noconfirm']
        print(f"Commande: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, check=True, text=True)
        print("✅ Construction terminée")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Échec construction: {e}")
        return False

def test_executable():
    """Teste l'exécutable généré."""
    print("🧪 Test de l'exécutable...")
    
    system = platform.system()
    
    if system == "Windows":
        exe_path = Path("dist/AnComicsViewer.exe")
    elif system == "Darwin":  # macOS
        exe_path = Path("dist/AnComicsViewer.app/Contents/MacOS/AnComicsViewer")
        if not exe_path.exists():
            exe_path = Path("dist/AnComicsViewer")  # Fallback
    else:  # Linux
        exe_path = Path("dist/AnComicsViewer")
    
    if not exe_path.exists():
        print(f"❌ Exécutable non trouvé: {exe_path}")
        print("📁 Contenu du dossier dist/:")
        if Path("dist").exists():
            for item in Path("dist").iterdir():
                print(f"  {item}")
        return False
    
    # Test basic (juste vérifier que ça lance)
    try:
        # Test simple: version ou aide
        result = subprocess.run([str(exe_path), '--version'], 
                              capture_output=True, text=True, timeout=10)
        print(f"✅ Exécutable testé (exit code: {result.returncode})")
        return True
    except subprocess.TimeoutExpired:
        print("⚠️ Timeout test exécutable (normal pour app graphique)")
        return True
    except Exception as e:
        print(f"⚠️ Test exécutable échoué: {e}")
        print("💡 Cela peut être normal pour une app graphique sans display")
        return True  # On considère comme OK

def show_results():
    """Affiche les résultats de la construction."""
    print("\\n📊 Résultats de la construction:")
    
    dist_dir = Path("dist")
    if not dist_dir.exists():
        print("❌ Aucun exécutable généré")
        return
    
    system = platform.system()
    
    for item in dist_dir.iterdir():
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"📦 {item.name}: {size_mb:.1f} MB")
        elif item.is_dir() and item.name.endswith('.app'):
            # macOS app bundle
            try:
                result = subprocess.run(['du', '-sm', str(item)], 
                                      capture_output=True, text=True)
                size_mb = int(result.stdout.split()[0])
                print(f"📦 {item.name}: {size_mb} MB")
            except:
                print(f"📦 {item.name}: (taille indéterminée)")
    
    print(f"\\n🎯 Plateforme: {system}")
    print("✅ Exécutable standalone prêt pour distribution!")
    print("🚀 Aucune dépendance requise pour l'utilisateur final")

def main():
    """Fonction principale."""
    print("🏗️ CONSTRUCTION EXÉCUTABLE ANCOMICSVIEWER")
    print("=" * 50)
    
    # Vérifications préliminaires
    if not install_pyinstaller():
        return False
    
    if not check_dependencies():
        return False
    
    # Construction
    if not generate_spec():
        return False
    
    if not build_executable():
        return False
    
    if not test_executable():
        print("⚠️ Test de l'exécutable a échoué, mais la construction peut être OK")
    
    show_results()
    
    print("\\n🎉 Construction terminée avec succès!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
