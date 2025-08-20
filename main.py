#!/usr/bin/env python3
"""
AnComicsViewer - Main Entry Point
==================================

📖 Lecteur PDF pour bandes dessinées avec détection intelligente de cases
🤖 Détection ML multi-styles (Golden City, Tintin, Pin-up) + navigation cross-page
🎯 Interface moderne PySide6 avec overlay interactif des panels

Auteur: Vincent Cruvellier
License: MIT
Repository: https://github.com/r45635/AnComicsViewer
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Ajouter le répertoire du script au PYTHONPATH
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

def get_version():
    """Récupère la version depuis Git ou fallback."""
    try:
        # Essayer de récupérer la version Git
        result = subprocess.run(
            ["git", "describe", "--tags", "--long", "--dirty"],
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            # Convertir v2.0.0-3-gbd5772c en v2.0.0+3.bd5772c
            if '-g' in version:
                parts = version.split('-')
                if len(parts) >= 3:
                    tag = parts[0]  # v2.0.0
                    commits = parts[1]  # 3
                    commit_hash = parts[2]  # gbd5772c
                    if parts[-1] == 'dirty':
                        return f"{tag}+{commits}.{commit_hash}.dirty"
                    else:
                        return f"{tag}+{commits}.{commit_hash}"
            return version
    except Exception:
        pass
    
    # Fallback vers une version statique
    return "v2.0.0+dev"

def check_environment():
    """Vérifie l'environnement Python et les dépendances critiques."""
    errors = []
    
    # Vérifier Python version
    if sys.version_info < (3, 8):
        errors.append(f"❌ Python 3.8+ requis, trouvé: {sys.version}")
    
    # Vérifier les dépendances critiques
    critical_deps = ['PySide6', 'numpy', 'cv2']
    for dep in critical_deps:
        try:
            __import__(dep)
        except ImportError:
            errors.append(f"❌ Module manquant: {dep}")
    
    return errors

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="AnComicsViewer",
        description="Lecteur PDF pour bandes dessinées avec détection intelligente de cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python main.py                                    # Interface graphique normale
  python main.py --preset fb mycomics.pdf           # Ouvrir avec preset Franco-Belge
  python main.py --detector multibd --page 5        # Commencer à la page 5 avec Multi-BD
  python main.py --dpi 300 --preset manga file.pdf # Manga haute résolution

Variables d'environnement supportées:
  ANCOMICS_PRESET   : fb|manga|newspaper
  ANCOMICS_DETECTOR : heur|yolo|multibd
  ANCOMICS_DPI      : résolution de détection (100-400)
  ANCOMICS_PDF      : chemin du fichier PDF à ouvrir
  ANCOMICS_PAGE     : page de démarrage (0-based)
        """
    )
    
    parser.add_argument("pdf_file", nargs="?", help="Fichier PDF à ouvrir au démarrage")
    parser.add_argument("--version", action="version", version=f"AnComicsViewer {get_version()}")
    
    # Options de configuration
    parser.add_argument("--preset", choices=["fb", "manga", "newspaper"], 
                       help="Preset de détection (fb=Franco-Belge, manga=Japonais, newspaper=US)")
    parser.add_argument("--detector", choices=["heur", "yolo", "multibd"], 
                       help="Type de détecteur (heur=Heuristique, yolo=YOLO, multibd=Multi-BD)")
    parser.add_argument("--dpi", type=int, metavar="N", 
                       help="Résolution de détection (100-400, défaut: 200)")
    parser.add_argument("--page", type=int, metavar="N", default=0,
                       help="Page de démarrage (0-based, défaut: 0)")
    
    return parser.parse_args()

def setup_environment(args):
    """Configure environment variables from command line arguments."""
    if args.preset:
        os.environ["ANCOMICS_PRESET"] = args.preset
    if args.detector:
        os.environ["ANCOMICS_DETECTOR"] = args.detector
    if args.dpi:
        os.environ["ANCOMICS_DPI"] = str(args.dpi)
    if args.pdf_file:
        os.environ["ANCOMICS_PDF"] = args.pdf_file
    if args.page > 0:
        os.environ["ANCOMICS_PAGE"] = str(args.page)


def main():
    """Point d'entrée principal avec gestion des arguments."""
    # Analyser les arguments de ligne de commande
    args = parse_arguments()
    
    # Configurer les variables d'environnement depuis les arguments
    setup_environment(args)
    
    print("🎨 AnComicsViewer - Lecteur PDF Comics Intelligent")
    print(f"📦 Version: {get_version()}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"📁 Répertoire: {SCRIPT_DIR}")
    if args.pdf_file:
        print(f"📂 Fichier: {args.pdf_file}")
    print("-" * 60)
    
    # Vérification de l'environnement
    env_errors = check_environment()
    if env_errors:
        print("💥 Erreurs d'environnement détectées:")
        for error in env_errors:
            print(f"   {error}")
        print("\n💡 Solutions:")
        print("   • Installer les dépendances: pip install -r requirements.txt")
        print("   • Utiliser l'environnement virtuel: ./run.sh")
        print("   • Vérifier la documentation: README.md")
        return 1
    
    # Import et lancement de l'application
    try:
        print("🚀 Chargement de l'application...")
        
        # Configurer l'icône pour l'application
        os.environ['ANCOMICSVIEWER_ICON'] = str(SCRIPT_DIR / "icon.ico")
        
        # Importer et lancer AnComicsViewer
        from src.ancomicsviewer import main as app_main
        
        print("✅ Interface utilisateur initialisée")
        print("📖 Prêt à ouvrir des fichiers PDF!")
        if args.pdf_file:
            print(f"📂 Ouverture de: {args.pdf_file}")
        else:
            print("💡 Utilisez Ctrl+O pour ouvrir un fichier ou glissez-déposez")
        
        return app_main()
        
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        print("💡 Vérifiez que toutes les dépendances sont installées")
        return 1
    except Exception as e:
        print(f"💥 Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    # Configurer l'environnement pour une meilleure compatibilité
    if sys.platform == "darwin":  # macOS
        # Améliorer la compatibilité Qt sur macOS
        if "QT_MAC_WANTS_LAYER" not in os.environ:
            os.environ["QT_MAC_WANTS_LAYER"] = "1"
    
    # Point d'entrée
    sys.exit(main())
