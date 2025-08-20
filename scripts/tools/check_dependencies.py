#!/usr/bin/env python3
"""
Script de vérification des dépendances pour AnComicsViewer.
Vérifie que tous les modules requis sont disponibles.
"""

import sys
import os

def check_module(name, optional=False):
    """Vérifie qu'un module peut être importé."""
    try:
        __import__(name)
        print(f"✅ {name:<20} - Disponible")
        return True
    except ImportError as e:
        status = "⚠️" if optional else "❌"
        print(f"{status} {name:<20} - Manquant" + (f" (optionnel)" if optional else ""))
        if not optional:
            print(f"   Erreur: {e}")
        return False

def main():
    print("🔍 Vérification des dépendances AnComicsViewer")
    print("=" * 50)
    
    # Dépendances principales
    print("\n📦 Dépendances principales:")
    deps_main = [
        "PySide6",
        "numpy", 
        "cv2",
        "PIL"
    ]
    
    main_ok = all(check_module(dep) for dep in deps_main)
    
    # Dépendances ML
    print("\n🤖 Dépendances Machine Learning:")
    deps_ml = [
        "torch",
        "torchvision", 
        "ultralytics",
        "matplotlib"
    ]
    
    ml_ok = all(check_module(dep, optional=True) for dep in deps_ml)
    
    # Résumé
    print("\n" + "=" * 50)
    if main_ok:
        print("✅ Toutes les dépendances principales sont disponibles")
        print("🚀 AnComicsViewer peut démarrer")
    else:
        print("❌ Dépendances principales manquantes")
        print("💡 Exécutez: pip install -r requirements.txt")
    
    if ml_ok:
        print("✅ Toutes les dépendances ML sont disponibles")
        print("🎯 Détecteurs Multi-BD et YOLOv8 fonctionnels")
    else:
        print("⚠️ Certaines dépendances ML manquantes")
        print("💡 Pour ML complet: pip install -r requirements-ml.txt")
    
    # Information environnement
    print(f"\n🐍 Python: {sys.version}")
    print(f"📁 Répertoire: {os.getcwd()}")
    
    return main_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
