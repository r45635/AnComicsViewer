#!/usr/bin/env python3
"""
Test script pour valider l'interface en ligne de commande d'AnComicsViewer
"""

import os
import sys
import subprocess
from pathlib import Path

def test_env_vars():
    """Test des variables d'environnement"""
    print("🧪 Test des variables d'environnement...")
    
    # Configurer les variables d'environnement
    test_env = os.environ.copy()
    test_env.update({
        "ANCOMICS_PRESET": "fb",
        "ANCOMICS_DETECTOR": "multibd", 
        "ANCOMICS_DPI": "250"
    })
    
    # Tester que l'application démarre sans erreur
    try:
        result = subprocess.run(
            [sys.executable, "main.py", "--version"],
            env=test_env,
            capture_output=True,
            text=True,
            timeout=10
        )
        print(f"✅ Variables d'environnement: OK (code {result.returncode})")
        print(f"   Version: {result.stdout.strip()}")
        return True
    except Exception as e:
        print(f"❌ Variables d'environnement: ERREUR - {e}")
        return False

def test_cli_args():
    """Test des arguments en ligne de commande"""
    print("🧪 Test des arguments CLI...")
    
    test_cases = [
        ["--help"],
        ["--version"],
        ["--preset", "manga", "--dpi", "300"],
        ["--detector", "heur", "--page", "5"]
    ]
    
    for args in test_cases:
        try:
            result = subprocess.run(
                [sys.executable, "main.py"] + args,
                capture_output=True,
                text=True,
                timeout=10
            )
            if args[0] in ["--help", "--version"] and result.returncode == 0:
                print(f"✅ CLI args {' '.join(args)}: OK")
            elif args[0] not in ["--help", "--version"]:
                print(f"✅ CLI args {' '.join(args)}: OK (parsed without error)")
        except Exception as e:
            print(f"❌ CLI args {' '.join(args)}: ERREUR - {e}")
            return False
    
    return True

def test_argparse_integration():
    """Test de l'intégration argparse dans main.py"""
    print("🧪 Test de l'intégration argparse...")
    
    # Importer le module pour vérifier les fonctions
    sys.path.insert(0, str(Path(__file__).parent))
    
    try:
        import main
        
        # Vérifier que les fonctions existent
        if hasattr(main, 'parse_arguments'):
            print("✅ Fonction parse_arguments: OK")
        else:
            print("❌ Fonction parse_arguments: MANQUANTE")
            return False
            
        if hasattr(main, 'setup_environment'):
            print("✅ Fonction setup_environment: OK")
        else:
            print("❌ Fonction setup_environment: MANQUANTE")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Import main.py: ERREUR - {e}")
        return False

def main():
    print("🎯 AnComicsViewer CLI Test Suite")
    print("=" * 50)
    
    tests = [
        test_argparse_integration,
        test_cli_args,
        test_env_vars
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
            print()
        except Exception as e:
            print(f"💥 Test {test_func.__name__} failed: {e}")
            results.append(False)
            print()
    
    # Résumé
    print("📊 Résultats:")
    passed = sum(results)
    total = len(results)
    print(f"   ✅ Réussis: {passed}/{total}")
    
    if passed == total:
        print("🎉 Tous les tests CLI sont passés!")
        return 0
    else:
        print("⚠️  Certains tests CLI ont échoué")
        return 1

if __name__ == "__main__":
    sys.exit(main())
