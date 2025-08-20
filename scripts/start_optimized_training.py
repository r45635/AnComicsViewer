#!/usr/bin/env python3
"""
Script de démarrage rapide pour l'entraînement optimisé Multi-BD Enhanced v2
Simplifie le lancement avec des paramètres pré-validés
"""

import sys
import subprocess
from pathlib import Path

def main():
    print("🚀 Démarrage Entraînement Optimisé Multi-BD Enhanced v2")
    print("=" * 60)
    
    # Vérifications préliminaires
    script_path = Path("train_enhanced_v2.py")
    if not script_path.exists():
        print(f"❌ Script d'entraînement non trouvé: {script_path}")
        return False
    
    config_path = Path("dataset/multibd_enhanced.yaml")
    if not config_path.exists():
        print(f"❌ Configuration dataset non trouvée: {config_path}")
        return False
    
    venv_python = Path(".venv/bin/python")
    if not venv_python.exists():
        print(f"❌ Environnement virtuel non trouvé: {venv_python}")
        print(f"💡 Exécuter: python -m venv .venv && .venv/bin/pip install -r requirements.txt")
        return False
    
    print("✅ Tous les prérequis sont satisfaits")
    print("\n🎯 Configuration d'entraînement optimisée:")
    print("   • Hyperparamètres: Learning rate adaptatif, batch size optimisé")
    print("   • Augmentations: Équilibrées pour BD (pas de rotation excessive)")
    print("   • Callbacks: Monitoring avancé, early stopping intelligent")
    print("   • Performance: Mixed precision, cache optimisé")
    print("   • Évaluation: Métriques complètes et copie automatique du modèle")
    
    # Confirmation utilisateur
    response = input("\n🤔 Lancer l'entraînement optimisé ? (o/N): ").strip().lower()
    if response not in ['o', 'oui', 'y', 'yes']:
        print("⏹️  Entraînement annulé")
        return False
    
    # Lancement
    print("\n🎬 Lancement de l'entraînement...")
    try:
        result = subprocess.run([
            str(venv_python), 
            str(script_path)
        ], check=True)
        
        print("\n🎊 Entraînement terminé avec succès!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Erreur pendant l'entraînement (code {e.returncode})")
        return False
    except KeyboardInterrupt:
        print(f"\n⏹️  Entraînement interrompu par l'utilisateur")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
