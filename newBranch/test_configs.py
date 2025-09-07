#!/usr/bin/env python3
"""
Test rapide des configurations de détection
"""

import os
import sys
import shutil

def test_config(config_file: str, description: str):
    """Test une configuration spécifique"""
    print(f"\n{'='*60}")
    print(f"🧪 TEST: {description}")
    print(f"📄 Config: {config_file}")
    print('='*60)

    # Sauvegarder la config actuelle
    if os.path.exists('config/detect.yaml'):
        shutil.copy('config/detect.yaml', 'config/detect_backup.yaml')

    # Copier la nouvelle config
    if os.path.exists(f'config/{config_file}'):
        shutil.copy(f'config/{config_file}', 'config/detect.yaml')
        print(f"✅ Configuration {config_file} chargée")
    else:
        print(f"❌ Configuration {config_file} introuvable")
        return

    # Tester avec un timeout
    print("\n🚀 Lancement de l'application...")
    print("💡 Ouvrez un PDF et observez les logs de détection")
    print("💡 Appuyez sur Ctrl+C pour arrêter le test")

    try:
        os.system("python main.py")
    except KeyboardInterrupt:
        print("\n⏹️  Test arrêté par l'utilisateur")

    # Restaurer la config originale
    if os.path.exists('config/detect_backup.yaml'):
        shutil.copy('config/detect_backup.yaml', 'config/detect.yaml')
        os.remove('config/detect_backup.yaml')
        print("✅ Configuration originale restaurée")

def main():
    print("🔧 TESTEUR DE CONFIGURATIONS DE DÉTECTION")
    print("=" * 50)

    configs = [
        ("detect.yaml", "CONFIG ACTUELLE (sans merging)"),
        ("detect_with_merge.yaml", "CONFIG AVEC MERGING"),
    ]

    while True:
        print("\n📋 Configurations disponibles:")
        for i, (file, desc) in enumerate(configs, 1):
            print(f"  {i}. {desc}")

        print("\n  0. Quitter")

        try:
            choice = input("\nChoisissez une configuration (0-2): ").strip()

            if choice == '0':
                break
            elif choice == '1':
                test_config("detect.yaml", "CONFIG SANS MERGING")
            elif choice == '2':
                test_config("detect_with_merge.yaml", "CONFIG AVEC MERGING")
            else:
                print("❌ Choix invalide")

        except KeyboardInterrupt:
            print("\n👋 Au revoir!")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")

if __name__ == "__main__":
    # Vérifier qu'on est dans le bon répertoire
    if not os.path.exists('config'):
        print("❌ Dossier config/ introuvable. Lancez depuis newBranch/")
        sys.exit(1)

    main()
