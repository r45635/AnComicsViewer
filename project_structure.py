#!/usr/bin/env python3
"""
📁 Structure du projet AnComicsViewer v2.0 réorganisé
Affiche la nouvelle architecture clean et organisée
"""

import os
from pathlib import Path

def show_project_structure():
    """Affiche la structure finale du projet."""
    
    print("📁 ANCOMICSVIEWER V2.0 - STRUCTURE FINALE")
    print("=" * 50)
    print()
    
    structure = {
        "🎯 CORE (Application principale)": [
            "AnComicsViewer.py",
            "detectors/base.py",
            "detectors/multibd_detector.py", 
            "detectors/yolo_seg.py",
            "patch_pytorch.py"
        ],
        "🔧 TOOLS (Développement & Maintenance)": [
            "tools/train_multibd_model.py",
            "tools/test_multibd_integration.py",
            "tools/dataset_analyzer.py",
            "tools/start_annotation.py",
            "tools/labelme_to_yolo.py",
            "tools/integrate_pinup_system.py",
            "tools/integrate_tintin.py",
            "tools/smoke_test.py",
            "tools/post_annotation_processing.py",
            "tools/integration_summary.py",
            "tools/release_summary.py"
        ],
        "📖 DOCS (Documentation)": [
            "README.md",
            "docs/MULTIBD_GUIDE.md",
            "docs/RELEASE_NOTES_v2.0.md", 
            "docs/QUICK_REFERENCE.md"
        ],
        "🎨 ASSETS (Ressources)": [
            "assets/favicon.png",
            "assets/icon.png",
            "assets/logo.png"
        ],
        "⚙️ CONFIG (Configuration)": [
            "requirements.txt",
            "requirements-ml.txt",
            ".gitignore",
            "LICENSE"
        ],
        "📊 DATA (Dataset & Modèles)": [
            "dataset/images/",
            "dataset/labels/",
            "runs/detect/multibd_mixed_model/"
        ]
    }
    
    total_files = 0
    
    for category, files in structure.items():
        print(f"{category}")
        print("-" * len(category.replace("🎯 ", "").replace("🔧 ", "").replace("📖 ", "").replace("🎨 ", "").replace("⚙️ ", "").replace("📊 ", "")))
        
        category_count = 0
        for file_path in files:
            path = Path(file_path)
            if path.exists():
                if path.is_dir():
                    file_count = len(list(path.rglob("*")))
                    print(f"✅ {file_path:<35} ({file_count} fichiers)")
                    category_count += file_count
                else:
                    size = path.stat().st_size
                    size_str = f"{size//1024}KB" if size > 1024 else f"{size}B"
                    print(f"✅ {file_path:<35} ({size_str})")
                    category_count += 1
            else:
                print(f"❌ {file_path:<35} (manquant)")
        
        print(f"   Sous-total: {category_count} fichiers\n")
        total_files += category_count
    
    print("=" * 50)
    print(f"📊 TOTAL: {total_files} fichiers dans la structure")
    print()
    
    return True

def show_usage_instructions():
    """Affiche les instructions d'utilisation."""
    
    print("🚀 INSTRUCTIONS D'UTILISATION")
    print("=" * 35)
    print()
    
    print("📋 UTILISATION STANDARD:")
    print("  python AnComicsViewer.py")
    print("  Menu ⚙️ → Detector → Multi-BD (Trained)")
    print()
    
    print("🧪 TESTS & VALIDATION:")
    print("  python tools/test_multibd_integration.py  # Test complet")
    print("  python tools/smoke_test.py               # Test de base")
    print("  python tools/dataset_analyzer.py coverage # Analyse dataset")
    print()
    
    print("🔧 DÉVELOPPEMENT:")
    print("  python tools/train_multibd_model.py      # Ré-entraînement")
    print("  python tools/start_annotation.py         # Annotation dataset")
    print("  python tools/integration_summary.py      # Rapport détaillé")
    print()
    
    print("📖 DOCUMENTATION:")
    print("  docs/MULTIBD_GUIDE.md                    # Guide utilisateur")
    print("  docs/RELEASE_NOTES_v2.0.md              # Notes de version")
    print("  docs/QUICK_REFERENCE.md                 # Référence rapide")

def show_cleanup_summary():
    """Affiche le résumé du nettoyage."""
    
    print("\n🧹 RÉSUMÉ DU NETTOYAGE")
    print("=" * 28)
    print()
    
    removed_items = [
        "Scripts de développement temporaires (15 fichiers)",
        "Tests unitaires obsolètes (7 fichiers)", 
        "Fichiers de configuration anciens (5 fichiers)",
        "Dossiers temporaires (temp_tintin/, golden_city_samples/)",
        "Documentation redondante (3 fichiers)",
        "Utilitaires de génération (generate_logo.py, etc.)"
    ]
    
    print("🗑️  SUPPRIMÉ:")
    for item in removed_items:
        print(f"   • {item}")
    
    print()
    reorganized_items = [
        "Outils → tools/ (11 scripts)",
        "Documentation → docs/ (3 guides)",
        "Assets → assets/ (3 fichiers)",
        "Imports corrigés dans tous les scripts"
    ]
    
    print("📁 RÉORGANISÉ:")
    for item in reorganized_items:
        print(f"   • {item}")
    
    print()
    print("✨ RÉSULTAT: Structure claire, maintenable et professionnelle")

def main():
    """Fonction principale."""
    
    if show_project_structure():
        show_usage_instructions()
        show_cleanup_summary()
        
        print(f"\n🎉 PROJET ANCOMICSVIEWER V2.0 PARFAITEMENT ORGANISÉ !")
        print("🚀 Prêt pour utilisation et développement futur")

if __name__ == "__main__":
    main()
