#!/usr/bin/env python3
"""
Créateur d'Archive Ultra-Simplifiée - Cœur du Programme
=======================================================
Archive minimale avec juste les fichiers essentiels pour analyse ChatGPT.
"""

import zipfile
import sys
from pathlib import Path
from datetime import datetime

def create_minimal_core_archive():
    """Crée une archive ultra-minimale avec juste le cœur."""
    script_dir = Path(__file__).parent.absolute()
    base_dir = script_dir.parent.absolute()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    archive_name = f"AnComicsViewer_MINIMAL_CORE_{timestamp}.zip"
    archive_path = script_dir / archive_name
    
    print(f"🎯 Création d'archive ULTRA-SIMPLIFIÉE pour analyse ChatGPT")
    print(f"📦 Archive: {archive_name}")
    print(f"📁 Source: {base_dir}")
    print("-" * 60)
    
    # SEULEMENT les fichiers critiques pour comprendre le problème
    core_files = [
        # Point d'entrée
        "main.py",
        
        # Interface principale (problème potentiel)
        "src/ancomicsviewer/main_app.py",
        
        # Détecteurs actuels
        "src/ancomicsviewer/detectors/ultra_robust_detector.py",
        "src/ancomicsviewer/detectors/ultra_panel_detector.py", 
        "src/ancomicsviewer/detectors/yolo_28h_detector.py",
        "src/ancomicsviewer/detectors/base.py",
        
        # Utils critiques
        "src/ancomicsviewer/ui/qimage_utils.py",
        "src/ancomicsviewer/panels_service.py",
        
        # Configuration
        "requirements.txt",
        "pyproject.toml",
        
        # Tests/Debug
        "tools/quickcheck.py",
        
        # Documentation du problème
        "PATCHES_DROPIN_ULTRA_ROBUST_REPORT.md",
    ]
    
    total_size = 0
    files_included = []
    files_missing = []
    
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        for rel_path in core_files:
            file_path = base_dir / rel_path
            if file_path.exists():
                zipf.write(file_path, rel_path)
                file_size = file_path.stat().st_size
                total_size += file_size
                files_included.append(rel_path)
                print(f"✅ {rel_path} ({file_size:,} bytes)")
            else:
                files_missing.append(rel_path)
                print(f"⚠️ MANQUANT: {rel_path}")
        
        # Ajouter un README spécifique pour l'analyse
        readme_content = f"""# AnComicsViewer - Archive Minimale pour Analyse ChatGPT

## 🎯 Contexte du Problème

Cette archive contient le CŒUR MINIMAL du programme AnComicsViewer après implémentation des patches drop-in ultra-robustes.

## 📊 État Actuel

### ✅ Ce qui fonctionne :
- Détecteur ultra-robuste intégré
- Application démarre sans crash
- Détection de panels opérationnelle (12 panels page 3)
- Navigation fluide entre panels

### ❓ Analyse Requise :
- Architecture générale du code
- Points d'amélioration potentiels
- Optimisations possibles
- Détection de code redondant ou problématique

## 📁 Fichiers Inclus

### Interface Principale
- `main.py` - Point d'entrée
- `src/ancomicsviewer/main_app.py` - Interface Qt principale (2037 lignes)

### Détecteurs
- `src/ancomicsviewer/detectors/ultra_robust_detector.py` - Nouveau détecteur drop-in
- `src/ancomicsviewer/detectors/ultra_panel_detector.py` - Architecture ultra-robuste
- `src/ancomicsviewer/detectors/yolo_28h_detector.py` - Ancien détecteur simplifié
- `src/ancomicsviewer/detectors/base.py` - Interface de base

### Utilitaires
- `src/ancomicsviewer/ui/qimage_utils.py` - Conversion QImage sécurisée
- `src/ancomicsviewer/panels_service.py` - Service de détection

### Debug/Test
- `tools/quickcheck.py` - Validation CLI

## 🚀 Instructions pour ChatGPT

1. **Analyser l'architecture générale** du code
2. **Identifier les redondances** entre détecteurs
3. **Suggérer des simplifications** possibles
4. **Détecter les anti-patterns** ou code problématique
5. **Proposer des améliorations** d'architecture

## 📊 Métriques

- **Fichiers inclus**: {len(files_included)}
- **Taille totale**: {total_size:,} bytes
- **Timestamp**: {timestamp}
- **Status**: Système fonctionnel mais à optimiser

## 🎯 Objectif

Obtenir une analyse externe pour identifier les améliorations possibles et simplifications du code actuel.

---
*Archive créée le {datetime.now().strftime('%d/%m/%Y à %H:%M')}*
*Système opérationnel mais demande analyse d'optimisation*
"""
        zipf.writestr("README_ANALYSIS.md", readme_content)
        files_included.append("README_ANALYSIS.md")
    
    # Statistiques finales
    print("\n" + "=" * 60)
    print(f"📊 ARCHIVE ULTRA-SIMPLIFIÉE CRÉÉE")
    print("=" * 60)
    print(f"✅ Fichiers inclus: {len(files_included)}")
    if files_missing:
        print(f"⚠️ Fichiers manquants: {len(files_missing)}")
    print(f"📏 Taille totale: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
    
    if archive_path.exists():
        archive_size = archive_path.stat().st_size
        print(f"🗜️ Archive: {archive_path}")
        print(f"📦 Taille archive: {archive_size:,} bytes ({archive_size/1024/1024:.2f} MB)")
    
    print(f"\n🎯 POUR CHATGPT:")
    print(f"   1. Analyser: unzip {archive_name}")
    print(f"   2. Lire: README_ANALYSIS.md")
    print(f"   3. Examiner: main_app.py + détecteurs")
    print(f"   4. Suggérer: améliorations et simplifications")
    
    return archive_path

def main():
    try:
        archive_path = create_minimal_core_archive()
        print(f"\n✅ Archive minimale créée: {archive_path}")
        return 0
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
