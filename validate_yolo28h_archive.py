#!/usr/bin/env python3
"""
Script de validation pour l'archive YOLO 28h Simplifiée
=======================================================
"""

import zipfile
import sys
from pathlib import Path

def validate_yolo28h_archive(archive_path):
    """Valide que l'archive contient tous les fichiers essentiels."""
    
    print(f"🔍 Validation de l'archive YOLO 28h Simplifiée")
    print(f"📦 Archive: {archive_path}")
    print("-" * 60)
    
    # Fichiers critiques requis
    critical_files = [
        "main.py",
        "src/ancomicsviewer/main_app.py", 
        "src/ancomicsviewer/detectors/yolo_28h_detector.py",
        "runs/multibd_enhanced_v2/yolov8s-mps-1280/weights/best.pt",
        "YOLO28H_SIMPLIFIED_README.md"
    ]
    
    # Vérification de l'existence de l'archive
    if not Path(archive_path).exists():
        print(f"❌ Archive non trouvée: {archive_path}")
        return False
    
    archive_size = Path(archive_path).stat().st_size
    print(f"📏 Taille archive: {archive_size:,} bytes ({archive_size/1024/1024:.2f} MB)")
    
    # Validation du contenu
    missing_files = []
    with zipfile.ZipFile(archive_path, 'r') as zipf:
        file_list = zipf.namelist()
        
        print(f"\n📋 Fichiers dans l'archive: {len(file_list)}")
        
        for critical_file in critical_files:
            if critical_file in file_list:
                info = zipf.getinfo(critical_file)
                print(f"✅ {critical_file} ({info.file_size:,} bytes)")
            else:
                missing_files.append(critical_file)
                print(f"❌ MANQUANT: {critical_file}")
    
    # Vérification du modèle YOLO
    model_file = "runs/multibd_enhanced_v2/yolov8s-mps-1280/weights/best.pt"
    if model_file in file_list:
        with zipfile.ZipFile(archive_path, 'r') as zipf:
            model_info = zipf.getinfo(model_file)
            model_size = model_info.file_size
            print(f"\n🔥 Modèle YOLO 28h: {model_size:,} bytes ({model_size/1024/1024:.2f} MB)")
            if model_size > 20_000_000:  # Plus de 20MB
                print("✅ Taille du modèle correcte")
            else:
                print("⚠️ Modèle peut-être incomplet")
    
    # Résultat final
    print("\n" + "=" * 60)
    if not missing_files:
        print("✅ VALIDATION RÉUSSIE - Archive complète!")
        print("🎯 Cette archive contient tous les fichiers pour YOLO 28h simplifié")
        return True
    else:
        print(f"❌ VALIDATION ÉCHOUÉE - {len(missing_files)} fichiers manquants")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python validate_yolo28h_archive.py <archive.zip>")
        return 1
    
    archive_path = sys.argv[1]
    success = validate_yolo28h_archive(archive_path)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
