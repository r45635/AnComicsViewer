#!/usr/bin/env python3
"""
Add Tintin Comic to Dataset
Extracts pages from Tintin PDF and adds them to the existing training dataset.
"""

import os
import sys
import subprocess
from pathlib import Path

def add_tintin_to_dataset():
    """Add Tintin pages to the existing dataset using the existing export tool."""
    
    print("📚 AJOUT DE TINTIN AU DATASET")
    print("=" * 35)
    
    # PDF file
    tintin_pdf = "Tintin - 161 - Le Lotus Bleu - .pdf"
    if not Path(tintin_pdf).exists():
        print(f"❌ PDF non trouvé: {tintin_pdf}")
        return False
    
    # Temporary directory for Tintin extraction
    temp_dir = Path("temp_tintin")
    temp_dir.mkdir(exist_ok=True)
    
    print(f"📖 Extraction: {tintin_pdf}")
    
    try:
        # Use the existing PDF export tool
        print("� Extraction des pages...")
        
        # Modify the export tool to work with different PDFs
        export_cmd = [
            sys.executable, 
            "tools/export_pdf_pages.py",
            "--pdf", tintin_pdf,
            "--output", str(temp_dir),
            "--prefix", "t",  # Tintin prefix
            "--dpi", "300"
        ]
        
        # Check if the export tool supports these parameters, if not use it directly
        try:
            result = subprocess.run(export_cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                # Fallback: use the tool as-is and rename later
                print("� Utilisation de l'outil d'export existant...")
                result = subprocess.run([
                    sys.executable, 
                    "-c", 
                    f"""
import sys
sys.path.insert(0, '.')
from tools.export_pdf_pages import export_pdf_pages
export_pdf_pages('{tintin_pdf}', '{temp_dir}', dpi=300, prefix='t')
"""
                ], capture_output=True, text=True)
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("⚠️  Outil d'export non disponible, extraction manuelle...")
            return False
            
        if result.returncode == 0:
            print("✅ Extraction réussie!")
        else:
            print(f"❌ Erreur d'extraction: {result.stderr}")
            return False
        
        # Move extracted pages to dataset
        extracted_files = list(temp_dir.glob("*.png"))
        if not extracted_files:
            print("❌ Aucune page extraite")
            return False
        
        print(f"📄 Pages extraites: {len(extracted_files)}")
        
        # Dataset directories
        train_images_dir = Path("dataset/images/train")
        val_images_dir = Path("dataset/images/val")
        
        train_images_dir.mkdir(parents=True, exist_ok=True)
        val_images_dir.mkdir(parents=True, exist_ok=True)
        
        # Move files with proper naming
        moved_train = 0
        moved_val = 0
        
        for i, src_file in enumerate(sorted(extracted_files)):
            # Use Tintin prefix with proper numbering
            new_name = f"t{i+1:04d}.png"
            
            # 90/10 split for train/val
            if i % 10 == 0:  # Every 10th page to validation
                dest_path = val_images_dir / new_name
                moved_val += 1
                set_name = "val"
            else:
                dest_path = train_images_dir / new_name
                moved_train += 1
                set_name = "train"
            
            # Move file
            src_file.rename(dest_path)
            print(f"📁 {src_file.name} → {new_name} ({set_name})")
        
        # Cleanup temp directory
        temp_dir.rmdir()
        
        print()
        print(f"✅ Ajout terminé!")
        print(f"📊 Nouvelles pages ajoutées:")
        print(f"   Train: +{moved_train} pages")
        print(f"   Val: +{moved_val} pages")
        
        # Show total dataset size
        total_train = len(list(train_images_dir.glob("*.png")))
        total_val = len(list(val_images_dir.glob("*.png")))
        tintin_train = len(list(train_images_dir.glob("t*.png")))
        tintin_val = len(list(val_images_dir.glob("t*.png")))
        
        print()
        print(f"📁 Dataset total:")
        print(f"   Train: {total_train} images ({tintin_train} Tintin)")
        print(f"   Val: {total_val} images ({tintin_val} Tintin)")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def update_class_definitions():
    """Update class definitions to reflect mixed dataset."""
    
    classes_file = Path("dataset/predefined_classes.txt")
    
    # Read current classes
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            current_classes = f.read().strip().split('\n')
    else:
        current_classes = ["panel", "text"]
    
    # Add comic-specific classes if needed
    new_classes = current_classes.copy()
    
    # Could add specific classes for different comic styles
    if "panel_tintin" not in new_classes:
        new_classes.append("panel_tintin")
    
    # Write updated classes
    with open(classes_file, 'w') as f:
        f.write('\n'.join(new_classes))
    
    print(f"📝 Classes mises à jour: {new_classes}")

def main():
    """Main function."""
    
    print("🎨 EXTENSION DU DATASET AVEC TINTIN")
    print("=" * 45)
    print()
    
    # Check current state
    current_train = len(list(Path("dataset/images/train").glob("*.png")))
    current_val = len(list(Path("dataset/images/val").glob("*.png")))
    
    print(f"📊 Dataset actuel:")
    print(f"   Train: {current_train} images")
    print(f"   Val: {current_val} images")
    print()
    
    # Add Tintin pages
    if add_tintin_to_dataset():
        print()
        print("🎯 PROCHAINES ÉTAPES:")
        print("=" * 20)
        print("1. 🏷️  Annoter les nouvelles pages Tintin:")
        print("   python start_annotation.py")
        print()
        print("2. 🔄 Régénérer le dataset YOLO:")
        print("   python tools/labelme_to_yolo.py")
        print()
        print("3. 🏋️  Réentraîner avec plus de données:")
        print("   python continue_training.py")
        print()
        print("💡 STRATÉGIE D'ANNOTATION:")
        print("   • Commencer par 5-10 pages Tintin")
        print("   • Style différent = meilleure généralisation")
        print("   • Panels plus ronds/organiques vs rectangulaires")
        
    else:
        print("❌ Échec de l'ajout de Tintin")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
