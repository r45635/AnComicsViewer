#!/usr/bin/env python3
"""
Simple Tintin PDF Extractor
Uses the existing export tool to add Tintin pages to dataset.
"""

import sys
import shutil
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, '.')

def extract_tintin_pages():
    """Extract Tintin pages using existing tools."""
    
    print("📚 EXTRACTION DU TINTIN - LE LOTUS BLEU")
    print("=" * 45)
    
    tintin_pdf = "Tintin - 161 - Le Lotus Bleu - .pdf"
    if not Path(tintin_pdf).exists():
        print(f"❌ PDF non trouvé: {tintin_pdf}")
        return False
    
    # Create temp directory
    temp_dir = Path("temp_tintin_pages")
    temp_dir.mkdir(exist_ok=True)
    
    print(f"📖 Extraction depuis: {tintin_pdf}")
    print(f"📁 Dossier temporaire: {temp_dir}")
    
    try:
        # Import and use existing export function
        from tools.export_pdf_pages import export_pdf_to_images
        
        # Extract pages to temp directory
        print("🔄 Extraction en cours...")
        success = export_pdf_to_images(
            pdf_path=tintin_pdf,
            output_dir=str(temp_dir),
            dpi=300,
            prefix="tintin"
        )
        
        if not success:
            print("❌ Échec de l'extraction")
            return False
        
        # Check extracted files
        extracted_files = list(temp_dir.glob("*.png"))
        if not extracted_files:
            print("❌ Aucune page extraite")
            return False
        
        print(f"✅ {len(extracted_files)} pages extraites")
        
        # Move to dataset directories
        train_dir = Path("dataset/images/train")
        val_dir = Path("dataset/images/val")
        
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        
        moved_train = 0
        moved_val = 0
        
        for i, file in enumerate(sorted(extracted_files)):
            # Rename with Tintin prefix
            new_name = f"t{i+1:04d}.png"
            
            # 90/10 split
            if i % 10 == 0:  # Every 10th to validation
                dest = val_dir / new_name
                moved_val += 1
                set_type = "val"
            else:
                dest = train_dir / new_name
                moved_train += 1
                set_type = "train"
            
            shutil.move(str(file), str(dest))
            print(f"📁 {file.name} → {new_name} ({set_type})")
        
        # Cleanup
        temp_dir.rmdir()
        
        print()
        print(f"✅ Ajout terminé!")
        print(f"📊 Pages ajoutées:")
        print(f"   Train: +{moved_train}")
        print(f"   Validation: +{moved_val}")
        
        # Show totals
        total_train = len(list(train_dir.glob("*.png")))
        total_val = len(list(val_dir.glob("*.png")))
        
        print(f"📁 Dataset total: {total_train} train + {total_val} val = {total_train + total_val} images")
        
        return True
        
    except ImportError:
        print("❌ Outil d'export non disponible")
        return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

if __name__ == "__main__":
    if extract_tintin_pages():
        print()
        print("🎯 PROCHAINES ÉTAPES:")
        print("1. 🏷️  Annoter quelques pages Tintin:")
        print("   python start_annotation.py")
        print("2. 🔄 Régénérer dataset YOLO:")
        print("   python tools/labelme_to_yolo.py")
        print("3. 🏋️  Continuer l'entraînement:")
        print("   python continue_training.py")
    else:
        print("❌ Extraction échouée")
