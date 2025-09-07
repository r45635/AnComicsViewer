#!/usr/bin/env python3
"""
Script pour analyser les annotations et identifier les classes invalides
"""

import os
import glob
from pathlib import Path

def analyze_annotations():
    """Analyse les annotations pour identifier les classes invalides"""
    dataset_path = Path("./dataset")

    # Chemins des labels
    train_labels = dataset_path / "labels" / "train"
    val_labels = dataset_path / "labels" / "val"

    print("🔍 Analyse des annotations...")
    print("=" * 50)

    invalid_files = []
    class_counts = {0: 0, 1: 0}

    for split_name, labels_path in [("Train", train_labels), ("Val", val_labels)]:
        if not labels_path.exists():
            print(f"⚠️  Dossier {labels_path} introuvable")
            continue

        print(f"\n📁 Analyse {split_name}: {labels_path}")

        txt_files = list(labels_path.glob("*.txt"))
        print(f"   Fichiers trouvés: {len(txt_files)}")

        for txt_file in txt_files:
            try:
                with open(txt_file, 'r') as f:
                    lines = f.readlines()

                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 5:
                        print(f"   ❌ {txt_file.name}:{line_num} - Format invalide: {line}")
                        invalid_files.append(str(txt_file))
                        continue

                    try:
                        class_id = int(parts[0])
                        if class_id not in [0, 1]:
                            print(f"   ❌ {txt_file.name}:{line_num} - Classe invalide {class_id}: {line}")
                            invalid_files.append(str(txt_file))
                        else:
                            class_counts[class_id] += 1

                        # Vérifier les coordonnées bbox
                        bbox = [float(x) for x in parts[1:5]]
                        if any(x < 0 or x > 1 for x in bbox):
                            print(f"   ❌ {txt_file.name}:{line_num} - Bbox hors limites: {bbox}")
                            invalid_files.append(str(txt_file))

                    except ValueError as e:
                        print(f"   ❌ {txt_file.name}:{line_num} - Erreur parsing: {line} ({e})")
                        invalid_files.append(str(txt_file))

            except Exception as e:
                print(f"   ❌ Erreur lecture {txt_file.name}: {e}")
                invalid_files.append(str(txt_file))

    print("\n📊 RÉSUMÉ:")
    print(f"   Classe 0 (panel): {class_counts[0]} annotations")
    print(f"   Classe 1 (balloon): {class_counts[1]} annotations")
    print(f"   Total annotations: {sum(class_counts.values())}")
    print(f"   Fichiers problématiques: {len(set(invalid_files))}")

    if invalid_files:
        print("\n❌ Fichiers à corriger:")
        for f in sorted(set(invalid_files)):
            print(f"   - {f}")

    return len(set(invalid_files)) == 0

if __name__ == "__main__":
    is_valid = analyze_annotations()
    if is_valid:
        print("\n✅ Toutes les annotations sont valides!")
    else:
        print("\n❌ Des problèmes ont été détectés dans les annotations.")
