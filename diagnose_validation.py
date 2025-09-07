#!/usr/bin/env python3
"""
Script de diagnostic avancé pour identifier les problèmes de validation YOLO
"""

import os
import glob
import numpy as np
from pathlib import Path

def diagnose_validation_issues():
    """Diagnostiquer les problèmes de validation YOLO"""

    print("🔍 DIAGNOSTIC AVANCÉ DES PROBLÈMES DE VALIDATION")
    print("=" * 60)

    # 1. Vérifier la structure des dossiers
    print("\n📁 STRUCTURE DES DOSSIERS:")
    for split in ['train', 'val']:
        img_dir = f"dataset/images/{split}"
        lbl_dir = f"dataset/labels/{split}"

        if os.path.exists(img_dir):
            img_count = len(glob.glob(f"{img_dir}/*.png"))
            print(f"   {split}/images: {img_count} fichiers")
        else:
            print(f"   ❌ {split}/images: dossier manquant")

        if os.path.exists(lbl_dir):
            lbl_count = len(glob.glob(f"{lbl_dir}/*.txt"))
            print(f"   {split}/labels: {lbl_count} fichiers")
        else:
            print(f"   ❌ {split}/labels: dossier manquant")

    # 2. Analyser les annotations en détail
    print("\n📊 ANALYSE DES ANNOTATIONS:")
    issues = []

    for split in ['train', 'val']:
        lbl_dir = f"dataset/labels/{split}"
        if not os.path.exists(lbl_dir):
            continue

        print(f"\n{split.upper()}:")
        files_analyzed = 0
        total_annotations = 0

        for lbl_file in glob.glob(f"{lbl_dir}/*.txt"):
            files_analyzed += 1
            basename = Path(lbl_file).stem

            try:
                with open(lbl_file, 'r') as f:
                    lines = f.readlines()

                if not lines:
                    issues.append(f"❌ {lbl_file}: fichier vide")
                    continue

                for i, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    total_annotations += 1

                    if len(parts) != 5:
                        issues.append(f"❌ {lbl_file}:{i+1}: {len(parts)} champs au lieu de 5")
                        continue

                    try:
                        cls_id = int(float(parts[0]))
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        # Vérifications
                        if cls_id not in [0, 1]:
                            issues.append(f"❌ {lbl_file}:{i+1}: classe {cls_id} invalide (doit être 0 ou 1)")

                        if not (0 <= x_center <= 1):
                            issues.append(f"⚠️  {lbl_file}:{i+1}: x_center={x_center} hors limites [0,1]")

                        if not (0 <= y_center <= 1):
                            issues.append(f"⚠️  {lbl_file}:{i+1}: y_center={y_center} hors limites [0,1]")

                        if not (0 < width <= 1):
                            issues.append(f"⚠️  {lbl_file}:{i+1}: width={width} invalide")

                        if not (0 < height <= 1):
                            issues.append(f"⚠️  {lbl_file}:{i+1}: height={height} invalide")

                        # Vérifier que la bbox ne dépasse pas les limites
                        x1 = x_center - width/2
                        y1 = y_center - height/2
                        x2 = x_center + width/2
                        y2 = y_center + height/2

                        if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
                            issues.append(f"⚠️  {lbl_file}:{i+1}: bbox dépasse les limites ({x1:.3f},{y1:.3f})->({x2:.3f},{y2:.3f})")

                    except ValueError as e:
                        issues.append(f"❌ {lbl_file}:{i+1}: erreur de parsing: {e}")

            except Exception as e:
                issues.append(f"❌ {lbl_file}: erreur de lecture: {e}")

        print(f"   Fichiers analysés: {files_analyzed}")
        print(f"   Annotations totales: {total_annotations}")

    # 3. Vérifier la cohérence image/annotation
    print("\n🔗 COHÉRENCE IMAGE/ANNOTATION:")
    for split in ['train', 'val']:
        img_dir = f"dataset/images/{split}"
        lbl_dir = f"dataset/labels/{split}"

        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            continue

        img_files = set()
        for img_path in glob.glob(f"{img_dir}/*.png"):
            img_files.add(Path(img_path).stem)

        lbl_files = set()
        for lbl_path in glob.glob(f"{lbl_dir}/*.txt"):
            lbl_files.add(Path(lbl_path).stem)

        orphan_images = img_files - lbl_files
        orphan_labels = lbl_files - img_files

        if orphan_images:
            print(f"   ❌ {split}: {len(orphan_images)} images sans annotation")
            for img in sorted(list(orphan_images))[:3]:
                issues.append(f"❌ {split}: image {img}.png sans annotation")

        if orphan_labels:
            print(f"   ❌ {split}: {len(orphan_labels)} annotations sans image")
            for lbl in sorted(list(orphan_labels))[:3]:
                issues.append(f"❌ {split}: annotation {lbl}.txt sans image")

    # 4. Résumé des problèmes
    print(f"\n📋 RÉSUMÉ:")
    print(f"   Problèmes détectés: {len(issues)}")

    if issues:
        print("\n🚨 PROBLÈMES DÉTECTÉS:")
        for issue in issues[:10]:  # Afficher les 10 premiers
            print(f"   {issue}")
        if len(issues) > 10:
            print(f"   ... et {len(issues) - 10} autres problèmes")
    else:
        print("✅ Aucune anomalie détectée dans les annotations")

    # 5. Recommandations
    print("\n💡 RECOMMANDATIONS:")
    if issues:
        print("   🔧 Les problèmes détectés peuvent causer l'erreur de validation")
        print("   📝 Vérifiez et corrigez les annotations problématiques")
    else:
        print("   ✅ Les annotations semblent correctes")
        print("   🔍 Le problème pourrait être dans la configuration YOLO ou les données d'entraînement")

    return len(issues) == 0

if __name__ == "__main__":
    diagnose_validation_issues()
