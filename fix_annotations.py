#!/usr/bin/env python3
"""
Script pour détecter et corriger les annotations YOLO corrompues
qui causent des erreurs de validation
"""

import os
import glob
import numpy as np

def validate_and_fix_annotations():
    """Valide et corrige les annotations YOLO"""
    
    label_files = glob.glob("/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/dataset/labels/**/*.txt", recursive=True)
    
    print(f"🔍 Validation de {len(label_files)} fichiers d'annotations...")
    
    issues_found = 0
    files_fixed = 0
    
    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            original_lines = lines.copy()
            fixed_lines = []
            file_has_issues = False
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue  # Ignorer les lignes vides
                
                parts = line.split()
                
                # Vérifier le nombre de champs
                if len(parts) != 5:
                    print(f"❌ {label_file}:{i+1} - Nombre de champs incorrect: {len(parts)} (attendu: 5)")
                    file_has_issues = True
                    issues_found += 1
                    continue
                
                try:
                    # Parser les valeurs
                    class_id = int(float(parts[0]))  # Convertir via float d'abord pour gérer "1.0"
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Vérifier l'ID de classe
                    if class_id < 0 or class_id > 1:
                        print(f"❌ {label_file}:{i+1} - ID classe invalide: {class_id} (doit être 0 ou 1)")
                        file_has_issues = True
                        issues_found += 1
                        # Corriger: forcer à 0 ou 1
                        class_id = max(0, min(1, class_id))
                    
                    # Vérifier les coordonnées (doivent être entre 0 et 1)
                    coords_fixed = False
                    if not (0 <= x_center <= 1):
                        print(f"⚠️  {label_file}:{i+1} - x_center hors limites: {x_center}")
                        x_center = max(0, min(1, x_center))
                        coords_fixed = True
                    
                    if not (0 <= y_center <= 1):
                        print(f"⚠️  {label_file}:{i+1} - y_center hors limites: {y_center}")
                        y_center = max(0, min(1, y_center))
                        coords_fixed = True
                    
                    if not (0 < width <= 1):
                        print(f"⚠️  {label_file}:{i+1} - width invalide: {width}")
                        width = max(0.001, min(1, width))
                        coords_fixed = True
                    
                    if not (0 < height <= 1):
                        print(f"⚠️  {label_file}:{i+1} - height invalide: {height}")
                        height = max(0.001, min(1, height))
                        coords_fixed = True
                    
                    if coords_fixed:
                        file_has_issues = True
                        issues_found += 1
                    
                    # Vérifier que la bbox ne dépasse pas les limites
                    x1 = x_center - width/2
                    y1 = y_center - height/2
                    x2 = x_center + width/2
                    y2 = y_center + height/2
                    
                    if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
                        print(f"⚠️  {label_file}:{i+1} - bbox dépasse les limites: ({x1:.3f},{y1:.3f}) -> ({x2:.3f},{y2:.3f})")
                        # Ajuster les coordonnées pour rester dans les limites
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(1, x2)
                        y2 = min(1, y2)
                        
                        # Recalculer center et size
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                        
                        file_has_issues = True
                        issues_found += 1
                    
                    # Reconstituer la ligne corrigée
                    fixed_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                    fixed_lines.append(fixed_line)
                    
                except ValueError as e:
                    print(f"❌ {label_file}:{i+1} - Erreur de parsing: {e}")
                    file_has_issues = True
                    issues_found += 1
                    continue
            
            # Si le fichier a été modifié, le sauvegarder
            if file_has_issues and fixed_lines:
                with open(label_file, 'w') as f:
                    f.writelines(fixed_lines)
                files_fixed += 1
                print(f"🔧 Fichier corrigé: {label_file}")
            elif file_has_issues and not fixed_lines:
                print(f"❌ Fichier vide après correction: {label_file}")
                
        except Exception as e:
            print(f"❌ Erreur lors de la lecture de {label_file}: {e}")
            issues_found += 1
    
    print(f"\n📊 RÉSUMÉ:")
    print(f"   Fichiers vérifiés: {len(label_files)}")
    print(f"   Issues trouvées: {issues_found}")
    print(f"   Fichiers corrigés: {files_fixed}")
    
    if issues_found == 0:
        print("✅ Toutes les annotations sont valides !")
    else:
        print("🔧 Annotations corrigées. Relancez l'entraînement.")

if __name__ == "__main__":
    validate_and_fix_annotations()
