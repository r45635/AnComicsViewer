#!/usr/bin/env python3
"""
Auto-Convert and Train Pipeline
Pipeline automatique : annotations → YOLO → entraînement quand assez de données.
"""

import time
import subprocess
from pathlib import Path
import json

def count_annotations():
    """Compte le nombre total d'annotations."""
    labels_dir = Path("dataset/labels/train")
    return len(list(labels_dir.glob("*.json")))

def convert_to_yolo():
    """Convertit les annotations en format YOLO."""
    try:
        print("🔄 Conversion LabelMe → YOLO...")
        result = subprocess.run(['python3', 'tools/labelme_to_yolo.py'], 
                              capture_output=True, text=True, check=True)
        print("✅ Conversion réussie")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur conversion: {e}")
        return False

def should_retrain(annotation_count):
    """Détermine s'il faut relancer l'entraînement."""
    total_images = len(list(Path("dataset/images/train").glob("*.png")))
    progress = annotation_count / total_images
    
    # Seuils pour déclencher l'entraînement
    return progress >= 0.35  # 35% d'annotations

def train_model():
    """Lance l'entraînement du modèle."""
    try:
        print("🏋️ Lancement de l'entraînement...")
        result = subprocess.run(['python3', 'continue_training.py'], 
                              input='2\n',  # Option 2: réentraîner avec nouvelles données
                              text=True, capture_output=True)
        if result.returncode == 0:
            print("✅ Entraînement terminé avec succès")
            return True
        else:
            print(f"❌ Erreur entraînement: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def main():
    """Pipeline principal."""
    print("🤖 PIPELINE AUTO - ANNOTATION → YOLO → TRAINING")
    print("=" * 55)
    print()
    
    last_annotation_count = count_annotations()
    last_training_count = 0
    conversion_done = False
    
    print(f"📊 Annotations actuelles: {last_annotation_count}")
    print("🔍 Surveillance des changements...")
    print("💡 Ctrl+C pour arrêter")
    print()
    
    try:
        while True:
            current_count = count_annotations()
            
            # Nouvelles annotations détectées
            if current_count > last_annotation_count:
                print(f"📈 Nouvelles annotations: {current_count} (+{current_count - last_annotation_count})")
                last_annotation_count = current_count
                conversion_done = False
                
                # Conversion automatique toutes les 5 nouvelles annotations
                if current_count % 5 == 0 and not conversion_done:
                    if convert_to_yolo():
                        conversion_done = True
                
                # Vérifier s'il faut réentraîner
                if should_retrain(current_count) and current_count >= last_training_count + 10:
                    print("🚀 Seuil d'entraînement atteint!")
                    
                    # Conversion finale avant entraînement
                    if not conversion_done:
                        convert_to_yolo()
                    
                    # Demander confirmation
                    response = input("Lancer l'entraînement maintenant ? (y/N): ")
                    if response.lower() == 'y':
                        if train_model():
                            last_training_count = current_count
                        else:
                            print("⚠️ Entraînement échoué, continuez l'annotation")
                    else:
                        print("⏸️ Entraînement reporté")
            
            time.sleep(10)  # Vérifier toutes les 10 secondes
            
    except KeyboardInterrupt:
        print("\n👋 Pipeline arrêté")
        
        # Conversion finale
        if not conversion_done and last_annotation_count > 0:
            print("🔄 Conversion finale...")
            convert_to_yolo()

if __name__ == "__main__":
    main()
