#!/usr/bin/env python3
"""
Script d'entraînement Multi-BD Enhanced v2 - Version Stable
Hyperparamètres optimisés pour éviter l'explosion des gradients
"""

import os
import sys
from pathlib import Path
import torch
from ultralytics import YOLO

def train_model_stable():
    """Lance l'entraînement avec des hyperparamètres stables."""
    print("🚀 Entraînement Multi-BD Enhanced v2 - Version Stable")
    print("=" * 60)
    
    # Configuration
    config_file = Path("dataset/multibd_enhanced.yaml")
    if not config_file.exists():
        print(f"❌ Configuration manquante: {config_file}")
        return False
    
    # Déterminer le device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"   🎮 GPU CUDA détecté")
    elif torch.backends.mps.is_available():
        device = "mps"
        print(f"   🍎 Apple Silicon MPS détecté")
    else:
        device = "cpu"
        print(f"   💻 CPU seulement")
    
    # Créer le modèle avec architecture vide
    print("\n📦 Création du modèle YOLOv8n...")
    try:
        model = YOLO("yolov8n.yaml")  # Modèle vide pour éviter les erreurs de sérialisation
        print("✅ Modèle YOLOv8n créé")
    except Exception as e:
        print(f"❌ Erreur création modèle: {e}")
        return False
    
    # Configuration d'entraînement optimisée
    train_args = {
        "data": str(config_file),
        "epochs": 200,
        "imgsz": 640,
        "batch": 8,  # Batch size réduit pour stabilité
        "name": "multibd_enhanced_v2_stable",
        "device": device,
        "patience": 50,  # Plus de patience
        "save": True,
        "save_period": 20,
        "cache": True,
        "augment": True,
        
        # Hyperparamètres de loss plus conservateurs
        "lr0": 0.001,  # Learning rate réduit
        "lrf": 0.001,  # Final learning rate
        "momentum": 0.9,  # Momentum standard
        "weight_decay": 0.0001,  # Weight decay réduit
        "warmup_epochs": 5.0,  # Plus de warmup
        "warmup_momentum": 0.5,
        "warmup_bias_lr": 0.05,
        
        # Loss weights ajustés
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        
        # Augmentation plus modérée
        "degrees": 3.0,  # Rotation réduite
        "translate": 0.05,  # Translation réduite
        "scale": 0.1,  # Scale réduit
        "shear": 1.0,  # Shear réduit
        "perspective": 0.0,  # Pas de perspective
        "flipud": 0.0,  # Pas de flip vertical
        "fliplr": 0.3,  # Flip horizontal modéré
        "mosaic": 0.5,  # Mosaic réduit
        "mixup": 0.0,  # Pas de mixup
        "copy_paste": 0.0,  # Pas de copy-paste
        
        # Couleurs conservatrices
        "hsv_h": 0.01,
        "hsv_s": 0.3,
        "hsv_v": 0.2,
        
        "crop_fraction": 1.0,
        "close_mosaic": 20,  # Fermer mosaic plus tôt
    }
    
    print(f"\n📋 Configuration d'entraînement stable:")
    for key, value in train_args.items():
        print(f"   • {key}: {value}")
    
    # Lancer l'entraînement
    print(f"\n🎯 Démarrage de l'entraînement stable...")
    try:
        results = model.train(**train_args)
        print(f"\n🎉 Entraînement terminé avec succès!")
        
        # Vérifier le modèle final
        best_model_path = Path("runs/detect/multibd_enhanced_v2_stable/weights/best.pt")
        if best_model_path.exists():
            print(f"✅ Modèle sauvé: {best_model_path}")
            
            # Valider sur le dataset
            try:
                print(f"\n🧪 Validation du modèle final...")
                val_results = model.val()
                print(f"✅ Validation terminée")
            except Exception as e:
                print(f"⚠️  Erreur validation: {e}")
        else:
            print(f"⚠️  Modèle final non trouvé: {best_model_path}")
            
        return True
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Entraînement interrompu par l'utilisateur")
        return False
    except Exception as e:
        print(f"\n❌ Erreur pendant l'entraînement: {e}")
        return False

def copy_best_model():
    """Copie le meilleur modèle vers le répertoire des détecteurs."""
    print("\n🔄 Copie du meilleur modèle...")
    
    # Chercher le meilleur modèle dans les différentes runs
    possible_paths = [
        "runs/detect/multibd_enhanced_v2_stable/weights/best.pt",
        "runs/detect/multibd_enhanced_v2/weights/best.pt"
    ]
    
    best_model = None
    for path_str in possible_paths:
        path = Path(path_str)
        if path.exists():
            best_model = path
            break
    
    if not best_model:
        print("❌ Aucun modèle trouvé")
        return False
    
    # Créer le répertoire de destination
    dest_dir = Path("detectors/models")
    dest_dir.mkdir(exist_ok=True)
    
    # Copier le modèle
    dest_path = dest_dir / "multibd_enhanced_v2.pt"
    
    try:
        import shutil
        shutil.copy2(best_model, dest_path)
        print(f"✅ Modèle copié: {best_model} -> {dest_path}")
        
        # Mettre à jour le détecteur pour utiliser la v2
        print(f"\n🔧 Pour utiliser le nouveau modèle:")
        print(f"   1. Le modèle est maintenant disponible: {dest_path}")
        print(f"   2. Redémarrer AnComicsViewer")
        print(f"   3. Le nouveau modèle sera utilisé automatiquement")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur copie: {e}")
        return False

def main():
    """Point d'entrée principal."""
    print("🎯 Multi-BD Enhanced v2 Training - Stable Version")
    print("=" * 60)
    
    # Changer vers le bon répertoire
    os.chdir("/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer")
    
    success = train_model_stable()
    
    if success:
        print(f"\n🎊 Entraînement Multi-BD Enhanced v2 stable réussi!")
        
        # Copier le meilleur modèle
        copy_best_model()
        
        print(f"\n📊 Résultats disponibles dans:")
        print(f"   runs/detect/multibd_enhanced_v2_stable/")
        
    else:
        print(f"\n❌ Échec de l'entraînement stable")
        print(f"\n🔄 Tentative de récupération du modèle précédent...")
        copy_best_model()
        
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
