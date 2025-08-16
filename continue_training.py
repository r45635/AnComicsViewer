#!/usr/bin/env python3
"""
Resume YOLO Training
Continue training from the existing model with more epochs or new data.
"""

import os
import torch
from pathlib import Path

# Apply PyTorch compatibility fix
original_torch_load = torch.load
def patched_torch_load(f, map_location=None, pickle_module=None, weights_only=None, **kwargs):
    if weights_only is None:
        weights_only = False
    return original_torch_load(f, map_location=map_location, pickle_module=pickle_module, 
                              weights_only=weights_only, **kwargs)
torch.load = patched_torch_load

from ultralytics import YOLO

def resume_training():
    """Continue training from existing model."""
    
    # Set environment for MPS
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    print("🔄 Reprise de l'Entraînement YOLO")
    print("=" * 40)
    
    # Check for existing model
    existing_model = "runs/detect/overfit_small/weights/best.pt"
    last_model = "runs/detect/overfit_small/weights/last.pt"
    
    if Path(existing_model).exists():
        model_path = existing_model
        print(f"📦 Utilisation du meilleur modèle: {model_path}")
    elif Path(last_model).exists():
        model_path = last_model
        print(f"📦 Utilisation du dernier modèle: {model_path}")
    else:
        print("❌ Aucun modèle pré-entraîné trouvé")
        return False
    
    # Check dataset
    data_yaml = "dataset/yolo/data.yaml"
    if not Path(data_yaml).exists():
        print("❌ Dataset YOLO non trouvé. Exécutez d'abord:")
        print("   python tools/labelme_to_yolo.py")
        return False
    
    try:
        # Load the trained model
        print("🔄 Chargement du modèle pré-entraîné...")
        model = YOLO(model_path)
        
        # Resume training with additional epochs
        print("🎯 Reprise de l'entraînement...")
        print()
        
        # Training configuration for continuation
        results = model.train(
            data=data_yaml,
            epochs=50,  # Additional epochs
            imgsz=1024,
            batch=4,
            workers=0,
            device='mps',
            name='continued_training',
            cache=False,
            resume=False,  # Start fresh from loaded weights
            # Adjusted parameters for fine-tuning
            lr0=0.005,  # Lower learning rate for fine-tuning
            warmup_epochs=2,
            patience=15,  # Early stopping
            # Data augmentation
            mosaic=0.3,  # Light augmentation
            mixup=0.1,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.2,
            fliplr=0.5
        )
        
        print("✅ Entraînement continué terminé!")
        print(f"📁 Résultats: runs/detect/continued_training/")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la reprise: {e}")
        return False

def train_with_more_data():
    """Train with updated dataset (if new annotations added)."""
    
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    print("📊 Entraînement avec Nouvelles Données")
    print("=" * 40)
    
    # First, regenerate YOLO dataset
    print("🔄 Régénération du dataset YOLO...")
    try:
        import subprocess
        result = subprocess.run(['python3', 'tools/labelme_to_yolo.py'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Erreur de conversion: {result.stderr}")
            return False
        print("✅ Dataset régénéré")
    except Exception as e:
        print(f"❌ Erreur de conversion: {e}")
        return False
    
    # Start fresh training with all data
    model = YOLO("yolov8n.pt")  # Start from pretrained
    
    results = model.train(
        data="dataset/yolo/data.yaml",
        epochs=100,
        imgsz=1024,
        batch=4,
        workers=0,
        device='mps',
        name='with_new_data',
        cache=False,
        # Optimized for more data
        lr0=0.01,
        warmup_epochs=3,
        patience=20,
        # Augmentation
        mosaic=0.5,
        mixup=0.2,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5
    )
    
    print("✅ Nouvel entraînement terminé!")
    return True

def main():
    """Menu principal."""
    print("🎨 Continuation de l'Entraînement - Menu")
    print("=" * 45)
    print()
    print("Options disponibles:")
    print("1. 📈 Continuer l'entraînement (50 epochs supplémentaires)")
    print("2. 📊 Réentraîner avec nouvelles annotations")
    print("3. 🧪 Tester le modèle actuel")
    print()
    
    choice = input("Choix (1-3): ").strip()
    
    if choice == "1":
        print("\n🔄 Continuation de l'entraînement...")
        resume_training()
    elif choice == "2":
        print("\n📊 Réentraînement avec nouvelles données...")
        train_with_more_data()
    elif choice == "3":
        print("\n🧪 Test du modèle...")
        os.system("python test_model.py")
    else:
        print("❌ Choix invalide")

if __name__ == "__main__":
    main()
