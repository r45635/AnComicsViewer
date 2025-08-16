#!/usr/bin/env python3
"""
Test et entraînement du modèle sur le dataset multi-BD enrichi
"""

import sys
from pathlib import Path
from ultralytics import YOLO
import shutil
import subprocess

# Appliquer le patch PyTorch dès le début
exec(open('patch_pytorch.py').read())

def convert_dataset_to_yolo():
    """Convertit le dataset LabelMe en format YOLO."""
    
    print("🔄 CONVERSION LABELME → YOLO")
    print("=" * 35)
    
    try:
        # Utiliser notre script de conversion existant
        from tools.labelme_to_yolo import main as convert_main
        
        print("📁 Conversion des annotations LabelMe vers YOLO...")
        
        # Lancer la conversion
        convert_main()
        
        print("✅ Conversion terminée!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur conversion: {e}")
        return False

def analyze_yolo_dataset():
    """Analyse le dataset YOLO converti."""
    
    print("\n📊 ANALYSE DU DATASET YOLO")
    print("=" * 30)
    
    yolo_dir = Path("dataset/yolo")
    if not yolo_dir.exists():
        print("❌ Dossier YOLO non trouvé")
        return False
    
    # Compter les fichiers
    train_images = list((yolo_dir / "images" / "train").glob("*.png"))
    train_labels = list((yolo_dir / "labels" / "train").glob("*.txt"))
    val_images = list((yolo_dir / "images" / "val").glob("*.png"))
    val_labels = list((yolo_dir / "labels" / "val").glob("*.txt"))
    
    print(f"📸 Images d'entraînement: {len(train_images)}")
    print(f"🏷️  Labels d'entraînement: {len(train_labels)}")
    print(f"📸 Images de validation: {len(val_images)}")
    print(f"🏷️  Labels de validation: {len(val_labels)}")
    
    # Analyser les annotations par série
    series_stats = {"Golden City": 0, "Tintin": 0, "Pin-up du B24": 0}
    total_annotations = 0
    
    for label_file in train_labels:
        # Identifier la série
        name = label_file.stem
        if name.startswith("pinup_"):
            series_stats["Pin-up du B24"] += 1
        elif name.startswith("tintin_"):
            series_stats["Tintin"] += 1
        elif name.startswith("p") and not ("tintin" in name or "pinup" in name):
            series_stats["Golden City"] += 1
        
        # Compter les annotations dans le fichier
        with open(label_file) as f:
            annotations = len(f.readlines())
            total_annotations += annotations
    
    print(f"\n📋 Répartition par série:")
    for series, count in series_stats.items():
        print(f"   • {series}: {count} images annotées")
    
    print(f"📊 Total annotations: {total_annotations}")
    
    # Vérifier data.yaml
    data_yaml = yolo_dir / "data.yaml"
    if data_yaml.exists():
        print(f"✅ Fichier data.yaml présent")
        with open(data_yaml) as f:
            content = f.read()
            if "nc: 1" in content:
                print("🎯 Configuré pour classe unique (panel)")
            else:
                print("⚠️  Configuration multi-classes détectée")
    
    return len(train_labels) > 10  # Au moins 10 images annotées pour l'entraînement

def train_multibd_model():
    """Entraîne un nouveau modèle sur le dataset multi-BD."""
    
    print("\n🚀 ENTRAÎNEMENT MODÈLE MULTI-BD")
    print("=" * 35)
    
    data_yaml = "dataset/yolo/data.yaml"
    if not Path(data_yaml).exists():
        print(f"❌ Fichier data.yaml non trouvé: {data_yaml}")
        return False
    
    try:
        # Charger un modèle pré-entraîné
        model = YOLO('yolov8n.pt')
        
        print("📈 Paramètres d'entraînement:")
        print("   • Époques: 50")
        print("   • Batch size: 4") 
        print("   • Learning rate: 0.003")
        print("   • Augmentations légères pour diversité")
        
        # Entraîner le modèle
        results = model.train(
            data=data_yaml,
            epochs=50,
            batch=4,
            lr0=0.003,
            name="multibd_mixed_model",
            patience=15,
            save_period=10,
            # Augmentations adaptées pour BD
            hsv_h=0.01,    # Peu de variation de teinte
            hsv_s=0.3,     # Variation de saturation modérée
            hsv_v=0.2,     # Variation de luminosité modérée
            degrees=0.0,   # Pas de rotation (BD ont orientation fixe)
            translate=0.05, # Légère translation
            scale=0.2,     # Légère variation d'échelle
            fliplr=0.0,    # Pas de retournement horizontal
            flipud=0.0,    # Pas de retournement vertical
            mosaic=0.5,    # Moins de mosaïque pour préserver la structure
            mixup=0.0      # Pas de mixup pour les BD
        )
        
        print("✅ Entraînement terminé!")
        print(f"📁 Résultats: runs/detect/multibd_mixed_model/")
        return True
        
    except Exception as e:
        print(f"❌ Erreur entraînement: {e}")
        return False

def test_multibd_model():
    """Test le nouveau modèle sur des échantillons de chaque série."""
    
    print("\n🧪 TEST DU MODÈLE MULTI-BD")
    print("=" * 28)
    
    model_path = "runs/detect/multibd_mixed_model/weights/best.pt"
    if not Path(model_path).exists():
        print(f"❌ Modèle non trouvé: {model_path}")
        return
    
    model = YOLO(model_path)
    
    # Images de test pour chaque série
    test_images = [
        ("dataset/images/train/p0003.png", "Golden City"),
        ("dataset/images/train/tintin_p0001.png", "Tintin"),
        ("dataset/images/train/pinup_p0001.png", "Pin-up du B24")
    ]
    
    print("📸 Test sur échantillons de chaque série:")
    print("-" * 45)
    
    for img_path, series in test_images:
        if not Path(img_path).exists():
            print(f"⚠️  {series}: Image test non trouvée")
            continue
            
        print(f"\n🎯 {series} - {Path(img_path).name}")
        
        # Test avec différents seuils
        for conf in [0.1, 0.2, 0.3]:
            results = model.predict(img_path, conf=conf, verbose=False)
            panels = len(results[0].boxes) if results[0].boxes is not None else 0
            
            # Analyser les scores de confiance
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                confidences = results[0].boxes.conf.cpu().numpy()
                avg_conf = confidences.mean()
                max_conf = confidences.max()
                print(f"   Seuil {conf}: {panels} panels (conf moy: {avg_conf:.3f}, max: {max_conf:.3f})")
            else:
                print(f"   Seuil {conf}: {panels} panels")

def compare_with_previous_models():
    """Compare avec les anciens modèles."""
    
    print("\n📊 COMPARAISON AVEC ANCIENS MODÈLES")
    print("=" * 40)
    
    models_to_compare = [
        ("runs/detect/overfit_small/weights/best.pt", "Golden City seul"),
        ("runs/detect/mixed_golden_tintin/weights/best.pt", "Golden+Tintin 2 classes"),
        ("runs/detect/single_class_panels/weights/best.pt", "Classe unique"),
        ("runs/detect/multibd_mixed_model/weights/best.pt", "Multi-BD nouveau")
    ]
    
    test_image = "dataset/images/train/p0003.png"  # Image Golden City
    
    if not Path(test_image).exists():
        print("❌ Image test non disponible")
        return
    
    print(f"🎯 Test sur: {Path(test_image).name}")
    print("-" * 50)
    
    for model_path, model_name in models_to_compare:
        if not Path(model_path).exists():
            print(f"⚠️  {model_name}: Modèle non trouvé")
            continue
        
        try:
            model = YOLO(model_path)
            results = model.predict(test_image, conf=0.2, verbose=False)
            panels = len(results[0].boxes) if results[0].boxes is not None else 0
            
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                confidences = results[0].boxes.conf.cpu().numpy()
                avg_conf = confidences.mean()
                print(f"{model_name:<20}: {panels:2d} panels (conf: {avg_conf:.3f})")
            else:
                print(f"{model_name:<20}: {panels:2d} panels")
                
        except Exception as e:
            print(f"{model_name:<20}: Erreur - {e}")

def main():
    """Fonction principale."""
    
    print("🎯 TEST ET ENTRAÎNEMENT - DATASET MULTI-BD")
    print("=" * 50)
    print()
    
    # 1. Convertir le dataset
    if not convert_dataset_to_yolo():
        print("❌ Échec de la conversion")
        return
    
    # 2. Analyser le dataset
    if not analyze_yolo_dataset():
        print("❌ Dataset insuffisant pour l'entraînement")
        return
    
    # 3. Entraîner le modèle
    if train_multibd_model():
        # 4. Tester le nouveau modèle
        test_multibd_model()
        
        # 5. Comparer avec les anciens
        compare_with_previous_models()
        
        print(f"\n✅ ENTRAÎNEMENT ET TEST TERMINÉS!")
        print(f"🎯 Nouveau modèle: runs/detect/multibd_mixed_model/weights/best.pt")
        print(f"📊 Testé sur 3 styles de BD différents")
    else:
        print("❌ Échec de l'entraînement")

if __name__ == "__main__":
    main()
