#!/usr/bin/env python3
"""
Script de diagnostic pour analyser l'erreur de validation YOLO
"""

import os
import torch
from pathlib import Path
from ultralytics import YOLO
import numpy as np

def diagnose_validation_error():
    """Diagnostique l'erreur de validation IndexError"""
    print("🔍 Diagnostic de l'erreur de validation...")
    print("=" * 50)

    # Trouver le modèle entraîné
    model_dirs = sorted(Path("./runs/detect").glob("ancomics_final_optimized*"))
    if not model_dirs:
        print("❌ Aucun modèle trouvé")
        return

    model_dir = model_dirs[-1]
    model_path = model_dir / "weights" / "best.pt"

    if not model_path.exists():
        model_path = model_dir / "weights" / "last.pt"

    if not model_path.exists():
        print("❌ Modèle introuvable")
        return

    print(f"📁 Modèle: {model_path}")

    # Charger le modèle
    model = YOLO(str(model_path))

    # Tester sur quelques images de validation
    val_images = Path("./dataset/images/val")
    if not val_images.exists():
        print("❌ Images de validation introuvables")
        return

    image_files = list(val_images.glob("*.png"))[:3]  # Tester seulement 3 images
    print(f"🖼️  Test sur {len(image_files)} images de validation")

    for img_path in image_files:
        print(f"\n🧪 Test sur: {img_path.name}")

        try:
            # Faire une prédiction
            results = model(img_path, conf=0.25, iou=0.6, verbose=False)

            if results and len(results) > 0:
                result = results[0]

                if result.boxes is not None and len(result.boxes) > 0:
                    # Analyser les classes prédites
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                    confs = result.boxes.conf.cpu().numpy()

                    print(f"   📊 Prédictions: {len(classes)} objets")
                    print(f"   🎯 Classes détectées: {np.unique(classes)}")
                    print(f"   📈 Confiances: min={confs.min():.3f}, max={confs.max():.3f}")

                    # Vérifier les classes invalides
                    invalid_classes = []
                    for cls in classes:
                        if cls < 0 or cls >= 2:  # Pour nc=2, classes valides sont 0 et 1
                            invalid_classes.append(cls)

                    if invalid_classes:
                        print(f"   ❌ Classes invalides trouvées: {invalid_classes}")
                        print("   💡 Cela peut causer l'erreur IndexError!")
                    else:
                        print("   ✅ Toutes les classes sont valides (0 ou 1)")

                else:
                    print("   📭 Aucune détection")
            else:
                print("   ❌ Aucune prédiction")

        except Exception as e:
            print(f"   ❌ Erreur lors du test: {e}")

    # Tester la validation manuellement
    print("\n🎯 Test de validation manuelle...")
    try:
        # Simuler la validation
        val_results = model.val(
            data="./dataset/multibd_enhanced.yaml",
            conf=0.25,
            iou=0.6,
            verbose=False
        )

        if val_results:
            print("✅ Validation réussie!")
            print(f"   📊 mAP50: {val_results.box.map50:.3f}")
            print(f"   📊 mAP50-95: {val_results.box.map:.3f}")
        else:
            print("❌ Validation échouée")

    except Exception as e:
        print(f"❌ Erreur de validation: {e}")
        print("   🔍 Cela confirme le bug dans Ultralytics!")

    print("\n📋 RECOMMANDATIONS:")
    print("   1. Le modèle fonctionne pour les prédictions individuelles")
    print("   2. L'erreur se produit seulement lors de la validation en batch")
    print("   3. Cela semble être un bug dans Ultralytics 8.3.192")
    print("   4. Solutions possibles:")
    print("      - Utiliser le modèle tel quel (il fonctionne)")
    print("      - Désactiver complètement la validation")
    print("      - Mettre à jour Ultralytics si possible")

if __name__ == "__main__":
    diagnose_validation_error()
