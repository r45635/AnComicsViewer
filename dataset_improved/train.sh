#!/bin/bash
# Script d'entraînement YOLO avec dataset amélioré

echo "🚀 DÉMARRAGE DE L'ENTRAÎNEMENT YOLO"
echo "==================================="

# Vérifier que YOLO est installé
if ! command -v yolo &> /dev/null; then
    echo "❌ YOLO CLI non trouvé. Installer avec: pip install ultralytics"
    exit 1
fi

# Configuration
MODEL="yolov8m.pt"
DATA_FILE="data.yaml"
EPOCHS=100
BATCH_SIZE=16
IMAGE_SIZE=640

echo "📊 Configuration:"
echo "   • Modèle: $MODEL"
echo "   • Dataset: $DATA_FILE"
echo "   • Epochs: $EPOCHS"
echo "   • Batch size: $BATCH_SIZE"
echo "   • Image size: $IMAGE_SIZE"

# Lancer l'entraînement
echo ""
echo "🎯 Lancement de l'entraînement..."
yolo train \
    model=$MODEL \
    data=$DATA_FILE \
    epochs=$EPOCHS \
    imgsz=$IMAGE_SIZE \
    batch=$BATCH_SIZE \
    name=ancomics_improved \
    save=True \
    save_period=10 \
    cache=True \
    workers=4 \
    device=mps  # Pour Mac avec GPU

echo ""
echo "✅ Entraînement terminé!"
echo "📁 Résultats dans: runs/train/ancomics_improved/"
