#!/bin/bash

echo "🧪 Test du modèle entraîné..."
echo "============================"

# Activer l'environnement virtuel
source .venv/bin/activate

# Trouver le modèle
MODEL_DIR=$(ls -td runs/detect/ancomics_final_optimized*/ | head -1)
MODEL_PATH="${MODEL_DIR}weights/best.pt"

if [ ! -f "$MODEL_PATH" ]; then
    MODEL_PATH="${MODEL_DIR}weights/last.pt"
fi

echo "📁 Modèle: $MODEL_PATH"

# Tester sur une image d'exemple
TEST_IMAGE="./dataset/images/val/$(ls ./dataset/images/val/ | head -1)"

if [ ! -f "$TEST_IMAGE" ]; then
    echo "❌ Aucune image de test trouvée"
    exit 1
fi

echo "🖼️  Image de test: $TEST_IMAGE"

# Prédiction simple
echo "🎯 Test de prédiction..."
yolo predict model="$MODEL_PATH" source="$TEST_IMAGE" imgsz=1280 conf=0.25 device=mps save=True verbose=True

echo "✅ Test terminé!"
