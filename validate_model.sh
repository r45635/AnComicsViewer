#!/bin/bash

echo "🔍 Validation du modèle entraîné..."
echo "=================================="

# Activer l'environnement virtuel
source .venv/bin/activate

# Trouver le dernier modèle entraîné
MODEL_DIR=$(ls -td runs/detect/ancomics_final_optimized*/ | head -1)
MODEL_PATH="${MODEL_DIR}weights/best.pt"

if [ ! -f "$MODEL_PATH" ]; then
    MODEL_PATH="${MODEL_DIR}weights/last.pt"
fi

echo "📁 Modèle trouvé: $MODEL_PATH"

# Validation avec paramètres optimisés
echo "🎯 Lancement de la validation..."
yolo val model="$MODEL_PATH" data=dataset/multibd_enhanced.yaml imgsz=1280 batch=4 conf=0.25 iou=0.6 device=mps plots=True save_txt=True save_conf=True verbose=True

echo "✅ Validation terminée!"
echo "📊 Résultats sauvegardés dans: $MODEL_DIR"
