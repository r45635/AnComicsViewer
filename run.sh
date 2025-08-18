#!/bin/bash
# Script de lancement d'AnComicsViewer avec l'environnement virtuel correct

cd "$(dirname "$0")"

# Vérifier que l'environnement virtuel existe
if [ ! -d ".venv" ]; then
    echo "❌ Environnement virtuel .venv non trouvé"
    echo "💡 Exécutez d'abord : python -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-ml.txt"
    exit 1
fi

# Vérifier que matplotlib est installé
if ! .venv/bin/pip show matplotlib >/dev/null 2>&1; then
    echo "⚠️ matplotlib manquant, installation en cours..."
    .venv/bin/pip install matplotlib
fi

echo "🚀 Lancement d'AnComicsViewer avec l'environnement virtuel..."
.venv/bin/python main.py
