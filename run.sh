#!/bin/bash
# Script de lancement AnComicsViewer avec environment Qt configuré

cd "$(dirname "$0")" || exit 1

# Utiliser Python système (pas venv pour éviter les problèmes Qt)
PYTHON=$(which python3)

# Ajouter le répertoire courant au path Python
export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "🎬 Lancement AnComicsViewer..."
echo "📌 Python: $PYTHON"
echo ""

# Lancer l'app
"$PYTHON" -m ancomicsviewer
