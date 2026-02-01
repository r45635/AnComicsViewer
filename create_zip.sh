#!/bin/bash

# Script pour créer une archive zip du projet AnComicsViewer
# Usage: ./create_zip.sh
#
# Ce script génère un fichier AnComicsViewer.zip contenant tout le code source du projet
# et les fichiers de debug, sans les fichiers volumineux inutiles.
#
# Inclus dans le zip:
# - ancomicsviewer/ (code source)
# - tests/ (scripts de test)
# - debug_output/ (logs d'analyse)
# - Documentation (.md files)
# - Configuration (requirements.txt, etc)
#
# Exclu du zip (pour réduire la taille):
# - samples_PDF/ (fichiers PDF de test - 156M)
# - .venv/ (environnement virtuel Python - 1.4G)
# - .git/ (historique git - 1.1G)
# - .github/ (workflows CI/CD)
# - .claude/ (fichiers temporaires)
# - __pycache__/ et .pytest_cache/ (cache Python)
# - Anciens fichiers .zip

cd "$(dirname "$0")" || exit 1

echo "Création du fichier AnComicsViewer.zip..."
rm -f AnComicsViewer.zip
zip -r AnComicsViewer.zip . \
  -x "samples_PDF/*" ".venv/*" ".git/*" ".github/*" ".claude/*" \
  "__pycache__/*" ".pytest_cache/*" "*.pyc" "venv/*" ".env*" "*.zip" \
  -q

echo "✓ Zip créé avec succès!"
ls -lh AnComicsViewer.zip
