#!/bin/bash
# Script de release automatique pour développeurs

set -e

echo "🚀 SCRIPT DE RELEASE ANCOMICSVIEWER"
echo "=================================="

# Vérifier qu'on est sur une branche clean
if [[ -n $(git status --porcelain) ]]; then
    echo "❌ Repository non propre. Committez d'abord vos changements."
    exit 1
fi

# Demander la version
echo "📝 Quelle version voulez-vous releaser ?"
echo "Format: v2.1.0, v2.0.1, etc."
read -p "Version: " VERSION

if [[ ! $VERSION =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "❌ Format de version invalide. Utilisez vX.Y.Z"
    exit 1
fi

# Vérifier que le tag n'existe pas
if git tag | grep -q "^${VERSION}$"; then
    echo "❌ Le tag ${VERSION} existe déjà"
    exit 1
fi

echo "✅ Création du tag ${VERSION}..."

# Créer le tag annoté
git tag -a "${VERSION}" -m "Release ${VERSION}

🎯 AnComicsViewer ${VERSION} - Multi-BD Revolution

Features:
- AI-powered Multi-BD detection (91.1% mAP50)
- Standalone executables for Windows/macOS/Linux
- Advanced reading order improvements (AR-A to AR-E)
- Enhanced title zone detection
- Real-time parameter tuning

Built automatically with GitHub Actions.
"

echo "✅ Tag créé. Push en cours..."

# Pousser le tag
git push origin "${VERSION}"

echo "🎉 Release ${VERSION} déclenchée !"
echo "📦 Les exécutables seront disponibles dans quelques minutes sur:"
echo "   https://github.com/r45635/AnComicsViewer/releases/tag/${VERSION}"
echo ""
echo "🔍 Suivez le build sur:"
echo "   https://github.com/r45635/AnComicsViewer/actions"
