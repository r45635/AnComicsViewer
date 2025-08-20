#!/usr/bin/env python3
"""Test de debug pour vérifier l'inclusion des modèles."""

import os
from pathlib import Path

def should_include_file(file_path, base_dir):
    """Version debug de la fonction should_include_file."""
    relative_path = file_path.relative_to(base_dir)
    path_str = str(relative_path).lower()
    
    print(f"🔍 Test: {relative_path}")
    
    # Extensions autorisées
    allowed_extensions = {'.py', '.ini', '.yaml', '.yml', '.toml', '.txt', '.ico', '.png', '.jpg', '.svg', '.pt', '.pth'}
    
    # Vérifier l'extension
    if file_path.suffix.lower() not in allowed_extensions:
        print(f"   ❌ Extension non autorisée: {file_path.suffix}")
        return False
    
    # Fichiers spécifiques à inclure (configuration et setup)
    core_files = {'requirements.txt', 'requirements-ml.txt', 'setup.py', 'pyproject.toml', 'MANIFEST.in', 'main.py', 'ARCHIVE_README.md'}
    if file_path.name in core_files:
        print(f"   ✅ Fichier core: {file_path.name}")
        return True
    
    # Pour les modèles dans runs/ : ne garder que les meilleurs modèles
    if relative_path.parts[0] == 'runs':
        print(f"   🔍 Dans runs/: {relative_path}")
        # Ne garder que les modèles finaux optimisés
        if file_path.suffix.lower() in {'.pt', '.pth', '.onnx', '.weights'}:
            # Ne garder que best.pt des modèles multibd_enhanced_v2
            if (file_path.name == 'best.pt' and 
                ('multibd_enhanced_v2' in str(relative_path) or 'multibd_enhanced_v2_stable' in str(relative_path))):
                print(f"   ✅ Modèle multibd: {file_path.name}")
                return True
            else:
                print(f"   ❌ Modèle non-multibd ou pas best.pt: {file_path.name}")
                return False
    
    print(f"   🤔 Logique générale...")
    return True

# Test avec le modèle spécifique
base_dir = Path("/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer")
model_path = base_dir / "runs" / "detect" / "multibd_enhanced_v2_stable" / "weights" / "best.pt"

print("🧪 Test du modèle multibd_enhanced_v2_stable:")
result = should_include_file(model_path, base_dir)
print(f"Résultat: {'✅ INCLUS' if result else '❌ EXCLU'}")
