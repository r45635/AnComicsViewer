#!/usr/bin/env python3
"""
CLI minimal pour AnComicsViewer - Test rapide des fonctionnalités principales
Usage: python3 scripts/cli_minimal.py [options]
"""

import sys
import os
import argparse
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_import():
    """Test d'import des modules principaux."""
    print("🔍 Test d'import des modules...")
    try:
        from src.ancomicsviewer.detectors.multibd_detector import MultiBDPanelDetector
        print("✅ MultiBDPanelDetector importé")
        
        from src.ancomicsviewer.utils.enhanced_cache import PanelCacheManager
        print("✅ PanelCacheManager importé")
        
        return True
    except Exception as e:
        print(f"❌ Erreur d'import: {e}")
        return False

def test_detector():
    """Test de création du détecteur."""
    print("\n🔧 Test de création du détecteur...")
    try:
        from src.ancomicsviewer.detectors.multibd_detector import MultiBDPanelDetector
        
        detector = MultiBDPanelDetector(device='cpu')
        print("✅ Détecteur créé")
        
        info = detector.get_model_info()
        print(f"✅ Modèle: {info['name']}")
        print(f"✅ Confidence: {info['confidence']}")
        
        return True
    except Exception as e:
        print(f"❌ Erreur détecteur: {e}")
        return False

def test_detection(pdf_path=None):
    """Test de détection sur un PDF."""
    if not pdf_path or not os.path.exists(pdf_path):
        print("⚠️ Pas de PDF fourni pour le test de détection")
        return True
    
    print(f"\n🎯 Test de détection sur: {pdf_path}")
    try:
        from src.ancomicsviewer.detectors.multibd_detector import MultiBDPanelDetector
        import fitz
        import numpy as np
        from PIL import Image
        import io
        
        # Ouvrir le PDF
        doc = fitz.open(pdf_path)
        page = doc[0]  # Première page
        
        # Convertir en image
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        img_data = pix.pil_tobytes("RGB")
        img_pil = Image.open(io.BytesIO(img_data))
        img_rgb = np.array(img_pil)
        
        print(f"✅ Image extraite: {img_rgb.shape}")
        
        # Test de détection
        detector = MultiBDPanelDetector(device='cpu')
        result = detector._predict_raw(img_rgb)
        
        print(f"✅ Détections trouvées: {len(result)}")
        
        doc.close()
        return True
        
    except Exception as e:
        print(f"❌ Erreur détection: {e}")
        return False

def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(description="CLI minimal AnComicsViewer")
    parser.add_argument("--pdf", help="Chemin vers un PDF pour test de détection")
    parser.add_argument("--skip-detection", action="store_true", help="Ignorer le test de détection")
    
    args = parser.parse_args()
    
    print("🚀 AnComicsViewer CLI - Test minimal")
    print("=" * 50)
    
    # Tests séquentiels
    success = True
    
    # 1. Test d'import
    if not test_import():
        success = False
    
    # 2. Test détecteur
    if success and not test_detector():
        success = False
    
    # 3. Test détection (optionnel)
    if success and not args.skip_detection:
        if not test_detection(args.pdf):
            success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Tous les tests ont réussi!")
        print("✅ AnComicsViewer est prêt à fonctionner")
    else:
        print("❌ Certains tests ont échoué")
        print("⚠️ Vérifiez l'installation et les dépendances")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
