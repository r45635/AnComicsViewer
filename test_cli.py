#!/usr/bin/env python3
"""
AnComicsViewer - Version CLI simple pour tester le nouveau modèle
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def test_on_comic_page(pdf_path: str | None = None, image_path: str | None = None):
    """Test du modèle sur une page de BD"""
    
    print("🚀 AnComicsViewer - Test du modèle YOLOv8 Multi-BD Enhanced v2")
    print("=" * 60)
    
    # Charger le modèle
    model_path = "detectors/models/multibd_enhanced_v2.pt"
    print(f"📦 Chargement du modèle: {model_path}")
    model = YOLO(model_path)
    
    # Déterminer l'image à tester
    if image_path:
        test_image = image_path
    elif pdf_path:
        print(f"📖 PDF spécifié: {pdf_path}")
        print("⚠️  Conversion PDF → image non implémentée dans cette version test")
        print("💡 Utilisez directement une image avec --image")
        return
    else:
        # Utiliser une image du dataset de validation
        test_image = "dataset/images/val/p0002.png"
        
    print(f"📸 Test sur: {test_image}")
    
    if not Path(test_image).exists():
        print(f"❌ Image introuvable: {test_image}")
        return
        
    # Charger et analyser l'image
    image = cv2.imread(test_image)
    if image is None:
        print(f"❌ Impossible de charger l'image: {test_image}")
        return
    height, width = image.shape[:2]
    print(f"🖼️  Dimensions: {width}x{height}")
    
    # Faire la détection
    print("🤖 Détection des panels en cours...")
    results = model(test_image, conf=0.15, iou=0.6, imgsz=1280, device='mps')
    
    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes
        
        if boxes is not None and len(boxes) > 0:
            print(f"🎯 {len(boxes)} panels détectés!")
            print("\n📋 Détails des panels:")
            
            for i, box in enumerate(boxes):
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                xyxy = box.xyxy[0].cpu().numpy()
                
                x1, y1, x2, y2 = xyxy
                w, h = x2 - x1, y2 - y1
                
                class_name = "panel" if cls == 0 else f"class_{cls}"
                print(f"  Panel {i+1}:")
                print(f"    • Classe: {class_name}")
                print(f"    • Confiance: {conf:.3f}")
                print(f"    • Position: ({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f})")
                print(f"    • Taille: {w:.0f}x{h:.0f} px")
                print()
                
            # Sauvegarder l'image avec les détections
            output_path = f"detection_result_{Path(test_image).stem}.jpg"
            annotated = result.plot()
            cv2.imwrite(output_path, annotated)
            print(f"💾 Résultat sauvegardé: {output_path}")
            
        else:
            print("⚠️  Aucun panel détecté")
    else:
        print("❌ Erreur lors de la détection")

def main():
    """Point d'entrée principal"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AnComicsViewer CLI - Test du modèle YOLOv8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python test_cli.py                                    # Test sur image par défaut
  python test_cli.py --image mon_comic.jpg              # Test sur image spécifique
  python test_cli.py --pdf mon_comic.pdf                # Test sur PDF (à venir)
        """
    )
    
    parser.add_argument("--image", "-i", 
                       help="Chemin vers une image de BD à analyser")
    parser.add_argument("--pdf", "-p",
                       help="Chemin vers un PDF de BD à analyser")
    
    args = parser.parse_args()
    
    try:
        test_on_comic_page(pdf_path=args.pdf, image_path=args.image)
        print("\n✅ Test terminé avec succès!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
