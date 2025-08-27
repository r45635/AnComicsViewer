# AnComicsViewer MINI

Viewer PDF minimal et robuste pour bandes dessinées avec overlays YOLO.
- **Aucun décalage**: l'inférence se fait sur **la même** QImage que celle affichée.
- **Overlays qui suivent**: rectangles **enfants du pixmap** → zoom/scroll/resize ne posent plus de problème.
- **HiDPI neutralisé**: `setDevicePixelRatio(1.0)` sur le pixmap.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# si PyTorch n'est pas installé, installe une version compatible Apple Silicon (MPS) :
# pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

## Utilisation

```bash
python main.py
```
1. Menu **Ouvrir PDF…** pour charger un album.
2. Menu **Charger modèle…** pour charger `multibd_enhanced_v2.pt`.
3. Défile, zoome : les boîtes restent parfaitement alignées.

## Notes
- Les classes par défaut sont `["panel", "balloon"]`. Si ton modèle expose d'autres noms, ils seront utilisés automatiquement.
- Les boîtes sont **cosmétiques** (1px) et les labels **ignorent les transformations** pour rester lisibles.

## Pourquoi ça corrige tes soucis ?
- **Une seule source de vérité** (QImage unique) ⇒ pas de remap ni d'approximation.
- **Parentage correct** des overlays ⇒ Qt applique la même transform au pixmap et aux boîtes.
- **DPI fixe** côté rendu PDF ⇒ le zoom se fait dans la vue, pas en re-rendant l'image.
