#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audit des annotations YOLO pour détection de chevauchements panel ↔ balloon.
- Lit un YAML Ultralytics (train/val/test) et tous les .txt associés
- Calcule IoU et containment panel→balloon et balloon→panel
- Rapports: console + CSV + JSON
- Option --autofix-plan : propose des suggestions (ne modifie rien par défaut)

Usage:
  python tools/validate_annotations.py --data dataset/multibd_enhanced.yaml --out out/audit --cls-panel 0 --cls-balloon 1 --iou 0.1 --cont 0.6
"""

import os, sys, json, csv, argparse, glob
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import math

try:
    import yaml
except Exception:
    print("pip install pyyaml", file=sys.stderr); raise

@dataclass
class Box:
    cls: int
    x: float
    y: float
    w: float
    h: float
    # absolute px for convenience
    ax1: float = 0.0
    ay1: float = 0.0
    ax2: float = 0.0
    ay2: float = 0.0

def iou(a: Box, b: Box) -> float:
    x1 = max(a.ax1, b.ax1); y1 = max(a.ay1, b.ay1)
    x2 = min(a.ax2, b.ax2); y2 = min(a.ay2, b.ay2)
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0: return 0.0
    ua = (a.ax2 - a.ax1) * (a.ay2 - a.ay1)
    ub = (b.ax2 - b.ax1) * (b.ay2 - b.ay1)
    return inter / max(1e-6, ua + ub - inter)

def containment(a: Box, b: Box) -> float:
    """portion de b couverte par a (a couvre b)"""
    x1 = max(a.ax1, b.ax1); y1 = max(a.ay1, b.ay1)
    x2 = min(a.ax2, b.ax2); y2 = min(a.ay2, b.ay1 + (b.ay2 - b.ay1))
    x2 = min(a.ax2, b.ax2); y2 = min(a.ay2, b.ay2)
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0: return 0.0
    ub = (b.ax2 - b.ax1) * (b.ay2 - b.ay1)
    return inter / max(1e-6, ub)

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def yolo_txt_to_boxes(txt_path: str, img_w: int, img_h: int) -> List[Box]:
    boxes: List[Box] = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) != 5: 
                # tolérer conf (6 tokens) -> prendre 5 premiers
                if len(parts) >= 5:
                    parts = parts[:5]
                else:
                    continue
            c = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
            # YOLO normalized -> abs
            ax = x * img_w; ay = y * img_h
            aw = w * img_w; ah = h * img_h
            ax1 = ax - aw/2; ay1 = ay - ah/2
            ax2 = ax + aw/2; ay2 = ay + ah/2
            boxes.append(Box(c, x, y, w, h, ax1, ay1, ax2, ay2))
    return boxes

def guess_image_size(img_path: str) -> Tuple[int,int]:
    """Heuristique: si pas d'API image dispo, lire via PIL si présent; fallback: essayer d'inférer depuis nom.
       Mieux: si vous pouvez, installez Pillow et lisez la vraie taille."""
    try:
        from PIL import Image
        with Image.open(img_path) as im:
            return im.width, im.height
    except Exception:
        # fallback (approx): 2000x3000 portrait
        return 2000, 3000

def collect_images(split: str, base_path: str = "") -> List[str]:
    # split peut être dossier ou glob
    # Si base_path est fourni, le combiner avec split
    if base_path:
        full_path = os.path.join(base_path, split)
    else:
        full_path = split
    
    if os.path.isdir(full_path):
        imgs = sorted(glob.glob(os.path.join(full_path, "**", "*.jpg"), recursive=True) + 
                      glob.glob(os.path.join(full_path, "**", "*.png"), recursive=True) +
                      glob.glob(os.path.join(full_path, "**", "*.jpeg"), recursive=True))
        return imgs
    # sinon, c'est peut-être un fichier list ou pattern
    return sorted(glob.glob(full_path))

def run_audit(data_yaml: str, outdir: str, cls_panel: int, cls_balloon: int, iou_thr: float, cont_thr: float, autofix_plan: bool):
    os.makedirs(outdir, exist_ok=True)
    cfg = load_yaml(data_yaml)
    
    # Extraire le chemin de base du YAML
    base_path = cfg.get('path', '')
    if base_path and not os.path.isabs(base_path):
        # Si le chemin est relatif, le résoudre par rapport au répertoire PARENT du YAML
        # car le YAML est dans dataset/ mais path pointe vers dataset/
        yaml_dir = os.path.dirname(os.path.abspath(data_yaml))
        parent_dir = os.path.dirname(yaml_dir)
        base_path = os.path.join(parent_dir, base_path.lstrip('./'))
    
    splits = []
    for k in ("train", "val", "test"):
        if k in cfg and cfg[k]:
            val = cfg[k]
            if isinstance(val, list):
                for v in val:
                    splits.append((k, v))
            else:
                splits.append((k, val))

    summary = {"tot_images": 0, "tot_txt": 0,
               "images_missing_txt": 0, "txt_missing_images": 0,
               "images_with_overlap": 0, "overlap_pairs": 0,
               "per_split": {}}

    csv_path = os.path.join(outdir, "overlaps.csv")
    json_path = os.path.join(outdir, "overlaps.json")
    issues = []

    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["split","image","pair_idx","panel_box","balloon_box","IoU","containment(panel->balloon)","containment(balloon->panel)"])

        for split_name, split in splits:
            imgs = collect_images(split, base_path)
            summary["per_split"].setdefault(split_name, {"images": 0, "overlaps": 0})
            for img in imgs:
                summary["tot_images"] += 1
                summary["per_split"][split_name]["images"] += 1

                # label path
                lbl_dir = None
                # convention Ultralytics: labels/… miroir de images/…
                # essayer à côté d'abord
                base, fn = os.path.split(img)
                name, _ = os.path.splitext(fn)
                candidates = [
                    os.path.join(base.replace("images","labels"), name + ".txt"),
                    os.path.join(base, name + ".txt")
                ]
                txt = None
                for c in candidates:
                    if os.path.exists(c):
                        txt = c; break

                if txt is None:
                    summary["images_missing_txt"] += 1
                    continue
                summary["tot_txt"] += 1

                W, H = guess_image_size(img)
                boxes = yolo_txt_to_boxes(txt, W, H)
                if not boxes: 
                    continue

                panels   = [b for b in boxes if b.cls == cls_panel]
                balloons = [b for b in boxes if b.cls == cls_balloon]

                pair_idx = 0
                img_has_overlap = False
                for pa in panels:
                    for ba in balloons:
                        i = iou(pa, ba)
                        c_pb = containment(pa, ba)
                        c_bp = containment(ba, pa)
                        # deux critères: beaucoup d'IoU ou panel couvre (ou est couvert par) le ballon
                        if i >= iou_thr or c_pb >= cont_thr or c_bp >= cont_thr:
                            img_has_overlap = True
                            summary["overlap_pairs"] += 1
                            summary["per_split"][split_name]["overlaps"] += 1
                            writer.writerow([split_name, img, pair_idx,
                                             f"[{pa.ax1:.1f},{pa.ay1:.1f},{pa.ax2:.1f},{pa.ay2:.1f}]",
                                             f"[{ba.ax1:.1f},{ba.ay1:.1f},{ba.ax2:.1f},{ba.ay2:.1f}]",
                                             f"{i:.3f}", f"{c_pb:.3f}", f"{c_bp:.3f}"])
                            issues.append({
                                "split": split_name,
                                "image": img,
                                "panel": [pa.ax1, pa.ay1, pa.ax2, pa.ay2],
                                "balloon": [ba.ax1, ba.ay1, ba.ax2, ba.ay2],
                                "iou": i, "contain_pb": c_pb, "contain_bp": c_bp
                            })
                            pair_idx += 1

                if img_has_overlap:
                    summary["images_with_overlap"] += 1

    with open(json_path, "w", encoding="utf-8") as fj:
        json.dump({"summary": summary, "issues": issues}, fj, indent=2, ensure_ascii=False)

    # Console report
    print("=== AUDIT SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nCSV: {csv_path}\nJSON: {json_path}")

    # Auto-fix plan (suggestions, no write)
    if autofix_plan and issues:
        plan_path = os.path.join(outdir, "autofix_plan.md")
        with open(plan_path, "w", encoding="utf-8") as f:
            f.write("# Autofix Plan (suggestions)\n")
            f.write(f"- Règle: un balloon ne doit pas être contenu par un panel (> {cont_thr*100:.0f}%)\n")
            f.write(f"- Règle: IoU panel↔balloon ne doit pas dépasser {iou_thr}\n\n")
            by_img: Dict[str, List[dict]] = {}
            for it in issues:
                by_img.setdefault(it["image"], []).append(it)
            for img, lst in by_img.items():
                f.write(f"## {img}\n")
                for k, it in enumerate(lst, 1):
                    px1,py1,px2,py2 = it["panel"]
                    bx1,by1,bx2,by2 = it["balloon"]
                    f.write(f"- [{k}] Panel {px1:.0f},{py1:.0f},{px2:.0f},{py2:.0f} vs Balloon {bx1:.0f},{by1:.0f},{bx2:.0f},{by2:.0f} | IoU={it['iou']:.2f} c(P→B)={it['contain_pb']:.2f} c(B→P)={it['contain_bp']:.2f}\n")
                f.write("\n")
        print(f"Autofix plan: {plan_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="YAML Ultralytics (train/val/test)")
    ap.add_argument("--out", required=True, help="Dossier de sortie des rapports")
    ap.add_argument("--cls-panel", type=int, default=0)
    ap.add_argument("--cls-balloon", type=int, default=1)
    ap.add_argument("--iou", type=float, default=0.10, help="Seuil IoU max autorisé panel↔balloon")
    ap.add_argument("--cont", type=float, default=0.60, help="Seuil containment max autorisé")
    ap.add_argument("--autofix-plan", action="store_true", help="Génère un plan de correction (ne modifie rien)")
    args = ap.parse_args()

    run_audit(args.data, args.out, args.cls_panel, args.cls_balloon, args.iou, args.cont, args.autofix_plan)

if __name__ == "__main__":
    main()
