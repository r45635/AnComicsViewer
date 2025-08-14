import os
import argparse
from PySide6.QtPdf import QPdfDocument
from PySide6.QtCore import QSizeF

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--out", default="dataset/images/train")
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = QPdfDocument()
    assert d.load(a.pdf) in (0, getattr(QPdfDocument.Error,"NoError",0))
    for i in range(d.pageCount()):
        pt = d.pagePointSize(i)
        sc = a.dpi/72.0
        qsize = QSizeF(pt.width()*sc, pt.height()*sc).toSize()
        img = d.render(i, qsize)
        out = os.path.join(a.out, f"p{i+1:04d}.png")
        img.save(out)
        print("wrote", out)

if __name__ == "__main__":
    main()
