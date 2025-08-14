import os
import argparse
from PySide6.QtPdf import QPdfDocument
from PySide6.QtCore import QSizeF
from PySide6.QtWidgets import QApplication

def main():
    # Ensure QApplication exists
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--out", default="dataset/images/train")
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = QPdfDocument()
    
    print(f"Attempting to load PDF: {a.pdf}")
    print(f"File exists: {os.path.exists(a.pdf)}")
    print(f"File size: {os.path.getsize(a.pdf) if os.path.exists(a.pdf) else 'N/A'} bytes")
    
    load_result = d.load(a.pdf)
    print(f"Load result: {load_result}")
    
    # Check if loading was successful - QPdfDocument.Error.None_ means no error
    success = (load_result == QPdfDocument.Error.None_)
    
    if not success:
        print(f"Error loading PDF: {a.pdf}")
        print(f"Error code: {load_result}")
        try:
            print(f"Error name: {load_result.name}")
        except:
            print("Could not get error name")
        return 1
        
    print(f"Loaded PDF: {a.pdf} ({d.pageCount()} pages)")
    
    for i in range(d.pageCount()):
        pt = d.pagePointSize(i)
        sc = a.dpi/72.0
        qsize = QSizeF(pt.width()*sc, pt.height()*sc).toSize()
        img = d.render(i, qsize)
        out = os.path.join(a.out, f"p{i+1:04d}.png")
        if img.save(out):
            print(f"wrote {out} ({img.width()}x{img.height()})")
        else:
            print(f"failed to write {out}")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
