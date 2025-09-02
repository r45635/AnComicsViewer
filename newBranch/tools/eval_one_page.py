import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path to allow importing main module
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from main import PdfYoloViewer

def eval_page(pdf_path: str, page_num: int):
    """
    Loads a PDF, runs detection on a single page, and prints the results as JSON.
    """
    if not Path(pdf_path).exists():
        print(f"Error: PDF file not found at '{pdf_path}'", file=sys.stderr)
        sys.exit(1)

    # Minimal app instance to access detection logic
    # This is a bit of a hack, but avoids duplicating the detection code.
    # In a real app, the detection logic would be in a separate, easily importable module.
    viewer = PdfYoloViewer()
    viewer.load_pdf(pdf_path)
    
    if not viewer.doc:
        print(f"Error: Could not load PDF '{pdf_path}'", file=sys.stderr)
        sys.exit(1)

    if not 0 <= page_num < viewer.doc.page_count:
        print(
            f"Error: Invalid page number {page_num}. "
            f"PDF has {viewer.doc.page_count} pages (0-indexed).",
            file=sys.stderr
        )
        sys.exit(1)
        
    # Manually set the page and trigger detection
    viewer.current_page = page_num
    viewer._run_detection()

    panels = viewer.panels
    balloons = viewer.balloons

    results = {
        "pdf_path": pdf_path,
        "page_index": page_num,
        "panel_count": len(panels),
        "balloon_count": len(balloons),
        "panels": [p.tolist() for p in panels],
        "balloons": [b.tolist() for b in balloons],
    }

    print(json.dumps(results, indent=2))

    if not panels:
        print(f"\nWarning: Zero panels detected on page {page_num}.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run panel and balloon detection on a single PDF page and output JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file.")
    parser.add_argument(
        "page_index",
        type=int,
        help="The 0-based index of the page to process.",
    )
    args = parser.parse_args()

    eval_page(args.pdf_path, args.page_index)
