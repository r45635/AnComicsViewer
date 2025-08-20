#!/usr/bin/env python3
"""
Golden City Comic Analysis Tool
Extracts pages from the Golden City PDF for analysis and dataset creation.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtPdf import QPdfDocument
from PySide6.QtCore import QSizeF
from PySide6.QtWidgets import QApplication

def extract_golden_city_pages(pdf_path: str, output_dir: str = "golden_city_pages", dpi: int = 300):
    """Extract pages from Golden City PDF for analysis."""
    
    # Ensure we have a QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Extracting pages from: {pdf_path}")
    print(f"Output directory: {output_path.absolute()}")
    print(f"DPI: {dpi}")
    
    # Load PDF
    doc = QPdfDocument()
    if doc.load(pdf_path) != 0:
        print(f"Error: Could not load PDF file: {pdf_path}")
        return False
    
    page_count = doc.pageCount()
    print(f"Total pages: {page_count}")
    
    for page_num in range(page_count):
        print(f"Extracting page {page_num + 1}/{page_count}...", end=" ")
        
        # Get page size in points
        page_size_pt = doc.pagePointSize(page_num)
        
        # Calculate render size based on DPI
        scale = dpi / 72.0  # 72 points per inch
        render_size = QSizeF(
            page_size_pt.width() * scale,
            page_size_pt.height() * scale
        ).toSize()
        
        # Render page
        image = doc.render(page_num, render_size)
        
        # Save page
        output_file = output_path / f"page_{page_num + 1:03d}.png"
        if image.save(str(output_file)):
            print(f"✓ Saved {output_file.name} ({image.width()}x{image.height()})")
        else:
            print(f"✗ Failed to save {output_file.name}")
    
    print(f"\nExtraction complete! Pages saved to: {output_path.absolute()}")
    return True

def analyze_page_structure(pdf_path: str):
    """Analyze the structure of Golden City pages."""
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    doc = QPdfDocument()
    if doc.load(pdf_path) != 0:
        print(f"Error: Could not load PDF file: {pdf_path}")
        return
    
    page_count = doc.pageCount()
    print(f"=== Golden City - Analysis ===")
    print(f"Total pages: {page_count}")
    
    # Analyze first few pages for structure
    for page_num in range(min(5, page_count)):
        page_size = doc.pagePointSize(page_num)
        print(f"Page {page_num + 1}: {page_size.width():.1f} x {page_size.height():.1f} pts")
        
        # Convert to common units
        width_mm = page_size.width() * 25.4 / 72
        height_mm = page_size.height() * 25.4 / 72
        print(f"           {width_mm:.1f} x {height_mm:.1f} mm")
        
        aspect_ratio = page_size.width() / page_size.height()
        print(f"           Aspect ratio: {aspect_ratio:.3f}")
        print()

def main():
    parser = argparse.ArgumentParser(description="Golden City comic analysis and page extraction")
    parser.add_argument("pdf_file", help="Path to Golden City PDF file")
    parser.add_argument("--extract", action="store_true", help="Extract pages as PNG files")
    parser.add_argument("--analyze", action="store_true", help="Analyze page structure")
    parser.add_argument("--output", "-o", default="golden_city_pages", help="Output directory for extracted pages")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for page extraction")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_file):
        print(f"Error: PDF file not found: {args.pdf_file}")
        return 1
    
    if args.analyze:
        analyze_page_structure(args.pdf_file)
    
    if args.extract:
        if not extract_golden_city_pages(args.pdf_file, args.output, args.dpi):
            return 1
    
    if not args.analyze and not args.extract:
        print("Please specify --analyze and/or --extract")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
