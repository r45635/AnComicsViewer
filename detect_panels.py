#!/usr/bin/env python3
"""
Panel Detection for Green-bordered Comics (Tintin-style)

Detects comic panels with green borders, filters false positives,
and exports bounding boxes in reading order.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Load image from file path.
    
    Args:
        path: Path to image file
        
    Returns:
        BGR image array
        
    Raises:
        FileNotFoundError: If image doesn't exist
        ValueError: If image can't be loaded
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    return img


def mask_green_borders(
    img_bgr: np.ndarray,
    hsv_lower: Tuple[int, int, int],
    hsv_upper: Tuple[int, int, int]
) -> np.ndarray:
    """Create binary mask for panel borders.
    
    Supports both green borders (HSV-based) and dark borders (grayscale-based).
    If hsv_lower is None, uses adaptive threshold on grayscale.
    
    Args:
        img_bgr: Input image in BGR
        hsv_lower: Lower HSV threshold (H, S, V) or None for grayscale
        hsv_upper: Upper HSV threshold (H, S, V)
        
    Returns:
        Binary mask (255 = border)
    """
    # If hsv_lower is None or (0,0,0), use grayscale adaptive threshold
    if hsv_lower is None or all(v == 0 for v in hsv_lower):
        # Convert to grayscale
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Use bilateral filter to preserve edges while smoothing text
        gray_smooth = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Adaptive threshold for dark borders
        mask = cv2.adaptiveThreshold(
            gray_smooth, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            99, 5
        )
        return mask
    else:
        # Original green border detection
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
        return mask


def postprocess_mask(
    mask: np.ndarray,
    close_kernel: int = 5,
    dilate_kernel: int = 5,
    dilate_iter: int = 1
) -> np.ndarray:
    """Post-process mask to connect broken borders.
    
    Args:
        mask: Binary mask
        close_kernel: Kernel size for morphological closing
        dilate_kernel: Kernel size for dilation
        dilate_iter: Number of dilation iterations
        
    Returns:
        Processed mask
    """
    # Close gaps in borders
    kernel_close = np.ones((close_kernel, close_kernel), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Thicken borders to better capture panels
    kernel_dilate = np.ones((dilate_kernel, dilate_kernel), np.uint8)
    mask = cv2.dilate(mask, kernel_dilate, iterations=dilate_iter)
    
    return mask


def find_panel_candidates(mask: np.ndarray) -> List[np.ndarray]:
    """Find contours in mask that could be panels.
    
    Args:
        mask: Binary mask
        
    Returns:
        List of contours (external only)
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def calculate_ink_ratio(roi_gray: np.ndarray) -> float:
    """Calculate ink density ratio for editorial box detection.
    
    Args:
        roi_gray: Grayscale ROI
        
    Returns:
        Ratio of ink pixels (0.0 to 1.0)
    """
    # Adaptive threshold to detect ink
    binary = cv2.adaptiveThreshold(
        roi_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 10
    )
    
    ink_pixels = np.count_nonzero(binary)
    total_pixels = roi_gray.shape[0] * roi_gray.shape[1]
    
    return ink_pixels / total_pixels if total_pixels > 0 else 0.0


def is_editorial_box(
    x: int, y: int, w: int, h: int,
    img_gray: np.ndarray,
    page_width: int,
    page_height: int,
    ink_threshold: float = 0.35
) -> bool:
    """Detect editorial boxes (e.g. "Nouvelles de Tintin").
    
    Heuristic: high ink density + short height + wide width
    
    Args:
        x, y, w, h: Bounding box
        img_gray: Full grayscale image
        page_width: Image width
        page_height: Image height
        ink_threshold: Minimum ink ratio to consider editorial
        
    Returns:
        True if likely an editorial box
    """
    # Extract ROI
    roi = img_gray[y:y+h, x:x+w]
    
    # Calculate ink density
    ink_ratio = calculate_ink_ratio(roi)
    
    # Editorial characteristics:
    # - High text density (ink_ratio > threshold)
    # - Short height (< 25% of page)
    # - Wide width (> 40% of page)
    is_dense = ink_ratio > ink_threshold
    is_short = h < 0.25 * page_height
    is_wide = w > 0.4 * page_width
    
    return is_dense and is_short and is_wide


def is_title_box(
    x: int, y: int, w: int, h: int,
    page_width: int,
    page_height: int
) -> bool:
    """Detect title boxes at top of page.
    
    Heuristic: in top 35% of page
    
    Args:
        x, y, w, h: Bounding box
        page_width: Image width
        page_height: Image height
        
    Returns:
        True if likely a title box
    """
    # Title characteristics: in top 35% of page (more permissive)
    # This catches title rows and page numbers
    in_top = (y + h) < 0.35 * page_height
    
    return in_top


def filter_boxes(
    boxes: List[Tuple[int, int, int, int]],
    img_gray: np.ndarray,
    img_shape: Tuple[int, int],
    min_area: int = 10000,
    aspect_min: float = 0.2,
    aspect_max: float = 5.0,
    editorial_ink_ratio: float = 0.35
) -> List[Tuple[int, int, int, int]]:
    """Filter bounding boxes by size, aspect ratio, and editorial detection.
    
    Args:
        boxes: List of (x, y, w, h) tuples
        img_gray: Grayscale image for ink analysis
        img_shape: (height, width) of image
        min_area: Minimum area in pixels
        aspect_min: Minimum width/height ratio
        aspect_max: Maximum width/height ratio
        editorial_ink_ratio: Ink threshold for editorial detection
        
    Returns:
        Filtered list of boxes
    """
    page_height, page_width = img_shape
    filtered = []
    
    for x, y, w, h in boxes:
        # Area filter
        area = w * h
        if area < min_area:
            continue
        
        # Aspect ratio filter
        aspect = w / h if h > 0 else 0
        if aspect < aspect_min or aspect > aspect_max:
            continue
        
        # Title box filter (remove boxes at top of page)
        if is_title_box(x, y, w, h, page_width, page_height):
            continue
        
        # Editorial box filter
        if is_editorial_box(x, y, w, h, img_gray, page_width, page_height, editorial_ink_ratio):
            continue
        
        filtered.append((x, y, w, h))
    
    return filtered


def sort_reading_order(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """Sort boxes in reading order (top→bottom, left→right with row clustering).
    
    Args:
        boxes: List of (x, y, w, h) tuples
        
    Returns:
        Sorted list in reading order
    """
    if not boxes:
        return []
    
    # Sort by y first
    sorted_by_y = sorted(boxes, key=lambda b: b[1])
    
    # Group into rows
    rows = []
    current_row = [sorted_by_y[0]]
    y_ref = sorted_by_y[0][1]
    h_ref = sorted_by_y[0][3]
    
    for box in sorted_by_y[1:]:
        y = box[1]
        h = box[3]
        
        # Same row if y within 0.5 * reference height
        if abs(y - y_ref) < 0.5 * h_ref:
            current_row.append(box)
        else:
            rows.append(current_row)
            current_row = [box]
            y_ref = y
            h_ref = h
    
    rows.append(current_row)
    
    # Sort each row by x and concatenate
    result = []
    for row in rows:
        row_sorted = sorted(row, key=lambda b: b[0])
        result.extend(row_sorted)
    
    return result


def draw_debug(
    img: np.ndarray,
    boxes: List[Tuple[int, int, int, int]]
) -> np.ndarray:
    """Draw bounding boxes with indices on image.
    
    Args:
        img: Input image
        boxes: List of (x, y, w, h) in reading order
        
    Returns:
        Debug image with annotations
    """
    debug = img.copy()
    
    for idx, (x, y, w, h) in enumerate(boxes, 1):
        # Draw red rectangle
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 0, 255), 3)
        
        # Draw blue index number
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = str(idx)
        text_size = cv2.getTextSize(text, font, 1.5, 3)[0]
        
        # Position text at top-left corner with background
        text_x = x + 10
        text_y = y + 40
        
        # Background rectangle for text
        cv2.rectangle(
            debug,
            (text_x - 5, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            (255, 255, 255),
            -1
        )
        
        # Draw text
        cv2.putText(debug, text, (text_x, text_y), font, 1.5, (255, 0, 0), 3)
    
    return debug


def detect_panels(
    img_path: str,
    hsv_lower: Tuple[int, int, int] = None,
    hsv_upper: Tuple[int, int, int] = None,
    close_kernel: int = 7,
    dilate_kernel: int = 7,
    dilate_iter: int = 1,
    min_area: int = 10000,
    aspect_min: float = 0.2,
    aspect_max: float = 5.0,
    editorial_ink_ratio: float = 0.35
) -> Tuple[List[Tuple[int, int, int, int]], Dict[str, Any]]:
    """Main panel detection pipeline.
    
    Args:
        img_path: Path to input image
        hsv_lower: Lower HSV threshold for green (None = use grayscale)
        hsv_upper: Upper HSV threshold for green (None = use grayscale)
        close_kernel: Kernel size for morphological closing
        dilate_kernel: Kernel size for dilation
        dilate_iter: Dilation iterations
        min_area: Minimum panel area
        aspect_min: Minimum aspect ratio
        aspect_max: Maximum aspect ratio
        editorial_ink_ratio: Ink threshold for editorial detection
        
    Returns:
        Tuple of (sorted_boxes, metadata)
    """
    print(f"Loading image: {img_path}")
    img = load_image(img_path)
    h, w = img.shape[:2]
    
    # Default to grayscale detection if no HSV specified
    if hsv_lower is None:
        hsv_lower = (0, 0, 0)
        hsv_upper = (0, 0, 0)
        print(f"Using grayscale adaptive threshold detection")
    else:
        print(f"HSV range: {hsv_lower} → {hsv_upper}")
    
    print(f"Image size: {w}x{h}")
    print(f"Kernels: close={close_kernel}, dilate={dilate_kernel}x{dilate_iter}")
    print(f"Filters: min_area={min_area}, aspect=[{aspect_min}, {aspect_max}]")
    print(f"Editorial ink ratio threshold: {editorial_ink_ratio}")
    
    # Create mask
    mask = mask_green_borders(img, hsv_lower, hsv_upper)
    
    # Post-process mask
    mask = postprocess_mask(mask, close_kernel, dilate_kernel, dilate_iter)
    
    # Find contours
    contours = find_panel_candidates(mask)
    print(f"Found {len(contours)} contours")
    
    # Extract bounding boxes
    boxes = [cv2.boundingRect(c) for c in contours]
    print(f"Extracted {len(boxes)} bounding boxes")
    
    # Filter boxes
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    boxes = filter_boxes(
        boxes, img_gray, (h, w),
        min_area, aspect_min, aspect_max, editorial_ink_ratio
    )
    print(f"After filtering: {len(boxes)} panels")
    
    # Sort in reading order
    boxes = sort_reading_order(boxes)
    print(f"Sorted in reading order")
    
    # Metadata
    metadata = {
        "image_path": str(img_path),
        "image_size": {"width": w, "height": h},
        "num_panels": len(boxes),
        "parameters": {
            "hsv_lower": hsv_lower,
            "hsv_upper": hsv_upper,
            "close_kernel": close_kernel,
            "dilate_kernel": dilate_kernel,
            "dilate_iter": dilate_iter,
            "min_area": min_area,
            "aspect_min": aspect_min,
            "aspect_max": aspect_max,
            "editorial_ink_ratio": editorial_ink_ratio
        }
    }
    
    return boxes, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Detect comic panels with green borders (Tintin-style)"
    )
    
    # Input/output
    parser.add_argument("--in", dest="input", required=True, help="Input image path")
    parser.add_argument("--out", dest="output", help="Output debug image path")
    parser.add_argument("--json", dest="json_out", help="Output JSON path")
    
    # Detection mode
    parser.add_argument("--mode", choices=["green", "gray"], default="gray",
                       help="Detection mode: 'green' for HSV green borders, 'gray' for dark borders (default: gray)")
    
    # HSV parameters (only for green mode)
    parser.add_argument("--hmin", type=int, default=35, help="HSV H min (default: 35)")
    parser.add_argument("--smin", type=int, default=40, help="HSV S min (default: 40)")
    parser.add_argument("--vmin", type=int, default=40, help="HSV V min (default: 40)")
    parser.add_argument("--hmax", type=int, default=85, help="HSV H max (default: 85)")
    parser.add_argument("--smax", type=int, default=255, help="HSV S max (default: 255)")
    parser.add_argument("--vmax", type=int, default=255, help="HSV V max (default: 255)")
    
    # Morphology parameters
    parser.add_argument("--close-kernel", type=int, default=7, help="Close kernel size (default: 7)")
    parser.add_argument("--dilate-kernel", type=int, default=7, help="Dilate kernel size (default: 7)")
    parser.add_argument("--dilate-iter", type=int, default=1, help="Dilate iterations (default: 1)")
    
    # Filter parameters
    parser.add_argument("--min-area", type=int, default=10000, help="Min panel area (default: 10000)")
    parser.add_argument("--aspect-min", type=float, default=0.2, help="Min aspect ratio (default: 0.2)")
    parser.add_argument("--aspect-max", type=float, default=5.0, help="Max aspect ratio (default: 5.0)")
    parser.add_argument("--editorial-ink-ratio", type=float, default=0.35, 
                       help="Editorial box ink threshold (default: 0.35)")
    
    args = parser.parse_args()
    
    # Set HSV parameters based on mode
    if args.mode == "green":
        hsv_lower = (args.hmin, args.smin, args.vmin)
        hsv_upper = (args.hmax, args.smax, args.vmax)
    else:
        # Grayscale mode
        hsv_lower = None
        hsv_upper = None
    
    # Detect panels
    boxes, metadata = detect_panels(
        args.input,
        hsv_lower=hsv_lower,
        hsv_upper=hsv_upper,
        close_kernel=args.close_kernel,
        dilate_kernel=args.dilate_kernel,
        dilate_iter=args.dilate_iter,
        min_area=args.min_area,
        aspect_min=args.aspect_min,
        aspect_max=args.aspect_max,
        editorial_ink_ratio=args.editorial_ink_ratio
    )
    
    # Export JSON
    if args.json_out:
        output_data = {
            **metadata,
            "panels": [
                {"index": i, "x": x, "y": y, "width": w, "height": h}
                for i, (x, y, w, h) in enumerate(boxes, 1)
            ]
        }
        
        with open(args.json_out, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Saved JSON to: {args.json_out}")
    
    # Export debug image
    if args.output:
        img = load_image(args.input)
        debug_img = draw_debug(img, boxes)
        cv2.imwrite(args.output, debug_img)
        print(f"Saved debug image to: {args.output}")
    
    # Print results
    print("\n=== RESULTS ===")
    print(f"Total panels detected: {len(boxes)}")
    for i, (x, y, w, h) in enumerate(boxes, 1):
        print(f"  Panel {i}: x={x}, y={y}, w={w}, h={h}")


def test_sample():
    """Quick test function to verify detection on a sample image."""
    import sys
    from pathlib import Path
    
    # Look for sample images
    samples_dir = Path("samples_PDF")
    test_images = list(samples_dir.glob("*.png")) + list(samples_dir.glob("*.jpg"))
    
    if not test_images:
        # Try current directory
        test_images = list(Path(".").glob("*.png")) + list(Path(".").glob("*.jpg"))
    
    if not test_images:
        print("No test images found. Please provide a path:")
        print("  python detect_panels.py --in <image.png>")
        return
    
    # Use first available image
    test_img = str(test_images[0])
    print(f"\n{'='*60}")
    print(f"RUNNING TEST ON: {test_img}")
    print(f"{'='*60}\n")
    
    try:
        boxes, metadata = detect_panels(test_img)
        
        print(f"\n{'='*60}")
        print(f"TEST RESULTS")
        print(f"{'='*60}")
        print(f"Image: {metadata['image_size']['width']}x{metadata['image_size']['height']}px")
        print(f"Panels detected: {len(boxes)}")
        print(f"\nBounding boxes (reading order):")
        for i, (x, y, w, h) in enumerate(boxes, 1):
            area = w * h
            aspect = w / h if h > 0 else 0
            print(f"  [{i}] x={x:4d} y={y:4d} w={w:4d} h={h:4d} | area={area:6d} aspect={aspect:.2f}")
        
        print(f"\n✓ Test completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    # If no arguments, run test
    if len(sys.argv) == 1:
        test_sample()
    else:
        main()


# Example usage:
#
# Basic usage:
#   python detect_panels.py --in page.png --out debug.png --json panels.json
#
# Custom HSV range for different green shades:
#   python detect_panels.py --in page.png --out debug.png --hmin 30 --hmax 90
#
# Adjust morphology for thicker/thinner borders:
#   python detect_panels.py --in page.png --out debug.png --close-kernel 7 --dilate-kernel 7
#
# Stricter filtering:
#   python detect_panels.py --in page.png --out debug.png --min-area 20000 --aspect-min 0.3
#
# Fine-tune editorial box detection:
#   python detect_panels.py --in page.png --out debug.png --editorial-ink-ratio 0.40
