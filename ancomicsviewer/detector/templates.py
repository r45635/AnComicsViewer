"""Template-based layout matching for panel detection.

Pre-defined common comic page layouts (2-strip, 3-strip, 2x2, 2x3,
L-shape, T-shape, staggered) are matched against detected content
distribution.

This acts as a strong structural prior - even when individual panel
detection is noisy, the most likely layout template can refine the result.

Pipeline:
1. Analyze page content distribution (where is the ink?)
2. Generate layout templates for common configurations
3. Score each template against content distribution
4. If best template scores well, use it to refine/replace detection
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from PySide6.QtCore import QRectF, QSizeF

from .utils import pdebug, HAS_CV2, HAS_NUMPY

if HAS_CV2:
    import cv2
if HAS_NUMPY:
    import numpy as np
    from numpy.typing import NDArray


@dataclass
class LayoutTemplate:
    """A predefined page layout template."""
    name: str
    panels: List[Tuple[float, float, float, float]]  # (x, y, w, h) normalized 0-1
    description: str


# Common comic page layouts (normalized coordinates 0-1)
LAYOUT_TEMPLATES: List[LayoutTemplate] = [
    # === 1 Panel ===
    LayoutTemplate(
        name="splash",
        panels=[(0.02, 0.02, 0.96, 0.96)],
        description="Full-page splash panel",
    ),

    # === 2 Panels ===
    LayoutTemplate(
        name="2_horizontal",
        panels=[(0.02, 0.02, 0.96, 0.47), (0.02, 0.51, 0.96, 0.47)],
        description="Two horizontal strips",
    ),
    LayoutTemplate(
        name="2_vertical",
        panels=[(0.02, 0.02, 0.47, 0.96), (0.51, 0.02, 0.47, 0.96)],
        description="Two vertical panels",
    ),

    # === 3 Panels ===
    LayoutTemplate(
        name="3_horizontal",
        panels=[
            (0.02, 0.02, 0.96, 0.30),
            (0.02, 0.35, 0.96, 0.30),
            (0.02, 0.68, 0.96, 0.30),
        ],
        description="Three horizontal strips",
    ),
    LayoutTemplate(
        name="3_vertical",
        panels=[
            (0.02, 0.02, 0.30, 0.96),
            (0.34, 0.02, 0.30, 0.96),
            (0.67, 0.02, 0.30, 0.96),
        ],
        description="Three vertical panels",
    ),
    LayoutTemplate(
        name="3_top1_bottom2",
        panels=[
            (0.02, 0.02, 0.96, 0.47),
            (0.02, 0.51, 0.47, 0.47),
            (0.51, 0.51, 0.47, 0.47),
        ],
        description="One wide top, two bottom",
    ),
    LayoutTemplate(
        name="3_top2_bottom1",
        panels=[
            (0.02, 0.02, 0.47, 0.47),
            (0.51, 0.02, 0.47, 0.47),
            (0.02, 0.51, 0.96, 0.47),
        ],
        description="Two top, one wide bottom",
    ),

    # === 4 Panels ===
    LayoutTemplate(
        name="4_grid",
        panels=[
            (0.02, 0.02, 0.47, 0.47),
            (0.51, 0.02, 0.47, 0.47),
            (0.02, 0.51, 0.47, 0.47),
            (0.51, 0.51, 0.47, 0.47),
        ],
        description="2x2 grid",
    ),
    LayoutTemplate(
        name="4_horizontal",
        panels=[
            (0.02, 0.02, 0.96, 0.22),
            (0.02, 0.27, 0.96, 0.22),
            (0.02, 0.52, 0.96, 0.22),
            (0.02, 0.77, 0.96, 0.21),
        ],
        description="Four horizontal strips",
    ),
    LayoutTemplate(
        name="4_L_shape",
        panels=[
            (0.02, 0.02, 0.60, 0.47),
            (0.64, 0.02, 0.34, 0.47),
            (0.02, 0.51, 0.34, 0.47),
            (0.38, 0.51, 0.60, 0.47),
        ],
        description="L-shape layout",
    ),

    # === 5 Panels ===
    LayoutTemplate(
        name="5_top2_mid1_bottom2",
        panels=[
            (0.02, 0.02, 0.47, 0.30),
            (0.51, 0.02, 0.47, 0.30),
            (0.02, 0.35, 0.96, 0.28),
            (0.02, 0.66, 0.47, 0.32),
            (0.51, 0.66, 0.47, 0.32),
        ],
        description="2-1-2 layout",
    ),
    LayoutTemplate(
        name="5_top1_mid2_bottom2",
        panels=[
            (0.02, 0.02, 0.96, 0.28),
            (0.02, 0.33, 0.47, 0.30),
            (0.51, 0.33, 0.47, 0.30),
            (0.02, 0.66, 0.47, 0.32),
            (0.51, 0.66, 0.47, 0.32),
        ],
        description="1-2-2 layout",
    ),

    # === 6 Panels ===
    LayoutTemplate(
        name="6_grid_2x3",
        panels=[
            (0.02, 0.02, 0.47, 0.30),
            (0.51, 0.02, 0.47, 0.30),
            (0.02, 0.35, 0.47, 0.28),
            (0.51, 0.35, 0.47, 0.28),
            (0.02, 0.66, 0.47, 0.32),
            (0.51, 0.66, 0.47, 0.32),
        ],
        description="2x3 grid",
    ),
    LayoutTemplate(
        name="6_grid_3x2",
        panels=[
            (0.02, 0.02, 0.30, 0.47),
            (0.34, 0.02, 0.30, 0.47),
            (0.67, 0.02, 0.30, 0.47),
            (0.02, 0.51, 0.30, 0.47),
            (0.34, 0.51, 0.30, 0.47),
            (0.67, 0.51, 0.30, 0.47),
        ],
        description="3x2 grid",
    ),

    # === 8 Panels === (common in franco-belge)
    LayoutTemplate(
        name="8_grid_2x4",
        panels=[
            (0.02, 0.02, 0.47, 0.22),
            (0.51, 0.02, 0.47, 0.22),
            (0.02, 0.27, 0.47, 0.22),
            (0.51, 0.27, 0.47, 0.22),
            (0.02, 0.52, 0.47, 0.22),
            (0.51, 0.52, 0.47, 0.22),
            (0.02, 0.77, 0.47, 0.21),
            (0.51, 0.77, 0.47, 0.21),
        ],
        description="2x4 grid (classic franco-belge)",
    ),

    # === 9 Panels ===
    LayoutTemplate(
        name="9_grid_3x3",
        panels=[
            (0.02, 0.02, 0.30, 0.30),
            (0.34, 0.02, 0.30, 0.30),
            (0.67, 0.02, 0.30, 0.30),
            (0.02, 0.35, 0.30, 0.28),
            (0.34, 0.35, 0.30, 0.28),
            (0.67, 0.35, 0.30, 0.28),
            (0.02, 0.66, 0.30, 0.32),
            (0.34, 0.66, 0.30, 0.32),
            (0.67, 0.66, 0.30, 0.32),
        ],
        description="3x3 grid",
    ),

    # === Asymmetric / Complex ===
    LayoutTemplate(
        name="T_shape",
        panels=[
            (0.02, 0.02, 0.96, 0.35),
            (0.02, 0.40, 0.47, 0.58),
            (0.51, 0.40, 0.47, 0.25),
            (0.51, 0.68, 0.47, 0.30),
        ],
        description="T-shape: wide top, split bottom",
    ),
    LayoutTemplate(
        name="staggered",
        panels=[
            (0.02, 0.02, 0.55, 0.32),
            (0.40, 0.20, 0.58, 0.30),
            (0.02, 0.36, 0.58, 0.30),
            (0.40, 0.52, 0.58, 0.28),
            (0.02, 0.68, 0.96, 0.30),
        ],
        description="Staggered overlapping layout",
    ),
]


def compute_content_map(
    gray: NDArray,
    grid_size: int = 20,
) -> NDArray:
    """Compute a content density map from the grayscale image.

    Divides the image into a grid and computes the "ink density"
    (ratio of dark pixels) in each cell.

    Args:
        gray: Grayscale image
        grid_size: Number of grid cells per dimension

    Returns:
        2D array of shape (grid_size, grid_size) with content density
    """
    if not HAS_NUMPY:
        return np.zeros((grid_size, grid_size))

    h, w = gray.shape[:2]
    content_map = np.zeros((grid_size, grid_size), dtype=np.float32)

    cell_h = h / grid_size
    cell_w = w / grid_size

    # Content = dark pixels (ink, drawings)
    for i in range(grid_size):
        y0 = int(i * cell_h)
        y1 = int((i + 1) * cell_h)
        for j in range(grid_size):
            x0 = int(j * cell_w)
            x1 = int((j + 1) * cell_w)

            cell = gray[y0:y1, x0:x1]
            if cell.size == 0:
                continue

            # Dark pixels = content (threshold at 200)
            dark_ratio = float(np.mean(cell < 200))
            content_map[i, j] = dark_ratio

    return content_map


def compute_edge_map(
    gray: NDArray,
    grid_size: int = 20,
) -> NDArray:
    """Compute an edge density map (border likelihood per cell).

    Args:
        gray: Grayscale image
        grid_size: Grid cells per dimension

    Returns:
        2D array with edge density per cell
    """
    if not HAS_CV2 or not HAS_NUMPY:
        return np.zeros((grid_size, grid_size))

    h, w = gray.shape[:2]
    edges = cv2.Canny(gray, 50, 150)

    edge_map = np.zeros((grid_size, grid_size), dtype=np.float32)
    cell_h = h / grid_size
    cell_w = w / grid_size

    for i in range(grid_size):
        y0 = int(i * cell_h)
        y1 = int((i + 1) * cell_h)
        for j in range(grid_size):
            x0 = int(j * cell_w)
            x1 = int((j + 1) * cell_w)

            cell = edges[y0:y1, x0:x1]
            if cell.size == 0:
                continue

            edge_map[i, j] = float(np.mean(cell > 0))

    return edge_map


def score_template(
    template: LayoutTemplate,
    content_map: NDArray,
    edge_map: NDArray,
    grid_size: int = 20,
) -> float:
    """Score how well a template matches the content distribution.

    Scoring criteria:
    1. Content inside panels: panels should have high ink density
    2. Low content between panels: gutters should be empty
    3. Edges at panel borders: borders should have high edge density
    4. Uniformity: content should be distributed across panels

    Args:
        template: Layout template to score
        content_map: Content density map
        edge_map: Edge density map
        grid_size: Grid size used for maps

    Returns:
        Score (0.0 to 1.0, higher = better match)
    """
    if not HAS_NUMPY:
        return 0.0

    n_panels = len(template.panels)
    if n_panels == 0:
        return 0.0

    # Create panel mask (which grid cells are inside panels)
    panel_mask = np.zeros((grid_size, grid_size), dtype=np.float32)
    panel_content_scores = []

    for px, py, pw, ph in template.panels:
        x0 = int(px * grid_size)
        y0 = int(py * grid_size)
        x1 = int((px + pw) * grid_size)
        y1 = int((py + ph) * grid_size)

        x0 = max(0, min(x0, grid_size - 1))
        y0 = max(0, min(y0, grid_size - 1))
        x1 = max(x0 + 1, min(x1, grid_size))
        y1 = max(y0 + 1, min(y1, grid_size))

        panel_mask[y0:y1, x0:x1] = 1.0

        # Content inside this panel
        panel_content = content_map[y0:y1, x0:x1]
        if panel_content.size > 0:
            panel_content_scores.append(float(np.mean(panel_content)))

    # Score 1: Content inside panels (should be high)
    avg_panel_content = np.mean(panel_content_scores) if panel_content_scores else 0.0
    content_inside_score = min(avg_panel_content / 0.3, 1.0)  # Full score at 30% density

    # Score 2: Low content in gutters (non-panel areas)
    gutter_mask = 1.0 - panel_mask
    gutter_content = content_map * gutter_mask
    gutter_pixels = gutter_mask.sum()
    if gutter_pixels > 0:
        avg_gutter_content = gutter_content.sum() / gutter_pixels
        gutter_emptiness = max(0, 1.0 - avg_gutter_content / 0.15)
    else:
        gutter_emptiness = 0.5

    # Score 3: Edge alignment with panel borders
    # Create border mask (edges of panels in the grid)
    border_mask = _make_border_mask(template, grid_size)
    edge_alignment = float(np.sum(edge_map * border_mask)) / max(1, float(np.sum(border_mask)))
    edge_score = min(edge_alignment / 0.15, 1.0)

    # Score 4: Content uniformity across panels
    if len(panel_content_scores) > 1:
        std_content = np.std(panel_content_scores)
        mean_content = np.mean(panel_content_scores)
        cv = std_content / mean_content if mean_content > 0 else 1.0
        uniformity_score = max(0, 1.0 - cv)
    else:
        uniformity_score = 0.5

    # Weighted combination
    score = (
        content_inside_score * 0.30 +
        gutter_emptiness * 0.30 +
        edge_score * 0.25 +
        uniformity_score * 0.15
    )

    return float(score)


def _make_border_mask(
    template: LayoutTemplate,
    grid_size: int,
) -> NDArray:
    """Create a mask of panel borders in the grid."""
    if not HAS_NUMPY:
        return np.zeros((grid_size, grid_size))

    border_mask = np.zeros((grid_size, grid_size), dtype=np.float32)

    for px, py, pw, ph in template.panels:
        x0 = int(px * grid_size)
        y0 = int(py * grid_size)
        x1 = min(int((px + pw) * grid_size), grid_size - 1)
        y1 = min(int((py + ph) * grid_size), grid_size - 1)

        # Top and bottom borders
        if 0 <= y0 < grid_size:
            border_mask[y0, max(0, x0):min(x1+1, grid_size)] = 1.0
        if 0 <= y1 < grid_size:
            border_mask[y1, max(0, x0):min(x1+1, grid_size)] = 1.0

        # Left and right borders
        if 0 <= x0 < grid_size:
            border_mask[max(0, y0):min(y1+1, grid_size), x0] = 1.0
        if 0 <= x1 < grid_size:
            border_mask[max(0, y0):min(y1+1, grid_size), x1] = 1.0

    return border_mask


def match_best_template(
    gray: NDArray,
    page_point_size: QSizeF,
    detected_rects: Optional[List[QRectF]] = None,
    config: "DetectorConfig" = None,
    grid_size: int = 20,
    min_score: float = 0.45,
) -> Tuple[Optional[str], List[QRectF], float]:
    """Find the best matching layout template for a page.

    Args:
        gray: Grayscale image
        page_point_size: Page size in points
        detected_rects: Optional existing detection result for comparison
        config: Detector configuration
        grid_size: Grid resolution for matching
        min_score: Minimum score to accept a template match

    Returns:
        (template_name, panel_rects, score) or (None, [], 0) if no match
    """
    if not HAS_CV2 or not HAS_NUMPY:
        return None, [], 0.0

    h, w = gray.shape[:2]

    # Compute feature maps
    content_map = compute_content_map(gray, grid_size)
    edge_map = compute_edge_map(gray, grid_size)

    # Optionally filter templates by panel count hint
    templates_to_try = LAYOUT_TEMPLATES
    if detected_rects and len(detected_rects) > 0:
        # Prioritize templates with similar panel count (±1)
        n_detected = len(detected_rects)
        templates_to_try = sorted(
            LAYOUT_TEMPLATES,
            key=lambda t: abs(len(t.panels) - n_detected),
        )

    # Score each template
    best_name = None
    best_rects: List[QRectF] = []
    best_score = 0.0

    for template in templates_to_try:
        score = score_template(template, content_map, edge_map, grid_size)

        if score > best_score:
            best_score = score
            best_name = template.name

    pdebug(f"[Template] Best match: {best_name} (score={best_score:.3f})")

    if best_score < min_score or best_name is None:
        pdebug(f"[Template] No good match (best={best_score:.3f} < min={min_score})")
        return None, [], best_score

    # Convert best template to QRectF
    best_template = next(t for t in LAYOUT_TEMPLATES if t.name == best_name)
    pw = page_point_size.width()
    ph = page_point_size.height()

    for px, py, tw, th in best_template.panels:
        best_rects.append(QRectF(px * pw, py * ph, tw * pw, th * ph))

    pdebug(f"[Template] Using '{best_name}': {len(best_rects)} panels")
    return best_name, best_rects, best_score


def refine_with_template(
    detected_rects: List[QRectF],
    gray: NDArray,
    page_point_size: QSizeF,
    config: "DetectorConfig" = None,
    min_template_score: float = 0.50,
    detection_score_threshold: float = 0.60,
) -> List[QRectF]:
    """Refine detection results using template matching.

    If the detection result is weak and a template matches well,
    use the template. If detection is strong, keep it.

    Args:
        detected_rects: Current detection result
        gray: Grayscale image
        page_point_size: Page dimensions
        config: Detector configuration
        min_template_score: Minimum template score to consider
        detection_score_threshold: Below this, prefer template

    Returns:
        Refined list of panel rectangles
    """
    from .multiscale import score_detection_result

    # Score current detection
    det_score = score_detection_result(detected_rects, page_point_size)

    # If detection is already good, don't override
    if det_score >= detection_score_threshold:
        pdebug(f"[Template] Detection strong ({det_score:.3f}), keeping as-is")
        return detected_rects

    # Try template matching
    template_name, template_rects, template_score = match_best_template(
        gray, page_point_size,
        detected_rects=detected_rects,
        config=config,
    )

    if template_name and template_score >= min_template_score:
        template_det_score = score_detection_result(template_rects, page_point_size)

        if template_det_score > det_score:
            pdebug(f"[Template] Template '{template_name}' beats detection "
                   f"({template_det_score:.3f} > {det_score:.3f})")
            return template_rects
        else:
            pdebug(f"[Template] Detection still better than template")

    return detected_rects


# Type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..config import DetectorConfig
