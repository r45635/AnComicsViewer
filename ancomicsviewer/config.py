"""Configuration dataclasses for AnComicsViewer.

Using dataclasses provides:
- Type safety and IDE autocompletion
- Easy serialization/deserialization
- Immutable defaults with mutable instances
- Clear documentation of all parameters
"""

from dataclasses import dataclass, field
from typing import Dict, Any
import copy


@dataclass
class DetectorConfig:
    """Configuration for the panel detection algorithm.

    All fractional parameters are relative to page dimensions for DPI stability.
    """

    # Adaptive threshold parameters
    adaptive_block: int = 51      # Window size for adaptive threshold (must be odd)
    adaptive_C: int = 5           # Offset for adaptive threshold

    # Morphology parameters
    morph_kernel: int = 3         # Kernel size for morphological operations (compromise for various page layouts)
    morph_iter: int = 1           # Number of morphology iterations
    morph_scale_with_dpi: bool = True  # Scale morphology kernel with image resolution

    # Base filter parameters (as fractions of page area/dimensions)
    min_area_pct: float = 0.008   # Minimum panel area (0.8% to catch very small panels)
    max_area_pct: float = 0.95    # Maximum panel area (exclude full-bleed)
    min_fill_ratio: float = 0.25  # Minimum contour fill ratio (permissive for gutter-detected panels)
    min_rect_px: int = 45         # Minimum panel dimension in pixels
    min_rect_frac: float = 0.05   # Minimum panel dimension as fraction of page

    # Gutter detection parameters (background-similarity based)
    min_gutter_px: int = 2        # Minimum gutter thickness in pixels
    min_gutter_frac: float = 0.010  # Minimum gutter as fraction of block
    max_gutter_px_frac: float = 0.08  # Maximum gutter as fraction of block
    gutter_cov_min: float = 0.45  # Minimum brightness coverage for gutter validation
    
    # Background similarity mask (for watercolor pages like Grémillet)
    gutter_bg_delta: float = 12.0        # Lab distance threshold for background mask (10-14 recommended)
    gutter_grad_percentile: int = 25    # Percentile of gradient (20-30 recommended, was 55)
    gutter_open_kernel_frac: float = 0.25  # Opening kernel size as fraction of image dimension
    gutter_min_stripe_width: int = 25  # Max width/height for stripe filtering (pixels)
    gutter_stripe_length_frac: float = 0.20  # Min length as fraction of image dimension
    
    edge_margin_frac: float = 0.02  # Margin from edges for gutter detection

    # Brightness-based split parameters
    light_col_rel: float = 0.12   # Column brightness threshold (relative)
    light_row_rel: float = 0.12   # Row brightness threshold (relative)
    proj_smooth_k: int = 15       # Projection smoothing kernel (must be odd)

    # Title row filter parameters
    filter_title_rows: bool = False  # Disabled - causes too many false positives
    title_row_top_frac: float = 0.20      # Title must be in top X% of page
    title_row_max_h_frac: float = 0.10    # Maximum title row height
    title_row_median_w_frac_max: float = 0.25  # Maximum median width
    title_row_min_boxes: int = 4          # Minimum boxes for "many small" pattern
    title_row_big_min_boxes: int = 2      # Minimum boxes for "few big" pattern
    title_row_big_w_min_frac: float = 0.16  # Minimum width for "big" boxes
    title_row_min_meanL: float = 0.88     # Minimum mean luminosity

    # Limits
    max_panels_per_page: int = 24  # Safety limit

    # Freeform detection parameters (for complex layouts/tinted backgrounds)
    freeform_bg_delta: float = 15.0  # Lab color distance threshold for freeform watershed
    sure_fg_ratio: float = 0.35   # Ratio of max distance for sure foreground in watershed (lowered for more seeds)
    min_area_ratio_freeform: float = 0.005  # Minimum panel area ratio for freeform detection (lowered)
    min_fill_ratio_freeform: float = 0.15  # Minimum fill ratio for freeform regions (lowered)
    iou_merge_thr: float = 0.20   # IoU threshold for merging overlapping regions
    approx_eps_ratio: float = 0.01  # Epsilon ratio for polygon approximation
    
    # Lab-based empty panel filtering (for tinted/watercolor pages)
    min_non_bg_ratio: float = 0.08  # Minimum ratio of non-background pixels (Lab distance > bg_delta)
    min_dim_ratio: float = 0.12    # Minimum dimension ratio vs median (to filter thin/gutter panels)

    # Line Segment Detection (LSD) parameters
    use_line_detection: bool = True        # Enable LSD-based line detection
    lsd_min_length_frac: float = 0.08      # Minimum line length as fraction of image dimension
    lsd_angle_tolerance: float = 8.0       # Maximum deviation from H/V in degrees
    lsd_min_gap_frac: float = 0.04         # Minimum gap between line clusters as fraction
    lsd_min_panel_frac: float = 0.03       # Minimum panel dimension as fraction

    # Multi-scale consensus parameters
    use_multiscale: bool = True            # Enable multi-scale consensus detection
    multiscale_factors: list = field(default_factory=lambda: [0.6, 1.0, 1.5])  # Scale factors
    multiscale_min_agreement: int = 2      # Minimum number of scales that must agree
    multiscale_iou_threshold: float = 0.30 # IoU threshold to consider two rects as matching

    # K-means background clustering parameters
    use_kmeans_bg: bool = True             # Enable k-means background estimation
    kmeans_k: int = 4                      # Number of color clusters
    kmeans_bg_delta_expand: float = 15.0   # Lab distance to expand bg mask around centers

    # Contour hierarchy analysis parameters
    use_hierarchy: bool = True             # Enable contour hierarchy analysis
    hierarchy_min_area_pct: float = 0.008  # Minimum panel area as fraction
    hierarchy_max_area_pct: float = 0.95   # Maximum panel area as fraction
    hierarchy_min_fill: float = 0.40       # Minimum fill ratio for panel contours
    hierarchy_min_rectangularity: float = 0.50  # Minimum rectangularity score

    # Template layout matching parameters
    use_template_matching: bool = True     # Enable template matching refinement
    template_min_score: float = 0.45       # Minimum template match score
    template_grid_size: int = 20           # Grid resolution for template matching
    template_detection_threshold: float = 0.60  # Below this det score, prefer template

    # Detection options
    use_canny_fallback: bool = True
    use_freeform_fallback: bool = True  # Enable freeform/watershed fallback
    reading_rtl: bool = False     # Right-to-left reading order (manga)
    panel_mode: str = "auto"      # Detection mode: "auto", "classic_franco_belge", "modern"
    debug: bool = False

    def copy(self) -> "DetectorConfig":
        """Return a deep copy of this configuration."""
        return copy.deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "adaptive_block": self.adaptive_block,
            "adaptive_C": self.adaptive_C,
            "morph_kernel": self.morph_kernel,
            "morph_iter": self.morph_iter,
            "morph_scale_with_dpi": self.morph_scale_with_dpi,
            "min_area_pct": self.min_area_pct,
            "max_area_pct": self.max_area_pct,
            "min_fill_ratio": self.min_fill_ratio,
            "min_rect_px": self.min_rect_px,
            "min_rect_frac": self.min_rect_frac,
            "min_gutter_px": self.min_gutter_px,
            "min_gutter_frac": self.min_gutter_frac,
            "max_gutter_px_frac": self.max_gutter_px_frac,
            "gutter_cov_min": self.gutter_cov_min,
            "edge_margin_frac": self.edge_margin_frac,
            "light_col_rel": self.light_col_rel,
            "light_row_rel": self.light_row_rel,
            "proj_smooth_k": self.proj_smooth_k,
            "filter_title_rows": self.filter_title_rows,
            "title_row_top_frac": self.title_row_top_frac,
            "title_row_max_h_frac": self.title_row_max_h_frac,
            "title_row_median_w_frac_max": self.title_row_median_w_frac_max,
            "title_row_min_boxes": self.title_row_min_boxes,
            "title_row_big_min_boxes": self.title_row_big_min_boxes,
            "title_row_big_w_min_frac": self.title_row_big_w_min_frac,
            "title_row_min_meanL": self.title_row_min_meanL,
            "max_panels_per_page": self.max_panels_per_page,
            "freeform_bg_delta": self.freeform_bg_delta,
            "sure_fg_ratio": self.sure_fg_ratio,
            "min_area_ratio_freeform": self.min_area_ratio_freeform,
            "min_fill_ratio_freeform": self.min_fill_ratio_freeform,
            "gutter_bg_delta": self.gutter_bg_delta,
            "gutter_grad_percentile": self.gutter_grad_percentile,
            "gutter_open_kernel_frac": self.gutter_open_kernel_frac,
            "gutter_min_stripe_width": self.gutter_min_stripe_width,
            "gutter_stripe_length_frac": self.gutter_stripe_length_frac,
            "iou_merge_thr": self.iou_merge_thr,
            "approx_eps_ratio": self.approx_eps_ratio,
            "min_non_bg_ratio": self.min_non_bg_ratio,
            "min_dim_ratio": self.min_dim_ratio,
            # LSD
            "use_line_detection": self.use_line_detection,
            "lsd_min_length_frac": self.lsd_min_length_frac,
            "lsd_angle_tolerance": self.lsd_angle_tolerance,
            "lsd_min_gap_frac": self.lsd_min_gap_frac,
            "lsd_min_panel_frac": self.lsd_min_panel_frac,
            # Multi-scale
            "use_multiscale": self.use_multiscale,
            "multiscale_factors": self.multiscale_factors,
            "multiscale_min_agreement": self.multiscale_min_agreement,
            "multiscale_iou_threshold": self.multiscale_iou_threshold,
            # K-means
            "use_kmeans_bg": self.use_kmeans_bg,
            "kmeans_k": self.kmeans_k,
            "kmeans_bg_delta_expand": self.kmeans_bg_delta_expand,
            # Hierarchy
            "use_hierarchy": self.use_hierarchy,
            "hierarchy_min_area_pct": self.hierarchy_min_area_pct,
            "hierarchy_max_area_pct": self.hierarchy_max_area_pct,
            "hierarchy_min_fill": self.hierarchy_min_fill,
            "hierarchy_min_rectangularity": self.hierarchy_min_rectangularity,
            # Template
            "use_template_matching": self.use_template_matching,
            "template_min_score": self.template_min_score,
            "template_grid_size": self.template_grid_size,
            "template_detection_threshold": self.template_detection_threshold,
            "use_canny_fallback": self.use_canny_fallback,
            "use_freeform_fallback": self.use_freeform_fallback,
            "reading_rtl": self.reading_rtl,
            "panel_mode": self.panel_mode,
            "debug": self.debug,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectorConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class AppConfig:
    """Application-level configuration."""

    detection_dpi: float = 150.0  # DPI for rendering pages for detection
    panel_framing: str = "fit"    # "fit", "fill", or "center"
    debug_panels: bool = True     # Show debug overlay

    def copy(self) -> "AppConfig":
        """Return a deep copy of this configuration."""
        return copy.deepcopy(self)


# Preset configurations for different comic styles
PRESETS: Dict[str, tuple[DetectorConfig, AppConfig]] = {
    "Franco-Belge": (
        DetectorConfig(
            adaptive_block=51,
            adaptive_C=5,
            morph_kernel=5,
            morph_iter=1,
            morph_scale_with_dpi=True,
            min_rect_px=40,
            min_fill_ratio=0.50,
            light_col_rel=0.10,
            light_row_rel=0.10,
            gutter_cov_min=0.82,
            min_gutter_px=4,
            max_gutter_px_frac=0.08,
            edge_margin_frac=0.02,
            filter_title_rows=True,
            title_row_top_frac=0.20,
            title_row_max_h_frac=0.12,
            title_row_min_boxes=4,
            title_row_min_meanL=0.80,
            max_panels_per_page=20,
            reading_rtl=False,
        ),
        AppConfig(detection_dpi=200.0),
    ),
    "Manga": (
        DetectorConfig(
            adaptive_block=51,
            adaptive_C=4,
            morph_kernel=5,
            morph_iter=1,
            morph_scale_with_dpi=True,
            min_rect_px=35,
            min_fill_ratio=0.45,
            light_col_rel=0.08,
            light_row_rel=0.08,
            gutter_cov_min=0.78,
            min_gutter_px=3,
            max_gutter_px_frac=0.12,
            edge_margin_frac=0.015,
            filter_title_rows=True,
            title_row_top_frac=0.18,
            title_row_max_h_frac=0.10,
            title_row_min_boxes=4,
            title_row_min_meanL=0.78,
            max_panels_per_page=24,
            reading_rtl=True,
        ),
        AppConfig(detection_dpi=200.0),
    ),
    "Newspaper": (
        DetectorConfig(
            adaptive_block=41,
            adaptive_C=6,
            morph_kernel=5,
            morph_iter=1,
            morph_scale_with_dpi=True,
            min_rect_px=40,
            min_fill_ratio=0.50,
            light_col_rel=0.12,
            light_row_rel=0.12,
            gutter_cov_min=0.80,
            min_gutter_px=4,
            max_gutter_px_frac=0.08,
            edge_margin_frac=0.02,
            filter_title_rows=False,
            max_panels_per_page=16,
            reading_rtl=False,
        ),
        AppConfig(detection_dpi=150.0),
    ),
}
