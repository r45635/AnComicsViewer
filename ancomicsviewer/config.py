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
    morph_kernel: int = 7         # Kernel size for morphological operations
    morph_iter: int = 2           # Number of morphology iterations

    # Base filter parameters (as fractions of page area/dimensions)
    min_area_pct: float = 0.015   # Minimum panel area (1.5% of page)
    max_area_pct: float = 0.95    # Maximum panel area (exclude full-bleed)
    min_fill_ratio: float = 0.55  # Minimum contour fill ratio
    min_rect_px: int = 80         # Minimum panel dimension in pixels
    min_rect_frac: float = 0.055  # Minimum panel dimension as fraction of page

    # Gutter detection parameters
    min_gutter_px: int = 10       # Minimum gutter thickness in pixels
    min_gutter_frac: float = 0.012  # Minimum gutter as fraction of block
    max_gutter_px_frac: float = 0.06  # Maximum gutter as fraction of block
    gutter_cov_min: float = 0.88  # Minimum brightness coverage for gutter
    edge_margin_frac: float = 0.03  # Margin from edges for gutter detection

    # Brightness-based split parameters
    light_col_rel: float = 0.12   # Column brightness threshold (relative)
    light_row_rel: float = 0.12   # Row brightness threshold (relative)
    proj_smooth_k: int = 15       # Projection smoothing kernel (must be odd)

    # Title row filter parameters
    filter_title_rows: bool = True
    title_row_top_frac: float = 0.20      # Title must be in top X% of page
    title_row_max_h_frac: float = 0.10    # Maximum title row height
    title_row_median_w_frac_max: float = 0.25  # Maximum median width
    title_row_min_boxes: int = 4          # Minimum boxes for "many small" pattern
    title_row_big_min_boxes: int = 2      # Minimum boxes for "few big" pattern
    title_row_big_w_min_frac: float = 0.16  # Minimum width for "big" boxes
    title_row_min_meanL: float = 0.88     # Minimum mean luminosity

    # Limits
    max_panels_per_page: int = 24  # Safety limit

    # Detection options
    use_canny_fallback: bool = True
    reading_rtl: bool = False     # Right-to-left reading order (manga)
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
            "use_canny_fallback": self.use_canny_fallback,
            "reading_rtl": self.reading_rtl,
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
            morph_kernel=7,
            morph_iter=2,
            min_rect_px=60,
            light_col_rel=0.12,
            light_row_rel=0.12,
            gutter_cov_min=0.90,
            min_gutter_px=8,
            max_gutter_px_frac=0.06,
            edge_margin_frac=0.03,
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
            morph_kernel=7,
            morph_iter=2,
            min_rect_px=50,
            light_col_rel=0.10,
            light_row_rel=0.10,
            gutter_cov_min=0.85,
            min_gutter_px=6,
            max_gutter_px_frac=0.10,
            edge_margin_frac=0.02,
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
            morph_iter=2,
            min_rect_px=50,
            light_col_rel=0.14,
            light_row_rel=0.14,
            gutter_cov_min=0.88,
            min_gutter_px=6,
            max_gutter_px_frac=0.06,
            edge_margin_frac=0.03,
            filter_title_rows=False,
            max_panels_per_page=16,
            reading_rtl=False,
        ),
        AppConfig(detection_dpi=150.0),
    ),
}
