"""Dialog windows for AnComicsViewer.

Provides:
- PanelTuningDialog: Interactive parameter adjustment
"""

from __future__ import annotations

import traceback
from typing import TYPE_CHECKING, Callable, Optional

from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QVBoxLayout,
    QHBoxLayout,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QPushButton,
    QGroupBox,
    QScrollArea,
    QWidget,
)

from .config import DetectorConfig
from .image_utils import pdebug

if TYPE_CHECKING:
    from .main_window import ComicsView


class PanelTuningDialog(QDialog):
    """Interactive dialog for tuning panel detection parameters.

    Provides real-time parameter adjustment with Apply & Re-run functionality.
    Parameters are organized into logical groups for better usability.
    """

    def __init__(
        self,
        parent: Optional[QWidget],
        config: DetectorConfig,
        det_dpi: float,
        on_apply: Optional[Callable[[DetectorConfig, float], None]] = None
    ):
        """Initialize tuning dialog.

        Args:
            parent: Parent widget
            config: Current detector configuration
            det_dpi: Current detection DPI
            on_apply: Callback when Apply is clicked (receives new config and DPI)
        """
        super().__init__(parent)
        self.setWindowTitle("Panel Detection Tuning")
        self.setMinimumWidth(400)

        self._config = config
        self._det_dpi = det_dpi
        self._on_apply = on_apply
        self._widgets: dict = {}

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the dialog UI."""
        # Create scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        main_layout = QVBoxLayout(content)

        # Add parameter groups
        main_layout.addWidget(self._create_general_group())
        main_layout.addWidget(self._create_threshold_group())
        main_layout.addWidget(self._create_filter_group())
        main_layout.addWidget(self._create_gutter_group())
        main_layout.addWidget(self._create_title_group())

        main_layout.addStretch()
        scroll.setWidget(content)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_apply = QPushButton("Apply && Re-run")
        btn_close = QPushButton("Close")
        btn_layout.addWidget(btn_apply)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_close)

        btn_apply.clicked.connect(self._apply)
        btn_close.clicked.connect(self.accept)

        # Main layout
        layout = QVBoxLayout(self)
        layout.addWidget(scroll)
        layout.addLayout(btn_layout)

    def _create_general_group(self) -> QGroupBox:
        """Create general settings group."""
        group = QGroupBox("General")
        form = QFormLayout(group)

        self._widgets["dpi"] = self._add_int_spin(
            form, "Detection DPI", int(self._det_dpi), 72, 400, 10
        )

        return group

    def _create_threshold_group(self) -> QGroupBox:
        """Create adaptive threshold settings group."""
        group = QGroupBox("Adaptive Threshold")
        form = QFormLayout(group)
        c = self._config

        self._widgets["adaptive_block"] = self._add_int_spin(
            form, "Block size (odd)", c.adaptive_block, 15, 101, 2
        )
        self._widgets["adaptive_C"] = self._add_int_spin(
            form, "C offset", c.adaptive_C, 0, 20, 1
        )
        self._widgets["morph_kernel"] = self._add_int_spin(
            form, "Morph kernel", c.morph_kernel, 3, 15, 2
        )
        self._widgets["morph_iter"] = self._add_int_spin(
            form, "Morph iterations", c.morph_iter, 1, 5, 1
        )

        return group

    def _create_filter_group(self) -> QGroupBox:
        """Create base filter settings group."""
        group = QGroupBox("Base Filters")
        form = QFormLayout(group)
        c = self._config

        self._widgets["min_area_pct"] = self._add_float_spin(
            form, "Min area %", c.min_area_pct, 0.001, 0.2, 0.005
        )
        self._widgets["max_area_pct"] = self._add_float_spin(
            form, "Max area %", c.max_area_pct, 0.3, 0.99, 0.01
        )
        self._widgets["min_fill_ratio"] = self._add_float_spin(
            form, "Min fill ratio", c.min_fill_ratio, 0.10, 0.90, 0.01
        )
        self._widgets["min_rect_px"] = self._add_int_spin(
            form, "Min rect px", c.min_rect_px, 10, 200, 2
        )
        self._widgets["min_rect_frac"] = self._add_float_spin(
            form, "Min rect frac", c.min_rect_frac, 0.01, 0.2, 0.005
        )
        self._widgets["max_panels_per_page"] = self._add_int_spin(
            form, "Max panels/page", c.max_panels_per_page, 4, 64, 1
        )

        return group

    def _create_gutter_group(self) -> QGroupBox:
        """Create gutter detection settings group."""
        group = QGroupBox("Gutter Detection")
        form = QFormLayout(group)
        c = self._config

        self._widgets["light_col_rel"] = self._add_float_spin(
            form, "Light col rel", c.light_col_rel, 0.0, 0.50, 0.01
        )
        self._widgets["light_row_rel"] = self._add_float_spin(
            form, "Light row rel", c.light_row_rel, 0.0, 0.50, 0.01
        )
        self._widgets["gutter_cov_min"] = self._add_float_spin(
            form, "Gutter coverage min", c.gutter_cov_min, 0.50, 0.99, 0.01
        )
        self._widgets["min_gutter_px"] = self._add_int_spin(
            form, "Min gutter px", c.min_gutter_px, 1, 50, 1
        )
        self._widgets["min_gutter_frac"] = self._add_float_spin(
            form, "Min gutter frac", c.min_gutter_frac, 0.001, 0.1, 0.001
        )
        self._widgets["max_gutter_px_frac"] = self._add_float_spin(
            form, "Max gutter frac", c.max_gutter_px_frac, 0.01, 0.20, 0.005
        )
        self._widgets["edge_margin_frac"] = self._add_float_spin(
            form, "Edge margin frac", c.edge_margin_frac, 0.0, 0.20, 0.005
        )
        self._widgets["proj_smooth_k"] = self._add_int_spin(
            form, "Proj. smooth (odd)", c.proj_smooth_k, 5, 51, 2
        )

        return group

    def _create_title_group(self) -> QGroupBox:
        """Create title row filter settings group."""
        group = QGroupBox("Title Row Filter")
        form = QFormLayout(group)
        c = self._config

        self._widgets["filter_title_rows"] = self._add_checkbox(
            form, "Enable filter", c.filter_title_rows
        )
        self._widgets["title_row_top_frac"] = self._add_float_spin(
            form, "Top frac", c.title_row_top_frac, 0.05, 0.40, 0.01
        )
        self._widgets["title_row_max_h_frac"] = self._add_float_spin(
            form, "Max height frac", c.title_row_max_h_frac, 0.05, 0.30, 0.01
        )
        self._widgets["title_row_min_boxes"] = self._add_int_spin(
            form, "Min boxes", c.title_row_min_boxes, 1, 12, 1
        )
        self._widgets["title_row_min_meanL"] = self._add_float_spin(
            form, "Min mean L", c.title_row_min_meanL, 0.40, 0.99, 0.01
        )
        self._widgets["title_row_median_w_frac_max"] = self._add_float_spin(
            form, "Median width max", c.title_row_median_w_frac_max, 0.05, 0.6, 0.01
        )

        return group

    def _add_int_spin(
        self, form: QFormLayout, label: str, value: int,
        min_val: int, max_val: int, step: int
    ) -> QSpinBox:
        """Add integer spinbox to form."""
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setSingleStep(step)
        spin.setValue(value)
        form.addRow(label, spin)
        return spin

    def _add_float_spin(
        self, form: QFormLayout, label: str, value: float,
        min_val: float, max_val: float, step: float
    ) -> QDoubleSpinBox:
        """Add float spinbox to form."""
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setDecimals(3)
        spin.setSingleStep(step)
        spin.setValue(value)
        form.addRow(label, spin)
        return spin

    def _add_checkbox(
        self, form: QFormLayout, label: str, checked: bool
    ) -> QCheckBox:
        """Add checkbox to form."""
        cb = QCheckBox()
        cb.setChecked(checked)
        form.addRow(label, cb)
        return cb

    def _apply(self) -> None:
        """Apply changes and trigger re-detection."""
        try:
            # Read values from widgets
            new_dpi = float(self._widgets["dpi"].value())

            # Create new config
            new_config = DetectorConfig(
                adaptive_block=int(self._widgets["adaptive_block"].value()) | 1,
                adaptive_C=int(self._widgets["adaptive_C"].value()),
                morph_kernel=int(self._widgets["morph_kernel"].value()),
                morph_iter=int(self._widgets["morph_iter"].value()),
                min_area_pct=float(self._widgets["min_area_pct"].value()),
                max_area_pct=float(self._widgets["max_area_pct"].value()),
                min_fill_ratio=float(self._widgets["min_fill_ratio"].value()),
                min_rect_px=int(self._widgets["min_rect_px"].value()),
                min_rect_frac=float(self._widgets["min_rect_frac"].value()),
                min_gutter_px=int(self._widgets["min_gutter_px"].value()),
                min_gutter_frac=float(self._widgets["min_gutter_frac"].value()),
                max_gutter_px_frac=float(self._widgets["max_gutter_px_frac"].value()),
                gutter_cov_min=float(self._widgets["gutter_cov_min"].value()),
                edge_margin_frac=float(self._widgets["edge_margin_frac"].value()),
                light_col_rel=float(self._widgets["light_col_rel"].value()),
                light_row_rel=float(self._widgets["light_row_rel"].value()),
                proj_smooth_k=int(self._widgets["proj_smooth_k"].value()) | 1,
                filter_title_rows=bool(self._widgets["filter_title_rows"].isChecked()),
                title_row_top_frac=float(self._widgets["title_row_top_frac"].value()),
                title_row_max_h_frac=float(self._widgets["title_row_max_h_frac"].value()),
                title_row_min_boxes=int(self._widgets["title_row_min_boxes"].value()),
                title_row_min_meanL=float(self._widgets["title_row_min_meanL"].value()),
                title_row_median_w_frac_max=float(self._widgets["title_row_median_w_frac_max"].value()),
                max_panels_per_page=int(self._widgets["max_panels_per_page"].value()),
                # Preserve non-UI settings from original config
                use_canny_fallback=self._config.use_canny_fallback,
                reading_rtl=self._config.reading_rtl,
                debug=self._config.debug,
            )

            # Call the callback
            if self._on_apply:
                self._on_apply(new_config, new_dpi)

        except Exception:
            pdebug(f"Apply tuning error:\n{traceback.format_exc()}")

    def get_config(self) -> DetectorConfig:
        """Get the current configuration from UI values."""
        self._apply()  # Update internal config
        return self._config

    def get_dpi(self) -> float:
        """Get the current DPI value."""
        return float(self._widgets["dpi"].value())
