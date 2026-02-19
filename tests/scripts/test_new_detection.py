#!/usr/bin/env python3
"""Test script for new detection improvements."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_imports():
    """Test all module imports."""
    print("=== Import Tests ===")

    from ancomicsviewer.config import DetectorConfig
    print("  OK DetectorConfig")

    from ancomicsviewer.detector import PanelDetector
    print("  OK PanelDetector")

    from ancomicsviewer.detector.line_detection import (
        detect_line_segments, line_based_detection, detect_gradient_borders,
        cluster_lines_to_positions, panels_from_line_grid
    )
    print("  OK line_detection (LSD)")

    from ancomicsviewer.detector.multiscale import (
        multiscale_detect, consensus_merge, score_detection_result, select_best_result
    )
    print("  OK multiscale")

    from ancomicsviewer.detector.clustering import (
        cluster_colors, identify_background_clusters,
        make_kmeans_background_mask, get_dominant_bg_lab
    )
    print("  OK clustering (k-means)")

    from ancomicsviewer.detector.contour_hierarchy import (
        analyze_contour_hierarchy, hierarchy_based_detection
    )
    print("  OK contour_hierarchy")

    from ancomicsviewer.detector.templates import (
        match_best_template, refine_with_template, LAYOUT_TEMPLATES,
        compute_content_map, compute_edge_map, score_template
    )
    print("  OK templates")

    return True


def test_config():
    """Test new config parameters."""
    print("\n=== Config Tests ===")
    from ancomicsviewer.config import DetectorConfig

    config = DetectorConfig()
    assert config.use_line_detection == True, "use_line_detection"
    assert config.use_multiscale == True, "use_multiscale"
    assert config.use_kmeans_bg == True, "use_kmeans_bg"
    assert config.use_hierarchy == True, "use_hierarchy"
    assert config.use_template_matching == True, "use_template_matching"
    assert config.multiscale_factors == [0.6, 1.0, 1.5], "multiscale_factors"
    assert config.kmeans_k == 4, "kmeans_k"
    assert config.lsd_min_length_frac == 0.08, "lsd_min_length_frac"
    print("  OK new params defaults")

    # Test serialization
    d = config.to_dict()
    assert "use_line_detection" in d
    assert "use_multiscale" in d
    assert "use_kmeans_bg" in d
    assert "use_hierarchy" in d
    assert "use_template_matching" in d
    print("  OK serialization")

    # Test deserialization
    config2 = DetectorConfig.from_dict(d)
    assert config2.use_line_detection == True
    assert config2.multiscale_factors == [0.6, 1.0, 1.5]
    print("  OK deserialization")

    return True


def test_templates():
    """Test template layout definitions."""
    print("\n=== Template Tests ===")
    from ancomicsviewer.detector.templates import LAYOUT_TEMPLATES

    assert len(LAYOUT_TEMPLATES) >= 15, f"Expected >= 15, got {len(LAYOUT_TEMPLATES)}"
    print(f"  OK {len(LAYOUT_TEMPLATES)} templates defined")

    # Verify all templates have valid coordinates
    for tmpl in LAYOUT_TEMPLATES:
        assert tmpl.name, "Template must have a name"
        assert len(tmpl.panels) > 0, f"Template {tmpl.name} has no panels"
        for px, py, pw, ph in tmpl.panels:
            assert 0 <= px <= 1, f"Invalid x in {tmpl.name}: {px}"
            assert 0 <= py <= 1, f"Invalid y in {tmpl.name}: {py}"
            assert 0 < pw <= 1, f"Invalid w in {tmpl.name}: {pw}"
            assert 0 < ph <= 1, f"Invalid h in {tmpl.name}: {ph}"
    print("  OK all template coordinates valid")

    return True


def test_multiscale_scoring():
    """Test detection scoring."""
    print("\n=== Multiscale Scoring Tests ===")
    from PySide6.QtCore import QRectF, QSizeF
    from ancomicsviewer.detector.multiscale import score_detection_result

    page = QSizeF(595, 842)

    # Good result: 4 panels covering most of page
    good = [
        QRectF(10, 10, 275, 400),
        QRectF(300, 10, 275, 400),
        QRectF(10, 425, 275, 400),
        QRectF(300, 425, 275, 400),
    ]
    score_good = score_detection_result(good, page)
    assert score_good > 0.5, f"Good result should score > 0.5, got {score_good:.3f}"
    print(f"  OK good result score = {score_good:.3f}")

    # Bad result: single small panel
    bad = [QRectF(100, 100, 50, 50)]
    score_bad = score_detection_result(bad, page)
    assert score_bad < score_good, f"Bad should score less: {score_bad:.3f} vs {score_good:.3f}"
    print(f"  OK bad result score = {score_bad:.3f}")

    # Empty result
    empty_score = score_detection_result([], page)
    assert empty_score == 0.0
    print("  OK empty result score = 0.0")

    return True


def test_line_detection_unit():
    """Test LSD line detection components."""
    print("\n=== Line Detection Unit Tests ===")
    import numpy as np
    from ancomicsviewer.detector.line_detection import (
        _make_segment, _filter_hv_lines, cluster_lines_to_positions,
        panels_from_line_grid,
    )
    from PySide6.QtCore import QSizeF

    # Test segment creation
    seg = _make_segment(0, 0, 100, 0)
    assert seg.is_horizontal, "Horizontal segment"
    assert not seg.is_vertical, "Not vertical"
    assert abs(seg.length - 100) < 1, f"Length should be 100, got {seg.length}"
    print("  OK segment creation (horizontal)")

    seg_v = _make_segment(0, 0, 0, 100)
    assert seg_v.is_vertical, "Vertical segment"
    assert not seg_v.is_horizontal, "Not horizontal"
    print("  OK segment creation (vertical)")

    # Test filtering
    diagonal = _make_segment(0, 0, 100, 100)
    filtered = _filter_hv_lines([seg, seg_v, diagonal], angle_tolerance=8.0)
    assert len(filtered) == 2, f"Should keep 2 HV lines, got {len(filtered)}"
    print("  OK HV filtering")

    # Test grid generation
    page = QSizeF(595, 842)
    rects = panels_from_line_grid([200, 400], [300], 600, 800, page)
    assert len(rects) > 0, "Should generate panels from grid"
    print(f"  OK grid generation: {len(rects)} panels from 2H+1V lines")

    return True


def test_clustering_unit():
    """Test k-means clustering."""
    print("\n=== Clustering Unit Tests ===")
    import numpy as np
    import cv2

    # Create a simple test image (white background with colored panels)
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255  # White background
    img[20:90, 20:90] = [30, 40, 50]   # Dark panel top-left
    img[110:190, 20:90] = [50, 60, 70]  # Dark panel bottom-left
    img[20:90, 110:190] = [40, 50, 60] # Dark panel top-right

    from ancomicsviewer.detector.clustering import (
        cluster_colors, identify_background_clusters, get_dominant_bg_lab
    )

    clusters = cluster_colors(img, k=3)
    assert len(clusters) > 0, "Should find clusters"
    print(f"  OK found {len(clusters)} clusters")

    bg_clusters = identify_background_clusters(clusters)
    assert len(bg_clusters) > 0, "Should identify background"
    print(f"  OK identified {len(bg_clusters)} background cluster(s)")

    bg_lab = get_dominant_bg_lab(img, k=3)
    assert bg_lab[0] > 200, f"Background should be bright, L={bg_lab[0]:.0f}"
    print(f"  OK dominant bg L={bg_lab[0]:.0f}")

    return True


def test_hierarchy_unit():
    """Test contour hierarchy analysis."""
    print("\n=== Hierarchy Unit Tests ===")
    import numpy as np
    import cv2
    from PySide6.QtCore import QSizeF
    from ancomicsviewer.config import DetectorConfig
    from ancomicsviewer.detector.contour_hierarchy import hierarchy_based_detection

    # Create image with clear rectangular panels
    img = np.ones((400, 300), dtype=np.uint8) * 255
    cv2.rectangle(img, (10, 10), (140, 190), 0, 2)
    cv2.rectangle(img, (160, 10), (290, 190), 0, 2)
    cv2.rectangle(img, (10, 210), (290, 390), 0, 2)

    # Add some content inside panels
    cv2.circle(img, (75, 100), 30, 0, -1)
    cv2.circle(img, (225, 100), 30, 0, -1)
    cv2.circle(img, (150, 300), 50, 0, -1)

    config = DetectorConfig()
    page = QSizeF(300, 400)
    rects = hierarchy_based_detection(img, 300, 400, page, config)
    print(f"  OK hierarchy detected {len(rects)} panels")

    return True


def test_template_matching():
    """Test template matching."""
    print("\n=== Template Matching Tests ===")
    import numpy as np
    from PySide6.QtCore import QSizeF
    from ancomicsviewer.detector.templates import (
        compute_content_map, compute_edge_map, match_best_template
    )

    # Create 2x2 grid-like image
    img = np.ones((400, 300), dtype=np.uint8) * 255
    # 4 dark panels in 2x2 grid
    img[10:190, 10:140] = 50
    img[10:190, 160:290] = 50
    img[210:390, 10:140] = 50
    img[210:390, 160:290] = 50

    content_map = compute_content_map(img, grid_size=20)
    assert content_map.shape == (20, 20), "Content map shape"
    print("  OK content map computed")

    edge_map = compute_edge_map(img, grid_size=20)
    assert edge_map.shape == (20, 20), "Edge map shape"
    print("  OK edge map computed")

    page = QSizeF(300, 400)
    name, rects, score = match_best_template(img, page, grid_size=20, min_score=0.0)
    print(f"  OK best template: {name} (score={score:.3f}, {len(rects)} panels)")

    return True


def test_smoke_detection():
    """Smoke test: run full detection pipeline on a synthetic image."""
    print("\n=== Smoke Test: Full Pipeline ===")
    import numpy as np
    import cv2
    from PySide6.QtCore import QSizeF
    from PySide6.QtGui import QImage
    from ancomicsviewer.detector import PanelDetector
    from ancomicsviewer.config import DetectorConfig

    # Create a synthetic comic page (RGBA)
    h, w = 800, 600
    img = np.ones((h, w, 4), dtype=np.uint8) * 255
    img[:, :, 3] = 255  # Full alpha

    # Draw panel borders (black lines)
    cv2.rectangle(img, (20, 20), (w//2 - 10, h//2 - 10), (0, 0, 0, 255), 3)
    cv2.rectangle(img, (w//2 + 10, 20), (w - 20, h//2 - 10), (0, 0, 0, 255), 3)
    cv2.rectangle(img, (20, h//2 + 10), (w - 20, h - 20), (0, 0, 0, 255), 3)

    # Add some content inside panels
    cv2.circle(img, (w//4, h//4), 50, (100, 50, 50, 255), -1)
    cv2.circle(img, (3*w//4, h//4), 50, (50, 100, 50, 255), -1)
    cv2.circle(img, (w//2, 3*h//4), 80, (50, 50, 100, 255), -1)

    # Convert to QImage
    qimage = QImage(img.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()

    config = DetectorConfig(debug=False)
    config.use_multiscale = False  # Faster for test
    detector = PanelDetector(config)
    page_size = QSizeF(w * 72 / 150, h * 72 / 150)

    rects = detector.detect_panels(qimage, page_size, page_num=0, dpi=150.0)

    print(f"  Detected {len(rects)} panels")
    for i, r in enumerate(rects):
        print(f"    Panel {i+1}: ({r.left():.0f},{r.top():.0f}) {r.width():.0f}x{r.height():.0f}")

    assert len(rects) >= 1, "Should detect at least 1 panel"
    print("  OK full pipeline works!")

    return True


if __name__ == "__main__":
    results = []
    tests = [
        test_imports,
        test_config,
        test_templates,
        test_multiscale_scoring,
        test_line_detection_unit,
        test_clustering_unit,
        test_hierarchy_unit,
        test_template_matching,
        test_smoke_detection,
    ]

    for test_fn in tests:
        try:
            ok = test_fn()
            results.append((test_fn.__name__, ok))
        except Exception as e:
            import traceback
            print(f"\n  FAILED: {e}")
            traceback.print_exc()
            results.append((test_fn.__name__, False))

    print("\n" + "=" * 50)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n{passed}/{total} tests passed")
    if passed == total:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
