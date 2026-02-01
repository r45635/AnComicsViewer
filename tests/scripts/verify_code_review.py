#!/usr/bin/env python3
"""Verification test for the code review changes."""

import sys
import os
import tempfile

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def test_imports():
    """Test all module imports."""
    print("=" * 50)
    print("Testing Imports")
    print("=" * 50)
    
    modules = [
        ("config", "ancomicsviewer.config", ["DetectorConfig", "AppConfig"]),
        ("cache", "ancomicsviewer.cache", ["PanelCache", "MemoryAwareLRUCache"]),
        ("image_utils", "ancomicsviewer.image_utils", ["qimage_to_numpy_rgba"]),
        ("panel_editor", "ancomicsviewer.panel_editor", ["PanelEditor", "PanelCorrections"]),
        ("async_detection", "ancomicsviewer.async_detection", ["AsyncDetectionManager", "DetectionTask"]),
        ("detector", "ancomicsviewer.detector", ["PanelDetector"]),
        ("detector.utils", "ancomicsviewer.detector.utils", ["PanelRegion", "DebugInfo"]),
        ("detector.classifier", "ancomicsviewer.detector.classifier", ["PageStyleClassifier"]),
        ("detector.adaptive", "ancomicsviewer.detector.adaptive", ["adaptive_threshold_route"]),
        ("detector.gutter", "ancomicsviewer.detector.gutter", ["gutter_based_detection"]),
        ("detector.freeform", "ancomicsviewer.detector.freeform", ["freeform_detection"]),
        ("detector.filters", "ancomicsviewer.detector.filters", ["filter_by_area"]),
        ("detector.base", "ancomicsviewer.detector.base", ["PanelDetector"]),
        ("main_window", "ancomicsviewer.main_window", ["ComicsView"]),
        ("pdf_view", "ancomicsviewer.pdf_view", ["PannablePdfView"]),
    ]
    
    all_ok = True
    for name, module_path, classes in modules:
        try:
            mod = __import__(module_path, fromlist=classes)
            for cls in classes:
                getattr(mod, cls)
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            all_ok = False
    
    return all_ok


def test_panel_editor():
    """Test PanelEditor functionality."""
    print("\n" + "=" * 50)
    print("Testing PanelEditor")
    print("=" * 50)
    
    from PySide6.QtCore import QRectF, QPointF, QSizeF
    from ancomicsviewer.panel_editor import PanelEditor, PanelCorrections, EditHandle
    
    # Test PanelEditor
    editor = PanelEditor()
    print("  ✓ PanelEditor created")
    
    # Set panels
    panels = [
        QRectF(10, 10, 100, 100),
        QRectF(120, 10, 100, 100),
    ]
    editor.set_panels(panels)
    assert len(editor.get_panels()) == 2
    print("  ✓ set_panels works")
    
    # Test selection
    pos = QPointF(50, 50)  # Inside first panel
    selected = editor.select_panel_at(pos)
    assert selected == True
    assert editor.selected_index == 0
    print("  ✓ select_panel_at works")
    
    # Test handle detection
    editor.edit_mode = True
    handle = editor.get_handle_at(QPointF(10, 10))  # Top-left corner
    assert handle == EditHandle.TOP_LEFT
    print("  ✓ get_handle_at works")
    
    # Test deletion
    deleted = editor.delete_selected()
    assert deleted == True
    assert len(editor.get_panels()) == 1
    print("  ✓ delete_selected works")
    
    # Test undo
    editor.undo()
    assert len(editor.get_panels()) == 2
    print("  ✓ undo works")
    
    # Test PanelCorrections
    with tempfile.TemporaryDirectory() as tmpdir:
        corrections = PanelCorrections(tmpdir)
        print("  ✓ PanelCorrections created")
        
        # Save
        saved = corrections.save("/test/comic.pdf", 0, panels, QSizeF(612, 792))
        assert saved == True
        print("  ✓ save works")
        
        # Load
        loaded = corrections.load("/test/comic.pdf", 0)
        assert loaded is not None
        assert len(loaded) == 2
        print("  ✓ load works")
        
        # Has corrections
        has = corrections.has_corrections("/test/comic.pdf", 0)
        assert has == True
        print("  ✓ has_corrections works")
        
        # Delete
        deleted = corrections.delete("/test/comic.pdf", 0)
        assert deleted == True
        has = corrections.has_corrections("/test/comic.pdf", 0)
        assert has == False
        print("  ✓ delete works")
    
    return True


def test_memory_cache():
    """Test MemoryAwareLRUCache functionality."""
    print("\n" + "=" * 50)
    print("Testing MemoryAwareLRUCache")
    print("=" * 50)
    
    from ancomicsviewer.cache import MemoryAwareLRUCache
    
    cache = MemoryAwareLRUCache(max_items=10, max_memory_mb=1)
    print("  ✓ MemoryAwareLRUCache created")
    
    # Put items
    cache.put("key1", [1, 2, 3, 4, 5])
    cache.put("key2", "hello world")
    assert len(cache) == 2
    print("  ✓ put works")
    
    # Get items
    val = cache.get("key1")
    assert val == [1, 2, 3, 4, 5]
    print("  ✓ get works")
    
    # Hit rate
    rate = cache.hit_rate
    assert rate > 0
    print(f"  ✓ hit_rate works: {rate:.2%}")
    
    # Memory usage
    mem = cache.memory_usage_mb
    print(f"  ✓ memory_usage_mb works: {mem:.4f} MB")
    
    # Stats
    stats = cache.get_stats()
    assert "items" in stats
    assert "memory_mb" in stats
    assert "hit_rate" in stats
    print("  ✓ get_stats works")
    
    # Clear
    cache.clear()
    assert len(cache) == 0
    print("  ✓ clear works")
    
    return True


def test_async_detection():
    """Test AsyncDetectionManager functionality."""
    print("\n" + "=" * 50)
    print("Testing AsyncDetectionManager")
    print("=" * 50)
    
    from ancomicsviewer.async_detection import AsyncDetectionManager, DetectionTask
    from ancomicsviewer.config import DetectorConfig
    
    config = DetectorConfig()
    manager = AsyncDetectionManager(config)
    print("  ✓ AsyncDetectionManager created")
    
    assert manager.is_busy == False
    print("  ✓ is_busy works")
    
    assert manager.pending_count == 0
    print("  ✓ pending_count works")
    
    # Test config update
    new_config = DetectorConfig(debug=True)
    manager.update_config(new_config)
    print("  ✓ update_config works")
    
    # Test shutdown
    manager.shutdown()
    print("  ✓ shutdown works")
    
    return True


def test_detector():
    """Test PanelDetector from new module."""
    print("\n" + "=" * 50)
    print("Testing PanelDetector (new module)")
    print("=" * 50)
    
    from ancomicsviewer.detector import PanelDetector
    from ancomicsviewer.config import DetectorConfig
    
    config = DetectorConfig()
    detector = PanelDetector(config)
    print("  ✓ PanelDetector created from new module")
    
    # Check methods exist
    assert hasattr(detector, "detect_panels")
    assert hasattr(detector, "last_debug")
    assert hasattr(detector, "config")
    print("  ✓ PanelDetector has required attributes")
    
    # Check functions are importable
    from ancomicsviewer.detector.adaptive import adaptive_threshold_route
    from ancomicsviewer.detector.gutter import gutter_based_detection
    from ancomicsviewer.detector.freeform import freeform_detection
    from ancomicsviewer.detector.filters import filter_by_area
    print("  ✓ All route functions importable")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 50)
    print("  AnComicsViewer Code Review Verification")
    print("=" * 50)
    
    all_ok = True
    
    try:
        all_ok = test_imports() and all_ok
    except Exception as e:
        print(f"  ✗ Import tests failed: {e}")
        all_ok = False
    
    try:
        all_ok = test_panel_editor() and all_ok
    except Exception as e:
        print(f"  ✗ PanelEditor tests failed: {e}")
        import traceback
        traceback.print_exc()
        all_ok = False
    
    try:
        all_ok = test_memory_cache() and all_ok
    except Exception as e:
        print(f"  ✗ MemoryAwareLRUCache tests failed: {e}")
        import traceback
        traceback.print_exc()
        all_ok = False
    
    try:
        all_ok = test_async_detection() and all_ok
    except Exception as e:
        print(f"  ✗ AsyncDetectionManager tests failed: {e}")
        import traceback
        traceback.print_exc()
        all_ok = False
    
    try:
        all_ok = test_detector() and all_ok
    except Exception as e:
        print(f"  ✗ PanelDetector tests failed: {e}")
        import traceback
        traceback.print_exc()
        all_ok = False
    
    print("\n" + "=" * 50)
    if all_ok:
        print("  ✓ ALL TESTS PASSED")
    else:
        print("  ✗ SOME TESTS FAILED")
    print("=" * 50)
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
