# Exact Diff: Gutter Detection & Nested Suppression

## 1. GUTTER DETECTION VALIDATION

### Location: `_gutter_based_detection()` lines 533-640

**Key Change (lines 603-608):**
```python
# Group peaks into gutter regions
h_lines = self._group_gutter_lines(h_peaks)
v_lines = self._group_gutter_lines(v_peaks)

pdebug(f"[gutters] h_lines raw={len(h_lines)} v_lines raw={len(v_lines)}")

# ✅ VALIDATE gutters: check they are continuous bright bands
h_lines = self._validate_h_gutters(binary, h_lines, min_cov=0.85, band=6)
v_lines = self._validate_v_gutters(binary, v_lines, min_cov=0.85, band=6)

pdebug(f"[gutters] h_lines valid={len(h_lines)} v_lines valid={len(v_lines)}")
```

**What it does:**
- Lines 603-604: Extract raw gutter peaks → group into regions
- Lines 606: Log raw counts before validation
- Lines 609-610: ✅ **Validate each gutter** with continuity check
- Lines 612: Log validated counts

---

## 2. VALIDATION FUNCTIONS

### `_validate_h_gutters()` - Lines 822-865

```python
def _validate_h_gutters(self, binary_bright: NDArray, h_lines: List[Tuple[int, int]], 
                       min_cov: float = 0.85, band: int = 6) -> List[Tuple[int, int]]:
    """Validate horizontal gutter candidates by checking continuity across WIDTH.
    
    For each horizontal gutter (y-coordinate):
    1. Extract ±6 pixel vertical band around center
    2. Count bright pixels (255) in that band
    3. Calculate coverage = bright_pixels / total_pixels
    4. REJECT if coverage < 85%  ← Filters false gutters inside panels
    
    Args:
        binary_bright: Binary mask (255=bright gutters, 0=dark content)
        h_lines: List of (y0, y1) horizontal candidates
        min_cov: 85% coverage threshold
        band: ±6 pixels around gutter center
    """
    # Extract band region around gutter center
    y_center = (y0 + y1) // 2
    band_region = binary_bright[y_min:y_max, :]  # All columns
    
    # Check if band is mostly bright (continuous gutter)
    coverage = bright_pixels / total_pixels
    
    if coverage >= min_cov:
        validated.append((y0, y1))  # KEEP: Strong gutter
    else:
        pdebug(f"[gutters] Rejected h-gutter at y={y_center}: coverage={coverage:.2f} < {min_cov}")
        # REJECT: Weak/broken gutter (bright band inside panel)
```

### `_validate_v_gutters()` - Lines 867-905

```python
def _validate_v_gutters(self, binary_bright: NDArray, v_lines: List[Tuple[int, int]], 
                       min_cov: float = 0.85, band: int = 6) -> List[Tuple[int, int]]:
    """Validate vertical gutter candidates by checking continuity across HEIGHT.
    
    For each vertical gutter (x-coordinate):
    1. Extract ±6 pixel horizontal band around center
    2. Count bright pixels in that band
    3. Calculate coverage
    4. REJECT if coverage < 85%
    
    Args:
        binary_bright: Binary mask
        v_lines: List of (x0, x1) vertical candidates
        min_cov: 85% coverage threshold
        band: ±6 pixels around gutter center
    """
    # Extract band region around gutter center
    x_center = (x0 + x1) // 2
    band_region = binary_bright[:, x_min:x_max]  # All rows
    
    # Check if band is mostly bright (continuous gutter)
    coverage = bright_pixels / total_pixels
    
    if coverage >= min_cov:
        validated.append((x0, x1))  # KEEP
    else:
        pdebug(f"[gutters] Rejected v-gutter at x={x_center}: coverage={coverage:.2f} < {min_cov}")
        # REJECT
```

---

## 3. LAB-BASED EMPTY PANEL FILTERING

### Location: `detect_panels()` lines 184-236

**Exact code:**
```python
# Lab-based empty filtering (for tinted/watercolor pages)
if rects:
    initial_count = len(rects)
    filtered_rects = []
    
    # Calculate median dimensions for thin panel detection
    if len(rects) > 1:
        widths = [r.width() for r in rects]
        heights = [r.height() for r in rects]
        median_w = np.median(widths)
        median_h = np.median(heights)
    else:
        median_w = median_h = 0
    
    for rect in rects:
        # Check non-background ratio (Lab color distance)
        x = int(rect.left() * (w / page_point_size.width()))
        y = int(rect.top() * (h / page_point_size.height()))
        rw = int(rect.width() * (w / page_point_size.width()))
        rh = int(rect.height() * (h / page_point_size.height()))
        
        # Clamp to image bounds
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        rw = max(1, min(rw, w - x))
        rh = max(1, min(rh, h - y))
        
        # Extract ROI
        roi_bgr = img_bgr[y:y+rh, x:x+rw]
        
        if roi_bgr.size == 0:
            continue
        
        # ✅ Calculate non-background ratio using Lab color space
        non_bg = _non_bg_ratio(roi_bgr, bg_lab, delta=self.config.bg_delta)
        
        # ✅ FILTER 1: Too empty (mostly background)
        if non_bg < self.config.min_non_bg_ratio:  # Default: 0.08 (8%)
            pdebug(f"[Lab] Dropped mostly-bg panel: non_bg={non_bg:.3f} < {self.config.min_non_bg_ratio:.3f}")
            continue
        
        # ✅ FILTER 2: Too thin (likely misdetected gutters)
        if median_w > 0 and median_h > 0:
            if (rect.height() < self.config.min_dim_ratio * median_h or  # Default: 0.12 (12%)
                rect.width() < self.config.min_dim_ratio * median_w):
                pdebug(f"[Lab] Dropped thin panel: {rect.width():.0f}x{rect.height():.0f}pt vs median {median_w:.0f}x{median_h:.0f}pt")
                continue
        
        filtered_rects.append(rect)
    
    rects = filtered_rects
    if len(rects) < initial_count:
        pdebug(f"[Lab] Lab-based filter: {initial_count} -> {len(rects)} panels")
```

**Two filters in sequence:**
1. **Empty check**: `non_bg < 8%` → mostly background → DROP
2. **Thin check**: `height < 12% of median` OR `width < 12% of median` → likely gutter → DROP

---

## 4. NESTED RECTANGLE SUPPRESSION

### Location: `detect_panels()` lines 238-250

**Call site:**
```python
# ✅ Suppress nested rectangles (small rects inside larger ones)
if rects:
    # Save debug image BEFORE nested suppression
    if self.config.debug:
        self._save_debug_panels(img_bgr, rects, w, h, page_point_size, "before")
    
    # ✅ CALL: Remove nested empty rectangles
    rects = self._suppress_nested_rects(rects, img_bgr, bg_lab, w, h, page_point_size,
                                        delta=self.config.bg_delta)
    pdebug(f"After nested suppression -> {len(rects)} rects")
    
    # Save debug image AFTER nested suppression
    if self.config.debug:
        self._save_debug_panels(img_bgr, rects, w, h, page_point_size, "after")
```

### `_suppress_nested_rects()` - Lines 1359-1461

**Algorithm:**
```python
def _suppress_nested_rects(self, rects: List[QRectF], img_bgr: NDArray, 
                           bg_lab: NDArray, w: int, h: int, page_point_size: QSizeF,
                           delta: float = 12.0,
                           contain_thr: float = 0.90,      # ← 90% containment
                           area_ratio_thr: float = 0.15,   # ← small < 15% of big
                           empty_ratio_thr: float = 0.10   # ← < 10% content
                           ) -> List[QRectF]:
    """For each pair (small_rect, big_rect):
    1. Check if small is > 90% contained in big
    2. Check if small is < 15% the area of big
    3. If both true:
       - Calculate non_bg_ratio for small rect
       - If non_bg < 10% → REMOVE small (it's empty inside big)
       
    This catches: Panel #3 (bright band) inside Panel #2 (large panel)
    """
    to_remove = set()
    
    for i, small in enumerate(rects):
        if i in to_remove:
            continue
        
        for j, big in enumerate(rects):
            if i == j or j in to_remove:
                continue
            
            # Check size: small must be smaller than big
            small_area = small.width() * small.height()
            big_area = big.width() * big.height()
            
            if small_area >= big_area:
                continue  # Not nested
            
            # Check area ratio: small < 15% of big
            area_ratio = small_area / big_area
            if area_ratio > 0.15:
                continue  # Not small enough
            
            # Check containment: > 90% of small is inside big
            inter = small.intersected(big)
            inter_area = inter.width() * inter.height()
            containment = inter_area / small_area
            
            if containment >= 0.90:
                # ✅ Small rect is highly nested in big
                # Now check: is it mostly EMPTY (background)?
                non_bg = _non_bg_ratio(roi_bgr, bg_lab, delta)
                
                if non_bg < 0.10:  # Less than 10% content
                    # ✅ REMOVE: It's a false gutter inside a panel
                    to_remove.add(i)
                    pdebug(f"[Nested] Removed nested empty rect at (...): "
                          f"containment={containment:.2f}, non_bg={non_bg:.3f}")
                    break
    
    return [r for i, r in enumerate(rects) if i not in to_remove]
```

**Three thresholds working together:**
- `contain_thr = 0.90`: 90% must be inside (strict containment)
- `area_ratio_thr = 0.15`: Small must be < 15% the area of big (small enough to be "debris")
- `empty_ratio_thr = 0.10`: Must have < 10% content to be considered "empty"

---

## 5. DEBUG OUTPUT

### Call Sites in `detect_panels()`

**Line 558: Bright mask (binary threshold)**
```python
if self.config.debug:
    self._save_debug_image(binary, "dbg_bright_mask.png")
```

**Line 614: Gutter lines overlay**
```python
if self.config.debug:
    self._save_debug_gutters(L, h_lines, v_lines, w, h, "dbg_hv_gutters.png")
```

**Lines 241 & 249: Panel rectangles before/after nested suppression**
```python
if self.config.debug:
    self._save_debug_panels(img_bgr, rects, w, h, page_point_size, "before")
    # [after suppression]
    self._save_debug_panels(img_bgr, rects, w, h, page_point_size, "after")
```

**Output files (in `debug_output/` folder):**
```
debug_output/
├── dbg_bright_mask.png         ← Binary threshold (200 intensity)
├── dbg_hv_gutters.png          ← Red (H) + Blue (V) gutter lines
├── dbg_panels_before.png       ← Green rectangles before suppression
└── dbg_panels_after.png        ← Green rectangles after suppression
```

---

## PIPELINE SEQUENCE

```
1. Image → L-channel → Threshold (200) → binary_bright
                              ↓
                        [DEBUG: dbg_bright_mask.png]
                              ↓
2. H/V projections → find peaks → group into lines (RAW)
                              ↓
3. ✅ _validate_h_gutters(binary)   (check 85% coverage across WIDTH)
   ✅ _validate_v_gutters(binary)   (check 85% coverage across HEIGHT)
                              ↓
                        [DEBUG: dbg_hv_gutters.png]
                              ↓
4. Panels from gutter intersections
                              ↓
5. Filter empty (grayscale) → Filter by area → etc.
                              ↓
6. ✅ Lab-based filtering:
   - Remove if non_bg < 8%  (mostly background)
   - Remove if too thin (< 12% median)
                              ↓
7.                    [DEBUG: dbg_panels_before.png]
                              ↓
8. ✅ _suppress_nested_rects(img_bgr, bg_lab):
   - For each pair: check containment (>90%) + area (<15%) + empty (<10%)
   - Remove nested empty rects
                              ↓
9.                    [DEBUG: dbg_panels_after.png]
                              ↓
10. Sort by reading order → RETURN
```

---

## CONFIGURATION

### New config.py parameters:

```python
# Lab-based empty panel filtering (for tinted/watercolor pages)
min_non_bg_ratio: float = 0.08  # Minimum ratio of non-background pixels (Lab distance > bg_delta)
min_dim_ratio: float = 0.12     # Minimum dimension ratio vs median (to filter thin/gutter panels)
```

### Thresholds in `_suppress_nested_rects()`:

```python
contain_thr: float = 0.90       # 90% intersection/small_area
area_ratio_thr: float = 0.15    # small_area < 15% * big_area
empty_ratio_thr: float = 0.10   # non_bg < 10% = "empty"
```

---

## RESULT: Grémillet Sisters Page 6

| Stage | Count | Notes |
|-------|-------|-------|
| After gutter detection | 4 | Including false #3 |
| After Lab filtering | 3 | False #3 still there (not thin enough) |
| **After nested suppression** | **2** | ✅ False #3 removed (nested + empty) |

**The false vertical band (panel #3) inside the large panel (#2):**
- Has low non-bg ratio (mostly background color)
- Is smaller than panel #2
- Is highly contained (>90%) in panel #2
- **→ Gets removed by `_suppress_nested_rects()`**
