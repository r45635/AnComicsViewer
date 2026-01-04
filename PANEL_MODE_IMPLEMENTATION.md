# Panel Detection Mode Implementation Summary

## Overview

Implemented a configurable panel detection system with three modes to address regressions in Tintin-style comics while improving modern watercolor page detection.

## Changes Made

### 1. Configuration (`config.py`)
- Added `panel_mode` parameter to `DetectorConfig`
- Values: `"auto"` (default), `"classic_franco_belge"`, `"modern"`
- Included in `to_dict()` serialization

### 2. CLI Interface (`AnComicsViewer.py`)
- Added `--panel-mode` argument
- Accepts: `auto`, `classic_franco_belge`, `modern`
- Stores choice in `ANCOMICS_PANEL_MODE` environment variable
- Example: `python AnComicsViewer.py --panel-mode classic_franco_belge my.pdf`

### 3. Panel Detector (`detector.py`)

#### Page Style Classifier (`_classify_page_style`)
Conservative classifier for AUTO mode that analyzes:
- **Rect count** and **small rect ratio**
- **Margin rect ratio** (page numbers/titles)
- **Gutter strength** (validated gutter count)
- **Background tint** (Lab distance from pure white)

Classification criteria (prefers classic when uncertain):
- Classic if: gutter_count ≥ 4, small_ratio < 0.25, 3-12 panels, white background
- Otherwise: modern

#### Detection Policies

**Classic Franco-Belge Policy** (`_classic_detection_policy`):
- Tintin-safe, preserves existing behavior
- Uses adaptive + gutter-based detection
- NO header/footer strip removal
- NO freeform watershed
- Keeps existing filtering

**Modern Policy** (`_modern_detection_policy`):
- For watercolor/complex layouts
- Uses adaptive + gutter-based
- **Enables** header/footer strip removal
- **Triggers** freeform for weak candidates
- Uses conservative selection with mega-panel prevention

#### Modern Selection Logic (`_select_best_route_modern`)
Prevents mega-panel collapse and applies strict criteria:

**Mega-panel check**: Reject freeform if ≤2 panels and any rect > 60% page area

**Candidate validity**:
- Coverage ≥ 0.55
- Count between 2-12
- Small ratio < 0.60

**Selection criteria** (prefer freeform only if):
- Valid candidate
- No mega-panel collapse
- Coverage improvement (≥ +0.07) OR significant small_ratio reduction (≥ -0.15)

#### Debug Output (`_export_decision_json`)
Writes `debug_output/decision.json` with:
- `panel_mode_input` and `panel_mode_used`
- `route_chosen` (adaptive/gutter/freeform)
- Final metrics: count, coverage, small_ratio, margin_ratio
- Candidate metrics for both adaptive and freeform
- Freeform mega-panel flag and selection status

### 4. Main Window (`main_window.py`)
- Reads `ANCOMICS_PANEL_MODE` environment variable
- Passes to `DetectorConfig` on initialization

### 5. Regression Test (`tests/scripts/regress_panels.py`)
Tests both classic and modern pages in all three modes:
- Validates AUTO classification
- Checks panel counts and metrics
- Exports results to `debug_output/regression_results.json`

Usage:
```bash
python tests/scripts/regress_panels.py <pdf> <classic_page> <modern_page>
```

## Usage Examples

### Classic Comics (Tintin)
```bash
# Force classic mode
python AnComicsViewer.py --panel-mode classic_franco_belge tintin.pdf

# Auto (should classify as classic)
python AnComicsViewer.py --panel-mode auto tintin.pdf
```

### Modern Comics (Watercolor)
```bash
# Force modern mode  
python AnComicsViewer.py --panel-mode modern gremillet.pdf

# Auto (may classify as modern)
python AnComicsViewer.py --panel-mode auto gremillet.pdf
```

### Regression Testing
```bash
# Test both page types
python tests/scripts/regress_panels.py "samples_PDF/Comic.pdf" 4 18

# Check decision.json for details
cat debug_output/decision.json
```

## Safety Guarantees

1. **Default AUTO is conservative**: Prefers classic when uncertain
2. **Classic mode unchanged**: Preserves exact Tintin behavior
3. **Modern mega-panel prevention**: Rejects 1-2 huge panel collapses
4. **Strict selection criteria**: Freeform only chosen with significant improvement
5. **Debug visibility**: decision.json shows reasoning for every choice

## Testing Checklist

- [ ] Tintin page with AUTO → classified as classic
- [ ] Tintin page with CLASSIC → same count as before
- [ ] Modern page with AUTO → may be classic or modern
- [ ] Modern page with MODERN → better coverage, fewer small rects
- [ ] decision.json created in debug mode
- [ ] regression_results.json produced by test script

## Metrics Reference

- **Coverage**: Union of all panels / page area
- **Small ratio**: Fraction of panels < 0.5% page area
- **Margin ratio**: Fraction of panels in top/bottom 6% with height < 12%
- **Gutter count**: Number of validated gutter runs (horizontal + vertical)
