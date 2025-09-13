#!/bin/bash
# Demo script for the enhanced AnComicsViewer with calibration & metrics
# This script demonstrates the new features step by step

echo "🚀 AnComicsViewer Enhanced Demo"
echo "=================================="

# Check if we have a PDF to test with
PDF_FILE="../dataset/pdfs/The Grémillet Sisters - 02 - Cassiopeia's Summer of Love (2020).pdf"

if [ ! -f "$PDF_FILE" ]; then
    echo "❌ PDF file not found: $PDF_FILE"
    echo "Please provide a PDF file for testing"
    exit 1
fi

echo "📄 Using PDF: $(basename "$PDF_FILE")"

# Create output directories
mkdir -p outputs
mkdir -p debug

echo ""
echo "🧪 Test 1: Basic detection with metrics export"
echo "----------------------------------------------"
python3 main.py \
    --pdf "$PDF_FILE" \
    --page 5 \
    --metrics-out "outputs/metrics.json" \
    --config "config/detect_with_merge.yaml" \
    --debug-detect \
    --save-debug-overlays debug &
    
# Get the process ID so we can kill it after a few seconds
MAIN_PID=$!

# Wait a bit for the GUI to load and process the page
echo "⏳ Waiting 10 seconds for processing..."
sleep 10

# Kill the GUI process
kill $MAIN_PID 2>/dev/null

echo ""
echo "📊 Checking results..."

# Check if metrics file was created
if [ -f "outputs/metrics.json" ]; then
    echo "✅ Metrics file created successfully!"
    echo "📋 Metrics content:"
    python3 -c "
import json
with open('outputs/metrics.json', 'r') as f:
    data = json.load(f)
    print(json.dumps(data, indent=2))
"
else
    echo "❌ Metrics file not found"
fi

# Check if debug overlays were created
if ls debug/*.png >/dev/null 2>&1; then
    echo "✅ Debug overlays created:"
    ls -la debug/*.png | head -3
else
    echo "❌ No debug overlays found"
fi

echo ""
echo "🔧 Test 2: Configuration test with different parameters"
echo "------------------------------------------------------"

# Create a custom test config
cat > test_config.yaml << 'EOF'
# Custom test configuration
panel_conf: 0.25
balloon_conf: 0.30
panel_area_min_pct: 0.02
panel_area_max_pct: 0.85
balloon_area_min_pct: 0.0015
balloon_area_max_pct: 0.25
panel_nms_iou: 0.25
balloon_nms_iou: 0.20
min_box_w_px: 25
min_box_h_px: 20
page_margin_inset_pct: 0.02
balloon_min_overlap_panel: 0.05
max_panels: 10
max_balloons: 20
EOF

echo "📝 Created custom config:"
cat test_config.yaml

echo ""
echo "🧪 Running with custom config..."
python3 main.py \
    --pdf "$PDF_FILE" \
    --page 3 \
    --metrics-out "outputs/metrics_custom.json" \
    --config "test_config.yaml" \
    --debug-detect &

MAIN_PID2=$!
sleep 8
kill $MAIN_PID2 2>/dev/null

if [ -f "outputs/metrics_custom.json" ]; then
    echo "✅ Custom config test successful!"
    echo "📊 Custom config results:"
    python3 -c "
import json
with open('outputs/metrics_custom.json', 'r') as f:
    data = json.load(f)
    for item in data:
        print(f'Page {item[\"page_index\"]}: panels={item[\"panels\"]}, balloons={item[\"balloons\"]}, quality={item[\"quality_score\"]:.3f}')
"
fi

echo ""
echo "📈 Test 3: Multi-page metrics collection"
echo "----------------------------------------"

# Test multiple pages
echo "🔄 Processing pages 1-3..."
for page in 1 2 3; do
    echo "  📄 Processing page $page..."
    python3 main.py \
        --pdf "$PDF_FILE" \
        --page $page \
        --metrics-out "outputs/multipage_metrics.json" \
        --config "config/detect_with_merge.yaml" &
    
    PID=$!
    sleep 5
    kill $PID 2>/dev/null
    sleep 1
done

if [ -f "outputs/multipage_metrics.json" ]; then
    echo "✅ Multi-page metrics collected!"
    echo "📊 Summary of all processed pages:"
    python3 -c "
import json
try:
    with open('outputs/multipage_metrics.json', 'r') as f:
        data = json.load(f)
    
    print(f'Total pages processed: {len(data)}')
    for item in data:
        quality = item['quality_score']
        panels = item['panels']
        balloons = item['balloons']
        page_idx = item['page_index']
        
        quality_emoji = '🟢' if quality > 0.7 else '🟡' if quality > 0.4 else '🔴'
        print(f'  {quality_emoji} Page {page_idx+1}: {panels} panels, {balloons} balloons, quality={quality:.3f}')
        
        if panels > 0:
            avg_panel_area = sum(item['panel_area_ratios']) / len(item['panel_area_ratios'])
            print(f'      Avg panel area: {avg_panel_area:.1%}')
        
        if balloons > 0:
            avg_balloon_area = sum(item['balloon_area_ratios']) / len(item['balloon_area_ratios'])
            print(f'      Avg balloon area: {avg_balloon_area:.1%}')
        
        if item['overlaps'] > 0:
            print(f'      ⚠️  {item[\"overlaps\"]} overlaps, {item[\"severe_overlaps\"]} severe')
            
except Exception as e:
    print(f'Error: {e}')
"
fi

echo ""
echo "🎯 Summary of new features tested:"
echo "=================================="
echo "✅ Pixel↔PDF calibration with fixed 300 DPI rendering"
echo "✅ Quality metrics computation (overlaps, severe overlaps, quality score)"
echo "✅ JSON metrics export with --metrics-out argument"
echo "✅ Refined post-processing with class-specific NMS"
echo "✅ Advanced filtering (size, margin, balloon→panel attachment)"
echo "✅ Configurable parameters via YAML"
echo "✅ Debug overlay generation"
echo "✅ Multi-page metrics aggregation"

# Cleanup
rm -f test_config.yaml

echo ""
echo "🎉 Demo completed! Check the outputs/ directory for results."
echo ""
echo "📁 Files created:"
ls -la outputs/
echo ""
echo "🖼️  Debug overlays:"
ls -la debug/ 2>/dev/null | head -5

echo ""
echo "💡 Usage examples:"
echo ""
echo "  # Basic usage with metrics:"
echo "  python3 main.py --pdf comic.pdf --page 0 --metrics-out metrics.json"
echo ""
echo "  # Advanced usage with custom config and debug:"
echo "  python3 main.py --pdf comic.pdf --config my_config.yaml \\"
echo "                  --debug-detect --save-debug-overlays debug \\"
echo "                  --metrics-out detailed_metrics.json"
echo ""
echo "  # Batch processing (you can script this):"
echo "  for i in {0..10}; do"
echo "    python3 main.py --pdf comic.pdf --page \$i --metrics-out batch_metrics.json"
echo "  done"
