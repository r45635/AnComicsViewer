#!/bin/bash
# Optimized training script for AnComicsViewer
# Generated after manual annotation review

echo "🚀 Starting Optimized YOLO Training"
echo "==================================="

# Check YOLO installation
if ! command -v yolo &> /dev/null; then
    echo "❌ YOLO CLI not found. Install with: pip install ultralytics"
    exit 1
fi

# Run training
#!/bin/bash
# Final Optimized Training Script for AnComicsViewer
# Based on manual annotation review and dataset analysis

echo "🚀 Starting FINAL Optimized YOLO Training"
echo "=========================================="

# Check YOLO installation
if ! command -v yolo &> /dev/null; then
    echo "❌ YOLO CLI not found. Install with: pip install ultralytics"
    exit 1
fi

# Check dataset exists
if [ ! -d "dataset/images/train" ]; then
    echo "❌ Dataset not found. Please ensure dataset is properly set up."
    exit 1
fi

# Validate dataset consistency
echo "🔍 Validating dataset consistency..."
python3 check_dataset.py
if [ $? -ne 0 ]; then
    echo "❌ Dataset validation failed. Please fix dataset issues first."
    exit 1
fi

# Validate annotations
echo "🔍 Validating annotations..."
python3 fix_annotations.py
if [ $? -ne 0 ]; then
    echo "❌ Annotation validation failed."
    exit 1
fi

# Display dataset info
echo ""
echo "📊 Dataset Information:"
echo "   Training images: $(find dataset/images/train -name "*.jpg" -o -name "*.png" | wc -l)"
echo "   Validation images: $(find dataset/images/val -name "*.jpg" -o -name "*.png" | wc -l)"
echo "   Training labels: $(find dataset/labels/train -name "*.txt" | wc -l)"
echo "   Validation labels: $(find dataset/labels/val -name "*.txt" | wc -l)"

# Configuration summary
echo ""
echo "🔧 Training Configuration:"
echo "   Model: YOLOv8m (medium)"
echo "   Epochs: 120"
echo "   Batch size: 6"
echo "   Image size: 1280px"
echo "   Optimizer: AdamW"
echo "   Device: MPS (Apple Silicon)"

# Create logs directory
LOG_DIR="training_logs"
mkdir -p "$LOG_DIR"

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"

# Run training with optimized parameters
echo ""
echo "🎯 Starting training..."
echo "📝 Training log will be saved to: $LOG_FILE"

# Run training and capture output to both terminal and log file
yolo train model=yolov8m.pt data=/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/dataset/multibd_enhanced.yaml epochs=120 imgsz=1280 batch=6 name=ancomics_final_optimized save=True save_period=20 cache=False workers=0 device=mps optimizer=AdamW lr0=0.0008 lrf=0.01 momentum=0.9 weight_decay=0.0004 box=6.0 cls=0.4 dfl=1.2 patience=25 seed=42 deterministic=True close_mosaic=10 mixup=0.05 copy_paste=0.05 warmup_epochs=2.0 warmup_momentum=0.8 warmup_bias_lr=0.08 amp=True plots=True verbose=True val=False 2>&1 | tee "$LOG_FILE"

# Save training completion status
TRAIN_EXIT_CODE=${PIPESTATUS[0]}
echo "TRAINING_EXIT_CODE: $TRAIN_EXIT_CODE" >> "$LOG_FILE"
echo "TRAINING_END_TIME: $(date)" >> "$LOG_FILE"

echo ""
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "📁 Results: runs/detect/ancomics_final_optimized/"
    
    # Check if best model was saved
    if [ -f "runs/detect/ancomics_final_optimized/weights/best.pt" ]; then
        echo "🏆 Best model saved: runs/detect/ancomics_final_optimized/weights/best.pt"
        echo "📊 Run validation with: yolo val model=runs/detect/ancomics_final_optimized/weights/best.pt data=dataset/multibd_enhanced.yaml"
    else
        echo "⚠️  Warning: Best model not found. Check training logs for issues."
    fi
else
    echo "❌ Training failed with exit code: $TRAIN_EXIT_CODE"
    echo "� Check the log file for details: $LOG_FILE"
fi

echo ""
echo "📝 Complete training log saved to: $LOG_FILE"
echo "🔍 Monitor training progress with: ./monitor_training.sh $LOG_FILE"
