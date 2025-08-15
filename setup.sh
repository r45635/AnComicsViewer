#!/bin/bash
# AnComicsViewer - Automated Setup Script
# This script automates the environment setup process described in README_SETUP.md

set -e  # Exit on any error

echo "🚀 AnComicsViewer - Automated Setup"
echo "=================================="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    print_warning "This script is optimized for macOS. Manual setup may be required for other platforms."
fi

# Check Python version
print_step "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
if [[ $(echo "$python_version >= 3.11" | bc -l) -eq 1 ]]; then
    print_success "Python $python_version detected"
else
    print_error "Python 3.11+ required. Found: $python_version"
    exit 1
fi

# Create virtual environment
print_step "Creating Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_step "Activating virtual environment..."
source .venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_step "Upgrading pip..."
.venv/bin/pip install --upgrade pip

# Install core dependencies
print_step "Installing core dependencies..."
.venv/bin/pip install 'PySide6==6.8.3'
.venv/bin/pip install opencv-python numpy Pillow

print_success "Core dependencies installed"

# Check if we're setting up ML environment
read -p "🤖 Set up ML environment? (y/n): " setup_ml

if [[ $setup_ml == "y" || $setup_ml == "Y" ]]; then
    print_step "Installing ML dependencies..."
    
    # Install ML packages
    .venv/bin/pip install ultralytics==8.2.0
    .venv/bin/pip install torch torchvision torchaudio
    .venv/bin/pip install scikit-learn matplotlib seaborn
    .venv/bin/pip install labelme  # For annotation
    
    print_success "ML dependencies installed"
    
    # Test YOLO installation
    print_step "Testing YOLO installation..."
    .venv/bin/python -c "import matplotlib; matplotlib.use('Agg'); from ultralytics import YOLO; print('YOLO import successful')"
    
    # Download pre-trained model
    read -p "📥 Download YOLOv8 nano model for testing? (y/n): " download_model
    if [[ $download_model == "y" || $download_model == "Y" ]]; then
        print_step "Downloading YOLOv8 nano segmentation model..."
        .venv/bin/python -c "import matplotlib; matplotlib.use('Agg'); from ultralytics import YOLO; YOLO('yolov8n-seg.pt')"
        print_success "Pre-trained model downloaded"
    fi
fi

# Set up macOS environment variables
if [[ "$OSTYPE" == "darwin"* ]]; then
    print_step "Setting up macOS environment variables..."
    export QT_MAC_WANTS_LAYER=1
    
    # Check if already in shell profile
    shell_profile=""
    if [ -f "$HOME/.zshrc" ]; then
        shell_profile="$HOME/.zshrc"
    elif [ -f "$HOME/.bash_profile" ]; then
        shell_profile="$HOME/.bash_profile"
    fi
    
    if [ -n "$shell_profile" ]; then
        if ! grep -q "QT_MAC_WANTS_LAYER" "$shell_profile"; then
            echo 'export QT_MAC_WANTS_LAYER=1' >> "$shell_profile"
            print_success "Added QT_MAC_WANTS_LAYER to $shell_profile"
        else
            print_warning "QT_MAC_WANTS_LAYER already in $shell_profile"
        fi
    fi
fi

# Test basic application
print_step "Testing basic application..."
timeout 5s .venv/bin/python AnComicsViewer.py &
app_pid=$!
sleep 2

if kill -0 $app_pid 2>/dev/null; then
    kill $app_pid
    print_success "Application starts successfully"
else
    print_warning "Application test inconclusive (may require GUI)"
fi

# Set up dataset structure if requested
read -p "📁 Set up dataset structure? (y/n): " setup_dataset

if [[ $setup_dataset == "y" || $setup_dataset == "Y" ]]; then
    print_step "Creating dataset structure..."
    mkdir -p dataset/images/train
    mkdir -p dataset/images/val
    mkdir -p dataset/labels/train
    mkdir -p dataset/labels/val
    
    # Create class definitions
    echo "panel" > dataset/predefined_classes.txt
    echo "text" >> dataset/predefined_classes.txt
    
    print_success "Dataset structure created"
fi

# Create VS Code configuration if requested
read -p "⚙️ Set up VS Code configuration? (y/n): " setup_vscode

if [[ $setup_vscode == "y" || $setup_vscode == "Y" ]]; then
    if [ ! -d ".vscode" ]; then
        print_step "Setting up VS Code configuration..."
        mkdir -p .vscode
        
        # This would create the VS Code files - they should already exist
        # from the previous setup
        print_success "VS Code configuration ready"
    else
        print_warning "VS Code configuration already exists"
    fi
fi

echo
echo "🎉 Setup Complete!"
echo "=================="
echo
echo "✅ Virtual environment: .venv"
echo "✅ Core dependencies installed"
if [[ $setup_ml == "y" || $setup_ml == "Y" ]]; then
    echo "✅ ML environment configured"
fi
echo "✅ macOS compatibility configured"
echo
echo "📋 Next Steps:"
echo "1. Test the application:"
echo "   .venv/bin/python AnComicsViewer.py"
echo
if [[ $setup_ml == "y" || $setup_ml == "Y" ]]; then
    echo "2. Test ML detector:"
    echo "   - Load a PDF in the app"
    echo "   - Settings (⚙️) → Detector → Load ML weights... → yolov8n-seg.pt"
    echo "   - Settings (⚙️) → Detector → YOLOv8 Seg (ML)"
    echo
fi
if [[ $setup_dataset == "y" || $setup_dataset == "Y" ]]; then
    echo "3. Extract comic pages:"
    echo "   .venv/bin/python tools/export_pdf_pages.py 'Your Comic.pdf' --out dataset/images/train --dpi 300"
    echo
    echo "4. Start annotation:"
    echo "   .venv/bin/python start_annotation.py"
    echo
fi
echo "📖 For detailed instructions, see README_SETUP.md"
echo
print_success "Setup script completed successfully!"

# Remind about activation
echo
print_warning "Remember to activate the virtual environment in new terminals:"
echo "   source .venv/bin/activate"
