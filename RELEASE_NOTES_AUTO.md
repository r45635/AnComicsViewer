# 🎯 AnComicsViewer v2.0.0-5-gada3b5e - Multi-BD Revolution

## 📦 **Downloads**

Choose the right package for your operating system:

- 🪟 **Windows** (64-bit): `AnComicsViewer-Windows.zip`
- 🍎 **macOS** (Intel/Apple Silicon): `AnComicsViewer-macOS.tar.gz`  
- 🐧 **Linux** (64-bit): `AnComicsViewer-Linux.tar.gz`

## 🚀 **Quick Start**

### Windows
1. Download `AnComicsViewer-Windows.zip`
2. Extract the archive
3. Double-click `AnComicsViewer.exe`

### macOS
1. Download `AnComicsViewer-macOS.tar.gz`
2. Extract the archive  
3. Double-click `AnComicsViewer.app`

### Linux
1. Download `AnComicsViewer-Linux.tar.gz`
2. Extract: `tar -xzf AnComicsViewer-Linux.tar.gz`
3. Run: `./AnComicsViewer`

## ✨ **Features**

### 🤖 **AI-Powered Multi-BD Detection**
- **91.1% mAP50 accuracy** on multi-style comic dataset
- **3 BD styles supported**: Golden City, Tintin, Pin-up du B24
- **Real-time processing**: ~35ms per page
- **Zero configuration**: Optimized parameters out-of-the-box

### 📖 **Advanced Reading Experience**
- **Smart panel navigation** with cross-page support
- **Intelligent reading order** (AR-A to AR-E improvements)
- **Title zone detection** with complete chapter text handling
- **Panel-by-panel zoom** for immersive reading

### 🔧 **Technical Excellence**
- **Standalone executables**: No dependencies required
- **Cross-platform**: Windows, macOS, Linux support
- **Modular architecture**: Swappable detectors
- **Professional UI**: Native PySide6 interface

## 🎯 **What's New in vv2.0.0-5-gada3b5e**

### 🆕 **AR-A to AR-E Reading Improvements**
- ✅ Intelligent row grouping with `row_band_frac` parameter
- ✅ Natural reading order respecting panel layout
- ✅ Enhanced title zone detection (28% of page height)
- ✅ Franco-Belge preset optimization

### 🤖 **Multi-BD Enhanced Detector**
- ✅ Smart title/panel differentiation
- ✅ Optimized parameters (conf=0.15, iou=0.4)
- ✅ Advanced post-processing filters
- ✅ Real-time parameter tuning interface

### 🚀 **Model Retraining Success**
- ✅ 50 annotated images from 3 BD series
- ✅ 377 annotations with 2 classes (panel, panel_inset)
- ✅ Excellent performance: 91.1% mAP50, 88.3% mAP50-95
- ✅ Production-ready model deployment

## 🔧 **System Requirements**

**Standalone Executables (Recommended)**:
- No additional software required
- Estimated disk space: ~150-200 MB per platform

**From Source (Developers)**:
- Python 3.8+ 
- Operating System: Windows 10+, macOS 10.15+, Ubuntu 18.04+

## 🛠️ **Advanced Usage**

### 🎯 **Panel Detection Options**
1. **Heuristic (OpenCV)**: Fast, rule-based detection
2. **Multi-BD (Trained)**: AI-powered, high accuracy ⭐
3. **Multi-BD Enhanced**: With smart post-processing ✨

### ⚙️ **Customization**
- **Panel Tuning**: Fine-tune detection parameters
- **Reading Order**: Adjust row grouping tolerance
- **Presets**: Franco-Belge, Manga, Custom configurations

## 🐛 **Known Issues & Solutions**

### **If the app doesn't start**:
- Windows: Check Windows Defender/Antivirus settings
- macOS: Right-click → Open (bypass Gatekeeper)
- Linux: Ensure executable permissions: `chmod +x AnComicsViewer`

### **Performance tips**:
- Use SSD storage for best performance
- Close other applications for large PDF files
- Use Multi-BD Enhanced for best accuracy

## 📊 **Performance Metrics**

| Metric | Value | Description |
|--------|-------|-------------|
| **Accuracy** | 91.1% mAP50 | Panel detection precision |
| **Speed** | ~35ms/page | Processing time |
| **Styles** | 3 BD series | Supported comic styles |
| **Size** | ~150MB | Standalone executable |

## 🤝 **Support & Feedback**

- 📘 **Documentation**: [GitHub Wiki](https://github.com/r45635/AnComicsViewer/wiki)
- 🐛 **Bug Reports**: [Issues](https://github.com/r45635/AnComicsViewer/issues)
- 💡 **Feature Requests**: [Discussions](https://github.com/r45635/AnComicsViewer/discussions)
- 📧 **Contact**: Create an issue for support

## 🏆 **Credits**

- **YOLOv8** (Ultralytics) - Object detection framework
- **PySide6** (Qt) - Cross-platform GUI framework  
- **OpenCV** - Computer vision library
- **PyTorch** - Machine learning backend

---

**Build Info**: 
- Version: `v2.0.0-5-gada3b5e`
- Commit: `ada3b5e`
- Built with: GitHub Actions automated pipeline

🎯 **AnComicsViewer - Revolutionizing comic reading with AI!** 🚀