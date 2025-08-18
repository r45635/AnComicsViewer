#!/usr/bin/env python3
"""
Configuration et outils pour les releases automatiques d'AnComicsViewer
"""

import os
import subprocess
from pathlib import Path
import json

def get_version_info():
    """Récupère les informations de version depuis Git."""
    try:
        # Version depuis git describe
        result = subprocess.run(['git', 'describe', '--tags', '--always'], 
                              capture_output=True, text=True, check=True)
        git_version = result.stdout.strip()
        
        # Version propre (sans commit hash si on est sur un tag)
        try:
            result = subprocess.run(['git', 'describe', '--tags', '--exact-match'], 
                                  capture_output=True, text=True, check=True)
            clean_version = result.stdout.strip()
        except subprocess.CalledProcessError:
            clean_version = git_version
        
        # Commit hash court
        result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], 
                              capture_output=True, text=True, check=True)
        commit_hash = result.stdout.strip()
        
        return {
            'git_version': git_version,
            'clean_version': clean_version,
            'commit_hash': commit_hash,
            'is_release': git_version == clean_version
        }
        
    except subprocess.CalledProcessError:
        return {
            'git_version': 'unknown',
            'clean_version': 'v2.0.0',
            'commit_hash': 'unknown',
            'is_release': False
        }

def generate_release_notes(version_info):
    """Génère les notes de release automatiques."""
    
    notes = f"""# 🎯 AnComicsViewer {version_info['clean_version']} - Multi-BD Revolution

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

## 🎯 **What's New in v{version_info['clean_version']}**

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
- Version: `{version_info['git_version']}`
- Commit: `{version_info['commit_hash']}`
- Built with: GitHub Actions automated pipeline

🎯 **AnComicsViewer - Revolutionizing comic reading with AI!** 🚀"""

    return notes

def create_release_config():
    """Crée la configuration pour les releases automatiques."""
    
    version_info = get_version_info()
    
    config = {
        "name": "AnComicsViewer",
        "version": version_info['clean_version'],
        "git_version": version_info['git_version'],
        "commit_hash": version_info['commit_hash'],
        "is_release": version_info['is_release'],
        "assets": [
            {
                "name": "AnComicsViewer-Windows.zip",
                "path": "AnComicsViewer-Windows.zip",
                "content_type": "application/zip",
                "platform": "Windows"
            },
            {
                "name": "AnComicsViewer-macOS.tar.gz", 
                "path": "AnComicsViewer-macOS.tar.gz",
                "content_type": "application/gzip",
                "platform": "macOS"
            },
            {
                "name": "AnComicsViewer-Linux.tar.gz",
                "path": "AnComicsViewer-Linux.tar.gz", 
                "content_type": "application/gzip",
                "platform": "Linux"
            }
        ],
        "release_notes": generate_release_notes(version_info)
    }
    
    return config

def save_release_config():
    """Sauvegarde la configuration de release."""
    config = create_release_config()
    
    config_file = Path("release_config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Configuration de release sauvée: {config_file}")
    
    # Sauvegarder aussi les notes de release
    notes_file = Path("RELEASE_NOTES_AUTO.md")
    with open(notes_file, 'w', encoding='utf-8') as f:
        f.write(config['release_notes'])
    
    print(f"✅ Notes de release sauvées: {notes_file}")
    
    return config

def main():
    """Point d'entrée principal."""
    print("🎉 GÉNÉRATION CONFIGURATION RELEASE")
    print("=" * 40)
    
    config = save_release_config()
    
    print(f"\n📋 Informations de version:")
    print(f"  • Version: {config['version']}")
    print(f"  • Git version: {config['git_version']}")
    print(f"  • Commit: {config['commit_hash']}")
    print(f"  • Is release: {config['is_release']}")
    
    print(f"\n📦 Assets configurés:")
    for asset in config['assets']:
        print(f"  • {asset['name']} ({asset['platform']})")
    
    print(f"\n🚀 Configuration prête pour GitHub Actions!")

if __name__ == "__main__":
    main()
