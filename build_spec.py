#!/usr/bin/env python3
"""
Configuration PyInstaller pour AnComicsViewer
Génère des exécutables standalone cross-platform
"""
import os
import sys
from pathlib import Path

# Configuration de base
APP_NAME = "AnComicsViewer"
MAIN_SCRIPT = "main.py"
ICON_PATH = "icon.ico"  # Windows
ICON_PATH_MAC = "assets/icon.png"  # macOS

def get_hidden_imports():
    """Liste des imports cachés nécessaires."""
    return [
        # PySide6 modules
        'PySide6.QtCore',
        'PySide6.QtGui', 
        'PySide6.QtWidgets',
        'PySide6.QtPdf',
        'PySide6.QtPdfWidgets',
        
        # OpenCV
        'cv2',
        
        # NumPy
        'numpy',
        'numpy.core._methods',
        'numpy.lib.format',
        
        # PIL/Pillow
        'PIL',
        'PIL.Image',
        
        # ML libraries (optionnels mais inclus)
        'torch',
        'torchvision',
        'ultralytics',
        'matplotlib',
        'matplotlib.backends.backend_qt5agg',
        
        # Détecteurs
        'detectors.base',
        'detectors.multibd_detector',
        'detectors.yolo_seg',
        
        # Autres
        'pathlib',
        'json',
        'subprocess',
        'platform'
    ]

def get_data_files():
    """Fichiers de données à inclure."""
    data_files = []
    
    # Assets (icônes, images)
    if Path("assets").exists():
        data_files.append(("assets", "assets"))
    
    # Modèles ML pré-entraînés
    if Path("runs/detect/multibd_mixed_model/weights/best.pt").exists():
        data_files.append((
            "runs/detect/multibd_mixed_model/weights/best.pt",
            "runs/detect/multibd_mixed_model/weights/"
        ))
    
    # Fichiers de configuration
    config_files = [
        "requirements.txt",
        "requirements-ml.txt", 
        "patch_pytorch.py"
    ]
    for config_file in config_files:
        if Path(config_file).exists():
            data_files.append((config_file, "."))
    
    # Documentation essentielle
    doc_files = [
        "README.md",
        "LICENSE"
    ]
    for doc_file in doc_files:
        if Path(doc_file).exists():
            data_files.append((doc_file, "."))
    
    return data_files

def get_exclude_modules():
    """Modules à exclure pour réduire la taille."""
    return [
        # Modules de développement
        'pytest',
        'setuptools', 
        'wheel',
        'pip',
        
        # Modules optionnels lourds
        'scipy',
        'pandas',
        'jupyter',
        'notebook',
        
        # Modules système non nécessaires
        'tkinter',
        'turtle',
        'test',
        'unittest',
        'doctest',
        'pdb',
        'profile',
        'pstats'
    ]

def create_spec_file():
    """Génère le fichier .spec pour PyInstaller."""
    
    # Détecter la plateforme
    is_windows = sys.platform.startswith('win')
    is_macos = sys.platform == 'darwin'
    is_linux = sys.platform.startswith('linux')
    
    # Icon selon la plateforme
    icon = None
    if is_windows and Path(ICON_PATH).exists():
        icon = ICON_PATH
    elif is_macos and Path(ICON_PATH_MAC).exists():
        icon = ICON_PATH_MAC
    
    spec_content = f'''# -*- mode: python ; coding: utf-8 -*-
# Fichier de configuration PyInstaller généré automatiquement pour {APP_NAME}

import sys
from pathlib import Path

# Configuration de base
a = Analysis(
    ['{MAIN_SCRIPT}'],
    pathex=['.'],
    binaries=[],
    datas={get_data_files()!r},
    hiddenimports={get_hidden_imports()!r},
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes={get_exclude_modules()!r},
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# Exclusions supplémentaires pour optimiser la taille
a.datas = [x for x in a.datas if not x[0].startswith('matplotlib/tests/')]
a.datas = [x for x in a.datas if not x[0].startswith('torch/test/')]
a.datas = [x for x in a.datas if not x[0].startswith('PIL/tests/')]

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# Configuration de l'exécutable
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='{APP_NAME}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # Compression UPX si disponible
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Application graphique
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    {'icon=' + repr(icon) + ',' if icon else ''}
)
'''

    # Ajout spécifique macOS (bundle .app)
    if is_macos:
        spec_content += f'''
# Bundle macOS .app
app = BUNDLE(
    exe,
    name='{APP_NAME}.app',
    icon='{ICON_PATH_MAC}',
    bundle_identifier='com.ancomicsviewer.app',
    info_plist={{
        'CFBundleName': '{APP_NAME}',
        'CFBundleDisplayName': '{APP_NAME}',
        'CFBundleVersion': '2.0.0',
        'CFBundleShortVersionString': '2.0.0',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '10.15',  # macOS Catalina minimum
    }},
)
'''
    
    return spec_content

def main():
    """Génère le fichier de configuration PyInstaller."""
    spec_content = create_spec_file()
    
    spec_filename = f"{APP_NAME}.spec"
    with open(spec_filename, 'w', encoding='utf-8') as f:
        f.write(spec_content)
    
    print(f"✅ Fichier {spec_filename} généré avec succès")
    print(f"📦 Données incluses: {len(get_data_files())} fichiers/dossiers")
    print(f"🔒 Imports cachés: {len(get_hidden_imports())} modules")
    print(f"🚫 Modules exclus: {len(get_exclude_modules())} modules")
    
    print(f"\n🚀 Pour construire l'exécutable:")
    print(f"   pyinstaller {spec_filename}")

if __name__ == "__main__":
    main()
