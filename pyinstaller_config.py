# Configuration optimisée pour PyInstaller
# Fichier de hook personnalisé pour AnComicsViewer

# Modules à précharger pour PySide6
hiddenimports = [
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtWidgets', 
    'PySide6.QtPdf',
    'PySide6.QtPdfWidgets',
    'PySide6.QtOpenGL',
    'PySide6.QtSvg',
]

# Exclusions pour réduire la taille
excludedimports = [
    'tkinter',
    'turtle', 
    'matplotlib.tests',
    'torch.test',
    'test',
    'unittest',
    'doctest',
    'pdb',
    'profile',
    'pstats',
    'scipy',  # Si pas utilisé
    'pandas', # Si pas utilisé
]

# Optimisations UPX (compression)
upx_exclude = [
    'vcruntime140.dll',  # Windows
    'python3.dll',       # Windows
    'QtCore.dll',        # Qt libs
    'QtGui.dll',
    'QtWidgets.dll',
]
