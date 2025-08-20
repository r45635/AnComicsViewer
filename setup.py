#!/usr/bin/env python3
"""
Setup script pour AnComicsViewer
===============================
"""

from pathlib import Path
from setuptools import setup, find_packages

# Lire le README
HERE = Path(__file__).parent
README = (HERE / "README.md").read_text(encoding='utf-8')

# Lire les requirements
def read_requirements(filename):
    """Lit les requirements depuis un fichier"""
    req_file = HERE / filename
    if req_file.exists():
        return req_file.read_text().strip().split('\n')
    return []

setup(
    name="ancomicsviewer",
    version="2.0.0",
    description="Lecteur PDF intelligent pour bandes dessinées avec détection ML de cases",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/r45635/AnComicsViewer",
    author="Vincent Cruvellier",
    author_email="vincent@example.com",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Multimedia :: Graphics :: Viewers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="pdf comics viewer panel detection machine learning",
    
    # Structure des packages
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    
    # Requirements
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "ml": read_requirements("requirements-ml.txt"),
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
    },
    
    # Points d'entrée
    entry_points={
        "console_scripts": [
            "ancomicsviewer=ancomicsviewer:main",
            "ancomicsviewer-cli=main:main",
        ],
    },
    
    # Scripts directs (compatible avec l'approche traditionnelle)
    scripts=["main.py"],
    
    # Données du package
    package_data={
        "ancomicsviewer": ["assets/*"],
    },
    include_package_data=True,
    
    # Métadonnées supplémentaires
    project_urls={
        "Bug Reports": "https://github.com/r45635/AnComicsViewer/issues",
        "Source": "https://github.com/r45635/AnComicsViewer",
        "Documentation": "https://github.com/r45635/AnComicsViewer/tree/main/docs",
    },
)
