# Interface en Ligne de Commande AnComicsViewer

AnComicsViewer propose une interface en ligne de commande complète permettant de contrôler l'application via des arguments et des variables d'environnement.

## 🚀 Utilisation de Base

```bash
# Lancement normal de l'interface graphique
python main.py

# Ouvrir un fichier PDF spécifique
python main.py moncomics.pdf

# Afficher l'aide complète
python main.py --help

# Afficher la version
python main.py --version
```

## 📋 Arguments de Ligne de Commande

### Options de Configuration

- `--preset {fb,manga,newspaper}` : Preset de détection optimisé
  - `fb` : Franco-Belge (BD européennes classiques)
  - `manga` : Style japonais (lecture RTL)
  - `newspaper` : Comics US/newspapers

- `--detector {heur,yolo,multibd}` : Type de détecteur
  - `heur` : Détecteur heuristique (rapide, léger)
  - `yolo` : Détecteur YOLO (précis, nécessite modèle)
  - `multibd` : Multi-BD Enhanced (dernière génération)

- `--dpi N` : Résolution de détection (100-400, défaut: 200)
- `--page N` : Page de démarrage (0-based, défaut: 0)

### Fichier PDF

- `pdf_file` : Chemin vers le fichier PDF à ouvrir au démarrage

## 🌍 Variables d'Environnement

Les variables d'environnement offrent une alternative aux arguments CLI et peuvent être utiles pour la configuration par défaut ou l'intégration dans des scripts.

### Variables Supportées

- `ANCOMICS_PRESET` : Preset de détection (`fb`, `manga`, `newspaper`)
- `ANCOMICS_DETECTOR` : Type de détecteur (`heur`, `yolo`, `multibd`)
- `ANCOMICS_DPI` : Résolution de détection (100-400)
- `ANCOMICS_PDF` : Chemin du fichier PDF à ouvrir
- `ANCOMICS_PAGE` : Page de démarrage (0-based)

### Ordre de Priorité

1. **Arguments CLI** (priorité maximale)
2. **Variables d'environnement**
3. **Valeurs par défaut** (priorité minimale)

## 📖 Exemples d'Utilisation

### Exemples avec Arguments CLI

```bash
# BD Franco-Belge avec détecteur Multi-BD
python main.py --preset fb --detector multibd tintin.pdf

# Manga en haute résolution, démarrer page 10
python main.py --preset manga --dpi 300 --page 10 onepiece.pdf

# Comics US avec détecteur heuristique
python main.py --preset newspaper --detector heur spiderman.pdf

# Configuration spécifique DPI seulement
python main.py --dpi 250 mycomics.pdf
```

### Exemples avec Variables d'Environnement

```bash
# Configuration via variables d'environnement
export ANCOMICS_PRESET=fb
export ANCOMICS_DETECTOR=multibd
export ANCOMICS_DPI=200
python main.py

# Configuration temporaire pour un fichier
ANCOMICS_PRESET=manga ANCOMICS_PAGE=5 python main.py manga_volume.pdf

# Script de lancement automatisé
#!/bin/bash
export ANCOMICS_PRESET=fb
export ANCOMICS_DETECTOR=multibd
export ANCOMICS_DPI=250
python main.py "$1"
```

### Exemples de Configuration Mixte

```bash
# Variable d'environnement + argument CLI (CLI prioritaire)
export ANCOMICS_PRESET=manga
python main.py --preset fb comics.pdf  # Utilisera 'fb' (CLI prioritaire)

# Configuration de base via env, fichier via CLI
export ANCOMICS_PRESET=fb
export ANCOMICS_DETECTOR=multibd
python main.py --page 20 comics.pdf
```

## 🔧 Intégration et Automatisation

### Script de Lancement Personnalisé

```bash
#!/bin/bash
# ancomics-fb.sh - Lancement optimisé Franco-Belge

export ANCOMICS_PRESET=fb
export ANCOMICS_DETECTOR=multibd
export ANCOMICS_DPI=200

if [ $# -eq 0 ]; then
    echo "Usage: $0 <fichier.pdf> [page]"
    exit 1
fi

if [ $# -eq 2 ]; then
    export ANCOMICS_PAGE=$2
fi

python /path/to/AnComicsViewer/main.py "$1"
```

### Alias Bash/Zsh

```bash
# Ajouter dans ~/.bashrc ou ~/.zshrc
alias comics-fb='ANCOMICS_PRESET=fb ANCOMICS_DETECTOR=multibd python /path/to/AnComicsViewer/main.py'
alias comics-manga='ANCOMICS_PRESET=manga ANCOMICS_DETECTOR=multibd python /path/to/AnComicsViewer/main.py'
alias comics-us='ANCOMICS_PRESET=newspaper ANCOMICS_DETECTOR=heur python /path/to/AnComicsViewer/main.py'

# Utilisation
comics-fb tintin.pdf
comics-manga --page 10 onepiece.pdf
comics-us spiderman.pdf
```

### Intégration dans un Gestionnaire de Fichiers

```bash
# Script pour intégration dans un gestionnaire de fichiers
#!/bin/bash
# ancomics-open.sh

PRESET="fb"
DETECTOR="multibd"

case "$1" in
    *manga*|*jp*) PRESET="manga" ;;
    *us*|*america*) PRESET="newspaper" ;;
esac

ANCOMICS_PRESET=$PRESET ANCOMICS_DETECTOR=$DETECTOR python /path/to/AnComicsViewer/main.py "$1"
```

## 🛠️ Méthodes Publiques (API Programmatique)

Pour l'intégration dans d'autres applications Python, AnComicsViewer expose des méthodes publiques :

```python
from AnComicsViewer import ComicsView

# Créer l'application
app = QApplication(sys.argv)
viewer = ComicsView()

# API publique
viewer.apply_preset("fb")           # Appliquer un preset
viewer.set_detector("multibd")      # Changer de détecteur
viewer.open_on_start("file.pdf", 5) # Ouvrir fichier à la page 5

viewer.show()
app.exec()
```

## 📋 Validation et Débogage

```bash
# Test de la configuration CLI
python test_cli.py

# Validation des arguments sans lancement GUI
python main.py --help

# Debug des variables d'environnement
env | grep ANCOMICS
```

## 🎯 Conseils d'Utilisation

1. **Performance** : Utilisez `--detector heur` pour un démarrage rapide
2. **Qualité** : Utilisez `--detector multibd` pour la meilleure précision
3. **Scripts** : Préférez les variables d'environnement pour l'automatisation
4. **Interactif** : Utilisez les arguments CLI pour l'usage ponctuel
5. **DPI** : Ajustez selon la résolution de vos PDFs (150-300 typique)

Cette interface CLI rend AnComicsViewer entièrement scriptable et intégrable dans des workflows automatisés tout en conservant sa facilité d'utilisation interactive.
