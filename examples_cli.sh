#!/bin/bash
# examples/launch-scripts.sh
# Scripts d'exemple pour lancer AnComicsViewer avec différentes configurations

echo "🎯 AnComicsViewer - Scripts de Lancement"
echo "========================================"
echo

# Fonction d'aide
show_help() {
    echo "Usage: $0 [COMMAND] [PDF_FILE]"
    echo
    echo "Commands:"
    echo "  fb       - Franco-Belge (BD européennes)"
    echo "  manga    - Style japonais (RTL)"
    echo "  us       - Comics US/newspaper"
    echo "  demo     - Démonstration des options"
    echo "  test     - Test de l'interface CLI"
    echo "  help     - Afficher cette aide"
    echo
    echo "Examples:"
    echo "  $0 fb tintin.pdf"
    echo "  $0 manga onepiece.pdf"
    echo "  $0 us spiderman.pdf"
    echo "  $0 demo"
}

# Fonction de lancement Franco-Belge
launch_fb() {
    echo "🇫🇷 Lancement mode Franco-Belge..."
    echo "   • Preset: Franco-Belge"
    echo "   • Detector: Multi-BD Enhanced"
    echo "   • DPI: 200"
    echo
    
    if [ "$1" ]; then
        python3 main.py --preset fb --detector multibd --dpi 200 "$1"
    else
        python3 main.py --preset fb --detector multibd --dpi 200
    fi
}

# Fonction de lancement Manga
launch_manga() {
    echo "🇯🇵 Lancement mode Manga..."
    echo "   • Preset: Manga (RTL)"
    echo "   • Detector: Multi-BD Enhanced"
    echo "   • DPI: 150"
    echo
    
    if [ "$1" ]; then
        python3 main.py --preset manga --detector multibd --dpi 150 "$1"
    else
        python3 main.py --preset manga --detector multibd --dpi 150
    fi
}

# Fonction de lancement US
launch_us() {
    echo "🇺🇸 Lancement mode US Comics..."
    echo "   • Preset: Newspaper"
    echo "   • Detector: Heuristique (rapide)"
    echo "   • DPI: 200"
    echo
    
    if [ "$1" ]; then
        python3 main.py --preset newspaper --detector heur --dpi 200 "$1"
    else
        python3 main.py --preset newspaper --detector heur --dpi 200
    fi
}

# Démonstration des options
demo_options() {
    echo "🧪 Démonstration des options CLI..."
    echo
    
    echo "1. Affichage de l'aide:"
    python3 main.py --help
    echo
    
    echo "2. Affichage de la version:"
    python3 main.py --version
    echo
    
    echo "3. Test avec variables d'environnement:"
    echo "   export ANCOMICS_PRESET=fb"
    echo "   export ANCOMICS_DETECTOR=multibd"
    echo "   export ANCOMICS_DPI=250"
    echo
    
    ANCOMICS_PRESET=fb ANCOMICS_DETECTOR=multibd ANCOMICS_DPI=250 python3 main.py --version
    echo
    
    echo "4. Configuration mixte (env + args):"
    echo "   ANCOMICS_PRESET=manga python3 main.py --preset fb --version"
    ANCOMICS_PRESET=manga python3 main.py --preset fb --version
    echo
}

# Test de l'interface CLI
test_cli() {
    echo "🔧 Test de l'interface CLI..."
    echo
    
    if [ -f "test_cli.py" ]; then
        python3 test_cli.py
    else
        echo "❌ Fichier test_cli.py non trouvé"
        echo "💡 Assurez-vous d'être dans le répertoire AnComicsViewer"
    fi
}

# Script principal
main() {
    case "$1" in
        "fb")
            launch_fb "$2"
            ;;
        "manga")
            launch_manga "$2"
            ;;
        "us")
            launch_us "$2"
            ;;
        "demo")
            demo_options
            ;;
        "test")
            test_cli
            ;;
        "help"|"--help"|"-h"|"")
            show_help
            ;;
        *)
            echo "❌ Commande inconnue: $1"
            echo
            show_help
            exit 1
            ;;
    esac
}

# Vérifier que nous sommes dans le bon répertoire
if [ ! -f "main.py" ]; then
    echo "❌ Fichier main.py non trouvé"
    echo "💡 Veuillez exécuter ce script depuis le répertoire AnComicsViewer"
    exit 1
fi

# Lancer le script principal
main "$@"
