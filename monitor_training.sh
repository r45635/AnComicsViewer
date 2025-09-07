#!/bin/bash
# Script de surveillance de l'entraînement YOLO
# Usage: ./monitor_training.sh [log_file]

# Configuration
REFRESH_INTERVAL=10  # secondes
RESULTS_DIR="runs/detect"

# Fonction pour afficher l'aide
show_help() {
    echo "Usage: $0 [OPTIONS] [LOG_FILE]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Afficher cette aide"
    echo "  -i, --interval SECONDS    Intervalle de rafraîchissement (défaut: 10)"
    echo "  -f, --follow   Suivre le log en temps réel (comme tail -f)"
    echo ""
    echo "Exemples:"
    echo "  $0                        # Surveiller le dernier entraînement"
    echo "  $0 training_logs/training_20241201_143000.log"
    echo "  $0 --follow training_logs/training_20241201_143000.log"
}

# Parse arguments
FOLLOW_MODE=false
LOG_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -i|--interval)
            REFRESH_INTERVAL="$2"
            shift 2
            ;;
        -f|--follow)
            FOLLOW_MODE=true
            shift
            ;;
        *)
            LOG_FILE="$1"
            shift
            ;;
    esac
done

# Fonction pour trouver le dernier log si non spécifié
find_latest_log() {
    if [ -d "training_logs" ]; then
        find training_logs -name "training_*.log" -type f -exec stat -f "%m %N" {} \; 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-
    fi
}

# Fonction pour trouver le dernier dossier de résultats
find_latest_results() {
    if [ -d "$RESULTS_DIR" ]; then
        find "$RESULTS_DIR" -maxdepth 1 -type d -name "*ancomics*" -exec stat -f "%m %N" {} \; 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-
    fi
}

# Déterminer le fichier de log à surveiller
if [ -z "$LOG_FILE" ]; then
    LOG_FILE=$(find_latest_log)
    if [ -z "$LOG_FILE" ]; then
        echo "❌ Aucun fichier de log trouvé dans training_logs/"
        echo "💡 Assurez-vous d'avoir lancé l'entraînement avec train_optimized.sh"
        exit 1
    fi
    echo "📝 Surveillance du dernier log: $LOG_FILE"
fi

# Vérifier que le fichier existe
if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Fichier de log introuvable: $LOG_FILE"
    exit 1
fi

# Mode suivi en temps réel
if [ "$FOLLOW_MODE" = true ]; then
    echo "🔍 Suivi en temps réel de: $LOG_FILE"
    echo "   (Ctrl+C pour arrêter)"
    echo ""
    tail -f "$LOG_FILE"
    exit 0
fi

# Fonction pour extraire les informations d'entraînement
extract_training_info() {
    local log_file="$1"
    
    echo "🚀 SURVEILLANCE DE L'ENTRAÎNEMENT YOLO"
    echo "======================================"
    echo "📝 Log: $(basename "$log_file")"
    echo "⏰ Dernière mise à jour: $(date)"
    echo ""
    
    # Statut général
    if grep -q "TRAINING_END_TIME" "$log_file"; then
        local exit_code=$(grep "TRAINING_EXIT_CODE" "$log_file" | tail -1 | cut -d: -f2 | tr -d ' ')
        local end_time=$(grep "TRAINING_END_TIME" "$log_file" | tail -1 | cut -d: -f2- | tr -d ' ')
        if [ "$exit_code" = "0" ]; then
            echo "✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS"
        else
            echo "❌ ENTRAÎNEMENT ÉCHOUÉ (code: $exit_code)"
        fi
        echo "🕐 Fin: $end_time"
        echo ""
    else
        echo "🔄 ENTRAÎNEMENT EN COURS..."
        echo ""
    fi
    
    # Progression des epochs
    local current_epoch=$(grep -o "Epoch [0-9]*/[0-9]*" "$log_file" | tail -1)
    if [ -n "$current_epoch" ]; then
        echo "📊 PROGRESSION:"
        echo "   $current_epoch"
        
        # Dernières métriques
        local last_metrics=$(grep -E "(train|val): " "$log_file" | tail -2)
        if [ -n "$last_metrics" ]; then
            echo ""
            echo "📈 DERNIÈRES MÉTRIQUES:"
            echo "$last_metrics" | sed 's/^/   /'
        fi
        echo ""
    fi
    
    # Métriques de validation récentes
    local recent_val=$(grep "val:" "$log_file" | tail -5)
    if [ -n "$recent_val" ]; then
        echo "🎯 VALIDATION (5 derniers):"
        echo "$recent_val" | sed 's/^/   /'
        echo ""
    fi
    
    # Erreurs récentes
    local errors=$(grep -iE "(error|fail|crash|exception)" "$log_file" | tail -3)
    if [ -n "$errors" ]; then
        echo "⚠️  ERREURS RÉCENTES:"
        echo "$errors" | sed 's/^/   /'
        echo ""
    fi
    
    # Informations sur les sauvegardes
    local saves=$(grep -E "(Saving|saved)" "$log_file" | tail -3)
    if [ -n "$saves" ]; then
        echo "💾 SAUVEGARDES:"
        echo "$saves" | sed 's/^/   /'
        echo ""
    fi
    
    # Dossier de résultats
    local results_dir=$(find_latest_results)
    if [ -n "$results_dir" ]; then
        echo "📁 DOSSIER DE RÉSULTATS: $results_dir"
        if [ -f "$results_dir/results.csv" ]; then
            echo "   📈 Graphiques: $results_dir/results.png"
            echo "   📊 Métriques: $results_dir/results.csv"
        fi
        if [ -f "$results_dir/weights/best.pt" ]; then
            echo "   🏆 Meilleur modèle: $results_dir/weights/best.pt"
        fi
        if [ -f "$results_dir/weights/last.pt" ]; then
            echo "   💾 Dernier modèle: $results_dir/weights/last.pt"
        fi
        echo ""
    fi
    
    # Processus en cours
    local process=$(ps aux | grep python | grep yolo | grep -v grep | head -1)
    if [ -n "$process" ]; then
        echo "🖥️  PROCESSUS:"
        echo "   ✅ Entraînement actif"
        local cpu=$(echo "$process" | awk '{print $3}')
        local mem=$(echo "$process" | awk '{print $4}')
        echo "   CPU: ${cpu}%, Mémoire: ${mem}%"
        echo ""
    fi
}

# Boucle principale de surveillance
echo "🔍 Surveillance de l'entraînement YOLO"
echo "   Fichier: $LOG_FILE"
echo "   Intervalle: ${REFRESH_INTERVAL}s"
echo "   (Ctrl+C pour arrêter)"
echo ""

while true; do
    clear
    extract_training_info "$LOG_FILE"
    
    # Vérifier si l'entraînement est terminé
    if grep -q "TRAINING_END_TIME" "$LOG_FILE"; then
        echo "🏁 Entraînement terminé. Surveillance arrêtée."
        break
    fi
    
    echo "🔄 Actualisation dans ${REFRESH_INTERVAL}s... (Ctrl+C pour arrêter)"
    sleep "$REFRESH_INTERVAL"
done
        PID=$(echo "$PROCESS" | awk '{print $2}')
        echo "📊 Resource Usage (PID: $PID):"
        ps -p $PID -o pid,pcpu,pmem,time,command | tail -1
        echo ""
        
        # Check for recent results files
        echo "📈 Recent Training Files:"
        find /Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/runs -name "*.csv" -o -name "*.pt" | head -5 | xargs ls -lat | head -5
        echo ""
        
        # Check GPU usage if possible
        if command -v nvidia-smi &> /dev/null; then
            echo "🎮 GPU Usage:"
            nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
        elif system_profiler SPDisplaysDataType 2>/dev/null | grep -q "Metal"; then
            echo "🎮 Metal GPU: Available (detailed stats not available via command line)"
        fi
        
    else
        echo "❌ Training Status: NOT RUNNING"
        echo ""
        echo "💡 To restart training, run:"
        echo "   source .venv/bin/activate && ./train_optimized.sh"
    fi
    
    echo "=========================================="
    echo "⏰ Next update in 30 seconds... (Ctrl+C to stop)"
    sleep 30
done
