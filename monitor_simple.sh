#!/bin/bash
# Script de surveillance de l'entraînement YOLO
# Usage: ./monitor_training.sh [log_file]

REFRESH_INTERVAL=10
RESULTS_DIR="runs/detect"

# Fonction pour trouver le dernier log
find_latest_log() {
    if [ -d "training_logs" ]; then
        ls -t training_logs/training_*.log 2>/dev/null | head -1
    fi
}

# Fonction pour trouver le dernier dossier de résultats
find_latest_results() {
    if [ -d "$RESULTS_DIR" ]; then
        ls -td "$RESULTS_DIR"/*ancomics* 2>/dev/null | head -1
    fi
}

# Déterminer le fichier de log
LOG_FILE="$1"
if [ -z "$LOG_FILE" ]; then
    LOG_FILE=$(find_latest_log)
    if [ -z "$LOG_FILE" ]; then
        echo "❌ Aucun fichier de log trouvé dans training_logs/"
        echo "💡 Assurez-vous d'avoir lancé l'entraînement avec train_optimized.sh"
        exit 1
    fi
    echo "📝 Surveillance du dernier log: $LOG_FILE"
fi

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Fichier de log introuvable: $LOG_FILE"
    exit 1
fi

# Fonction pour extraire les informations d'entraînement
extract_training_info() {
    local log_file="$1"
    
    echo "🚀 SURVEILLANCE DE L'ENTRAÎNEMENT YOLO"
    echo "======================================"
    echo "📝 Log: $(basename "$log_file")"
    echo "⏰ $(date)"
    echo ""
    
    # Statut général
    if grep -q "TRAINING_END_TIME" "$log_file"; then
        local exit_code=$(grep "TRAINING_EXIT_CODE" "$log_file" | tail -1 | cut -d: -f2 | tr -d ' ')
        if [ "$exit_code" = "0" ]; then
            echo "✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS"
        else
            echo "❌ ENTRAÎNEMENT ÉCHOUÉ (code: $exit_code)"
        fi
        echo ""
    else
        echo "🔄 ENTRAÎNEMENT EN COURS..."
        echo ""
    fi
    
    # Progression des epochs
    local current_epoch=$(grep -o "Epoch [0-9]*/[0-9]*" "$log_file" | tail -1)
    if [ -n "$current_epoch" ]; then
        echo "📊 PROGRESSION: $current_epoch"
        echo ""
    fi
    
    # Dernières métriques d'entraînement
    local last_train=$(grep "train:" "$log_file" | tail -1)
    if [ -n "$last_train" ]; then
        echo "🏋️  TRAIN: $last_train"
    fi
    
    # Dernières métriques de validation
    local last_val=$(grep "val:" "$log_file" | tail -1)
    if [ -n "$last_val" ]; then
        echo "🎯 VAL:   $last_val"
        echo ""
    fi
    
    # Erreurs récentes
    local errors=$(grep -iE "(error|fail|crash|exception)" "$log_file" | tail -2)
    if [ -n "$errors" ]; then
        echo "⚠️  ERREURS:"
        echo "$errors" | sed 's/^/   /'
        echo ""
    fi
    
    # Processus actif
    local process=$(ps aux | grep python | grep yolo | grep -v grep | head -1)
    if [ -n "$process" ]; then
        local cpu=$(echo "$process" | awk '{print $3}')
        local mem=$(echo "$process" | awk '{print $4}')
        echo "🖥️  PROCESSUS: Actif (CPU: ${cpu}%, Mem: ${mem}%)"
        echo ""
    fi
    
    # Dossier de résultats
    local results_dir=$(find_latest_results)
    if [ -n "$results_dir" ]; then
        echo "📁 RÉSULTATS: $results_dir"
        if [ -f "$results_dir/weights/best.pt" ]; then
            echo "   🏆 best.pt disponible"
        fi
        if [ -f "$results_dir/results.csv" ]; then
            echo "   📊 results.csv disponible"
        fi
        echo ""
    fi
}

# Boucle principale
echo "🔍 Surveillance de l'entraînement (Ctrl+C pour arrêter)"
echo "Fichier: $LOG_FILE"
echo ""

while true; do
    clear
    extract_training_info "$LOG_FILE"
    
    if grep -q "TRAINING_END_TIME" "$LOG_FILE"; then
        echo "🏁 Entraînement terminé."
        break
    fi
    
    echo "🔄 Prochaine mise à jour dans ${REFRESH_INTERVAL}s..."
    sleep "$REFRESH_INTERVAL"
done
