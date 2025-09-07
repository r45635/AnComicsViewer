#!/bin/bash
# Script simple pour suivre les logs en temps réel
# Usage: ./follow_logs.sh [log_file]

LOG_FILE="$1"

# Trouver le dernier log si non spécifié
if [ -z "$LOG_FILE" ]; then
    if [ -d "training_logs" ]; then
        LOG_FILE=$(ls -t training_logs/training_*.log 2>/dev/null | head -1)
    fi
fi

if [ -z "$LOG_FILE" ] || [ ! -f "$LOG_FILE" ]; then
    echo "❌ Fichier de log introuvable"
    echo "💡 Usage: $0 [log_file]"
    echo "💡 Ou lancez d'abord l'entraînement avec train_optimized.sh"
    exit 1
fi

echo "🔍 Suivi en temps réel: $LOG_FILE"
echo "   (Ctrl+C pour arrêter)"
echo ""

tail -f "$LOG_FILE"
