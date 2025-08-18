# AnComicsViewer Launcher Script pour Windows PowerShell
# Usage: .\run.ps1

$ErrorActionPreference = "Stop"

Write-Host "🎨 AnComicsViewer - Lecteur PDF Comics Intelligent" -ForegroundColor Cyan
Write-Host "🪟 Windows PowerShell Launcher" -ForegroundColor Yellow

# Changer vers le répertoire du script
Set-Location $PSScriptRoot

# Vérifier l'environnement virtuel
if (-not (Test-Path ".venv")) {
    Write-Host "❌ Environnement virtuel .venv non trouvé" -ForegroundColor Red
    Write-Host "💡 Exécutez d'abord :" -ForegroundColor Yellow
    Write-Host "   python -m venv .venv" -ForegroundColor Gray
    Write-Host "   .venv\Scripts\pip install -r requirements.txt -r requirements-ml.txt" -ForegroundColor Gray
    exit 1
}

# Vérifier matplotlib
$matplotlibCheck = & .venv\Scripts\pip show matplotlib 2>$null
if (-not $matplotlibCheck) {
    Write-Host "⚠️ matplotlib manquant, installation en cours..." -ForegroundColor Yellow
    & .venv\Scripts\pip install matplotlib
}

Write-Host "🚀 Lancement d'AnComicsViewer avec l'environnement virtuel..." -ForegroundColor Green
try {
    & .venv\Scripts\python main.py
} catch {
    Write-Host "❌ Erreur lors du lancement: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "💡 Vérifiez que toutes les dépendances sont installées" -ForegroundColor Yellow
    exit 1
}
