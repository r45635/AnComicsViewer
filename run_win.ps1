<#
Small PowerShell helper to run the app on Windows while ensuring Qt finds the platform plugin.
Usage: .\run_win.ps1
#>
param()

# Resolve Python executable
$py = & python -c "import sys; print(sys.executable)"
Write-Host "Using python: $py"

# Resolve PySide6 platforms dir
$plugs = & $py -c "import PySide6, os; print(os.path.join(os.path.dirname(PySide6.__file__), 'Qt','plugins','platforms'))"
Write-Host "Qt plugins: $plugs"

$env:QT_QPA_PLATFORM_PLUGIN_PATH = $plugs
& $py AnComicsViewer.py
