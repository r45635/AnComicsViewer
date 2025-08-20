#!/usr/bin/env python3
"""Test simple des chemins"""

from pathlib import Path

# Le script est dans archive_core/
script_dir = Path(__file__).parent.absolute()
base_dir = script_dir.parent.absolute()

print(f"📁 Script dir: {script_dir}")
print(f"📁 Base dir: {base_dir}")
print(f"📁 Exists main.py: {(base_dir / 'main.py').exists()}")
print(f"📁 Exists src/: {(base_dir / 'src').exists()}")

# Test git
try:
    import subprocess
    result = subprocess.run(
        ["git", "describe", "--tags", "--long", "--dirty"],
        cwd=base_dir,
        capture_output=True,
        text=True,
        timeout=5
    )
    print(f"🔧 Git result: {result.returncode}")
    if result.returncode == 0:
        print(f"📦 Version: {result.stdout.strip()}")
except Exception as e:
    print(f"❌ Git error: {e}")

print("✅ Test terminé")
