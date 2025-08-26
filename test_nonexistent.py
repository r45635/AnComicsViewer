#!/usr/bin/env python3
"""Test script for the auto-reopen functionality with a non-existent file."""

import os
import sys
from pathlib import Path
from PySide6.QtCore import QSettings

def test_nonexistent_file():
    """Test what happens when the last file no longer exists."""
    
    # Set a non-existent file path in the settings
    settings = QSettings("AnComicsViewer", "AnComicsViewer")
    fake_path = "/non/existent/file.pdf"
    settings.setValue("lastFile", fake_path)
    print(f"🧪 Set fake last file: {fake_path}")
    
    # Read it back to confirm
    last_file = settings.value("lastFile", "")
    print(f"📖 Confirmed saved: {last_file}")

if __name__ == "__main__":
    test_nonexistent_file()
