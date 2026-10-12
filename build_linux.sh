#!/bin/bash
# Build FunAutoScripter binaries for Linux using PyInstaller.
# Run this script outside of nix-shell, in a standard Python 3.11+ environment.
# ffmpeg must be installed on the target machine separately (not bundled).
#
# Usage:
#   chmod +x build_linux.sh
#   ./build_linux.sh
#
# Output: dist/funautoscripter  and  dist/funautoscripter-gui

set -e

echo "Building FunAutoScripter for Linux..."

pip install pyinstaller numpy scipy opencv-python matplotlib dearpygui

echo ""
echo "Building CLI binary..."
pyinstaller funautoscripter.spec

echo ""
echo "Building GUI binary..."
pyinstaller funautoscripter_gui.spec

echo ""
echo "Done. Binaries are in dist/:"
ls -lh dist/funautoscripter dist/funautoscripter-gui 2>/dev/null || true
