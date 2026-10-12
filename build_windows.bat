@echo off
REM Build FunAutoScripter binaries for Windows using PyInstaller.
REM Run this script in a standard Python 3.11+ environment (Command Prompt or PowerShell).
REM ffmpeg must be installed on the target machine separately (not bundled).
REM Download ffmpeg from https://ffmpeg.org/download.html and add it to PATH.
REM
REM Usage: build_windows.bat
REM Output: dist\funautoscripter.exe  and  dist\funautoscripter-gui.exe

echo Building FunAutoScripter for Windows...

pip install pyinstaller numpy scipy opencv-python matplotlib dearpygui
if errorlevel 1 (
    echo ERROR: pip install failed.
    exit /b 1
)

echo.
echo Building CLI binary...
pyinstaller funautoscripter.spec
if errorlevel 1 (
    echo ERROR: CLI build failed.
    exit /b 1
)

echo.
echo Building GUI binary...
pyinstaller funautoscripter_gui.spec
if errorlevel 1 (
    echo ERROR: GUI build failed.
    exit /b 1
)

echo.
echo Done. Binaries are in dist\:
dir dist\funautoscripter.exe dist\funautoscripter-gui.exe 2>nul
