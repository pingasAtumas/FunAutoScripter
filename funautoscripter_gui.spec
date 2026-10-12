# PyInstaller spec for the GUI build (includes DearPyGui)
# Build with: pyinstaller funautoscripter_gui.spec
# Output: dist/funautoscripter-gui (Linux/macOS) or dist/funautoscripter-gui.exe (Windows)
#
# NOTE: ffmpeg is a system dependency and is NOT bundled here.
# The user must have ffmpeg installed and available on their PATH.
# The GUI requires a display (X11 on Linux, native on Windows/macOS).

import sys

block_cipher = None

a = Analysis(
    ['funautoscripter_gui.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'dearpygui',
        'dearpygui.dearpygui',
        'cv2',
        'scipy',
        'scipy.signal',
        'scipy.ndimage',
        'numpy',
        'matplotlib',
        'matplotlib.backends.backend_agg',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=['tkinter', 'PyQt5', 'PyQt6', 'wx'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='funautoscripter-gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    # console=False on Windows hides the terminal window; True on Linux keeps stderr visible for debugging
    console=(sys.platform != 'win32'),
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
