# PyInstaller spec for the CLI build (no GUI dependencies)
# Build with: pyinstaller funautoscripter.spec
# Output: dist/funautoscripter (Linux/macOS) or dist/funautoscripter.exe (Windows)
#
# NOTE: ffmpeg is a system dependency and is NOT bundled here.
# The user must have ffmpeg installed and available on their PATH.

block_cipher = None

a = Analysis(
    ['funautoscripter.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['cv2', 'scipy', 'scipy.signal', 'scipy.ndimage', 'numpy', 'matplotlib', 'matplotlib.backends.backend_agg'],
    hookspath=[],
    runtime_hooks=[],
    excludes=['dearpygui', 'tkinter', 'PyQt5', 'PyQt6', 'wx'],
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
    name='funautoscripter',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
