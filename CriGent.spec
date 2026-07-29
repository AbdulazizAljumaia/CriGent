# PyInstaller spec for the portable CriGent build.
#   python -m PyInstaller CriGent.spec --noconfirm
#
# Deliberately does NOT bundle Ollama or any model: those are gigabytes, and the
# first-run wizard finds an existing Ollama or fetches one into the user's app
# folder. The exe stays a self-contained ~100 MB download.

block_cipher = None

a = Analysis(
    ['crigent.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['bs4', 'ddgs'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter', 'matplotlib', 'numpy', 'scipy', 'pandas', 'PIL',
        'PyQt6.QtQml', 'PyQt6.QtQuick', 'PyQt6.QtQuick3D', 'PyQt6.QtMultimedia',
        'PyQt6.QtWebEngineCore', 'PyQt6.QtWebEngineWidgets', 'PyQt6.QtBluetooth',
        'PyQt6.QtNfc', 'PyQt6.QtPositioning', 'PyQt6.QtSql', 'PyQt6.QtTest',
        'PyQt6.QtDesigner', 'PyQt6.QtHelp', 'PyQt6.QtCharts', 'PyQt6.QtDataVisualization',
    ],
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
    name='CriGent',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # GUI app: no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='crigent.ico',
)
