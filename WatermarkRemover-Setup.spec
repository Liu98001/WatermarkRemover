# -*- mode: python ; coding: utf-8 -*-
"""WatermarkRemover 安装包(Setup)构建 spec。

前置步骤(在项目根目录):
    pyinstaller --noconfirm --clean WatermarkRemover.spec   # onedir 应用 -> dist\WatermarkRemover
    python packaging/make_payload.py                        # -> packaging/app.zip
    pyinstaller --noconfirm --clean wmremover-unins.spec    # -> dist\wmremover-unins.exe
    pyinstaller --noconfirm --clean WatermarkRemover-Setup.spec
产物: dist\WatermarkRemover-Setup.exe

注意: binaries 里显式带上 conda Library\bin 的 tk/tcl 与 liblzma 等 DLL ——
conda 布局下 PyInstaller 不会自动收集, 缺了安装向导会报
"ImportError: DLL load failed while importing _tkinter"; app.zip 为 LZMA
压缩, 缺 liblzma 会在解压时失败。
"""
from PyInstaller.utils.hooks import collect_all
import os
import sys

datas = []
binaries = []
hiddenimports = []

_libbin = os.path.join(sys.prefix, "Library", "bin")
_TK_DLLS = ("tk86t.dll", "tcl86t.dll", "liblzma.dll", "libbz2.dll",
            "libmpdec-4.dll", "libcrypto-3-x64.dll", "libssl-3-x64.dll")
binaries += [(os.path.join(_libbin, n), ".") for n in _TK_DLLS
             if os.path.isfile(os.path.join(_libbin, n))]

hiddenimports += ['lzma', 'bz2']

# 卸载器路径: 默认 dist\wmremover-unins.exe; 用非默认 --distpath 构建时,
# 通过环境变量 WMR_UNINS 指定实际位置。
_unins = 'dist/wmremover-unins.exe'
if not os.path.isfile(_unins):
    _unins = os.environ.get("WMR_UNINS", _unins)

a = Analysis(
    ['packaging/installer.py'],
    pathex=[],
    binaries=binaries,
    datas=[('packaging/app.zip', '.'),
           (_unins, '.')],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='WatermarkRemover-Setup',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['packaging/icon.ico'],
)
