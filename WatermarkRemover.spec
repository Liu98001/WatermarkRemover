# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

import os
import sys

datas = []
binaries = []
hiddenimports = []
tmp_ret = collect_all('imageio_ffmpeg')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('yt_dlp')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# conda 布局下 PyInstaller 的 tkinter hook 收集不到 Library\bin 里的
# tk/tcl DLL, GUI 启动会报 "ImportError: DLL load failed while importing
# _tkinter"; liblzma/libbz2/libmpdec/libcrypto 是 _lzma/_bz2/_decimal/
# _hashlib 的依赖。显式带上(非 conda 布局自动跳过)。
_libbin = os.path.join(sys.prefix, "Library", "bin")
_APP_DLLS = ("tk86t.dll", "tcl86t.dll", "liblzma.dll", "libbz2.dll",
             "libmpdec-4.dll", "libcrypto-3-x64.dll", "libssl-3-x64.dll",
             "ffi.dll", "libexpat.dll", "sqlite3.dll")
binaries += [(os.path.join(_libbin, n), ".") for n in _APP_DLLS
             if os.path.isfile(os.path.join(_libbin, n))]


a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
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
    [],
    exclude_binaries=True,
    name='WatermarkRemover',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['packaging/icon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='WatermarkRemover',
)
