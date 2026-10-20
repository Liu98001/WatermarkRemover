# -*- coding: utf-8 -*-
r"""打包前置: 把 onedir 应用目录(默认 dist\WatermarkRemover)压成 packaging/app.zip"""
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "dist",
                                                         "WatermarkRemover")
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.zip")

if not os.path.isdir(SRC):
    raise SystemExit(f"应用目录不存在: {SRC} (请先 pyinstaller --onedir 构建)")

with zipfile.ZipFile(DST, "w", compression=zipfile.ZIP_LZMA,
                     compresslevel=6) as z:
    for r, _d, fs in os.walk(SRC):
        for f in fs:
            p = os.path.join(r, f)
            z.write(p, os.path.relpath(p, SRC))
print(f"app.zip: {os.path.getsize(DST) / 1e6:.1f} MB")
