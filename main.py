# -*- coding: utf-8 -*-
"""统一入口: python main.py [remove|detect|download|batch|gui|selftest] …

打包说明(pyinstaller, onedir 启动最快; onefile 启动需解压 150MB 会慢):
    pyinstaller --noconfirm --clean --onedir --windowed --icon packaging/icon.ico ^
        --name WatermarkRemover --paths . ^
        --collect-all imageio_ffmpeg --collect-all yt_dlp main.py
"""
import io
import os
import subprocess
import sys

# --windowed 模式下没有 stdout/stderr, 打印类代码(下载进度等)需要兜底
if sys.stdout is None:
    sys.stdout = io.open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = io.open(os.devnull, "w", encoding="utf-8")

# 控制台为 GBK(中文 Windows 默认)时, 标题里的 emoji 会让 print 崩溃,
# 统一切到 UTF-8 并容错替换。
for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass

# 全局抑制子进程控制台窗口: ffmpeg/ffprobe/yt-dlp 等都是控制台程序,
# 双击 exe(窗口模式, 无控制台)时每次调用都会闪出黑色命令窗。
# 仅在打包后的 exe 里启用 —— 源码/控制台方式运行时行为与从前完全一致。
# 注意: 必须用"子类"而非函数替换, 否则 asyncio.windows_utils 等模块
# 执行 class Popen(subprocess.Popen) 时会因基类是函数而崩溃。
if os.name == "nt" and getattr(sys, "frozen", False):
    class _QuietPopen(subprocess.Popen):
        def __init__(self, *args, **kwargs):
            kwargs["creationflags"] = (kwargs.get("creationflags", 0)
                                       | 0x08000000)      # CREATE_NO_WINDOW
            super().__init__(*args, **kwargs)

    subprocess.Popen = _QuietPopen

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wmremove.cli import main          # noqa: E402

if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:                       # 双击 exe / 无参数时直接打开界面
        argv = ["gui"]
    sys.exit(main(argv))
