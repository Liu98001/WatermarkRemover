# -*- coding: utf-8 -*-
"""WatermarkRemover 卸载器: 删除快捷方式/注册表项/安装目录, 最后自清理。

全程零弹窗(系统"应用和功能"本身已有确认步骤), --silent 参数保留兼容。
"""
import os
import shutil
import subprocess
import sys
import winreg

APP = "WatermarkRemover"
KEY = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\WatermarkRemover"
START_MENU = os.path.join(os.environ.get("APPDATA", ""),
                          r"Microsoft\Windows\Start Menu\Programs", APP)
SELF_DIR = os.environ.get("UNINS_DIR") or os.path.dirname(
    os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))


def _ps(lnk, target, workdir):
    cmd = (f"$ws=(New-Object -ComObject WScript.Shell);"
           f"$s=$ws.CreateShortcut('{lnk}');"
           f"$s.TargetPath='{target}';$s.WorkingDirectory='{workdir}';"
           f"$s.IconLocation='{target},0';$s.Save()")
    subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                   creationflags=0x08000000)


def real_desktop() -> str:
    """系统真实桌面(兼容重定向, 与安装器同逻辑)"""
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft"
                            r"\Windows\CurrentVersion\Explorer"
                            r"\User Shell Folders") as k:
            v, _t = winreg.QueryValueEx(k, "Desktop")
            return os.path.expandvars(v)
    except OSError:
        return os.path.join(os.environ.get("USERPROFILE", ""), "Desktop")


def remove_shortcuts():
    desk = real_desktop()
    for lnk in (os.path.join(desk, f"{APP}.lnk"),
                os.path.join(desk, "WatermarkRemover.lnk")):
        try:
            if os.path.isfile(lnk):
                os.remove(lnk)
        except OSError:
            pass
    try:
        if os.path.isdir(START_MENU):
            shutil.rmtree(START_MENU, ignore_errors=True)
    except OSError:
        pass


def remove_registry():
    try:
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, KEY)
    except OSError:
        pass


def read_install_location() -> str:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, KEY) as k:
            v, _t = winreg.QueryValueEx(k, "InstallLocation")
            return v
    except OSError:
        return ""


def uninstall():
    install_dir = read_install_location()
    remove_shortcuts()
    remove_registry()
    # 立即删除安装目录(此时应用与卸载确认均已结束, 无文件锁)
    if install_dir and os.path.isdir(install_dir):
        for _i in range(3):
            shutil.rmtree(install_dir, ignore_errors=True)
            if not os.path.isdir(install_dir):
                break
    # 卸载器自身所在目录(含运行中的 uninstall.exe)由延时脚本补删。
    # 必须用 CREATE_NO_WINDOW(隐藏控制台)而不是 DETACHED_PROCESS:
    # DETACHED 下 cmd 自己无控制台, 但它启动的 ping 会新建一个 ——
    # 卸载时弹出 CMD 黑框(实测)。CREATE_NO_WINDOW 下 cmd 拥有隐藏
    # 控制台, ping/rd/del 全部继承之, 全程无窗口。
    here = os.path.dirname(os.path.abspath(sys.executable))
    tmp = os.environ.get("TEMP", here)
    script = os.path.join(tmp, "_wmr_cleanup.cmd")
    with open(script, "w", encoding="gbk", errors="ignore") as f:
        f.write("@echo off\r\n"
                "ping -n 3 127.0.0.1 >nul\r\n"
                f"rd /s /q \"{here}\"\r\n"
                f"del /q /f \"%~s0\"\r\n")
    subprocess.Popen(["cmd", "/c", script], creationflags=0x08000000)


def main():
    uninstall()
    return 0


if __name__ == "__main__":
    sys.exit(main())
