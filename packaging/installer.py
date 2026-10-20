# -*- coding: utf-8 -*-
r"""WatermarkRemover 安装器: Tk 向导 + 静默模式

    WatermarkRemover-Setup.exe            # 图形向导
    WatermarkRemover-Setup.exe --silent   # 静默安装到默认路径(供自动化测试)

安装动作: 解压应用到 %LOCALAPPDATA%\Programs\WatermarkRemover,
创建桌面/开始菜单快捷方式, 写入 HKCU 卸载信息(控制面板"应用和功能"可见)。
卸载器被复制到 %LOCALAPPDATA%\WatermarkRemover\uninstall.exe(独立于安装目录)。
"""
import os
import shutil
import subprocess
import sys
import threading
import zipfile
import winreg

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

APP = "WatermarkRemover"
DISPLAY = "WatermarkRemover 去水印工具"
VERSION = "1.0.0"
KEY = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\WatermarkRemover"
ACCENT = "#2e6bff"
INK = "#1c2333"
MUT = "#7a8499"
BG = "#f4f6fa"
FONT = "Microsoft YaHei UI"


def res(name: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)


def default_dir() -> str:
    la = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    return os.path.join(la, "Programs", APP)


def local_data() -> str:
    la = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    return os.path.join(la, APP)


def _ps_new(lnk: str, target: str, workdir: str):
    cmd = (f"$ws=(New-Object -ComObject WScript.Shell);"
           f"$s=$ws.CreateShortcut('{lnk}');"
           f"$s.TargetPath='{target}';$s.WorkingDirectory='{workdir}';"
           f"$s.IconLocation='{target},0';$s.Save()")
    subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                   creationflags=0x08000000, check=False)


def real_desktop() -> str:
    """系统真实桌面(兼容 OneDrive/厂商重定向, 如华为 MoveData 迁移到 D 盘)"""
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft"
                            r"\Windows\CurrentVersion\Explorer"
                            r"\User Shell Folders") as k:
            v, _t = winreg.QueryValueEx(k, "Desktop")
            return os.path.expandvars(v)
    except OSError:
        return os.path.join(os.environ.get("USERPROFILE", ""), "Desktop")


def make_shortcuts(install_dir: str):
    exe = os.path.join(install_dir, f"{APP}.exe")
    desk = real_desktop()
    _ps_new(os.path.join(desk, f"{APP}.lnk"), exe, install_dir)
    sm = os.path.join(os.environ.get("APPDATA", ""),
                      r"Microsoft\Windows\Start Menu\Programs", APP)
    os.makedirs(sm, exist_ok=True)
    _ps_new(os.path.join(sm, f"{APP}.lnk"), exe, install_dir)
    unins = os.path.join(local_data(), "uninstall.exe")
    if os.path.isfile(unins):
        _ps_new(os.path.join(sm, "Uninstall.lnk"), unins, local_data())


def register_uninstall(install_dir: str):
    exe = os.path.join(install_dir, f"{APP}.exe")
    size_kb = 0
    for r, _d, fs in os.walk(install_dir):
        for f in fs:
            try:
                size_kb += os.path.getsize(os.path.join(r, f))
            except OSError:
                pass
    size_kb //= 1024
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, KEY) as k:
        winreg.SetValueEx(k, "DisplayName", 0, winreg.REG_SZ, DISPLAY)
        winreg.SetValueEx(k, "DisplayVersion", 0, winreg.REG_SZ, VERSION)
        winreg.SetValueEx(k, "DisplayIcon", 0, winreg.REG_SZ, exe)
        winreg.SetValueEx(k, "InstallLocation", 0, winreg.REG_SZ, install_dir)
        winreg.SetValueEx(k, "UninstallString", 0, winreg.REG_SZ,
                          f'"{os.path.join(local_data(), "uninstall.exe")}"')
        winreg.SetValueEx(k, "EstimatedSize", 0, winreg.REG_DWORD, size_kb)
        winreg.SetValueEx(k, "NoModify", 0, winreg.REG_DWORD, 1)
        winreg.SetValueEx(k, "NoRepair", 0, winreg.REG_DWORD, 1)
        winreg.SetValueEx(k, "Publisher", 0, winreg.REG_SZ, "wmremove")


def do_install(target: str, progress=None):
    """解压 + 卸载器 + 快捷方式 + 注册表; progress(frac, msg) 用于界面"""
    os.makedirs(target, exist_ok=True)
    os.makedirs(local_data(), exist_ok=True)
    if progress:
        progress(0.02, "正在准备…")
    with zipfile.ZipFile(res("app.zip")) as z:
        names = z.namelist()
        total = len(names)
        for i, name in enumerate(names):
            z.extract(name, target)
            if progress and (i % 20 == 0 or i == total - 1):
                progress(0.05 + 0.85 * (i + 1) / total,
                         f"正在安装… {i + 1}/{total}")
    if progress:
        progress(0.92, "写入卸载信息…")
    shutil.copy2(res("wmremover-unins.exe"),
                 os.path.join(local_data(), "uninstall.exe"))
    make_shortcuts(target)
    register_uninstall(target)
    if progress:
        progress(1.0, "安装完成")


class Wizard:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title(f"安装 {DISPLAY}")
        root.configure(bg=BG)
        root.resizable(False, False)
        w, h = 520, 330
        root.update_idletasks()
        x = (root.winfo_screenwidth() - w) // 2
        y = (root.winfo_screenheight() - h) // 2 - 30
        root.geometry(f"{w}x{h}+{x}+{y}")
        f = (FONT, 10)
        self.dir_var = tk.StringVar(value=default_dir())
        self.run_var = tk.BooleanVar(value=True)
        head = tk.Frame(root, bg=ACCENT, height=64)
        head.pack(fill=tk.X)
        head.pack_propagate(False)
        tk.Label(head, text=f"安装 {DISPLAY}", bg=ACCENT, fg="white",
                 font=(FONT, 14, "bold")).pack(side=tk.LEFT, padx=18)
        tk.Label(head, text=f"v{VERSION}", bg=ACCENT, fg="#dbe6ff",
                 font=(FONT, 9)).pack(side=tk.RIGHT, padx=18)
        body = tk.Frame(root, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=18, pady=14)
        tk.Label(body, text="安装到:", bg=BG, fg=INK, font=f)\
            .pack(anchor=tk.W)
        row = tk.Frame(body, bg=BG)
        row.pack(fill=tk.X, pady=(4, 2))
        e = tk.Entry(row, textvariable=self.dir_var, font=(FONT, 9),
                     relief="flat", highlightthickness=1,
                     highlightbackground="#cdd6e4")
        e.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)
        tk.Button(row, text="浏览…", font=f, relief="flat", bg="#e6eaf2",
                  activebackground="#dfe5ef", command=self.browse)\
            .pack(side=tk.LEFT, padx=(8, 0), ipady=3)
        tk.Label(body, text="将创建桌面与开始菜单快捷方式，"
                            "可在系统“应用和功能”中卸载。",
                 bg=BG, fg=MUT, font=(FONT, 9)).pack(anchor=tk.W, pady=(8, 0))
        self.bar = ttk.Progressbar(body, maximum=100)
        self.bar.pack(fill=tk.X, pady=(16, 6))
        self.msg = tk.Label(body, text="准备就绪", bg=BG, fg=MUT,
                            font=(FONT, 9))
        self.msg.pack(anchor=tk.W)
        self.btn = tk.Button(body, text="立即安装", font=(FONT, 10, "bold"),
                             bg=ACCENT, fg="white", relief="flat",
                             activebackground="#2257d6", activeforeground="white",
                             command=self.start, padx=16, pady=6)
        self.btn.pack(anchor=tk.E, pady=(10, 0))
        self.run_chk = tk.Checkbutton(body, text="安装完成后立即运行",
                                      variable=self.run_var, bg=BG, fg=INK,
                                      font=f)
        self.run_chk.pack(anchor=tk.E, pady=(6, 0))

    def browse(self):
        d = filedialog.askdirectory(initialdir=self.dir_var.get() or
                                    os.path.expanduser("~"))
        if d:
            self.dir_var.set(os.path.join(d, APP))

    def start(self):
        self.btn.config(state=tk.DISABLED)
        target = self.dir_var.get().strip()
        if not target:
            messagebox.showwarning("提示", "请选择安装路径")
            self.btn.config(state=tk.NORMAL)
            return
        threading.Thread(target=self._work, args=(target,),
                         daemon=True).start()

    def _work(self, target: str):
        try:
            def prog(frac, msg):
                self.root.after(0, lambda: (self.bar.config(value=frac * 100),
                                            self.msg.config(text=msg)))
            do_install(target, prog)
        except Exception as e:                                  # noqa: BLE001
            self.root.after(0, lambda: messagebox.showerror(
                "安装失败", str(e)))
            self.root.after(0, lambda: self.btn.config(state=tk.NORMAL))
            return
        self.root.after(0, self._done, target)

    def _done(self, target: str):
        self.msg.config(text="安装完成!")
        self.bar.config(value=100)
        self.btn.pack_forget()
        self.run_chk.config(state=tk.DISABLED)
        if self.run_var.get():
            try:
                os.startfile(os.path.join(target, f"{APP}.exe"))  # noqa: S606
            except OSError:
                pass
        tk.Button(self.root, text="完成", font=(FONT, 10, "bold"),
                  bg=ACCENT, fg="white", relief="flat",
                  activebackground="#2257d6", padx=18, pady=5,
                  command=self.root.destroy).pack(side=tk.BOTTOM,
                                                  anchor=tk.E,
                                                  padx=18, pady=12)


def main() -> int:
    if "--silent" in sys.argv:
        import io
        if sys.stdout is None:
            sys.stdout = io.StringIO()
        if sys.stderr is None:
            sys.stderr = io.StringIO()
        try:
            target = default_dir()
            for d in sys.argv:
                if d.startswith("--dir="):
                    target = d[6:]
            do_install(target)
        finally:
            pass
        return 0
    root = tk.Tk()
    Wizard(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
