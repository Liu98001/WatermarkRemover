# -*- coding: utf-8 -*-
"""框选界面: 打开视频/图片 → 拖框(矩形/椭圆/多边形/涂抹) → 预览 → 一键去水印

操作
----
* 左侧工具条选形状: 矩形 / 椭圆 / 多边形(逐点单击, 双击结束) / 涂抹(按住拖动)
* 视频可用滑杆选择参考帧; "预览"看当前帧效果; 满意后"处理并保存"
* "自动检测"按平台/AI 角标位置提示自动找水印, 结果可再手动调整
"""

from __future__ import annotations

import base64

import os
import threading
import traceback

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from . import Options, Shape, engine
from .detect import detect_image, detect_video, describe_hits, select_hits
from .media import (FrameReader, default_out_path, imwrite_u, is_image, probe)
from .shapes import shapes_from_json, shapes_to_json

TOOLS = ("矩形", "椭圆", "多边形", "涂抹", "选择")
COLORS = {"矩形": "#ff4444", "椭圆": "#44aaff", "多边形": "#44dd66",
          "涂抹": "#ffcc00", "选择": "#ffffff"}

# 窗口图标(与安装图标同源, 48x48 PNG 内嵌, 避免依赖外部文件)
_ICON_B64 = (
    "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")


class Toggle(tk.Canvas):
    """现代拨动开关: 替代 ttk.Checkbutton 自带的"x"指示器"""

    OFF = "#c9d1de"

    def __init__(self, master, variable: tk.BooleanVar, command=None,
                 bg: str = "#ffffff", size: tuple[int, int] = (36, 20)):
        super().__init__(master, width=size[0], height=size[1], bg=bg,
                         highlightthickness=0, cursor="hand2")
        self.var = variable
        self.command = command
        self._tw, self._th = size
        self.bind("<Button-1>", self._flip)
        self._draw()

    def _flip(self, _e=None):
        self.var.set(not self.var.get())
        self._draw()
        if self.command:
            self.command()

    def _draw(self):
        w, h = self._tw, self._th
        r = h
        self.delete("all")
        color = App.ACCENT if self.var.get() else self.OFF
        self.create_oval(0, 0, r, h, fill=color, outline="")
        self.create_oval(w - r, 0, w, h, fill=color, outline="")
        self.create_rectangle(r // 2, 0, w - r // 2, h, fill=color, outline="")
        kx = w - h + 2 if self.var.get() else 2
        self.create_oval(kx, 2, kx + h - 4, h - 2, fill="#ffffff", outline="")


class App:
    # ---- 配色(浅色专业风) ----
    BG = "#eef1f6"          # 窗口底
    CARD = "#ffffff"        # 卡片
    INK = "#1c2333"         # 主文字
    MUT = "#7a8499"         # 次要文字
    LINE = "#dfe5ee"        # 分隔线
    ACCENT = "#2e6bff"      # 主色
    ACCENT_D = "#2257d6"    # 主色(按下)
    CANVAS_BG = "#171a21"   # 画布
    LOG_BG = "#12151c"      # 日志底
    LOG_FG = "#c9d4e8"      # 日志文字
    FONT = "Microsoft YaHei UI"

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("通用去水印工具")
        self._setup_style()
        self._apply_window_icon(root)
        w, h = 1280, 860
        root.minsize(1060, 700)
        root.update_idletasks()
        x = max(0, (root.winfo_screenwidth() - w) // 2)
        y = max(0, (root.winfo_screenheight() - h) // 2 - 24)
        root.geometry(f"{w}x{h}+{x}+{y}")
        self.path: str | None = None
        self.is_video = False
        self.info = None
        self.frame: np.ndarray | None = None      # 当前帧(原图坐标)
        self.frame_idx = tk.IntVar(value=0)
        self.shapes: list[Shape] = []
        self.tool = tk.StringVar(value="矩形")
        self.brush_r = tk.DoubleVar(value=14)
        self.sel_idx: int | None = None
        self.drag_start = None
        self.poly_pts: list[tuple[int, int]] = []
        self.stroke: list[tuple[int, int]] = []
        self.show_after = tk.BooleanVar(value=False)
        self.after_img: np.ndarray | None = None
        self._photo = None
        self.mask_overlay = None
        self.interactive = True                  # 自动化测试时可关掉弹窗
        self.method = tk.StringVar(value="auto")
        self._method_disp = tk.StringVar(value="自动选择(推荐)")
        self.tight = tk.BooleanVar(value=True)
        self.colorfit = tk.BooleanVar(value=True)
        self.temporal = tk.BooleanVar(value=True)
        self.dilate = tk.IntVar(value=0)
        self.feather = tk.DoubleVar(value=3.0)
        self.grain = tk.DoubleVar(value=0.0)
        self._build_ui()

    # ------------------------------------------------------------------ 样式
    def _apply_window_icon(self, win) -> None:
        """窗口图标与安装图标一致(主窗口与全部弹窗共用)。"""
        try:
            win.iconphoto(True, tk.PhotoImage(
                data=base64.b64decode(_ICON_B64)))
        except Exception:
            pass

    def _setup_style(self):
        self.root.configure(bg=self.BG)
        st = ttk.Style(self.root)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        f = self.FONT
        st.configure(".", background=self.BG, foreground=self.INK,
                     font=(f, 10), borderwidth=0, focuscolor=self.BG)
        st.configure("TFrame", background=self.BG)
        st.configure("Card.TFrame", background=self.CARD)
        st.configure("TLabel", background=self.BG, foreground=self.INK)
        st.configure("Card.TLabel", background=self.CARD, foreground=self.INK)
        st.configure("Mut.TLabel", background=self.BG, foreground=self.MUT,
                     font=(f, 9))
        st.configure("CardMut.TLabel", background=self.CARD, foreground=self.MUT,
                     font=(f, 9))
        st.configure("Title.TLabel", background=self.BG, foreground=self.INK,
                     font=(f, 15, "bold"))
        st.configure("TButton", background=self.CARD, foreground=self.INK,
                     padding=(10, 5), borderwidth=0, font=(f, 10))
        st.map("TButton",
               background=[("pressed", "#dfe5f2"), ("active", "#e8edf7")],
               foreground=[("disabled", "#9aa3b5")])
        st.configure("Accent.TButton", background=self.ACCENT, foreground="#ffffff",
                     padding=(12, 8), borderwidth=0, font=(f, 11, "bold"))
        st.map("Accent.TButton",
               background=[("disabled", "#9db4f7"), ("pressed", self.ACCENT_D),
                           ("active", self.ACCENT_D)],
               foreground=[("disabled", "#eef3ff")])
        st.configure("Tool.TRadiobutton", background=self.CARD, foreground=self.INK,
                     padding=(8, 5), font=(f, 10), indicatorcolor=self.CARD)
        st.map("Tool.TRadiobutton",
               background=[("selected", "#e7eeff"), ("active", "#eef2fa")],
               indicatorcolor=[("selected", self.ACCENT)])
        st.configure("TCheckbutton", background=self.CARD, foreground=self.INK,
                     padding=(2, 3), indicatorcolor=self.CARD, font=(f, 10))
        st.map("TCheckbutton",
               background=[("active", self.CARD)],
               indicatorcolor=[("selected", self.ACCENT)])
        st.configure("Card.TLabelframe", background=self.CARD, borderwidth=0,
                     relief="flat")
        st.configure("Card.TLabelframe.Label", background=self.CARD,
                     foreground=self.MUT, font=(f, 9, "bold"))
        st.configure("TSpinbox", background=self.CARD, foreground=self.INK,
                     arrowsize=11, padding=(4, 2), bordercolor=self.LINE,
                     lightcolor=self.LINE, darkcolor=self.LINE,
                     buttonbackground=self.CARD)
        st.configure("TEntry", padding=(4, 3))
        st.configure("TProgressbar", troughcolor=self.LINE, background=self.ACCENT,
                     borderwidth=0, thickness=8)
        st.configure("Horizontal.TScale", troughcolor=self.LINE,
                     background=self.ACCENT, borderwidth=0, lightcolor=self.ACCENT,
                     darkcolor=self.ACCENT)
        st.configure("TCombobox", arrowsize=12, padding=(4, 2))
        st.configure("TSeparator", background=self.LINE)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        # 底部状态栏(先 pack, 避免被扩展区挤掉)
        ttk.Separator(self.root, orient=tk.HORIZONTAL).pack(side=tk.BOTTOM,
                                                            fill=tk.X)
        sb = ttk.Frame(self.root, padding=(14, 5))
        sb.pack(side=tk.BOTTOM, fill=tk.X)
        self.status = ttk.Label(sb, text="请打开视频或图片", style="Mut.TLabel")
        self.status.pack(side=tk.LEFT)
        ttk.Label(sb, text="wmremove", style="Mut.TLabel").pack(side=tk.RIGHT)

        # 顶栏: 标题 + 工具按钮
        head = ttk.Frame(self.root, padding=(14, 10, 14, 2))
        head.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(head, text="去水印工具", style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(head, text="框选任意形状 · 颜色无痕融合 · 视频 / 图片 / 链接",
                  style="Mut.TLabel").pack(side=tk.LEFT, padx=10, pady=(4, 0))
        bar = ttk.Frame(self.root, padding=(12, 4, 12, 8))
        bar.pack(side=tk.TOP, fill=tk.X)
        for txt, cmd, dx in (("打开文件", self.open_file, 0),
                             ("打开链接", self.open_url, 6),
                             ("自动检测", self.auto_detect, 6),
                             ("保存区域", self.save_regions, 14),
                             ("载入区域", self.load_regions, 6)):
            ttk.Button(bar, text=txt, command=cmd).pack(side=tk.LEFT, padx=(dx, 0))

        # 中部
        mid = ttk.Frame(self.root)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

        # 左: 工具与参数(白色卡片)
        left = ttk.Frame(mid, width=272, style="Card.TFrame")
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8), pady=2)
        left.pack_propagate(False)
        inner = ttk.Frame(left, style="Card.TFrame", padding=10)
        inner.pack(fill=tk.BOTH, expand=True)
        lf1 = ttk.Labelframe(inner, text=" 工具 ", style="Card.TLabelframe")
        lf1.pack(fill=tk.X, pady=(0, 8))
        self.toolbox = []
        for t in TOOLS:
            b = ttk.Radiobutton(lf1, text=t, value=t, variable=self.tool,
                                style="Tool.TRadiobutton", command=self.on_tool)
            b.pack(fill=tk.X)
            self.toolbox.append(b)
        lf2 = ttk.Labelframe(inner, text=" 参数 ", style="Card.TLabelframe")
        lf2.pack(fill=tk.X, pady=(0, 8))

        def _prow(parent, label):
            r = ttk.Frame(parent, style="Card.TFrame")
            r.pack(fill=tk.X, pady=2, padx=2)
            ttk.Label(r, text=label, style="Card.TLabel").pack(side=tk.LEFT)
            return r

        r1 = _prow(lf2, "笔刷半径")
        ttk.Spinbox(r1, from_=2, to=80, textvariable=self.brush_r,
                    width=5).pack(side=tk.RIGHT)
        r2 = _prow(lf2, "掩码扩张")
        ttk.Spinbox(r2, from_=-20, to=60, textvariable=self.dilate,
                    width=5).pack(side=tk.RIGHT)
        r3 = _prow(lf2, "羽化(px)")
        ttk.Spinbox(r3, from_=0, to=20, increment=0.5, textvariable=self.feather,
                    width=5).pack(side=tk.RIGHT)
        r4 = _prow(lf2, "补颗粒")
        ttk.Spinbox(r4, from_=0, to=3, increment=0.1, textvariable=self.grain,
                    width=5).pack(side=tk.RIGHT)
        for txt, var in (("紧致掩码(只修笔画)", self.tight),
                         ("颜色/亮度对齐周围", self.colorfit),
                         ("时序分析(AI 淡入水印)", self.temporal)):
            r = ttk.Frame(lf2, style="Card.TFrame")
            r.pack(fill=tk.X, pady=3, padx=2)
            Toggle(r, var, bg=self.CARD).pack(side=tk.LEFT)
            ttk.Label(r, text=txt, style="Card.TLabel").pack(side=tk.LEFT,
                                                             padx=(8, 0))
        lf2b = ttk.Labelframe(inner, text=" 修复方式 ", style="Card.TLabelframe")
        lf2b.pack(fill=tk.X, pady=(0, 8))
        self.method = tk.StringVar(value="auto")
        self._method_disp = tk.StringVar(value="自动选择(推荐)")
        _mnames = ["自动选择(推荐)", "纹理搬运", "Telea 修复(文字)", "谐波填充",
                   "逐块合成(慢)", "模糊遮盖", "马赛克遮盖"]
        _mvals = ["auto", "texture", "telea", "harmonic", "synthesis", "blur",
                  "pixelate"]
        cb = ttk.Combobox(lf2b, textvariable=self._method_disp, values=_mnames,
                          state="readonly", font=(self.FONT, 9))
        cb.pack(fill=tk.X, padx=2, pady=2)
        cb.bind("<<ComboboxSelected>>",
                lambda _e: self.method.set(
                    _mvals[_mnames.index(self._method_disp.get())]))
        lf3 = ttk.Labelframe(inner, text=" 区域列表 ", style="Card.TLabelframe")
        lf3.pack(fill=tk.BOTH, expand=True)
        self.listbox = tk.Listbox(lf3, height=8, exportselection=False,
                                  bg=self.CARD, fg=self.INK, relief="flat",
                                  highlightthickness=0, selectbackground="#e7eeff",
                                  selectforeground=self.INK, activestyle="none",
                                  font=(self.FONT, 9))
        self.listbox.pack(fill=tk.BOTH, expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.on_select)
        lrow = ttk.Frame(lf3, style="Card.TFrame")
        lrow.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(lrow, text="删除选中", command=self.delete_selected)\
            .pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(lrow, text="全部清空", command=self.clear_shapes)\
            .pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

        # 中: 画布
        center = ttk.Frame(mid)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(center, bg=self.CANVAS_BG, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Double-Button-1>", self.on_double)
        self.canvas.bind("<Button-3>", self.on_right)
        self.root.bind("<Delete>", lambda e: self.delete_selected())
        self.root.bind("<Escape>", lambda e: self.cancel_draft())
        self.slider = ttk.Scale(center, from_=0, to=0, orient=tk.HORIZONTAL,
                                command=self.on_slider)
        self.slider.pack(fill=tk.X, pady=(6, 0))
        self.var_label = ttk.Label(center, text="", style="Mut.TLabel")
        self.var_label.pack(anchor=tk.W, pady=(2, 0))

        # 右: 预览与执行(白色卡片)
        right = ttk.Frame(mid, width=230, style="Card.TFrame")
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0), pady=2)
        right.pack_propagate(False)
        rin = ttk.Frame(right, style="Card.TFrame", padding=10)
        rin.pack(fill=tk.BOTH, expand=True)
        ttk.Button(rin, text="预览当前帧", command=self.preview).pack(fill=tk.X)
        rshow = ttk.Frame(rin, style="Card.TFrame")
        rshow.pack(fill=tk.X, pady=(8, 10))
        Toggle(rshow, self.show_after, command=self.redraw,
               bg=self.CARD).pack(side=tk.LEFT)
        ttk.Label(rshow, text="显示处理结果",
                  style="Card.TLabel").pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(rin, text="处 理 并 保 存", style="Accent.TButton",
                   command=self.process).pack(fill=tk.X, ipady=2)
        self.progress = ttk.Progressbar(rin, maximum=100)
        self.progress.pack(fill=tk.X, pady=(10, 12))
        self.log = tk.Text(rin, height=16, bg=self.LOG_BG, fg=self.LOG_FG,
                           relief="flat", highlightthickness=0, wrap="word",
                           insertbackground=self.LOG_FG,
                           selectbackground="#33467a",
                           font=("Consolas", 9))
        self.log.pack(fill=tk.BOTH, expand=True)
        self.log.tag_configure("mut", foreground="#8b97ad")
        self.log.insert("end", "提示:\n1. 打开视频/图片\n2. 拖框选择水印区域\n"
                               "3. 预览满意后点处理\n\n支持多区域、多形状;\n"
                               "AI 淡入水印会自动用时序分析精确去除。", "mut")

    def _ui(self, fn):
        """线程安全地回到主线程; 无主循环(自动化测试)时静默忽略"""
        try:
            self.root.after(0, fn)
        except RuntimeError:
            pass

    def logf(self, msg: str):
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.root.update_idletasks()

    # ------------------------------------------------------------- 文件载入
    def open_file(self):
        p = filedialog.askopenfilename(title="选择视频或图片",
                                       filetypes=[("媒体", "*.mp4 *.mov *.mkv "
                                                   "*.avi *.webm *.jpg *.jpeg "
                                                   "*.png *.webp *.bmp"),
                                                  ("所有文件", "*.*")])
        if p:
            self.load(p)

    def open_url(self):
        win = tk.Toplevel(self.root); win.title("从链接打开")
        win.geometry("620x250")
        ttk.Label(win, text="粘贴视频链接或分享文本，选择平台后下载\n"
                            "(支持抖音 / 小红书 / B站 / 快手 / 视频号 / "
                            "爱奇艺 / 优酷 / 腾讯视频 / 直链…)"
                  ).pack(anchor=tk.W, padx=10, pady=6)
        e = ttk.Entry(win, width=72); e.pack(padx=10, pady=4); e.focus()
        row = ttk.Frame(win)
        row.pack(fill=tk.X, padx=10, pady=(6, 0))
        ttk.Label(row, text="平台:").pack(side=tk.LEFT)
        pf = tk.StringVar(value="自动识别")
        ttk.Combobox(row, textvariable=pf, state="readonly", width=12,
                     values=["自动识别", "抖音", "小红书", "B站", "快手",
                             "微信视频号", "爱奇艺", "优酷", "腾讯视频",
                             "直链/其他"]).pack(side=tk.LEFT, padx=(4, 14))
        ttk.Label(row, text="清晰度:").pack(side=tk.LEFT)
        q = tk.StringVar(value="best")
        ttk.Combobox(row, textvariable=q, values=["best", "1080", "720", "480"],
                     width=8, state="readonly").pack(side=tk.LEFT, padx=4)
        row2 = ttk.Frame(win)
        row2.pack(fill=tk.X, padx=10, pady=(6, 0))
        ttk.Label(row2, text="保存到:").pack(side=tk.LEFT)
        sv = tk.StringVar(value="")
        ttk.Entry(row2, textvariable=sv, width=52).pack(side=tk.LEFT,
                                                        padx=(4, 4))

        def browse():
            d = filedialog.askdirectory(initialdir=sv.get() or os.path
                                        .expanduser("~"),
                                        title="选择保存目录")
            if d:
                sv.set(d)

        ttk.Button(row2, text="浏览…", width=8, command=browse).pack(
            side=tk.LEFT)

        def go():
            url = e.get().strip()
            plat = pf.get()
            out_dir = sv.get().strip()
            win.destroy()
            if not url:
                return
            if not out_dir:
                # 未填保存目录: 弹目录选择框; 取消则中止下载
                out_dir = filedialog.askdirectory(title="选择下载保存目录")
                if not out_dir:
                    self.status.config(text="已取消(未选择保存目录)")
                    return
            self.status.config(text="下载中…")
            self.logf(f"下载[{plat}]: {url[:70]}")

            def work():
                try:
                    from .downloader import download, extract_url
                    u = extract_url(url) or url
                    os.makedirs(out_dir, exist_ok=True)
                    p = download(u, out_dir, q.get(),
                                 platform=None if plat == "自动识别" else plat)
                    # 记住平台: 下载平台的水印位置先验直接用于"自动检测"
                    if plat != "自动识别":
                        self.last_platform = plat
                    self.root.after(0, lambda: self._after_download(p, plat))
                except Exception as ex:                       # noqa: BLE001
                    err = str(ex)
                    self.root.after(0, lambda: (self.status.config(text="下载失败"),
                                                messagebox.showerror("下载失败",
                                                                     err)))
            threading.Thread(target=work, daemon=True).start()

        ttk.Button(win, text="下载并打开", command=go).pack(pady=10)

    def _after_download(self, p: str, plat: str):
        """下载完成: 打开视频; 平台已知时自动按平台先验框选水印。

        新手流程: 粘贴链接 → 选目录 → 点「下载并打开」→ 框已画好 →
        点「处理并保存」。没找到水印时明确提示手动框选。
        """
        self.status.config(text=f"下载完成: {p}")
        self.logf(f"已保存: {p}")
        self.load(p)
        auto_name = {"抖音": "抖音", "小红书": "小红书", "B站": "B站",
                     "快手": "快手", "微信视频号": "视频号",
                     "爱奇艺": "爱奇艺", "优酷": "优酷",
                     "腾讯视频": "腾讯视频"}.get(plat)
        if auto_name and self.frame is not None:
            n = self.run_detect(auto_name)
            if n:
                self.status.config(text=f"已自动框选 {n} 处水印, "
                                        f"可调整后点「处理并保存」")
                return
        self.status.config(text="已打开, 请框选水印或点「自动检测」")

    def load(self, path: str):
        try:
            self.path = path
            self.is_video = not is_image(path)
            self.shapes.clear()
            self.after_img = None
            self.show_after.set(False)
            if self.is_video:
                self.info = probe(path)
                self.slider.configure(from_=0, to=max(0, self.info.n_frames - 1))
                self.frame_idx.set(int(self.info.n_frames * 0.25))
                self.load_video_frame()
                self.status.config(
                    text=f"{os.path.basename(path)}  {self.info.out_width}x"
                         f"{self.info.out_height}  {self.info.n_frames}帧")
            else:
                self.slider.configure(from_=0, to=0)
                self.frame = engine.imread_u(path)
                self.status.config(text=f"{os.path.basename(path)}  "
                                        f"{self.frame.shape[1]}x{self.frame.shape[0]}")
            self.refresh_list()
            self.redraw()
        except Exception as e:                                # noqa: BLE001
            messagebox.showerror("打开失败", str(e))

    def load_video_frame(self):
        idx = int(self.frame_idx.get())
        with FrameReader(self.info) as r:
            r.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            r.index = idx
            try:
                _, f = next(r)
                self.frame = f
            except StopIteration:
                self.frame = engine.first_frame(self.path)

    # ------------------------------------------------------------- 鼠标交互
    def to_img(self, ex, ey):
        return ((ex - self.dx) / self.scale, (ey - self.dy) / self.scale)

    def on_press(self, e):
        if self.frame is None:
            return
        self.canvas.focus_set()
        t = self.tool.get()
        x, y = self.to_img(e.x, e.y)
        if t == "选择":
            self.sel_idx = self.hit_shape(e.x, e.y)
            self.refresh_list()
            return
        if t == "多边形":
            self.poly_pts.append((x, y))
            self.redraw()
            return
        if t == "涂抹":
            self.stroke = [(x, y)]
            return
        self.drag_start = (x, y)

    def on_motion(self, e):
        t = self.tool.get()
        if t == "涂抹" and self.stroke:
            x, y = self.to_img(e.x, e.y)
            self.stroke.append((x, y))
            self.redraw()
        elif self.drag_start and t in ("矩形", "椭圆"):
            x, y = self.to_img(e.x, e.y)
            self.redraw(draft=(self.drag_start, (x, y), t))

    def on_release(self, e):
        t = self.tool.get()
        if not self.drag_start:
            self.redraw()
            return
        x0, y0 = self.drag_start
        x1, y1 = self.to_img(e.x, e.y)
        self.drag_start = None
        if t == "矩形":
            x0, x1 = sorted((x0, x1)); y0, y1 = sorted((y0, y1))
            if abs(x1 - x0) > 3 and abs(y1 - y0) > 3:
                self.shapes.append(Shape("rect", (x0, y0, x1 - x0, y1 - y0)))
        elif t == "椭圆":
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            rx, ry = abs(x1 - x0) / 2, abs(y1 - y0) / 2
            if rx > 3 and ry > 3:
                self.shapes.append(Shape("ellipse", (cx, cy, rx, ry)))
        elif t == "涂抹" and self.stroke:
            self.shapes.append(Shape("brush", list(self.stroke),
                                     radius=float(self.brush_r.get())))
            self.stroke = []
        self.refresh_list()
        self.redraw()

    def on_double(self, _e):
        if self.tool.get() == "多边形" and len(self.poly_pts) >= 3:
            self.shapes.append(Shape("polygon", list(self.poly_pts)))
            self.poly_pts = []
            self.refresh_list()
            self.redraw()

    def on_right(self, _e):
        self.cancel_draft()

    def cancel_draft(self):
        self.poly_pts = []
        self.stroke = []
        self.drag_start = None
        self.redraw()

    def hit_shape(self, ex, ey):
        x, y = self.to_img(ex, ey)
        for i, s in enumerate(self.shapes):
            m = s.to_mask(self.frame.shape[1], self.frame.shape[0])
            if 0 <= int(y) < m.shape[0] and 0 <= int(x) < m.shape[1] and \
                    m[int(y), int(x)] > 0:
                return i
        return None

    # ------------------------------------------------------------- 列表操作
    def refresh_list(self):
        self.listbox.delete(0, tk.END)
        for i, s in enumerate(self.shapes):
            self.listbox.insert(tk.END, f"{i + 1}. {s.to_spec()}")
        if self.sel_idx is not None and self.sel_idx < len(self.shapes):
            self.listbox.selection_set(self.sel_idx)

    def on_select(self, _e=None):
        sel = self.listbox.curselection()
        self.sel_idx = sel[0] if sel else None
        self.redraw()

    def delete_selected(self):
        if self.sel_idx is not None and self.sel_idx < len(self.shapes):
            del self.shapes[self.sel_idx]
            self.sel_idx = None
            self.refresh_list()
            self.redraw()

    def clear_shapes(self):
        self.shapes.clear()
        self.poly_pts = []
        self.sel_idx = None
        self.refresh_list()
        self.redraw()

    def save_regions(self):
        if not self.shapes:
            messagebox.showinfo("提示", "还没有任何区域")
            return
        p = filedialog.asksaveasfilename(defaultextension=".json",
                                         initialfile="regions.json")
        if p:
            shapes_to_json(self.shapes, p)
            self.logf(f"区域已保存: {p}")

    def load_regions(self):
        p = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not p:
            return
        try:
            self.shapes = shapes_from_json(p)
            self.refresh_list()
            self.redraw()
        except Exception as e:                                # noqa: BLE001
            messagebox.showerror("载入失败", str(e))

    # ------------------------------------------------------------- 自动检测
    def run_detect(self, name: str) -> int:
        """按位置提示检测水印并自动加框, 返回加框数量。

        供「自动检测」弹窗与链接下载后的自动框选共用(下载平台已知时
        无需再让用户选一遍)。
        """
        if self.frame is None:
            return 0
        try:
            hits = (detect_video(self.info, name) if self.is_video
                    else detect_image(self.frame, name))
        except Exception as e:                                # noqa: BLE001
            messagebox.showerror("检测失败", str(e))
            return 0
        chosen = select_hits(hits, 3)
        self.logf("自动检测: " + describe_hits(chosen))
        if not chosen:
            return 0
        for h in chosen:
            x0, y0, x1, y1 = h["box"]
            self.shapes.append(Shape("rect", (x0, y0, x1 - x0, y1 - y0),
                                     note=f"自动检测 {h['tag']}"))
        self.refresh_list()
        self.redraw()
        return len(chosen)

    def auto_detect(self):
        if self.frame is None:
            return
        win = tk.Toplevel(self.root); win.title("自动检测水印位置")
        ttk.Label(win, text="选择位置提示(也可选 四角 全扫描):\n"
                            "从链接下载时选择的平台会作为默认项").pack(
            anchor=tk.W, padx=10, pady=6)
        names = ["四角", "豆包AI", "即梦AI", "可灵AI", "通义/千问", "Sora",
                 "抖音", "视频号", "腾讯视频", "爱奇艺", "优酷", "快手",
                 "小红书", "剪映", "B站", "微博", "顶部横幅", "底部横幅",
                 "全扫描"]
        last = getattr(self, "last_platform", None)
        default = last if last in names else "四角"
        var = tk.StringVar(value=default)
        ttk.Combobox(win, textvariable=var, values=names, width=16,
                     state="readonly").pack(padx=10, pady=4)

        def go():
            name = var.get()
            win.destroy()
            if not self.run_detect(name):
                messagebox.showinfo("自动检测", "未检测到明显水印, 请手动框选")

        ttk.Button(win, text="检测", command=go).pack(pady=6)

    # ------------------------------------------------------------------ 渲染
    def on_slider(self, v):
        if self.is_video and self.info:
            self.frame_idx.set(int(float(v)))
            self.load_video_frame()
            self.after_img = None
            self.show_after.set(False)
            self.redraw()

    def on_tool(self):
        self.redraw()

    def _display_img(self):
        if self.frame is None:
            return None
        img = self.after_img if (self.show_after.get() and
                                 self.after_img is not None) else self.frame
        H, W = img.shape[:2]
        cw = max(200, self.canvas.winfo_width())
        ch = max(150, self.canvas.winfo_height())
        self.scale = min(cw / W, ch / H, 1.6)
        self.dx = (cw - W * self.scale) / 2
        self.dy = (ch - H * self.scale) / 2
        disp = cv2.resize(img, (int(W * self.scale), int(H * self.scale)),
                          interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)

    def redraw(self, draft=None):
        if self.frame is None:
            return
        disp = self._display_img()
        if disp is None:
            return
        self._photo = ImageTk.PhotoImage(Image.fromarray(disp))
        self.canvas.delete("all")
        self.canvas.create_image(self.dx, self.dy, image=self._photo, anchor=tk.NW)
        W, H = self.frame.shape[1], self.frame.shape[0]

        def sx(pt):
            return self.dx + pt[0] * self.scale, self.dy + pt[1] * self.scale

        for i, s in enumerate(self.shapes):
            col = COLORS.get(s.kind, "#ffffff")
            width = 3 if i == self.sel_idx else 2
            if s.kind == "rect":
                x, y, w, h = s.data
                self.canvas.create_rectangle(*sx((x, y)), *sx((x + w, y + h)),
                                             outline=col, width=width)
            elif s.kind == "ellipse":
                cx, cy, rx, ry = s.data
                self.canvas.create_oval(*sx((cx - rx, cy - ry)),
                                           *sx((cx + rx, cy + ry)),
                                           outline=col, width=width)
            elif s.kind == "polygon":
                pts = []
                for p in s.data:
                    pts += sx(p)
                if len(pts) >= 6:
                    self.canvas.create_polygon(*pts, fill="", outline=col,
                                               width=width)
            elif s.kind == "brush":
                pts = []
                for p in s.data:
                    pts += sx(p)
                if len(pts) >= 4:
                    self.canvas.create_line(*pts, fill=col, width=max(
                        2, int(s.radius * self.scale * 2)), capstyle=tk.ROUND,
                        joinstyle=tk.ROUND)
        if self.poly_pts:
            pts = []
            for p in self.poly_pts:
                pts += sx(p)
            if len(pts) >= 4:
                self.canvas.create_line(*pts, fill=COLORS["多边形"], width=2)
        if draft:
            (x0, y0), (x1, y1), t = draft
            col = COLORS.get(t, "#ffffff")
            if t == "矩形":
                self.canvas.create_rectangle(*sx((x0, y0)), *sx((x1, y1)),
                                             outline=col, width=2, dash=(4, 2))
            else:
                self.canvas.create_oval(*sx((x0, y0)), *sx((x1, y1)),
                                           outline=col, width=2, dash=(4, 2))
        if self.stroke and len(self.stroke) >= 2:
            pts = []
            for p in self.stroke:
                pts += sx(p)
            self.canvas.create_line(*pts, fill=COLORS["涂抹"],
                                    width=max(2, int(self.brush_r.get() *
                                                     self.scale * 2)),
                                    capstyle=tk.ROUND)
        self.var_label.config(text=f"缩放 {self.scale:.2f}x   "
                                   f"帧 {self.frame_idx.get()}"
                                   f"{'' if not self.is_video else '/' + str(self.info.n_frames)}"
                                   f"   区域 {len(self.shapes)} 个   "
                                   f"({'处理后' if self.show_after.get() else '原图'})")

    # ------------------------------------------------------------------ 预览
    def current_opts(self) -> Options:
        return Options(method=self.method.get(), feather=float(self.feather.get()),
                       tight=bool(self.tight.get()),
                       colorfit=bool(self.colorfit.get()),
                       temporal=bool(self.temporal.get()),
                       dilate_px=int(self.dilate.get()), grain=float(self.grain.get()))

    def preview(self):
        if self.frame is None or not self.shapes:
            messagebox.showinfo("提示", "请先打开文件并框选区域")
            return
        H, W = self.frame.shape[:2]
        try:
            mask, _w, _raw = engine.build_mask(self.shapes, W, H,
                                               self.current_opts(),
                                               ref_frames=[self.frame],
                                               verbose=False)
        except Exception as e:                                # noqa: BLE001
            messagebox.showerror("掩码错误", str(e)); return
        self.after_img, _mask, rep = engine.repair_image(self.frame, self.shapes,
                                                         self.current_opts())
        self.show_after.set(True)
        self.redraw()
        self.logf(f"预览: 色差 {rep['色差(区域-环)']:.2f}, 边界台阶 "
                  f"{rep['边界台阶']:.2f} 灰度 (<1.5 基本无痕)")

    # ------------------------------------------------------------------ 处理
    def process(self):
        if self.frame is None or not self.shapes:
            messagebox.showinfo("提示", "请先打开文件并框选区域")
            return
        stem = os.path.splitext(os.path.basename(self.path))[0]
        ext = ".mp4" if self.is_video else ".png"
        if stem.endswith("_no_watermark"):          # 工具下载的文件: 不叠加后缀
            initialfile = stem + "_edited" + ext
        else:
            initialfile = stem + "_no_watermark" + ext
        dst = filedialog.asksaveasfilename(
            defaultextension=ext, initialfile=initialfile)
        if not dst:
            return
        # 保护: 不允许把输出写到源文件自身(避免处理中断/失败时丢掉原片)
        if os.path.abspath(dst) == os.path.abspath(self.path):
            messagebox.showerror(
                "不能覆盖原文件",
                "输出路径与源文件相同, 这样会破坏原片。\n"
                "请另选一个文件名保存。")
            return
        self.start_process(dst)

    def start_process(self, dst: str):
        """真正启动处理(线程 + 进度条); 供按钮与自动化测试调用"""
        self.progress["value"] = 0
        state = {"n": 0, "total": self.info.n_frames if self.is_video else 1,
                 "done": False}
        self._state = state
        opts = self.current_opts()

        def work():
            try:
                if self.is_video:
                    engine.process_video(self.path, dst, list(self.shapes), opts,
                                         verbose=False,
                                         progress=lambda a, b: state.update(
                                             n=a, total=b))
                else:
                    engine.process_image(self.path, dst, list(self.shapes), opts,
                                         verbose=False)
                state["done"] = True
                self._ui(lambda: self._done(dst, None))
            except Exception as e:                            # noqa: BLE001
                traceback.print_exc()
                state["done"] = True
                self._ui(lambda: self._done(None, str(e)))

        def poll():
            if state.get("done"):
                return
            t = state["total"] or 1
            self.progress["value"] = state["n"] / t * 100
            self.root.after(200, poll)

        threading.Thread(target=work, daemon=True).start()
        poll()

    def _done(self, dst, err):
        self.progress["value"] = 100
        if err:
            if self.interactive:
                messagebox.showerror("处理失败", err)
            self.logf("失败: " + err)
        else:
            self.logf(f"完成 -> {dst}")
            self.status.config(text="完成: " + dst)
            if self.interactive and messagebox.askyesno(
                    "完成", "处理完成, 打开输出文件夹?"):
                os.startfile(os.path.dirname(os.path.abspath(dst)))

    # ------------------------------------------------------------------ 其它
    def on_tool_change(self):
        pass


def main() -> int:
    root = tk.Tk()
    app = App(root)
    root.after(300, app.redraw)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
