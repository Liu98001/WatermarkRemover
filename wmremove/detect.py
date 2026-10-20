# -*- coding: utf-8 -*-
"""自动找水印: 已知位置提示 + 画面内容分析(局部叠加检测) + 紧致掩码

设计思路
--------
平台台标 / AI 工具水印的位置大体固定在四角或边缘, 所以先用"已知位置提示框"
(相对比例)圈出候选区域, 再在这些窗口里用内容分析找出真正的叠加像素:
  * 图片: 用形态学背景模型, 找出比周围"更亮/更暗的一层细结构"(文字/描边/logo);
  * 视频: 额外用"时域标准差"——静态水印所在像素的时域波动明显低于周围动态背景,
          且带有稳定的锐利边缘, 两者结合就能把半透明水印圈出来。
"""

from __future__ import annotations

import cv2
import numpy as np

from .media import FrameReader, VideoInfo
from .shapes import dilate, erode, mask_bbox

# 已知水印位置先验(相对比例 x, y, w, h) —— 紧致框: 检测失败时直接用它作为
# 框选结果, 检测成功时在框内做内容分析并返回更紧的精修框。
# 位置按真实视频实测校准(优酷/YOUKU、小红书/底部条带等)。
KNOWN: dict[str, list[tuple[float, float, float, float, str]] | None] = {
    "豆包AI": [(0.01, 0.01, 0.30, 0.12, "左上"), (0.68, 0.88, 0.31, 0.11, "右下")],
    "即梦AI": [(0.70, 0.87, 0.29, 0.12, "右下")],
    "可灵AI": [(0.68, 0.87, 0.31, 0.12, "右下")],
    "通义/千问": [(0.68, 0.87, 0.31, 0.12, "右下")],
    "文心一言": [(0.68, 0.87, 0.31, 0.12, "右下")],
    "Sora": [(0.68, 0.87, 0.31, 0.12, "右下")],
    "Runway": [(0.68, 0.87, 0.31, 0.12, "右下")],
    "Pika": [(0.68, 0.87, 0.31, 0.12, "右下")],
    "抖音": [(0.02, 0.87, 0.30, 0.12, "左下用户名"), (0.76, 0.87, 0.23, 0.12, "右下logo")],
    "视频号": [(0.50, 0.87, 0.49, 0.12, "右下"), (0.01, 0.87, 0.34, 0.12, "左下")],
    "腾讯视频": [(0.70, 0.01, 0.29, 0.10, "右上"), (0.62, 0.87, 0.37, 0.12, "右下")],
    "爱奇艺": [(0.70, 0.01, 0.29, 0.10, "右上")],
    "优酷": [(0.83, 0.01, 0.16, 0.12, "右上YOUKU")],
    "快手": [(0.01, 0.01, 0.26, 0.11, "左上logo"), (0.02, 0.87, 0.30, 0.12, "左下ID")],
    "小红书": [(0.01, 0.90, 0.36, 0.10, "左下用户名"), (0.80, 0.91, 0.19, 0.08, "右下徽标")],
    "剪映": [(0.62, 0.87, 0.37, 0.12, "右下")],
    "B站": [(0.01, 0.01, 0.28, 0.11, "左上投稿标"), (0.68, 0.01, 0.31, 0.13, "右上"),
            (0.58, 0.87, 0.41, 0.12, "右下")],
    "微博": [(0.62, 0.87, 0.37, 0.12, "右下")],
    "顶部横幅": [(0.0, 0.0, 1.0, 0.08, "顶部")],
    "底部横幅": [(0.0, 0.92, 1.0, 0.08, "底部")],
    "四角": [],       # 特殊: 四个角都扫
    "全扫描": None,   # 特殊: 整幅画面
}

CORNERS = [(0.0, 0.0, 0.34, 0.16, "左上"), (0.66, 0.0, 0.34, 0.16, "右上"),
           (0.0, 0.84, 0.34, 0.16, "左下"), (0.66, 0.84, 0.34, 0.16, "右下")]


def hint_windows(name: str, W: int, H: int) -> list[tuple[int, int, int, int, str]]:
    if name in ("四角", "auto", ""):
        items = CORNERS
    elif name == "全扫描":
        items = [(0.0, 0.0, 1.0, 1.0, "全画面")]
    else:
        items = KNOWN.get(name, []) or CORNERS
    out = []
    for (rx, ry, rw, rh, tag) in items:
        x0 = max(0, int(rx * W))
        y0 = max(0, int(ry * H))
        x1 = min(W, int((rx + rw) * W))
        y1 = min(H, int((ry + rh) * H))
        if x1 - x0 > 8 and y1 - y0 > 6:
            out.append((x0, y0, x1, y1, tag))
    return out


# ------------------------------------------------------------ 局部叠加检测
def _local_noise(gray: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """局部噪声尺度 —— 用中值(而非均值)估计, 避免被水印自身的高对比边缘带偏"""
    hf = np.abs(gray.astype(np.float32) -
                cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma))
    med = cv2.medianBlur(hf.astype(np.float32), 5)
    return (1.4826 * med).astype(np.float32)


def _drop_small(mask: np.ndarray, min_px: int = 6) -> np.ndarray:
    n, lab, st, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    out = np.zeros_like(mask)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_px:
            out[lab == i] = 255
    return out


def overlay_mask(gray: np.ndarray, k: int = 9, thr: float = 5.0,
                 polarity: str = "auto", pad: int = 2,
                 region: np.ndarray | None = None,
                 noise_mult: float = 3.0) -> np.ndarray:
    """找出"叠在背景上的一层细结构"(文字/描边/角标) 的像素掩码。

    用 close+open 形态学核(比笔画宽)估计"没有这层结构时"的背景, 再做差;
    阈值 = max(thr, noise_mult × 局部鲁棒噪声), 所以细淡的水印也能抓到,
    同时不会被画面纹理误检。
    """
    g = gray.astype(np.float32)
    kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, k), max(3, k)))
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, kk)
    bg = cv2.morphologyEx(bg, cv2.MORPH_OPEN, kk)
    diff = g - bg
    noise = _local_noise(g)
    t = np.maximum(float(thr), noise_mult * noise)
    strong = np.abs(diff) > t
    if region is not None:
        strong &= region
    if polarity == "auto":
        # 取"成片"的那一侧极性(水印整体偏亮或偏暗)
        pos = float(diff[strong & (diff > 0)].sum()) if np.any(strong & (diff > 0)) else 0.0
        neg = float(-diff[strong & (diff < 0)].sum()) if np.any(strong & (diff < 0)) else 0.0
        polarity = "bright" if pos >= neg else "dark"
    if polarity == "bright":
        strong &= diff > 0
    elif polarity == "dark":
        strong &= diff < 0
    sm = (strong.astype(np.uint8)) * 255
    if pad > 0:
        sm = dilate(sm, pad)
    sm = cv2.morphologyEx(sm, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    sm = _drop_small(sm, 4)
    if region is not None:
        sm = cv2.bitwise_and(sm, (region.astype(np.uint8)) * 255)
    return sm


def refine_mask(frame: np.ndarray, region_mask: np.ndarray, k: int = 45,
                thr: float = 4.0, pad: int = 2, min_keep: float = 0.01,
                iters: int = 3) -> np.ndarray:
    """把粗框细化成"只覆盖水印"的紧致掩码(修复面积小 → 几乎看不出痕迹)。

    鲁棒迭代背景法:
      1) 先用形态学背景差分得到初始水印猜测;
      2) 把猜测掩码膨胀后用周边像素修复出"没有水印时的背景";
      3) 再做差分 → 得到更完整的水印掩码; 重复迭代直至收敛。
    比单纯形态学可靠得多: 不会漏掉淡色笔画, 也不会把背景纹理当水印。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    g = gray.astype(np.float32)
    reg = region_mask > 0
    # 噪声/纹理水平用"框外环带"估计(单一标量, 鲁棒): 决定阈值的高低
    hf_all = np.abs(g - cv2.GaussianBlur(g, (0, 0), 2.0))
    ring = dilate(region_mask, 10) > 0
    ring &= ~reg
    noise = float(np.median(hf_all[ring])) * 1.4826 if ring.any() else 2.0
    t_scalar = max(float(thr), 2.5 * noise)
    seed = overlay_mask(g, k=k, thr=t_scalar, pad=0, region=reg)
    if (seed > 0).sum() < 12:                # 框内几乎没有叠加结构
        return region_mask.copy()
    m = seed
    for _ in range(max(1, iters)):
        m_d = dilate(m, 9)                   # 大幅外扩: 让背景模型完全不含水印
        bg = cv2.inpaint(np.clip(np.round(g), 0, 255).astype(np.uint8), m_d, 15,
                         cv2.INPAINT_NS)
        bg = cv2.GaussianBlur(bg.astype(np.float32), (0, 0), 2.0)
        diff = g - bg
        t = t_scalar
        strong = np.abs(diff) > t
        # 极性与初始猜测保持一致
        pos = float(diff[strong & (diff > 0)].sum()) if np.any(strong & (diff > 0)) else 0
        neg = float(-diff[strong & (diff < 0)].sum()) if np.any(strong & (diff < 0)) else 0
        pol = "bright" if pos >= neg else "dark"
        strong &= (diff > 0) if pol == "bright" else (diff < 0)
        strong &= reg
        m = (strong.astype(np.uint8)) * 255
    keep = (m > 0).sum() / max(1.0, reg.sum())
    if keep < min_keep or keep > 0.9:        # 几乎没找到 / 几乎整框都是: 用原框
        return region_mask.copy()
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    m = _drop_small(m, 6)
    if pad > 0:
        m = dilate(m, pad)
    return cv2.bitwise_and(m, region_mask)


def box_of(mask: np.ndarray, pad: int = 2):
    return mask_bbox(mask, pad=pad)


# ---------------------------------------------------------------- 图片检测
def detect_image(img: np.ndarray, name: str = "四角", k: int = 9, thr: float = 6.0,
                 min_density: float = 0.01, min_edge: float = 0.6) -> list[dict]:
    """在提示窗口里找水印, 返回 [{box, mask, score, tag}]"""
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    named = name not in ("四角", "auto", "")
    out = []
    for (x0, y0, x1, y1, tag) in hint_windows(name, W, H):
        win = np.zeros((H, W), bool)
        win[y0:y1, x0:x1] = True
        m = overlay_mask(gray, k=k, thr=thr, pad=2, region=win)
        if (m > 0).sum() < 24:
            if named:                     # 指定平台: 内容分析失败也给先验框
                out.append({"box": (x0, y0, x1, y1), "mask": None,
                            "score": 0.1, "tag": f"{tag}(先验)", "name": name})
            continue
        dens = float((m > 0).sum()) / max(1.0, (x1 - x0) * (y1 - y0))
        edge = float(grad[m > 0].mean()) / (float(grad[win].mean()) + 1e-3)
        if dens < min_density or edge < min_edge:
            if named:                     # 找到但置信度低 → 用先验框兜底
                out.append({"box": (x0, y0, x1, y1), "mask": None,
                            "score": 0.1, "tag": f"{tag}(先验)", "name": name})
            continue
        bb = mask_bbox(m, pad=2)
        score = dens * 10 + edge
        out.append({"box": bb, "mask": m, "score": float(score), "tag": tag,
                    "name": name})
    out.sort(key=lambda d: -d["score"])
    return out


# ---------------------------------------------------------------- 视频检测
def detect_video(info: VideoInfo, name: str = "四角", n_samples: int = 20,
                 k: int = 45, thr: float = 5.0,
                 min_density: float = 0.002) -> list[dict]:
    """视频水印检测: 时域中值帧 + 空间叠加检测 + 时域稳定性评分。

    在各提示窗口里找"静态的细结构"(台标/AI 水印文字), 输出候选框。
    """
    n = info.n_frames
    idxs = np.linspace(0, max(0, n - 1), max(4, min(n_samples, n))).round().astype(int)
    frames = []
    want = set(int(i) for i in idxs)
    with FrameReader(info) as r:
        for i, f in r:
            if i in want:
                g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
                frames.append(g)
            if len(frames) >= len(want):
                break
    if len(frames) < 3:
        return []
    st = np.stack(frames)
    med = np.median(st, axis=0)
    tstd = st.std(axis=0)
    bg_std = cv2.GaussianBlur(tstd, (0, 0), 12) + 1e-3
    H, W = med.shape
    named = name not in ("四角", "auto", "")
    out = []
    for (x0, y0, x1, y1, tag) in hint_windows(name, W, H):
        win = np.zeros((H, W), bool)
        win[y0:y1, x0:x1] = True
        m = overlay_mask(med, k=k, thr=thr, pad=2, region=win)
        quiet = tstd < np.maximum(1.5 * bg_std, 2.5)
        m = cv2.bitwise_and(m, (quiet.astype(np.uint8)) * 255)
        found = (m > 0).sum() >= 40
        dens = float((m > 0).sum()) / max(1.0, (x1 - x0) * (y1 - y0))
        if not found and not named:
            continue                      # 通用扫描: 没把握就不报
        if found and dens >= min_density:
            # 内容分析找到了水印 → 返回紧致的"内容精修框"(轻微外扩盖住软边)
            bb = mask_bbox(m, pad=6)
            bb = (max(x0, bb[0]), max(y0, bb[1]), min(x1, bb[2]), min(y1, bb[3]))
        else:
            # 内容分析失败(背景晃动/复杂) → 用该平台的紧致先验框
            bb = (x0, y0, x1, y1)
            m = None
        score = (dens * 8 if found else 0.2) + \
            float((tstd[win].mean() - tstd[bb[1]:bb[3], bb[0]:bb[2]].mean()) /
                  (tstd[win].mean() + 1e-3))
        out.append({"box": bb, "mask": m, "score": float(score), "tag": tag,
                    "name": name})
    if not named:
        out.sort(key=lambda d: -d["score"])
    else:
        out.sort(key=lambda d: -d["score"])   # 内容证实的框排前, 先验兜底排后
    return out


def select_hits(hits: list[dict], max_n: int = 2) -> list[dict]:
    """选择最终采用的命中: 优先"内容证实"的框(评分>=1), 先验框仅兜底。

    避免把没有水印的先验区域也送去修复(空修 = 无谓损伤画面)。
    """
    solid = [h for h in hits if h.get("score", 0) >= 1.0]
    if solid:
        return solid[:max_n]
    return hits[:1]


def describe_hits(hits: list[dict]) -> str:
    if not hits:
        return "未检测到明显水印"
    s = []
    for h in hits[:4]:
        x0, y0, x1, y1 = h["box"]
        s.append(f"{h['tag']}({x0},{y0},{x1 - x0}x{y1 - y0}) 评分{h['score']:.2f}")
    return "; ".join(s)
