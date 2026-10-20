# -*- coding: utf-8 -*-
"""让"修好的区域"和周围颜色/亮度/颗粒一致 —— 消除违和感的关键步骤

三步:
  1) 低频对齐: 把区域内的整体明暗与色温, 平滑地对齐到紧邻干净环的水平
     (用谐波延展得到"该有的低频", 而不是硬把两者平均);
  2) 颗粒匹配: 区域内若比周围"太干净", 补上与周围同强度的细颗粒;
  3) 接缝评估: 量化边界处的梯度突变与色差, 用于自检/调参。
"""

from __future__ import annotations

import cv2
import numpy as np

from . import methods as M
from .shapes import dilate, erode


def gray(img: np.ndarray) -> np.ndarray:
    return (0.114 * img[..., 0] + 0.587 * img[..., 1] + 0.299 * img[..., 2])


def ring_of(mask, inner: int = 2, outer: int = 14) -> np.ndarray:
    a = dilate(mask.astype(np.uint8), inner)
    b = dilate(mask.astype(np.uint8), outer)
    return cv2.subtract(b, a) > 0


def region_mean(img: np.ndarray, mask) -> np.ndarray:
    m = mask.astype(bool)
    if m.sum() == 0:
        return np.zeros(3, np.float32)
    return img.astype(np.float32)[m].mean(axis=0)


def ring_mean(img: np.ndarray, mask, inner: int = 2, outer: int = 14) -> np.ndarray:
    r = ring_of(mask, inner, outer)
    if r.sum() == 0:
        return region_mean(img, mask)
    return img.astype(np.float32)[r].mean(axis=0)


def align_lowfreq(img: np.ndarray, mask, sigma: float = 25.0, strength: float = 1.0,
                  steps: int = 1, guard: bool = True,
                  verbose: bool = False) -> np.ndarray:
    """把掩码内的低频明暗/色温对齐到紧邻环带(逐通道), 边界处平滑无痕。

    两部分:
      1) 形状项 (ref - lf): ref 是低频图在掩码内的谐波延展, 在边界上与外面严格相等,
         所以这一项在边界自动归零 —— 只把区域内的明暗"纹理化地"拉向周边;
      2) 电平项 delta * f: delta = 环带均色 - 掩码内均色, f 是"内部为1、边界为0"的
         平滑斜坡, 用来把整体电平掰到和周围一致(边界处依旧为 0, 不会留台阶)。
    带 guard: 若对齐后"区域 vs 环带"的色差反而变大, 就放弃对齐(宁可不做也别做坏)。
    """
    src = img.astype(np.float32)
    m = mask.astype(bool)
    if not m.any():
        return img.copy()
    d0 = float(np.abs(region_mean(src, m) - ring_mean(src, m)).mean())
    s0 = seam_step(src, m)[0]
    cur = src
    dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
    mx = float(dist.max()) + 1e-6
    ramp = np.clip(dist / (mx * 0.5), 0, 1)[..., None]      # 边界 0 -> 内部 1
    for _ in range(max(1, steps)):
        lf = cv2.GaussianBlur(cur, (0, 0), float(sigma))
        ref = M.harmonic_fill(lf, m, levels=6, iters=40).astype(np.float32)
        shape = (ref - lf)
        delta = ring_mean(cur, m) - region_mean(ref, m)
        corr = (shape + delta.reshape(1, 1, 3) * ramp) * float(strength)
        w = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5)
        w = np.clip(w * 1.5, 0, 1)[..., None]
        cur = cur + corr * w
    out = src.copy()
    out[m] = cur[m]
    if guard:
        d1 = float(np.abs(region_mean(out, m) - ring_mean(out, m)).mean())
        s1 = seam_step(out, m)[0]
        if d1 > d0 + 0.05 or s1 > max(0.35, s0) * 1.5:
            if verbose:
                print(f"    低频对齐: 反而变差(色差 {d0:.2f}->{d1:.2f}, "
                      f"台阶 {s0:.2f}->{s1:.2f}), 已跳过")
            return img.copy()
    if verbose:
        print(f"    低频对齐: sigma={sigma} 区域均色 {region_mean(src, m).round(1)}"
              f" -> {region_mean(out, m).round(1)} (环外 {ring_mean(src, m).round(1)})")
    return np.clip(out, 0, 255).astype(np.uint8)


def match_grain(img: np.ndarray, mask, amount: float = 1.0, sigma: float = 22.0,
                seed: int = 3) -> np.ndarray:
    """区域内高频能量若低于周围, 补同强度细颗粒(否则会显得"糊了一块")。

    颗粒权重用较大尺度平滑淡出, 免得边界处"颗粒忽多忽少"反而露出痕迹。
    """
    src = img.astype(np.float32)
    m = mask.astype(bool)
    if not m.any() or amount <= 0:
        return img
    hf = src - cv2.GaussianBlur(src, (0, 0), sigma)
    r = ring_of(m, 2, 14)
    if r.sum() == 0:
        return img
    sig_ring = float(hf[r].std())
    sig_in = float(hf[m].std())
    add = max(0.0, sig_ring - sig_in)
    if add < 0.35:
        return img
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, add * amount, src.shape[:2]).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 0.7)
    w = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 3.0)
    w = np.clip(w * 1.6, 0, 1)[..., None]
    out = src + noise[..., None] * w
    out[~m] = src[~m]
    return np.clip(out, 0, 255).astype(np.uint8)


def color_match(img: np.ndarray, mask, strength: float = 1.0, sigma: float = 25.0,
                match_contrast: bool = False) -> np.ndarray:
    """颜色一致性总入口: 低频对齐(+可选对比度匹配)"""
    out = align_lowfreq(img, mask, sigma=sigma, strength=strength)
    if match_contrast:
        src = out.astype(np.float32)
        m = mask.astype(bool)
        r = ring_of(m, 2, 14)
        for c in range(3):
            a = float(src[..., c][m].std()) + 1e-3
            b = float(src[..., c][r].std()) + 1e-3
            k = float(np.clip(b / a, 0.7, 1.4))
            mu_i = float(src[..., c][m].mean())
            mu_o = float(src[..., c][r].mean())
            src[..., c][m] = (src[..., c][m] - mu_i) * k + mu_o
        out = np.clip(src, 0, 255).astype(np.uint8)
    return out


def composite(orig: np.ndarray, fixed: np.ndarray, mask, weight: np.ndarray | None = None,
              feather: float = 3.0) -> np.ndarray:
    """只替换掩码内(含羽化过渡), 掩码外原样保留"""
    m = mask.astype(bool)
    if weight is None:
        w = cv2.GaussianBlur(m.astype(np.float32), (0, 0), max(0.0, float(feather)))
        inside = m.astype(np.float32)
        w = np.maximum(w, inside)          # 内部严格为 1
    else:
        w = np.clip(weight, 0, 1)
    w = w[..., None]
    out = orig.astype(np.float32) * (1 - w) + fixed.astype(np.float32) * w
    return np.clip(out, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ 质量评估
def seam_step(img: np.ndarray, mask, band: int = 3) -> tuple[float, float, float]:
    """跨越掩码边界的"台阶"程度。

    返回 (台阶灰度, 周边自然起伏, 比值):
      * 台阶灰度: 边界内侧均值 - 外侧均值 (单位: 灰度)。<1 基本看不出, >2 开始显眼;
      * 比值: 台阶 / 周边同样尺度下的自然起伏, ≈1 表示完全融合。
    """
    g = gray(img.astype(np.float32))
    m = (mask.astype(bool)).astype(np.float32)
    md = (m > 0).astype(np.uint8)
    k = 2 * band + 1
    inner = cv2.blur(g * m, (k, k)) / np.maximum(cv2.blur(m, (k, k)), 1e-3)
    om = (1 - m)
    outer = cv2.blur(g * om, (k, k)) / np.maximum(cv2.blur(om, (k, k)), 1e-3)
    edge = cv2.subtract(dilate(md, 1), erode(md, 1)) > 0
    if edge.sum() == 0:
        return 0.0, 0.0, 1.0
    step = float(np.abs(inner - outer)[edge].mean())
    near = ring_of(md, 4, 16)
    if near.sum() == 0:
        return step, 0.0, 1.0
    b = cv2.blur(g, (k, k))
    d1 = np.abs(b - np.roll(b, band, axis=1))
    d2 = np.abs(b - np.roll(b, band, axis=0))
    nat = float((d1[near].mean() + d2[near].mean()) * 0.5) + 0.02
    return step, nat, step / nat


def seam_ratio(img: np.ndarray, mask, band: int = 3) -> float:
    """接缝比值(台阶/自然起伏), ≈1 无痕, >3 明显"""
    _s, _n, r = seam_step(img, mask, band)
    return r


def texture_energy(img: np.ndarray, mask, sigma: float = 6.0) -> float:
    """区域内的"细节能量"(高频标准差), 用于判断纹理有没有被抹平"""
    g = img.astype(np.float32)
    hf = g - cv2.GaussianBlur(g, (0, 0), sigma)
    m = mask.astype(bool)
    return float(hf[m].std()) if m.any() else 0.0


def report(src: np.ndarray, fixed: np.ndarray, mask) -> dict:
    """返回修复质量指标: 色差(区域 vs 环)、边界台阶(灰度)、接缝比、纹理能量比"""
    em = region_mean(fixed, mask)
    rm = ring_mean(fixed, mask)
    step, nat, ratio = seam_step(fixed, mask)
    step_src = seam_step(src, mask)[0] if src is not None else step
    e_in = texture_energy(fixed, mask)
    e_out = texture_energy(fixed, ring_of(mask, 2, 14))
    return {
        "色差(区域-环)": float(np.abs(em - rm).mean()),
        "灰度差": float(abs(gray(fixed.astype(np.float32))[mask.astype(bool)].mean() -
                           gray(fixed.astype(np.float32))[ring_of(mask, 2, 14)].mean())),
        "边界台阶": step,
        "台阶倍率": step / max(0.15, step_src),
        "接缝比": ratio,
        "纹理能量比": e_in / max(1e-3, e_out),
    }
