# -*- coding: utf-8 -*-
"""修复算法集合: 谐波填充 / OpenCV inpaint / 多尺度纹理块合成 / 邻域克隆 / 模糊 / 马赛克

所有函数签名统一为  f(img, mask, **kw) -> out
img:  uint8 或 float32 的 HxWx3 (BGR); mask: uint8 0/255 或 bool
"""

from __future__ import annotations

import time

import cv2
import numpy as np


# --------------------------------------------------------------- 基础工具
def _f(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) if img.dtype != np.float32 else img


def _u8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def _m(mask) -> np.ndarray:
    if mask.dtype == np.uint8:
        return mask > 127
    return mask.astype(bool)


def _kernel(r: int) -> np.ndarray:
    r = max(1, int(r))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


# --------------------------------------------------- 谐波填充(平滑低频补全)
def harmonic_fill(img: np.ndarray, mask, levels: int = 6, iters: int = 40,
                  refine: int = 40, verbose: bool = False) -> np.ndarray:
    """把掩码内平滑地"补"成周边颜色/明暗的自然延续(求解 Laplace 问题)。

    先用 cv2.inpaint(Navier-Stokes, 收敛快) 得到初值, 再做 refine 轮
    Jacobi 扩散, 抹掉 NS 的轻微条纹并逼近真正的谐波解 —— 大块区域也不会拖影。
    """
    src = _f(img)
    m = _m(mask)
    H, W = m.shape
    if not m.any():
        return _u8(src)
    if not (~m).any():                      # 整幅都是待修区域, 无从下手
        return _u8(src)
    md = (m.astype(np.uint8)) * 255
    u8 = np.clip(np.round(src), 0, 255).astype(np.uint8)
    arr = cv2.inpaint(u8, md, max(3, int(min(H, W) // 40)),
                      cv2.INPAINT_NS).astype(np.float32)
    # 多尺度 Jacobi 精修: 先粗后细, 每级接着上一级的结果
    cur = arr
    for lv in range(max(0, levels - 1), -1, -1):
        scale = 2 ** lv
        h2, w2 = max(8, H // scale), max(8, W // scale)
        if (h2, w2) != (H, W):
            small = cv2.resize(cur, (w2, h2), interpolation=cv2.INTER_AREA)
            msmall = cv2.resize(md, (w2, h2), interpolation=cv2.INTER_NEAREST) > 127
        else:
            small, msmall = cur, m
        n_it = max(6, refine // (lv + 1))
        for _ in range(n_it):
            blur = cv2.blur(small, (3, 3))
            small[msmall] = blur[msmall]
        if (h2, w2) != (H, W):
            up = cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)
            up[~m] = src[~m]
            cur = up
        else:
            cur = small
    out = src.copy()
    out[m] = np.asarray(cur, np.float32)[m]
    if verbose:
        print(f"    harmonic_fill: NS初值 + {refine} 轮精修, mask={int(m.sum())}px")
    return _u8(out)


# ------------------------------------------------- 单侧均匀带延展
def side_fill(img: np.ndarray, mask, band: int = 24, smooth: int = 9,
              sd_max: float = 55.0, verbose: bool = False) -> np.ndarray | None:
    """从掩码旁"最均匀"的一条干净带做单侧延展; 不可行返回 None。

    大块区域的 telea 会从异质边界扩散颜色, 涂出污渍/发白; harmonic 是
    四面边界的调和平均, 边界含水印软边时结果整体发白。而掩码区背景往往
    与某一侧带同质(黑板/桌面/墙面), 从那一侧延展颜色最贴背景 —— 修复区
    与背景颜色一致的关键。带的均匀性与轮廓只取"与掩码相邻的那一段"。
    """
    src = _f(img)
    m = _m(mask)
    H, W = m.shape
    if not m.any():
        return None
    ys, xs = np.where(m)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    g = cv2.cvtColor(_u8(src), cv2.COLOR_BGR2GRAY).astype(np.float32)
    cands = []
    if x0 >= 6:
        cands.append((float(g[y0:y1 + 1, max(0, x0 - band):x0].std()), "left"))
    if x1 <= W - 7:
        cands.append((float(g[y0:y1 + 1,
                              x1 + 1:min(W, x1 + 1 + band)].std()), "right"))
    if y0 >= 6:
        cands.append((float(g[max(0, y0 - band):y0, x0:x1 + 1].std()), "top"))
    if y1 <= H - 7:
        cands.append((float(g[y1 + 1:min(H, y1 + 1 + band),
                              x0:x1 + 1].std()), "bottom"))
    if not cands:
        return None
    cands.sort(key=lambda c: c[0])
    sd, side = cands[0]
    if sd > sd_max:
        return None
    k = max(3, int(smooth)) | 1
    fill = np.zeros((H, W, 3), np.float32)
    if side in ("left", "right"):
        sl = (src[y0:y1 + 1, max(0, x0 - band):x0] if side == "left"
              else src[y0:y1 + 1, x1 + 1:min(W, x1 + 1 + band)])
        prof = cv2.GaussianBlur(np.median(sl, axis=1), (1, k), 0)   # (seg,3)
        fill[y0:y1 + 1] = np.repeat(prof[:, None, :], W, axis=1)
    else:
        sl = (src[max(0, y0 - band):y0, x0:x1 + 1] if side == "top"
              else src[y1 + 1:min(H, y1 + 1 + band), x0:x1 + 1])
        prof = cv2.GaussianBlur(np.median(sl, axis=0), (k, 1), 0)   # (seg,3)
        fill[y0:y1 + 1, x0:x1 + 1] = np.repeat(prof[None, :, :],
                                               y1 - y0 + 1, axis=0)
    # 颜色一致性: 延展色须与"所选侧"的紧邻环带(掩码外4px同侧段)一致。
    if side in ("left", "right"):
        rseg = (src[y0:y1 + 1, max(0, x0 - 4):x0] if side == "left"
                else src[y0:y1 + 1, x1 + 1:min(W, x1 + 5)])
    else:
        rseg = (src[max(0, y0 - 4):y0, x0:x1 + 1] if side == "top"
                else src[y1 + 1:min(H, y1 + 5), x0:x1 + 1])
    if rseg.size:
        pv = np.median(prof.reshape(-1, 3), axis=0)
        rv = np.median(rseg.reshape(-1, 3), axis=0)
        if float(np.abs(pv - rv).max()) > 30.0:
            return None
    out = src.copy()
    out[m] = fill[m]
    if verbose:
        print(f"    side_fill: side={side} 带差异={sd:.1f}")
    return np.clip(out, 0, 255).astype(np.uint8)


# ------------------------------------------------------- OpenCV inpaint
def inpaint_cv(img: np.ndarray, mask, radius: int = 3, telea: bool = True) -> np.ndarray:
    m = (_m(mask).astype(np.uint8)) * 255
    flags = cv2.INPAINT_TELEA if telea else cv2.INPAINT_NS
    out = cv2.inpaint(_u8(_f(img)), m, max(1, int(radius)), flags)
    return out


# ------------------------------------------------- 竖向结构保持填充
def vertical_fill(img: np.ndarray, mask, band: int = 16, smooth: int = 5,
                  band_mask=None, verbose: bool = False) -> np.ndarray:
    """逐列竖向外推: 用参考带上/下方的干净带做竖向线性插值, 填充 mask 区域。

    band_mask: 参考(决定上下带位置)的掩码, 缺省 = mask。
    关键: 参考带必须"紧贴水印区域"(那里的玻璃与水印处最同质);
    而 mask(填充范围)可以比参考掩码大 —— 把同一模型外延到水印软边缘。

    适用: 竖直结构为主的表面(玻璃瓶身的反光带/瓶壁暗边、柱面、幕墙等)。
    因为是"逐列"填充, 竖直结构被原样延续; 上下带各取 band 行做逐列均值,
    再在两带之间按行线性过渡; 最后对"填充结果"做竖向平滑消除行间噪声。
    """
    src = _f(img)
    m = _m(mask)
    H, W = m.shape
    out = src.copy()
    if not m.any():
        return _u8(out)
    ref = _m(band_mask) if band_mask is not None else m
    if not ref.any():
        ref = m
    band = max(2, int(band))
    ys, _xs = np.where(ref)
    y0, y1 = int(ys.min()), int(ys.max())
    top = src[max(0, y0 - band):y0, :, :].mean(axis=0) if y0 > 0 else None
    bot = src[y1 + 1:min(H, y1 + 1 + band), :, :].mean(axis=0) \
        if y1 + 1 < H else None
    if top is None and bot is None:
        return inpaint_cv(img, m, 3)
    if top is None:
        top = bot
    if bot is None:
        bot = top
    # 参考带里"很暗"的列 = 导管/深色竖管所在列 → 用左右邻列线性插值补齐,
    # 避免把深色结构外推进填充区(阈值 = 带内中位数 - 60, 只剔真正的深色结构)
    for _bi, arr in enumerate((top, bot)):
        gcol = 0.114 * arr[..., 0] + 0.587 * arr[..., 1] + 0.299 * arr[..., 2]
        bad = gcol < (float(np.median(gcol)) - 60.0)
        if bad.sum() == 0 or bad.sum() >= 0.4 * len(gcol):
            continue
        idx = np.arange(len(gcol))
        good = ~bad
        for x in idx[bad]:
            l = idx[(idx < x) & good]
            r = idx[(idx > x) & good]
            if len(l) and len(r):
                k = (x - l[-1]) / (r[0] - l[-1])
                arr[x] = arr[l[-1]] * (1 - k) + arr[r[0]] * k
            elif len(l):
                arr[x] = arr[l[-1]]
            else:
                arr[x] = arr[r[0]]
    ks = max(1, int(smooth)) | 1                 # 奇数
    top = cv2.blur(top[None, ...], (ks, 1))[0]   # 轻度横向平滑去噪
    bot = cv2.blur(bot[None, ...], (ks, 1))[0]
    t = np.zeros(H, np.float32)
    if y1 > y0:
        t[y0:y1 + 1] = np.linspace(0.0, 1.0, y1 - y0 + 1)
    fill = top[None, :, :] * (1 - t[:, None, None]) + bot[None, :, :] * t[:, None, None]
    fill = fill.astype(np.float32)
    fill = cv2.GaussianBlur(fill, (1, max(3, smooth) | 1), 0)   # 竖向平滑去噪
    out[m] = fill[m]
    if verbose:
        print(f"    vertical_fill: band={band} y {y0}~{y1}")
    return _u8(out)


# ------------------------------------------------- 横向结构保持填充
def horizontal_fill(img: np.ndarray, mask, band: int = 24, smooth: int = 5,
                    band_mask=None, verbose: bool = False) -> np.ndarray:
    """逐行横向外推: 用参考带左/右方的干净带做横向线性插值填充。

    适用: 贴着画面左右边或底边的横条水印(台标/用户名/平台徽标) ——
    此时竖向填充没有下方参考带, 而地面/墙面等背景的纹理是横向延续的。
    band_mask: 决定左右参考带位置的掩码(紧贴水印), 缺省 = mask;
    mask(填充范围)可比参考掩码大, 把同一模型外延到水印软边缘。
    """
    src = _f(img)
    m = _m(mask)
    H, W = m.shape
    out = src.copy()
    if not m.any():
        return _u8(out)
    ref = _m(band_mask) if band_mask is not None else m
    if not ref.any():
        ref = m
    band = max(2, int(band))
    _ys, xs = np.where(ref)
    x0, x1 = int(xs.min()), int(xs.max())
    left = src[:, max(0, x0 - band):x0, :].mean(axis=1) if x0 > 0 else None
    right = src[:, x1 + 1:min(W, x1 + 1 + band), :].mean(axis=1) \
        if x1 + 1 < W else None
    if left is None and right is None:
        return inpaint_cv(img, m, 3)
    if left is None:
        left = right
    if right is None:
        right = left
    # 参考带里"很暗"的行 = 深色横线/结构 → 上下邻行线性插值补齐
    for arr in (left, right):
        grow = 0.114 * arr[..., 0] + 0.587 * arr[..., 1] + 0.299 * arr[..., 2]
        bad = grow < (float(np.median(grow)) - 60.0)
        if bad.sum() == 0 or bad.sum() >= 0.4 * len(grow):
            continue
        idx = np.arange(len(grow))
        good = ~bad
        for y in idx[bad]:
            u = idx[(idx < y) & good]
            d = idx[(idx > y) & good]
            if len(u) and len(d):
                k = (y - u[-1]) / (d[0] - u[-1])
                arr[y] = arr[u[-1]] * (1 - k) + arr[d[0]] * k
            elif len(u):
                arr[y] = arr[u[-1]]
            else:
                arr[y] = arr[d[0]]
    ks = max(1, int(smooth)) | 1
    left = cv2.blur(left[:, None, :], (1, ks))[:, 0, :]
    right = cv2.blur(right[:, None, :], (1, ks))[:, 0, :]
    t = np.zeros(W, np.float32)
    if x1 > x0:
        t[x0:x1 + 1] = np.linspace(0.0, 1.0, x1 - x0 + 1)
    fill = left[:, None, :] * (1 - t[None, :, None]) + right[:, None, :] * t[None, :, None]
    fill = fill.astype(np.float32)
    fill = cv2.GaussianBlur(fill, (max(3, smooth) | 1, 1), 0)   # 横向平滑去噪
    out[m] = fill[m]
    if verbose:
        print(f"    horizontal_fill: band={band} x {x0}~{x1}")
    return _u8(out)


# ------------------------------------------------- 多尺度纹理块合成(补纹理)
def _pad(img: np.ndarray, mask: np.ndarray, pad: int):
    img_p = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    m_p = cv2.copyMakeBorder(mask.astype(np.uint8), pad, pad, pad, pad,
                             cv2.BORDER_REPLICATE) > 0
    return img_p, m_p


def texture_fill(img: np.ndarray, mask, sigma: float = 6.0, max_shift: int = 56,
                 offsets: tuple = (), n_best: int = 3, verbose: bool = False) -> np.ndarray:
    """"低频谐波填充 + 高频纹理搬运" —— 快速、通用、不会糊成一片。

    * 低频/结构: harmonic_fill 保证区域内明暗/颜色/大结构是周边的自然延续;
    * 高频纹理: 从周围"最吻合的几处干净区域"把细节纹理平移搬过来, 按吻合度加权融合。
      (sigma 取小一点, 只搬运细纹理, 避免把别处的结构错位搬过来)
    """
    t0 = time.time()
    src = _f(img)
    m = _m(mask)
    H, W = m.shape
    if not m.any():
        return _u8(src)
    base = harmonic_fill(src, m, verbose=False).astype(np.float32)
    md = m.astype(np.uint8)
    band = cv2.subtract(cv2.dilate(md, _kernel(10)), cv2.dilate(md, _kernel(4))) > 0
    grow = cv2.dilate(md, _kernel(2)) > 0            # 不允许把水印附近像素搬进来
    # 周围本来就"很光"的话, 就别硬搬纹理进去(否则反而像糊了一层噪点)
    ring = band
    hf_ring = src - cv2.GaussianBlur(src, (0, 0), 6.0)
    ring_energy = float(hf_ring[ring].std()) if ring.any() else 0.0
    tex_gain = float(np.clip((ring_energy - 1.2) / 2.2, 0.0, 1.0))

    # 候选平移: 四方向 + 对角, 距离覆盖到"比水印本身更远"
    cand = []
    if offsets:
        cand = [(int(dx), int(dy)) for dx, dy in offsets]
    else:
        span = int(max(max_shift, 1.15 * max(md.shape))) if md.any() else max_shift
        for d in (6, 10, 16, 24, 34, 48, 66, 88, 112):
            if d <= span:
                cand += [(d, 0), (-d, 0), (0, d), (0, -d)]
        for d in (12, 22, 36, 56, 80):
            if d <= span:
                cand += [(d, d), (-d, -d), (d, -d), (-d, d)]
    scored = []
    for dx, dy in cand:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        sh = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)
        shm = cv2.warpAffine(md, M, (W, H), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
        valid = ~(shm & grow)                       # 该平移对这个像素是否"干净可用"
        cov = float((valid & m).sum()) / max(1.0, float(m.sum()))
        if cov < 0.05:
            continue
        # 亮度一致性惩罚: 平移源在掩码区的均值必须接近谐波底色, 否则会
        # 把明暗完全不同的内容(如画面顶部的黑色灯架)整片搬进修复区,
        # 形成大块色斑(实测舞台场景顶部黑斑)。
        lum_pen = float(np.abs(sh[m].mean() - base[m].mean())) * 0.8
        score = float(np.abs(sh[band] - src[band]).mean()) + (1 - cov) * 30 \
            + lum_pen
        scored.append((score, dx, dy, sh, valid))
    if not scored:
        if verbose:
            print("    texture_fill: 无可用平移, 仅用谐波填充")
        out = src.copy()
        out[m] = base[m]
        return _u8(out)
    scored.sort(key=lambda x: x[0])
    chosen = scored[:max(1, n_best)]
    hf_acc = np.zeros_like(src)
    wsum = np.zeros((H, W, 1), np.float32)
    for rank, (score, _dx, _dy, sh, valid) in enumerate(chosen):
        hf = sh - cv2.GaussianBlur(sh, (0, 0), sigma)
        wgt = (1.0 / (score + 2.0)) * valid.astype(np.float32)
        hf_acc += hf * wgt[..., None]
        wsum += wgt[..., None]
    hf_acc = hf_acc / np.maximum(wsum, 1e-6)
    # 边界 2~3px 内的纹理稍作平滑, 让搬运来的纹理与外面自然接上
    near_edge = cv2.subtract(md * 255, cv2.erode(md * 255, _kernel(3))) > 0
    if near_edge.any():
        soft = cv2.GaussianBlur(hf_acc, (0, 0), 1.6)
        hf_acc[near_edge] = soft[near_edge]
    hf_acc = np.where(wsum > 0, hf_acc, 0.0) * tex_gain
    # 边界处纹理淡入(由谐波填充接管边界), 避免搬运纹理在边界产生可见断层
    dist = cv2.distanceTransform(md, cv2.DIST_L2, 3)
    taper = np.clip(dist / 5.0, 0, 1)[..., None]
    hf_acc = hf_acc * taper
    out = base + hf_acc
    res = src.copy()
    res[m] = out[m]
    if verbose:
        print(f"    texture_fill: {time.time() - t0:.2f}s, 周边纹理能量={ring_energy:.2f} "
              f"搬运系数={tex_gain:.2f}, 采用平移 "
              f"{[(s[1], s[2]) for s in chosen]}")
    return _u8(res)


def patch_synth(img: np.ndarray, mask, patch: int = 5, search: int = 36,
                max_mask_px: int = 24000, verbose: bool = False) -> np.ndarray:
    """基于样例的逐块合成(Exemplar-based): 按"离已知区域由近及远"的顺序,
    为每个掩码像素在其邻域窗口内搜索最相似的干净图块并把未知像素填进去。

    掩码太大时自动退回 texture_fill(速度考虑)。
    """
    t0 = time.time()
    src = _f(img)
    m = _m(mask)
    if int(m.sum()) > max_mask_px:
        if verbose:
            print(f"    patch_synth: 掩码 {int(m.sum())}px 过大, 退回 texture_fill")
        return texture_fill(img, m, verbose=verbose)
    pad = search + patch
    img_p, m_p = _pad(src, m, pad)
    h, w = m_p.shape
    r = patch // 2
    known = (~m_p).astype(np.float32)
    out = img_p.copy()
    if m_p.any():
        out[m_p] = harmonic_fill(img_p, m_p, levels=5, iters=40)[m_p].astype(np.float32)
    dist = cv2.distanceTransform((~m_p).astype(np.uint8), cv2.DIST_L2, 3)
    ys_m, xs_m = np.where(m_p)
    order = np.argsort(dist[m_p])
    ys_m, xs_m = ys_m[order], xs_m[order]
    # 候选源像素(离掩码足够远), 抽样控制规模
    src_ok = cv2.erode((1 - known).astype(np.uint8), _kernel(3)) > 0
    cy_all, cx_all = np.where(src_ok)
    if len(cy_all) > 4000:
        sel = np.random.RandomState(7).choice(len(cy_all), 4000, replace=False)
        cy_all, cx_all = cy_all[sel], cx_all[sel]
    if len(cy_all) == 0:
        return _u8(out[pad:pad + src.shape[0], pad:pad + src.shape[1]])
    # 预取所有候选图块 (N, patch, patch, 3)
    stacked = np.empty((len(cy_all), patch, patch, 3), np.float32)
    for i, (yy, xx) in enumerate(zip(cy_all, cx_all)):
        stacked[i] = out[yy - r:yy - r + patch, xx - r:xx - r + patch]
    for cx, cy in zip(xs_m, ys_m):
        x0, y0 = cx - r, cy - r
        tgt = out[y0:y0 + patch, x0:x0 + patch]
        tk = known[y0:y0 + patch, x0:x0 + patch]
        if tk.sum() < 3:
            continue
        near = (np.abs(cy_all - cy) <= search) & (np.abs(cx_all - cx) <= search)
        if near.sum() < 8:
            near = np.ones(len(cy_all), bool)
        sub = stacked[near]
        kk = tk[..., None]
        d = (((sub - tgt[None]) ** 2) * kk).sum(axis=(1, 2, 3)) / max(1e-6, kk.sum())
        b = int(np.argmin(d))
        best = sub[b]
        unk = (tk < 0.5)
        tmpl = out[y0:y0 + patch, x0:x0 + patch]
        tmpl[unk] = best[unk]
        out[y0:y0 + patch, x0:x0 + patch] = tmpl
        known[y0:y0 + patch, x0:x0 + patch] = 1.0
    res = src.copy()
    res[m] = out[pad:pad + src.shape[0], pad:pad + src.shape[1]][m]
    if verbose:
        print(f"    patch_synth: {time.time() - t0:.2f}s mask={int(m.sum())}px")
    return _u8(res)


# --------------------------------------------------------------- 邻域克隆
def clone_best(img: np.ndarray, mask, max_shift: int = 40, step: int = 1,
               verbose: bool = False) -> np.ndarray:
    """在掩码周围找一块"最像周围环带"的干净区域整块搬过来(纹理重复时效果最好)"""
    src = _f(img)
    m = _m(mask)
    band = cv2.subtract(cv2.dilate(m.astype(np.uint8), _kernel(6)),
                        cv2.dilate(m.astype(np.uint8), _kernel(2))) > 0
    best, best_d = (0, 0), None
    for dy in range(-max_shift, max_shift + 1, step):
        for dx in range(-max_shift, max_shift + 1, step):
            if dx == 0 and dy == 0:
                continue
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            sh = cv2.warpAffine(src, M, (src.shape[1], src.shape[0]),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            d = np.abs(sh[band] - src[band]).mean()
            if best_d is None or d < best_d:
                best_d, best = d, (dx, dy)
    M = np.float32([[1, 0, best[0]], [0, 1, best[1]]])
    sh = cv2.warpAffine(src, M, (src.shape[1], src.shape[0]),
                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    out = src.copy()
    out[m] = sh[m]
    if verbose:
        print(f"    clone_best: offset={best} 差异={best_d:.2f}")
    return _u8(out)


# ------------------------------------------------------------- 遮掩类方法
def blur_region(img: np.ndarray, mask, ksize: int = 31, strength: float = 1.0,
                weight: np.ndarray | None = None) -> np.ndarray:
    k = max(3, int(ksize) | 1)
    src = _f(img)
    b = cv2.GaussianBlur(src, (k, k), 0)
    w = weight if weight is not None else _m(mask).astype(np.float32)
    w = (w * float(strength))[..., None] if w.ndim == 2 else w
    return _u8(src * (1 - w) + b * w)


def pixelate_region(img: np.ndarray, mask, block: int = 16, strength: float = 1.0,
                    weight: np.ndarray | None = None) -> np.ndarray:
    src = _f(img)
    b = max(2, int(block))
    h, w = src.shape[:2]
    small = cv2.resize(src, (max(1, w // b), max(1, h // b)), interpolation=cv2.INTER_AREA)
    pix = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    wm = weight if weight is not None else _m(mask).astype(np.float32)
    wm = (wm * float(strength))[..., None] if wm.ndim == 2 else wm
    return _u8(src * (1 - wm) + pix * wm)


def darken_region(img: np.ndarray, mask, value: float = 0.0, strength: float = 1.0,
                  weight: np.ndarray | None = None) -> np.ndarray:
    src = _f(img)
    wm = weight if weight is not None else _m(mask).astype(np.float32)
    wm = (wm * float(strength))[..., None] if wm.ndim == 2 else wm
    return _u8(src * (1 - wm) + float(value) * wm)


# ------------------------------------------------------------------ 调度
METHODS = {
    "auto": "自动选择(默认: 细笔画用 telea, 大块用 texture)",
    "telea": "OpenCV Telea 修复(细笔画文字最佳)",
    "ns": "OpenCV Navier-Stokes 修复",
    "texture": "谐波低频 + 邻域纹理搬运(推荐的通用方法)",
    "synthesis": "基于样例的逐块合成(较慢, 复杂纹理)",
    "harmonic": "纯谐波填充(平滑背景)",
    "vertical": "竖向结构保持填充(玻璃/柱面等竖直纹理)",
    "horizontal": "横向结构保持填充(贴边横条水印: 台标/用户名/徽标)",
    "clone": "邻域克隆(重复纹理/规则背景)",
    "blur": "高斯模糊遮盖",
    "pixelate": "马赛克遮盖",
    "dark": "纯色遮盖",
}


def thin_ratio(mask) -> float:
    """掩码"细长程度": 掩码面积 / 其外接框面积, 用于判断是文字笔画还是实心块"""
    m = _m(mask).astype(np.uint8)
    if m.sum() == 0:
        return 0.0
    _x, _y, w, h = cv2.boundingRect(m)
    return float(m.sum()) / float(max(1, w * h))


def fill(img: np.ndarray, mask, method: str = "auto", weight: np.ndarray | None = None,
         verbose: bool = False, **kw) -> np.ndarray:
    """按方法名执行修复, 返回修好的整幅图(掩码外保持不变)。

    只在掩码外接框四周留 24px 的邻域内计算, 小掩码 + 大画面时显著提速。
    """
    m = _m(mask)
    if not m.any():
        return _u8(_f(img)) if img.dtype != np.uint8 else img.copy()
    H, W = m.shape[:2]
    x, y, w, h = cv2.boundingRect(m.astype(np.uint8))
    if method not in ("blur", "pixelate", "dark"):
        pad = 32
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        if x1 - x0 < W or y1 - y0 < H:        # 小掩码 + 大画面: 裁剪提速
            sub = img[y0:y1, x0:x1]
            sub_m = m[y0:y1, x0:x1]
            sub_w = None if weight is None else weight[y0:y1, x0:x1]
            out = fill(sub, sub_m, method=method, weight=sub_w,
                       verbose=verbose, **kw)
            res = img.copy() if img.dtype == np.uint8 else _u8(_f(img))
            res[y0:y1, x0:x1] = out
            return res
    if method == "auto":
        ratio = thin_ratio(m)
        npx = int(m.sum())
        md = m.astype(np.uint8)
        band = cv2.subtract(cv2.dilate(md, _kernel(12)), cv2.dilate(md, _kernel(4))) > 0
        hf = _f(img) - cv2.GaussianBlur(_f(img), (0, 0), 6.0)
        tex = float(hf[band].std()) if band.any() else 0.0
        if ratio < 0.60 and npx < 8000:
            method = "telea"                 # 文字/细结构: Telea 最干净
        else:
            # 大块区域: telea 从异质边界扩散会涂出污渍/发白, harmonic 的
            # 边界若含水印软边整体发白 —— 先试"最均匀单侧带延展"(黑板/
            # 桌面/墙面等平滑背景最贴背景色); 不可行再按周边纹理选。
            side = side_fill(img, m) if npx >= 8000 else None
            if side is not None:
                if verbose:
                    print(f"    auto -> side (掩码 {npx}px, 占比 {ratio:.2f})")
                return side
            method = "harmonic" if tex < 2.2 else "texture"
        if verbose:
            print(f"    auto -> {method} (掩码占比 {ratio:.2f}, 周边纹理能量 {tex:.2f})")
    if method == "telea":
        return inpaint_cv(img, m, kw.get("radius", 3), telea=True)
    if method == "ns":
        return inpaint_cv(img, m, kw.get("radius", 3), telea=False)
    if method == "harmonic":
        return harmonic_fill(img, m, verbose=verbose)
    if method == "side":
        r = side_fill(img, m, verbose=verbose)
        return r if r is not None else harmonic_fill(img, m, verbose=verbose)
    if method == "vertical":
        return vertical_fill(img, m, band=kw.get("band", 16),
                             smooth=kw.get("smooth", 5),
                             band_mask=kw.get("band_mask"), verbose=verbose)
    if method == "horizontal":
        return horizontal_fill(img, m, band=kw.get("band", 24),
                               smooth=kw.get("smooth", 5),
                               band_mask=kw.get("band_mask"), verbose=verbose)
    if method == "texture":
        # sigma 必须小: 只搬运细纹理。sigma 大会把平移源里的大尺度结构
        # (黄色导管/明暗边界)整个搬进来, 形成色斑(实测左上角黄色噪块)。
        return texture_fill(img, m, sigma=kw.get("sigma", 6.0),
                            max_shift=kw.get("max_shift", 56),
                            n_best=kw.get("n_best", 3), verbose=verbose)
    if method == "synthesis":
        return patch_synth(img, m, patch=kw.get("patch", 5),
                           search=kw.get("search", 36), verbose=verbose)
    if method == "clone":
        return clone_best(img, m, max_shift=kw.get("max_shift", 40), verbose=verbose)
    if method == "blur":
        return blur_region(img, m, ksize=kw.get("ksize", 31), weight=weight)
    if method == "pixelate":
        return pixelate_region(img, m, block=kw.get("block", 16), weight=weight)
    if method == "dark":
        return darken_region(img, m, value=kw.get("value", 0), weight=weight)
    raise ValueError(f"未知方法: {method} (可用: {', '.join(METHODS)})")
