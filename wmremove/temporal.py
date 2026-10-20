# -*- coding: utf-8 -*-
"""视频专用: 利用"时间维度"把静态水印精确抠掉(图像方法之外的第二条腿)

两种武器
--------
A. 淡入淡出层法(plan_fade_layer)
   AI 工具(豆包/通义等)的水印往往在开头淡入: 前几帧还没有水印, 场景又基本静止。
   于是"有水印帧 - 无水印帧"就得到了逐像素精确的水印层, 之后每帧按
   k_t(逐帧淡入系数) 减掉即可 —— 连半透明的文字/描边都能干净去除。

B. 仿射分离法(plan_temporal_unmix)
   水印是静态的、半透明叠加:  obs = (1-α)·背景 + α·颜色 。
   只要背景在动, 就能用"水印外圈干净像素的时域均值"当作背景代理,
   在同一小块内做时域线性回归  obs_t = a·x_t + b  (a=1-α, b=α·颜色),
   再把每帧反解回背景:  bg_t = (obs_t - b)/a。
   对平台台标、AI 半透明水印都成立; 不成立(背景不动/水印不透明)的像素
   自动退回空间修复。

C. 静态层替换法(plan_static_replace)
   大面积半透明水印(如瓶身上的圆盘 logo)时, 仿射分离的"环带背景代理"
   在圆盘内部失效。改用"参考均值替换":
     * 在掩码内内容最稳定的连续帧窗上取时域均值 M(等于含水印的底图);
     * 用结构保持填充(竖向外推/纹理搬运)构造"无水印目标" T,
       并把 T 的低频明暗对齐到掩码外紧邻环的水平(避免亮盘/暗盘);
     * 注入场 L = M - T, 逐帧 out = frame - L·w·gate:
       - gate = 内容变化门控: 每帧与 M 的差异越大(如液体涨进圆盘),
         注入越弱 —— 内容被改动的地方自动放弃处理;
       - 暗结构保护: 掩码内比周边暗很多的像素(如深色导管)不注入。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import methods
from .detect import overlay_mask
from .media import FrameReader, VideoInfo
from .shapes import dilate, erode


def sample_indices(n: int, k: int, head: int = 0) -> list[int]:
    if n <= 0:
        return []
    k = max(1, min(k, n))
    idx = np.linspace(0, n - 1, k).round().astype(int).tolist()
    out = sorted(set(idx + list(range(min(head, n)))))
    return out


def grab(info: VideoInfo, idxs: list[int]) -> dict[int, np.ndarray]:
    want = set(int(i) for i in idxs)
    got = {}
    with FrameReader(info) as r:
        for i, f in r:
            if i in want:
                got[i] = f
            if got and i >= max(want):
                break
    return got


# --------------------------------------------------------------- A. 淡入层法
@dataclass
class FadeLayer:
    layer: np.ndarray          # HxWx3 float32, 水印层(有水印 - 无水印)
    pre_mean: np.ndarray       # HxWx3 无水印参考
    alphas: np.ndarray         # 每帧的淡入系数 k_t
    pre_frames: list[int]
    post_frame: int
    quality: float             # 背景一致性(越小越可信)

    def k_at(self, idx: int) -> float:
        if idx < len(self.alphas):
            return float(self.alphas[idx])
        return float(self.alphas[-1]) if len(self.alphas) else 1.0


def plan_fade_layer(info: VideoInfo, mask: np.ndarray, head: int = 60,
                    stable_tol: float = 3.5, stable_step: float = 1.2,
                    max_bg_diff: float = 8.0,
                    min_layer: float = 3.0, resid_tol: float = 5.0,
                    verbose: bool = False) -> FadeLayer | None:
    """层配对法: 在全片寻找"同一背景、水印出现/消失"的相邻帧对。

    AI 工具水印常见形态: 在某段时间淡入、停留、再淡出(不限片头)。
    只要能找到一对"背景几乎相同、水印不同"的帧, 两者之差就是逐像素精确的
    水印层; 再对每一帧用最小二乘估计该帧的水印系数 k_t, 每帧减掉 k_t×层 即可。

    防误判:
      * 层强度必须显著(> min_layer);
      * 背景差必须小(掩码外 < max_bg_diff);
      * 模型校验: "帧 ≈ 无水印参考 + k×层" 在背景差合格的采样帧上残差要小。
    """
    n = info.n_frames
    if n < 8:
        return None
    m = mask.astype(bool)
    outer = ~dilate(m.astype(np.uint8), 6).astype(bool)
    fps = info.fps or 25.0
    stride = max(1, int(round(fps * 0.5)))
    if n // max(1, stride) < 16:             # 短视频: 加密采样
        stride = max(1, n // 16)
    idxs = list(range(0, n, stride))
    if len(idxs) > 90:                       # 控制采样规模
        idxs = list(np.linspace(0, n - 1, 90).round().astype(int))
    frames = grab(info, idxs)
    seq = [i for i in idxs if i in frames]
    if len(seq) < 4:
        return None

    def bg_diff(i: int, j: int) -> float:
        a = frames[i].astype(np.float32)
        b = frames[j].astype(np.float32)
        return float(np.abs(a[outer] - b[outer]).mean())

    best = None
    for gap in (1, 2, 4, 8):
        for i, j in zip(seq, seq[gap:]):
            if j - i > stride * 8:
                continue
            a = frames[i].astype(np.float32)
            b = frames[j].astype(np.float32)
            bgmap = np.abs(
                cv2.GaussianBlur(
                    0.114 * (b - a)[..., 0] + 0.587 * (b - a)[..., 1]
                    + 0.299 * (b - a)[..., 2], (0, 0), 4.0))
            bg_p95 = float(np.percentile(bgmap[outer], 95))
            if bg_p95 > 6.0:                 # 背景在动: 任何帧对都不可信
                continue
            bg = float(bgmap[outer].mean())
            layer = b - a
            L = layer[m]
            strength = float(np.abs(L).mean())
            if strength < min_layer:
                continue
            score = strength - bg * 2.0
            if best is None or score > best[0]:
                best = (score, i, j, layer, bg, bgmap)
    if best is None:
        return None
    _score, i0, j0, layer0, bg, bgmap = best
    pre_ref0 = frames[i0].astype(np.float32)
    # 逐像素背景一致性: 帧对间"背景没变"的地方层才可信;
    # 掩码内的可信度由外环的可信度向内插值得到。
    tol = max(3.0, float(np.percentile(bgmap[outer], 90)) * 1.3)
    valid_out = (bgmap <= tol).astype(np.uint8) * 255
    valid_fill = cv2.inpaint(valid_out, (m > 0).astype(np.uint8) * 255, 3,
                             cv2.INPAINT_TELEA)
    valid_m = valid_fill > 127
    cover = float((valid_m & m).sum()) / max(1.0, float(m.sum()))
    if cover < 0.6:
        if verbose:
            print(f"    层配对法: 背景一致区域只覆盖掩码 {cover * 100:.0f}%, 放弃")
        return None
    layer0[~(m & valid_m)] = 0.0            # 背景变了的像素不参与层估计
    L0 = layer0[m]
    den0 = float((L0 * L0).sum()) + 1e-6
    # 该层可能只是"部分淡入差", 先估计各帧系数, 再归一化到完整水印层
    k_raw = np.array([float(((frames[i].astype(np.float32)[m] - pre_ref0[m]) * L0).sum()
                            / den0) for i in seq])
    kmin, kmax = float(np.percentile(k_raw, 5)), float(np.percentile(k_raw, 95))
    if kmax - kmin < 0.35:                   # 层只覆盖很小一段变化, 精度不足
        if verbose:
            print(f"    层配对法: 层覆盖系数区间过小({kmin:.2f}~{kmax:.2f}), 放弃")
        return None
    scale = float(kmax - kmin)               # 完整水印层 = 观察差 / 系数差
    layer = layer0 * scale
    pre_ref = pre_ref0 + kmin * layer0       # 外推出"完全无水印"的参考帧
    pre_ref[~m] = frames[i0].astype(np.float32)[~m]
    L = layer[m]
    denom = float((L * L).sum()) + 1e-6

    # 逐帧系数 k_t(在采样帧上估计, 其余线性插值)
    ks = np.clip((k_raw - kmin) / max(1e-6, (kmax - kmin)), 0.0, 1.5)
    kis = seq
    resids = []
    for i, k in zip(seq, ks):
        f = frames[i].astype(np.float32)
        b = float(np.abs(f[outer] - pre_ref[outer]).mean())
        if b <= max_bg_diff * 1.2:
            resids.append(float(np.sqrt(np.mean((f[m] - pre_ref[m] - k * L) ** 2))))
    resid = float(np.median(resids)) if len(resids) >= 3 else 99.0
    if resid > resid_tol:
        if verbose:
            print(f"    层配对法: 模型残差过大({resid:.2f}>{resid_tol:.2f}), 放弃")
        return None
    # 显著性: 真水印层的强度必须明显(>8 灰度), 且 k_t 序列应随时间平滑
    # (淡入淡出是缓变的; 若"层"其实是噪声/场景变化, k 序列会剧烈跳动)
    strength_full = float(np.abs(L).mean())
    if strength_full < 8.0:
        if verbose:
            print(f"    层配对法: 层强度过弱({strength_full:.2f}<8), 放弃")
        return None
    tv = float(np.abs(np.diff(ks)).sum())
    span = float(ks.max() - ks.min()) + 1e-6
    if tv / span > 3.0:
        if verbose:
            print(f"    层配对法: 系数序列不平滑(波动/跨度={tv / span:.1f}), 放弃")
        return None
    alphas = np.interp(np.arange(n), kis, ks).astype(np.float32)
    # 时间平滑
    if n > 6:
        kernel = np.ones(5) / 5.0
        alphas = np.convolve(np.pad(alphas, 2, mode="edge"), kernel, mode="valid")
    alphas = np.clip(alphas, 0.0, 1.4).astype(np.float32)
    if verbose:
        print(f"    层配对法: 帧对 ({i0},{j0}) 层强 {float(np.abs(L).mean()):.2f}, "
              f"背景差 {bg:.2f}, 模型残差 {resid:.2f}, "
              f"系数范围 [{alphas.min():.2f},{alphas.max():.2f}]")
    return FadeLayer(layer, pre_ref, alphas, [i0], j0, bg)


# ------------------------------------------------------------ B. 仿射分离法
@dataclass
class UnmixPlan:
    a: np.ndarray        # HxWx3 float32, 斜率 = 1-α
    b: np.ndarray        # HxWx3 截距 = α·颜色
    ok: np.ndarray       # HxW bool, 该块是否可用
    block: int
    stats: dict

    def apply(self, frame: np.ndarray, mask: np.ndarray, weight: np.ndarray) -> np.ndarray:
        a = np.clip(self.a, 0.08, 1.3)
        rec = (frame.astype(np.float32) - self.b) / a
        rec = np.clip(rec, 0, 255)
        w = (mask.astype(bool) & self.ok).astype(np.float32)
        w = np.minimum(weight, w)[..., None]
        out = frame.astype(np.float32) * (1 - w) + rec * w
        return np.clip(out, 0, 255).astype(np.uint8)


def plan_temporal_unmix(info: VideoInfo, mask: np.ndarray, n_samples: int = 40,
                        block: int = 24, ring_r: int = 16, min_var: float = 2.0,
                        max_resid: float = 4.5, min_alpha: float = 0.15,
                        verbose: bool = False) -> UnmixPlan | None:
    """按小块做时域线性回归, 估计 obs = a·(周边干净均值) + b 的 a、b"""
    H, W = mask.shape
    m = mask.astype(np.uint8)
    idxs = sample_indices(info.n_frames, n_samples)
    if len(idxs) < 6:
        return None
    # 块网格
    nx = max(1, W // block)
    ny = max(1, H // block)
    bmask = cv2.resize(m, (nx, ny), interpolation=cv2.INTER_AREA) > 40
    # 环带: 每块周围 ring_r 内的干净像素
    ring_all = (dilate(m, ring_r) > 0) & (m == 0)
    ys, xs = np.where(bmask)
    if len(ys) == 0:
        return None
    nblk = len(ys)
    sx = np.zeros((nblk, 3), np.float64)
    sy = np.zeros((nblk, 3), np.float64)
    sxx = np.zeros((nblk, 3), np.float64)
    sxy = np.zeros((nblk, 3), np.float64)
    syy = np.zeros((nblk, 3), np.float64)
    cnt = np.zeros(nblk, np.float64)
    block_pix = []
    for b in range(nblk):
        y0, y1 = ys[b] * block, min(H, (ys[b] + 1) * block)
        x0, x1 = xs[b] * block, min(W, (xs[b] + 1) * block)
        bm = np.zeros((H, W), bool)
        bm[y0:y1, x0:x1] = True
        block_pix.append((y0, y1, x0, x1))
        ins = bm & (m > 0)
        ring = bm & ring_all
        if ins.sum() < 4 or ring.sum() < 4:
            cnt[b] = 0
            continue
        cnt[b] = 1
    frames = grab(info, idxs)
    order = [i for i in idxs if i in frames]
    for i in order:
        f = frames[i].astype(np.float32)
        for b in range(nblk):
            if cnt[b] == 0:
                continue
            y0, y1, x0, x1 = block_pix[b]
            bm = np.zeros((H, W), bool)
            bm[y0:y1, x0:x1] = True
            ins = bm & (m > 0)
            ring = bm & ring_all
            x = f[ring].mean(axis=0)
            y = f[ins].mean(axis=0)
            sx[b] += x
            sy[b] += y
            sxx[b] += x * x
            sxy[b] += x * y
            syy[b] += y * y
    k = float(len(order))
    good = cnt > 0
    if good.sum() == 0:
        return None
    a = np.ones((nblk, 3), np.float32)
    bvec = np.zeros((nblk, 3), np.float32)
    ok = np.zeros(nblk, bool)
    for bi in range(nblk):
        if not good[bi]:
            continue
        mx = sx[bi] / k
        my = sy[bi] / k
        vx = sxx[bi] / k - mx * mx
        vxy = sxy[bi] / k - mx * my
        vy = syy[bi] / k - my * my
        for c in range(3):
            if vx[c] < min_var * min_var:
                continue
            slope = float(vxy[c] / max(1e-6, vx[c]))
            inter = float(my[c] - slope * mx[c])
            resid = float(np.sqrt(max(0.0, vy[c] - slope * vxy[c])))
            if 0.05 < slope < 1.45 and resid < max_resid and (1 - slope) >= min_alpha * 0:
                a[bi, c] = slope
                bvec[bi, c] = inter
        # 三通道都合理才算可用
        ok[bi] = bool(np.all((a[bi] > 0.05) & (a[bi] < 1.42)))
    if ok.sum() == 0:
        return None
    # 块网格 -> 全分辨率(双线性插值 + 平滑)
    def up(grid, c=None):
        g = grid if c is None else grid[:, c]
        small = np.zeros((ny, nx), np.float32)
        small[ys, xs] = g
        big = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
        return big

    a_map = np.stack([up(a, c) for c in range(3)], axis=-1)
    b_map = np.stack([up(bvec, c) for c in range(3)], axis=-1)
    oksmall = np.zeros((ny, nx), np.float32)
    oksmall[ys[ok], xs[ok]] = 1
    ok_map = cv2.resize(oksmall, (W, H), interpolation=cv2.INTER_LINEAR) > 0.6
    stats = {"块数": int(nblk), "可用块": int(ok.sum()),
             "斜率均值": float(a[ok].mean()) if ok.sum() else 0.0}
    # 显著性: 大部分可用块必须真的检测到"半透明叠加"(a 明显小于 1)
    if ok.sum():
        cover = float(ok.sum()) / max(1.0, float(nblk))
        mean_a = stats["斜率均值"]
        stats["可用占比"] = cover
        if cover < 0.3 or mean_a > 0.94:      # 基本没检测到叠加 → 不要硬套
            if verbose:
                print(f"    仿射分离: 可用块占比 {cover * 100:.0f}%, 平均 a={mean_a:.2f} "
                      f"→ 未检测到明显半透明叠加, 放弃")
            return None
    # 后验校验: 块平均斜率模型对"稀疏不透明内容"(文字/贴片)会把整块只减掉
    # 一小部分, 水印仍然清晰。把模型套到中值帧上, 用叠加结构检测量残余 ——
    # 残余结构仍明显 → 该内容不适合仿射分离, 交给静态层替换/空间修复。
    plan0 = UnmixPlan(a_map, b_map, ok_map, block, stats)
    med_f = np.median(np.stack([frames[i] for i in order[:12]])
                      .astype(np.float32), axis=0).astype(np.uint8)
    rec0 = plan0.apply(med_f, mask, np.ones_like(mask, np.float32))
    g0 = cv2.cvtColor(rec0, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gmed = cv2.cvtColor(med_f, cv2.COLOR_BGR2GRAY).astype(np.float32)
    resid_m = overlay_mask(g0, k=9, thr=6.0, pad=0, region=mask > 0)
    dens_out = float((resid_m > 0).sum()) / max(1.0, float((mask > 0).sum()))
    # 分离有效性: 在掩码内找"明显偏离周边环带亮度"的像素(水印文字/贴片),
    # 分离后这些像素应回到环带水平; 若仍保留大半偏离 → 模型没真正还原
    # 背景(稀疏不透明内容的典型失败: 块平均斜率被背景主导, 只减一小部分,
    # 文字仍然清晰可读)。
    keep = 0.0
    mb_mask = mask > 0
    if ring_all.sum() >= 32 and int(mb_mask.sum()) >= 200:
        base = float(np.median(gmed[ring_all]))
        wm_px = mb_mask & (np.abs(gmed - base) > 25.0)
        if int(wm_px.sum()) >= 100:
            d_in = float(gmed[wm_px].mean()) - base
            d_out = float(g0[wm_px].mean()) - base
            if abs(d_in) > 12:
                keep = abs(d_out) / abs(d_in)
    # 逐块一致性: 分离正确的块, 其内部均值应回到周边环带水平;
    # "压黑背景/残留文字"的块会显著偏离 → 计入坏块。
    bad = 0
    tot = 0
    for (y0, y1, x0, x1) in block_pix:
        bm = np.zeros((H, W), bool)
        bm[y0:y1, x0:x1] = True
        ins = bm & (mask > 0) & ok_map
        rng = bm & ring_all
        if ins.sum() < 8 or rng.sum() < 8:
            continue
        tot += 1
        if abs(float(g0[ins].mean()) - float(gmed[rng].mean())) > 14:
            bad += 1
    eff = 1.0 - bad / max(1, tot)
    stats["分离后残余"] = round(dens_out, 3)
    stats["块一致率"] = round(eff, 2)
    stats["残余超亮比"] = round(keep, 2)
    if dens_out > 0.04 or eff < 0.6 or keep > 0.45:
        if verbose:
            print(f"    仿射分离: 分离后残余结构 {dens_out * 100:.0f}%, "
                  f"块一致率 {eff * 100:.0f}%, 残余超亮比 {keep:.2f} "
                  f"(内容为不透明/稀疏, 模型不适用), 放弃")
        return None
    if verbose:
        print(f"    仿射分离: 块 {stats['块数']} 个, 可用 {stats['可用块']} 个, "
              f"平均 a(1-α)={stats['斜率均值']:.2f} (相当于不透明度 "
              f"{1 - stats['斜率均值']:.2f}, 残余 {dens_out * 100:.1f}%, "
              f"一致率 {eff * 100:.0f}%, 超亮比 {keep:.2f})")
    return UnmixPlan(a_map, b_map, ok_map, block, stats)


def auto_plan(info: VideoInfo, mask: np.ndarray, allow_fade: bool = True,
              allow_unmix: bool = True, raw_mask: np.ndarray | None = None,
              opts=None, verbose: bool = True):
    """按优先级挑选时序方案:
    淡入层法 > 仿射分离法 > 静态层替换(大面积半透明) > None(逐帧空间修复)"""
    if allow_fade:
        try:
            fl = plan_fade_layer(info, mask, verbose=verbose)
        except Exception as e:
            fl = None
            if verbose:
                print(f"    淡入层法不可用: {e}")
        if fl is not None:
            # 层覆盖率: 淡入层只解释了掩码的一小部分时, 其余的恒定贴片
            # (层中为 0)会被原样留下 —— 实测 3.mp4 层只覆盖 11%, 左上角
            # 账号贴片完全没处理。覆盖率过低不如交给整框空间修复(平坦
            # 背景上 harmonic 无痕, 对不透明贴片同样有效)。
            cover = float(((mask > 0) & (
                np.abs(fl.layer.astype(np.float32)).max(axis=2) > 2.5)).sum()) \
                / max(1.0, float((mask > 0).sum()))
            if cover < 0.5:
                if verbose:
                    print(f"    淡入层只覆盖掩码 {cover * 100:.0f}%, "
                          f"放弃(改用空间修复)")
            else:
                return ("fade", fl)
    if allow_unmix:
        try:
            up = plan_temporal_unmix(info, mask, verbose=verbose)
        except Exception as e:
            up = None
            if verbose:
                print(f"    仿射分离法不可用: {e}")
        if up is not None and up.stats.get("可用块", 0) > 0:
            return ("unmix", up)
    if raw_mask is not None:                  # 大面积静态水印(独立于 unmix 开关)
        try:
            sp = plan_static_replace(info, raw_mask, opts=opts, verbose=verbose)
        except Exception as e:
            sp = None
            if verbose:
                print(f"    静态层替换不可用: {e}")
        if sp is not None:
            return ("static", sp)
    return ("spatial", None)


# ----------------------------------------------------------- C. 静态层替换
@dataclass
class StaticPlan:
    L: np.ndarray              # HxWx3 注入场 = M - T(含水印均值 - 干净目标)
    T: np.ndarray              # HxWx3 无水印目标图(运动补偿替换用)
    M: np.ndarray              # 参考均值帧(含水印)
    protect: np.ndarray        # HxW 0/1 保护掩码(暗结构等, 不注入)
    ref_window: tuple          # 参考均值所用帧窗 (i0, i1)
    repl: np.ndarray | None = None   # HxW 实测水印替换掩码(水印超出框选时非 None)
    stats: dict = field(default_factory=dict)


def _stable_windows(info: VideoInfo, mask: np.ndarray, max_samples: int = 60
                    ) -> list[tuple[int, int]]:
    """找出掩码内"内容稳定"的连续帧段(按出现顺序返回, 最早优先)。

    内容变化 = 相邻采样帧在掩码内的平均绝对差; 阈值取
    max(1.5, 40 分位 × 1.6), 即"比大多数时段更安静"的段才算稳定。
    """
    n = info.n_frames
    if n < 8:
        return []
    stride = max(1, n // max_samples)
    idxs = list(range(0, n, stride))
    frames = grab(info, idxs)
    seq = [i for i in idxs if i in frames]
    if len(seq) < 5:
        return []
    m = mask.astype(bool)
    d = []
    for a, b in zip(seq, seq[1:]):
        fa = frames[a].astype(np.float32)
        fb = frames[b].astype(np.float32)
        d.append(float(np.abs(fa[m] - fb[m]).mean()))
    d = np.array(d, np.float32)
    tol = max(1.5, float(np.percentile(d, 40)) * 1.6)
    ok = d <= tol
    runs = []
    cur = None
    for j, good in enumerate(ok):
        if good and cur is None:
            cur = j
        elif not good and cur is not None:
            runs.append((cur, j))
            cur = None
    if cur is not None:
        runs.append((cur, len(ok)))
    out = []
    for a, b in runs:
        if b - a < 2:
            continue
        i0 = int(seq[a])
        i1 = int(seq[min(b + 1, len(seq) - 1)])
        out.append((i0, i1))
    return out


def ecc_warp(ref_gray: np.ndarray, cur_gray: np.ndarray,
             input_mask: np.ndarray | None = None):
    """ECC 估计 cur→ref 的欧氏变换(平移+旋转+缩放); 失败返回 None。

    ref_gray/cur_gray: 同尺寸单通道 float32。
    input_mask: 8-bit 有效像素掩码(非零=参与估计), 用于排除水印区
    (台标固定在屏幕上而背景在动, 不排除的话 ECC 找到的是折中解)。
    """
    try:
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)
        _, warp = cv2.findTransformECC(ref_gray, cur_gray, warp,
                                       cv2.MOTION_EUCLIDEAN, criteria,
                                       input_mask, 5)
        # 平移过大 = 场景切换后在无关内容上找到的"伪对齐"(实测 r5 MV:
        # 切换后 ECC 病态解把 T 搬到错误位置, 水印区被注入别处内容)。
        # 正常手持晃动/缓移远小于 32px, 超过即视为对齐失败。
        if float(np.hypot(warp[0, 2], warp[1, 2])) > 32.0:
            return None
        return warp
    except cv2.error:
        return None


def _choose_fill_axis(M: np.ndarray, m: np.ndarray,
                      touches_v: bool, touches_h: bool,
                      verbose: bool = False) -> str:
    """按四条参考带(紧贴掩码的上/下/左/右)的"结构轮廓"相似度选填充轴。

    只比均值会把"均值接近但内容完全不同"的带判为相似(如框上方 5px 的
    游戏画面与下方深蓝 UI 均值接近 → 误选竖向, 把绿色画面拖成条纹)。
    这里比较每列/每行亮度轮廓的归一化形状差异, 并对过薄的带(<8px)
    施加重罚 —— 薄带代表性不足。
    竖向填充假设"上下带属于同一表面"(808 瓶身玻璃), 横向填充假设
    "左右带属于同一表面"(B站台标的深色面板/小红书底部横条)。
    """
    H, W = m.shape
    g = cv2.cvtColor(M.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    ys, xs = np.where(m)
    band = 14

    def _cmp(p1, p2):
        """两条一维轮廓的归一化形状差异(越小越相似)。"""
        if p1 is None or p2 is None or len(p1) < 4 or len(p2) < 4:
            return None
        n = max(len(p1), len(p2))
        x = np.linspace(0.0, 1.0, n)
        q1 = np.interp(x, np.linspace(0.0, 1.0, len(p1)), p1)
        q2 = np.interp(x, np.linspace(0.0, 1.0, len(p2)), p2)
        q1 = q1 - q1.mean()
        q2 = q2 - q2.mean()
        s1 = q1.std() + 1e-3
        s2 = q2.std() + 1e-3
        return float(np.abs(q1 / s1 - q2 / s2).mean())

    t_sl = g[max(0, ys.min() - band):ys.min(), xs.min():xs.max() + 1]
    b_sl = g[ys.max() + 1:min(H, ys.max() + 1 + band), xs.min():xs.max() + 1]
    l_sl = g[ys.min():ys.max() + 1, max(0, xs.min() - band):xs.min()]
    r_sl = g[ys.min():ys.max() + 1, xs.max() + 1:min(W, xs.max() + 1 + band)]
    ev = _cmp(t_sl.mean(axis=0) if t_sl.size else None,
              b_sl.mean(axis=0) if b_sl.size else None)
    eh = _cmp(l_sl.mean(axis=1) if l_sl.size else None,
              r_sl.mean(axis=1) if r_sl.size else None)
    # 薄带惩罚: 带厚 <8px 时该轴的参考不可靠
    thin_v = (ys.min() < 8) or (H - 1 - ys.max() < 8)
    thin_h = (xs.min() < 8) or (W - 1 - xs.max() < 8)
    err_v = 1e9 if (ev is None or touches_v or thin_v) else ev
    err_h = 1e9 if (eh is None or touches_h or thin_h) else eh
    axis = "vertical" if err_v <= err_h else "horizontal"
    if verbose:
        sv = f"{err_v:.2f}" if err_v < 1e8 else "不可用"
        sh = f"{err_h:.2f}" if err_h < 1e8 else "不可用"
        print(f"    静态层替换: 填充轴 自动 → {axis} (上下轮廓差 {sv}, "
              f"左右轮廓差 {sh})")
    return axis


def plan_static_replace(info: VideoInfo, mask: np.ndarray, opts=None,
                        min_overlay: float = 2.5, max_overlay: float = 60.0,
                        verbose: bool = False) -> StaticPlan | None:
    """大面积静态半透明水印: 参考均值 M + 结构保持目标 T → 注入场 L = M - T。

    逐稳定窗尝试(最早优先 —— 水印"原始状态"通常在片头):
      * 注入场强度 |L| 必须在 [min_overlay, max_overlay] 灰度之间:
        太小说明没有可去的叠加层(放弃, 交给空间修复),
        太大说明该窗是"内容已变"的状态(如液体灌满), 不是水印的参考状态。
    """
    from . import Options
    opts = opts or Options()
    m = mask.astype(bool)
    H, W = m.shape
    n = info.n_frames
    if n < 8 or m.sum() < 400:
        return None
    # 多分离区域直接拒绝: T 是"整幅单一背景的重建", 多域(实测 B站视频
    # 左上水印+右下误检框, r5 MV 左上+右上)各域背景完全不同, 一套 T
    # 必然把某个域的重建涂到另一个域上, 产生凭空色块。
    n_lab0, _lab0, _st0, _ = cv2.connectedComponentsWithStats(
        (m * 255).astype(np.uint8), connectivity=8)
    if n_lab0 - 1 > 1:
        if verbose:
            print(f"    静态层替换: 掩码含 {n_lab0 - 1} 个分离区域, "
                  f"各域背景不同, 放弃(改空间修复)")
        return None
    # 全片镜头切换计数: static 假设"背景全程静止"。MV/混剪类视频镜头
    # 频繁切换(实测 r5 NGGYU MV), T 只属于参考窗那个镜头, 播放到其他
    # 镜头时门控在"色调巧合相似"的场景会失效(紫窗场景被注入白块)。
    # 掩码外环带(纯背景)的相邻采样帧差出现 >=2 次跳变(>10 灰度)即判定
    # 为多镜头视频, 整体放弃 static。
    # 为多镜头检测与环带稳定性检查提前定义结构元
    _K8 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    _K30 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    ring_chk = (cv2.dilate((m * 255).astype(np.uint8), _K30) > 0) \
        & ~(cv2.dilate((m * 255).astype(np.uint8), _K8) > 0)
    if ring_chk.sum() > 100:
        step0 = max(1, n // 36)
        idxs0 = list(range(0, n, step0))
        fr0 = grab(info, idxs0)
        seq0 = [i for i in idxs0 if i in fr0]
        devs0 = []
        for a, b in zip(seq0, seq0[1:]):
            ga = cv2.cvtColor(fr0[a], cv2.COLOR_BGR2GRAY).astype(np.float32)
            gb = cv2.cvtColor(fr0[b], cv2.COLOR_BGR2GRAY).astype(np.float32)
            devs0.append(float(np.abs(ga[ring_chk] - gb[ring_chk]).mean()))
        n_sw = sum(1 for d in devs0 if d > 10.0)
        if n_sw >= 2:
            if verbose:
                print(f"    静态层替换: 环带背景检出 {n_sw} 次镜头切换, "
                      f"多镜头视频不适合整体替换, 放弃")
            return None
    wins = _stable_windows(info, mask)
    if not wins:
        if verbose:
            print("    静态层替换: 未找到稳定帧窗, 放弃")
        return None
    fill_mode = getattr(opts, "static_fill", "auto")
    # 外扩量: 水印的软边缘/描边会超出目测边界(≈8%√面积, 专用版 1.10 倍椭圆),
    # 目标图与注入场必须覆盖到外扩区, 否则圆盘边缘留下一圈弧线残留。
    grow = max(6, int(0.08 * np.sqrt(m.sum())))
    m_big = dilate((m * 255).astype(np.uint8), grow) > 0
    tried = []
    # ECC 对齐用补丁: 掩码外扩 100px 的灰度区域(掩码周边背景主导运动估计)
    ys_m, xs_m = np.where(m)
    ex0 = max(0, int(xs_m.min()) - 100)
    ey0 = max(0, int(ys_m.min()) - 100)
    ex1 = min(W, int(xs_m.max()) + 100)
    ey1 = min(H, int(ys_m.max()) + 100)
    for (i0, i1) in wins:
        span = max(2, (i1 - i0) // 3)
        lo, hi = i0 + span // 2, max(i0 + span // 2 + 1, i1 - span // 2)
        idxs = list(range(lo, hi, max(1, (hi - lo) // 24))) or [i0]
        frames = grab(info, idxs)
        seq = [i for i in idxs if i in frames]
        if len(seq) < 2:
            continue
        # 运动补偿平均: 手持拍摄时先做 ECC 欧氏对齐(在掩码周边背景上估计,
        # 排除水印区), 再取中值 —— 否则 M 是"运动混叠"的糊图, 后续全部失准。
        # 估计用半分辨率(ECC 是主要耗时), 矩阵平移×2 还原到全分辨率。
        crop0 = frames[seq[0]][ey0:ey1, ex0:ex1]
        base_g = cv2.cvtColor(crop0, cv2.COLOR_BGR2GRAY).astype(np.float32)
        base_g = cv2.resize(base_g, (base_g.shape[1] // 2, base_g.shape[0] // 2))
        o = np.float32([[ex0], [ey0]])
        m_big_crop = (~m_big[ey0:ey1, ex0:ex1]).astype(np.uint8) * 255
        m_big_crop = cv2.resize(m_big_crop, (base_g.shape[1], base_g.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
        aligned = [frames[seq[0]]]
        for fi in seq[1:]:
            cur_g = cv2.cvtColor(frames[fi][ey0:ey1, ex0:ex1],
                                 cv2.COLOR_BGR2GRAY).astype(np.float32)
            cur_g = cv2.resize(cur_g, (base_g.shape[1], base_g.shape[0]))
            warp = ecc_warp(base_g, cur_g, input_mask=m_big_crop)
            if warp is None:
                aligned.append(frames[fi])
                continue
            # 裁剪坐标 → 全帧坐标: 估计在半分辨率上, 平移×2 还原;
            # t_full = 2·t2 + o - A·o
            A = warp[:, :2]
            t_full = 2.0 * warp[:, 2:3] + o - A @ o
            warp_full = np.hstack([A, t_full]).astype(np.float32)
            aligned.append(cv2.warpAffine(
                frames[fi], warp_full, (W, H),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE))
        M = np.median(np.stack(aligned).astype(np.float32), axis=0)
        # 窗内环带静止性(绝对阈值): ECC 对齐能吸收整体平移, 吸收不了
        # 内容本身的变化 —— 云雾飘动/光斑游走/虚焦呼吸(实测 r8 微博
        # 片头)在环带上差异显著。_stable_windows 的相对阈值会被
        # "全片都在缓动"的视频骗过, 这里按对齐后残余差异硬判。
        if ring_chk.sum() > 100:
            gM_chk = cv2.cvtColor(np.clip(M, 0, 255).astype(np.uint8),
                                  cv2.COLOR_BGR2GRAY).astype(np.float32)
            dev_max = 0.0
            for fi2 in seq[::max(1, len(seq) // 8)]:
                g2 = cv2.cvtColor(np.clip(aligned[seq.index(fi2)], 0, 255)
                                  .astype(np.uint8),
                                  cv2.COLOR_BGR2GRAY).astype(np.float32)
                dev_max = max(dev_max, float(
                    np.abs(g2[ring_chk] - gM_chk[ring_chk]).mean()))
            if dev_max > 1.6:
                tried.append((i0, i1, -dev_max))
                if verbose:
                    print(f"    静态层替换: 窗 {i0}~{i1} 环带背景仍在变化 "
                          f"(dev {dev_max:.2f}), 非静止背景, 跳过")
                continue
        # 双中值: 对齐中值的背景是清晰的(给 T 的参考带), 未对齐中值的
        # 水印是清晰的(水印固定在屏幕上, 对齐反而使其错位稀释) ——
        # L 与门控参考用未对齐 M, T 的参考带用对齐 M。
        M_raw = np.median(np.stack([frames[i] for i in seq])
                          .astype(np.float32), axis=0)

        if fill_mode == "auto":
            ys_m, xs_m = np.where(m)
            touches_v = (ys_m.max() >= H - 3) or (ys_m.min() <= 2)
            touches_h = (xs_m.max() >= W - 3) or (xs_m.min() <= 2)
            use = _choose_fill_axis(M, m, touches_v, touches_h, verbose)
        else:
            use = fill_mode
        Mf = M_raw.astype(np.float32)
        r_out = max(10, int(0.15 * np.sqrt(m.sum())))
        gM = cv2.cvtColor(M_raw.astype(np.uint8),
                          cv2.COLOR_BGR2GRAY).astype(np.float32)

        def _build_T(fill_big):
            """按填充范围 fill_big 构造 T, 并做紧邻环带低频对齐(限幅)。

            参考带(band_mask=原掩码)不变 —— 紧贴水印的带子最同质;
            环带对齐的环带随填充范围外移, 对齐公式与限幅保持一致。
            """
            if use == "vertical":
                t = methods.vertical_fill(M, fill_big, band=16,
                                          smooth=5, band_mask=mask)
            elif use == "horizontal":
                t = methods.horizontal_fill(M, fill_big, band=32,
                                            smooth=5, band_mask=mask)
            elif fill_mode == "texture":
                t = methods.texture_fill(M, fill_big, sigma=18.0)
            else:
                t = methods.harmonic_fill(M, fill_big)
            t = t.astype(np.float32)
            ann = (dilate((m * 255).astype(np.uint8), r_out) > 0) & ~fill_big
            gT = cv2.cvtColor(np.clip(t, 0, 255).astype(np.uint8),
                              cv2.COLOR_BGR2GRAY).astype(np.float32)
            floor = max(60.0, float(np.median(gT[m])) - 25.0)
            a = ann & (gM >= floor) if ann.sum() > 100 else ann
            if a.sum() > 100:
                for c in range(3):
                    d = float(Mf[..., c][a].mean() - t[..., c][m].mean())
                    t[..., c] += float(np.clip(d, -2.5, 2.5))
            return t

        T = _build_T(m_big)
        L = Mf - T
        grayL = 0.114 * L[..., 0] + 0.587 * L[..., 1] + 0.299 * L[..., 2]
        sig = float(np.abs(grayL[m]).mean())
        tried.append((i0, i1, sig))
        if sig <= min_overlay:
            continue
        if sig > max_overlay:              # 该窗内容已变(如液体灌满), 不是参考状态
            if verbose:
                print(f"    静态层替换: 窗 {i0}~{i1} 注入场过强({sig:.0f}), "
                      f"是内容变化不是水印, 跳过")
            continue
        stats = {"帧窗": (i0, i1), "层强": round(sig, 2),
                 "蓝偏": round(float((L[..., 0] - L[..., 2])[m].mean()), 2)}
        # 水印完整性: 掩码外扩环带的 |L| 仍显著 → 水印延伸出用户框选
        # (典型: 框到圆盘 4/5, 下边缘留在外面)。参考带不变、只扩大填充
        # 范围重造 T/L, 替换掩码沿实测水印由内向外逐环生长到消退处。
        # 判据(实测标定, 防误扩):
        #   * 幅度窗口 2.5 < amp < 13 —— 水印是淡色叠加, 边缘衰减处约
        #     5~12 灰度; 液体/结构等"填充无法复原的真实内容"差异通常
        #     >15 灰度(实测液体 ~17), 幅度过大说明环带是内容不是水印;
        #   * 显著占比 ≥35% 且高频中位数 ≤3 —— 挡细长结构条;
        #   * 逐环幅度 <20 —— 液体区差异爆发即停, 避免把液体扩进来
        #     (液体被纳入替换后, 门控半开会在液体上留下玻璃色斑)。
        repl = None
        band_chk = m_big & ~m
        if band_chk.sum() > 50:
            band_amp = float(np.abs(grayL[band_chk]).mean())
            band_hf = float(np.median(
                np.abs(grayL - cv2.GaussianBlur(grayL, (0, 0), 4.0))[band_chk]))
        else:
            band_amp, band_hf = 0.0, 99.0
        if 2.5 < band_amp < 13.0 and band_hf < 3.0:
            m_ext = dilate((m * 255).astype(np.uint8), grow + 88) > 0
            T2 = _build_T(m_ext)
            L2 = Mf - T2
            grayL2 = 0.114 * L2[..., 0] + 0.587 * L2[..., 1] \
                + 0.299 * L2[..., 2]
            sig2 = np.abs(grayL2) > 4.0
            hf2 = np.abs(grayL2 - cv2.GaussianBlur(grayL2, (0, 0), 4.0))
            grown = (m * 255).astype(np.uint8)
            total_add = 0
            for _step in range(7):                    # 最多外扩 7×8=56px
                ring = ((cv2.dilate(grown, _K8) > 0)
                        & ~(grown > 0)) & m_ext
                if not ring.any():
                    break
                if (float(sig2[ring].mean()) < 0.35
                        or float(np.median(hf2[ring])) > 3.0
                        or float(np.abs(grayL2[ring]).mean()) > 20.0):
                    break
                grown = np.where(ring, 255, grown).astype(np.uint8)
                total_add += int(ring.sum())
            if 30 < total_add < 0.8 * m.sum():
                # 开运算去掉细碎的误差条(保留成片的水印外缘)
                keep = cv2.morphologyEx(
                    grown, cv2.MORPH_OPEN,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                repl = np.maximum(keep, (m * 255).astype(np.uint8)) > 0
                T, L, grayL = T2, L2, grayL2
                stats["水印外扩px"] = total_add
                if verbose:
                    print(f"    静态层替换: 框选外缘层强 {band_amp:.1f}"
                          f"(高频 {band_hf:.1f}), 水印超出框选, "
                          f"自动外扩 {total_add}px")
        # 暗结构保护: 只保护"延伸到掩码外"的深色结构(如穿过水印区的导管) —
        # 它们是真实背景物体, 注入会把它们错误提亮。水印自身的不透明深色
        # 图案(头像斑点/徽标文字)完全在掩码内、与外部深色区不相连, 不保护,
        # 否则永远去不掉。
        ann = (dilate((m * 255).astype(np.uint8), r_out) > 0) & ~m_big
        gT = cv2.cvtColor(np.clip(T, 0, 255).astype(np.uint8),
                          cv2.COLOR_BGR2GRAY).astype(np.float32)
        floor = max(60.0, float(np.median(gT[m])) - 25.0)
        if ann.sum() > 100:
            ann = ann & (gM >= floor)
        protect = np.ones((H, W), np.float32)
        if ann.sum() > 100:
            dark_thr = max(60.0, float(np.median(gM[ann])) - 70.0)
            dark_in = (gM < dark_thr) & m
            dark_out = (gM < dark_thr) & ~m_big
            dark = np.zeros((H, W), bool)
            if dark_in.any() and dark_out.any():
                near = dilate((dark_out * 255).astype(np.uint8), grow + 8) > 0
                dark = dark_in & near
            if dark.any() and dark.sum() < 0.5 * m.sum():
                protect[dark] = 0.0
                protect = cv2.GaussianBlur(protect, (0, 0), 2.0)
                stats["暗结构保护"] = int(dark.sum())
        if verbose:
            print(f"    静态层替换: 参考窗 {i0}~{i1}, 层强 {sig:.2f}, "
                  f"目标填充 {fill_mode}")
        return StaticPlan(L=L, T=T, M=Mf, protect=protect, ref_window=(i0, i1),
                          repl=repl, stats=stats)
    if verbose and tried:
        print(f"    静态层替换: 各稳定窗层强 {[t for t in tried]} 不满足 "
              f"[{min_overlay}, {max_overlay}], 放弃")
    return None
