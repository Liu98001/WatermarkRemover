# -*- coding: utf-8 -*-
"""调度引擎: 图片/视频去水印的统一入口

对每一帧/每张图:
   1) 按形状生成掩码(可选"紧致化"只覆盖水印笔画)
   2) 选方法: 视频优先用时域法(淡入层 / 仿射分离)精确去除;
      不适用时用空间修复(Telea 细笔画 / 纹理搬运)
   3) 颜色一致性: 低频对齐 + 可选颗粒匹配, 保证和周围无缝
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import cv2
import numpy as np

from . import colorfit, detect, methods
from .media import (FrameReader, VideoInfo, VideoWriter, grab_frames,
                    imread_u, imwrite_u, probe)
from .shapes import Shape, combine_masks, dilate, erode, feather_weights, mask_bbox
from .temporal import StaticPlan, UnmixPlan, FadeLayer, auto_plan, ecc_warp

SPATIAL_METHODS = {"auto", "telea", "ns", "texture", "synthesis", "harmonic",
                   "vertical", "horizontal", "clone", "blur", "pixelate", "dark"}
COVER_METHODS = {"blur", "pixelate", "dark"}


@dataclass
class Options:
    method: str = "auto"
    feather: float = 3.0
    tight: bool = True
    tight_k: int = 9
    tight_thr: float = 4.0
    tight_pad: int = 3
    colorfit: bool = True
    lf_sigma: float = 25.0
    grain: float = 0.0
    dilate_px: int = 0
    erode_px: int = 0
    temporal: bool = True
    fade: bool = True
    unmix: bool = True
    static_fill: str = "auto"       # 静态层替换的目标填充: auto/vertical/horizontal/texture/harmonic
    gate_tau: float = 45.0          # 静态层替换的内容变化门控阈值(灰度)
    per_frame: bool | None = None   # None=自动(背景不动用差值注入, 在动才逐帧)
    crf: int = 17
    preset: str = "medium"
    params: dict = field(default_factory=dict)

    def kw(self, **over):
        d = dict(self.params)
        d.update(over)
        return d


# ------------------------------------------------------------------ 掩码构建
def build_mask(shapes: Sequence[Shape], W: int, H: int, opts: Options,
               ref_frames: Sequence[np.ndarray] = (), verbose: bool = False):
    """返回 (最终掩码, 羽化权重, 用户原始掩码)。

    原始掩码用于"静态层替换"这类需要整个半透明区域(而非仅笔画)的方案。
    """
    if not shapes:
        raise ValueError("没有指定任何区域(请用 --box/--ellipse/--polygon 或界面框选)")
    mask = None
    for s in shapes:
        m = s.to_mask(W, H)
        mask = m if mask is None else np.maximum(mask, m)
    if mask is None or int(mask.sum()) == 0:
        raise ValueError("区域为空(坐标是否超出画面?)")
    if opts.erode_px:
        mask = erode(mask, int(opts.erode_px))
    if opts.dilate_px:
        mask = dilate(mask, int(opts.dilate_px))
    raw = mask.copy()
    if opts.tight and ref_frames:
        acc = None
        votes = None
        n_voted = 0
        per_frame = []
        if True:
            reg_px = int((mask > 0).sum())
            for f in ref_frames:
                t = detect.refine_mask(f, mask, k=opts.tight_k,
                                       thr=opts.tight_thr,
                                       pad=opts.tight_pad)
                # 只采纳"真的更紧致"的结果; 检测失败回退整框的帧跳过
                if 0 < int((t > 0).sum()) < 0.6 * reg_px:
                    acc = t if acc is None else np.maximum(acc, t)
                    v = (t > 0).astype(np.float32)
                    votes = v if votes is None else votes + v
                    n_voted += 1
                    per_frame.append(t > 0)
            # 背景剧烈变化(舞台灯光/烟花/人群)时, 单帧形态学检测会把亮
            # 纹理全抓成"笔画", 多帧并集可膨胀到接近整框(实测 4.mp4 右
            # 上台标 87%), 整框修复会把头发/舞台细节糊掉。并集占比过高
            # (>60%)说明并集不可信: 收束到"过半帧都检出的稳定笔画"及其
            # 邻域 —— 恒定水印逐帧同位必然高票, 亮纹理各帧位置不同低票。
            if (acc is not None and votes is not None and n_voted >= 2
                    and int((acc > 0).sum()) > 0.60 * reg_px):
                # "差一票全票": 恒定水印几乎每帧都检出; 亮纹理只是偶然
                # 命中(实测 4.mp4 右上: 15286 -> 3606px, 接近笔画规模)。
                # 但必须逐域并上"该域检出最完整帧": 间歇水印(如 4.mp4
                # 左上 logo 只在前 93 帧)在多数采样帧不在场, 低票会被
                # 收束裁掉 —— 它至少会在一个采样帧被整体检出。
                stable = votes >= max(2, n_voted - 1)
                if int(stable.sum()) >= 40:
                    zone = dilate(stable.astype(np.uint8) * 255, 12) > 0
                    acc2 = np.maximum(
                        dilate(stable.astype(np.uint8) * 255, 5) > 0,
                        (acc > 0) & zone)
                    n_lab2, lab2, _st2, _ = \
                        cv2.connectedComponentsWithStats(
                            (mask > 0).astype(np.uint8), connectivity=8)
                    for i2 in range(1, n_lab2):
                        comp2 = lab2 == i2
                        best_m = None
                        best_px = 0
                        for pm in per_frame:
                            px = int((pm & comp2).sum())
                            if px > best_px:
                                best_px, best_m = px, pm & comp2
                        if best_m is not None and best_px >= 40:
                            acc2 = np.maximum(acc2, dilate(
                                best_m.astype(np.uint8) * 255, 3) > 0)
                    acc = acc2.astype(np.uint8) * 255
                    if verbose:
                        print(f"[掩码] 背景剧烈变化: 并集收束到稳定笔画 "
                              f"{int(stable.sum())}px + 各域最完整帧")
            if acc is not None and int((acc > 0).sum()) > 0:
                if verbose:
                    print(f"[掩码] 紧致化: {reg_px}px -> "
                          f"{int((acc > 0).sum())}px "
                          f"({int((acc > 0).sum()) / max(1, reg_px) * 100:.0f}%)")
                # 不透明贴片检测(按连通域): 某域的紧致笔画(白色文字/logo)
                # 与其外扩环带的亮度差 >60 灰度 → 该域贴片不透明, 笔画修复
                # 会留下暗边/彩色残迹(实测 3.mp4 抖音账号贴片, 差 ~135),
                # 而其背景多为平坦深色, 整框 harmonic 无痕 —— 该域退回用户
                # 整框。半透明水印笔画与背景差通常 <40, 不受影响。
                # 判定必须跨帧投票: 单帧"笔画最亮"会撞上"该帧背景恰好
                # 很暗"(实测 r8 微博片头暗色云雾, 白字差 >60 而正片背景
                # 明亮, 单帧误判 → 整框 113kpx 保留 → static 替换在飘动
                # 云雾上贴出凭空剪影)。恒定不透明贴片在每个在场帧都差>60;
                # 差>60 的帧占比 >=70% 才判不透明。
                gs_all = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
                          for f in ref_frames]
                # 多区域框选的稳健性: 某个框的检测可能为空(水印太淡/检测
                # 失败) —— 整体采纳紧致结果会把该区域静默丢弃, 水印原样
                # 留下(实测 3.mp4 右上角 @化学汤老师)。按用户掩码的连通域
                # 分别检查: 检测几乎为空(<15% 覆盖)的域保留原框。
                n_lab, lab, _st, _ = cv2.connectedComponentsWithStats(
                    (mask > 0).astype(np.uint8), connectivity=8)
                # 必须拷贝: 后续 keep[comp]=... 若直接引用 acc 会把"不透明
                # 域整框"写进 acc, 污染后续笔画延续的"贴近 acc 笔画"判定
                # (字幕域被误判贴近)与紧致化统计。
                keep = acc.copy()
                dropped = 0
                opaque_n = 0
                for i in range(1, n_lab):
                    comp = lab == i
                    cov = float((acc[comp] > 0).sum()) \
                        / max(1, int(comp.sum()))
                    if cov < 0.15:
                        keep[comp] = np.maximum(keep[comp], mask[comp])
                        dropped += 1
                        continue
                    # 不透明判定(域级, 多帧投票): 各帧"域内紧致笔画均值
                    # vs 紧邻环带中位"差 >60 的帧占比
                    comp_u8 = comp.astype(np.uint8) * 255
                    ring_i = cv2.dilate(comp_u8, np.ones((25, 25),
                                                         np.uint8)) > 0
                    ring_i &= ~(cv2.dilate(comp_u8, np.ones((5, 5),
                                                            np.uint8)) > 0)
                    stroke_i = comp & (acc > 0)
                    if not gs_all or not ring_i.any() or not stroke_i.any():
                        continue
                    n_hit = 0
                    for g in gs_all:
                        base_i = float(np.median(g[ring_i]))
                        v_i = float(g[stroke_i].mean())
                        if v_i - base_i > 60.0:
                            n_hit += 1
                    if n_hit >= 0.7 * len(gs_all):
                        keep[comp] = np.maximum(keep[comp], raw[comp])
                        opaque_n += 1
                if dropped and verbose:
                    print(f"[掩码] {dropped} 个区域检测失败, 保留原框")
                if opaque_n and verbose:
                    print(f"[掩码] {opaque_n} 个区域为不透明贴片, "
                          f"保留整框修复")
                mask = keep
                # 检测偏向高亮核心, 软边缘/暗边/描边常漏检(实测 3.mp4 抖音
                # logo 边缘彩色残迹); 统一外扩补上 —— 修复量多出的部分落在
                # 框外平坦背景上, 羽化后无痕。不透明整框域不受影响。
                mask = dilate(mask, 10)
                # 框选偏小的笔画延续: 字符串/图标的尾部常被留在框外(实测
                # 3.mp4 "抖音号…104"的尾部残留在框边界右侧, 修复后残字
                # 显眼)。检查掩码边界外侧 2~14px 环内的高对比像素(与本地
                # 背景差 >55), 以"强笔画膨胀桥接后的连通域"为单位并入。
                # 域级判据: 该域必须贴近掩码内的紧致笔画(acc 距离 ≤16px)
                # 才是"被框截断的同一串水印"; 贴着框边的视频内容(字幕等,
                # 实测 3.mp4 字幕距任何笔画 90px+)整体放弃 —— 保留内容
                # 完整(希望效果保留字幕)。字符笔画间存在 diff≤55 的细小
                # 间隙(压缩平滑), strong 膨胀桥接后再判连通。单帧票即计
                # (间歇贴片只在少数采样帧在场)。
                try:
                    ring_o = (dilate(mask, 14) > 0) & ~(dilate(mask, 2) > 0)
                    if ring_o.any():
                        votes_o = None
                        n_o = 0
                        for f in ref_frames:
                            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) \
                                .astype(np.float32)
                            base_loc = cv2.blur(g, (61, 61))
                            cand = ((np.abs(g - base_loc) > 55.0) & ring_o) \
                                .astype(np.float32)
                            votes_o = cand if votes_o is None \
                                else votes_o + cand
                            n_o += 1
                        if votes_o is not None and n_o:
                            strong = (votes_o >= 1) & ring_o
                            strong_d = dilate(
                                strong.astype(np.uint8) * 255, 6) > 0
                            dist_acc = cv2.distanceTransform(
                                ((acc > 0).astype(np.uint8)) ^ 1,
                                cv2.DIST_L2, 3)
                            n_s, lab_s, _st, _ = \
                                cv2.connectedComponentsWithStats(
                                    strong_d.astype(np.uint8), connectivity=8)
                            add = np.zeros(strong.shape, bool)
                            for iu in range(1, n_s):
                                comp_s = lab_s == iu
                                if int((comp_s & strong).sum()) < 8:
                                    continue
                                if float(dist_acc[comp_s].min()) > 16.0:
                                    continue
                                add |= comp_s
                            if add.any() and int(add.sum()) \
                                    < 0.6 * max(1, int((mask > 0).sum())):
                                # 外扩 8px: 字符端点的 anti-alias 边缘在
                                # 压缩后 diff≤55 检不出(实测 3.mp4 "4"的
                                # 右上/右下端点残留在并区外 1~4px)。
                                mask = np.maximum(mask, dilate(
                                    add.astype(np.uint8) * 255, 8))
                                if verbose:
                                    print(f"[掩码] 笔画延续: 边界外并 "
                                          f"{int(add.sum())}px")
                except Exception:
                    pass
            elif verbose:
                print("[掩码] 紧致化未获有效结果, 保留原框")
    # 高亮核心孔洞填充: 紧致化对"比背景亮极多的核心"(logo 星星/角尖,
    # 实测 4.mp4 f30/f60 左上两个白点残留)可能只描到轮廓环, 内部留洞
    # —— 修复后洞里是水印最亮的部分, 极其显眼。floodFill 从加边框的
    # 外圈出发, 未被淹没的内部 0 区即孔洞; 只填小洞(<=2000px),
    # 大片空洞是检测失败域, 交给上层逻辑。
    try:
        hh_, ww_ = mask.shape
        ff2 = np.zeros((hh_ + 2, ww_ + 2), np.uint8)
        ff2[1:-1, 1:-1] = (mask > 0).astype(np.uint8)
        cv2.floodFill(ff2, np.zeros((hh_ + 4, ww_ + 4), np.uint8),
                      (0, 0), 1)
        holes = (ff2[1:-1, 1:-1] == 0)
        if holes.any():
            n_h, lab_h, st_h, _ = cv2.connectedComponentsWithStats(
                holes.astype(np.uint8), connectivity=8)
            fill = np.zeros(holes.shape, bool)
            for ih in range(1, n_h):
                if 8 <= int(st_h[ih, cv2.CC_STAT_AREA]) <= 2000:
                    fill |= lab_h == ih
            if fill.any():
                mask = np.maximum(mask, fill.astype(np.uint8) * 255)
                if verbose:
                    print(f"[掩码] 孔洞填充 {int(fill.sum())}px")
    except Exception:
        pass
    w = feather_weights(mask, opts.feather)
    if verbose:
        bb = mask_bbox(mask, pad=0)
        print(f"[掩码] 最终 {int((mask > 0).sum())}px, 外接框 {bb}")
    return mask, w, raw


# 这些方法的结果自带"边界一致性"(谐波类), 无需再做低频对齐
SELF_CONSISTENT = {"harmonic", "texture", "blur", "pixelate", "dark"}


def _extend_fill(frame: np.ndarray, m: np.ndarray, band: int = 24,
                 smooth: int = 9, sd_max: float = 55.0) -> np.ndarray | None:
    """镜头切换后的逐帧重建: 从掩码旁"最均匀"的一条干净带做单侧延展。

    实现移至 methods.side_fill(fill 的 auto 也复用); 失败返回 None,
    由调用方回退 telea。
    """
    return methods.side_fill(frame, m, band=band, smooth=smooth,
                             sd_max=sd_max)
    src = frame.astype(np.float32)
    H, W = m.shape
    ys, xs = np.where(m)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    seg = (slice(y0, y1 + 1), slice(x0, x1 + 1))     # 掩码相邻段的窗口
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
    # 不能用整圈环带 —— 异质环带(如顶部白天空+底部深蓝集装箱)的中值是
    # 无意义的混合色, 会把正确的单侧延展误拒掉。
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
    return np.clip(out, 0, 255).astype(np.uint8)


def _spatial_repair(frame: np.ndarray, mask: np.ndarray, weight: np.ndarray,
                    opts: Options, verbose: bool = False) -> np.ndarray:
    fixed = methods.fill(frame, mask, method=opts.method, weight=weight,
                         verbose=verbose, **opts.params)
    if opts.colorfit and opts.method not in COVER_METHODS and opts.method not in SELF_CONSISTENT:
        fixed = colorfit.align_lowfreq(fixed, mask, sigma=opts.lf_sigma,
                                       strength=0.9, verbose=False)
    if opts.grain > 0:
        fixed = colorfit.match_grain(fixed, mask, amount=opts.grain)
    return colorfit.composite(frame, fixed, mask, None, feather=opts.feather)


def prepare_spatial(ref: np.ndarray, mask: np.ndarray, weight: np.ndarray,
                    opts: Options, verbose: bool = False) -> dict:
    """在参考帧上一次性算好"颜色校准场/颗粒场", 逐帧复用(速度关键)。"""
    prep = {"corr": None, "noise": None}
    base = methods.fill(ref, mask, method=opts.method, weight=weight,
                        verbose=verbose, **opts.params)
    if opts.colorfit and opts.method not in COVER_METHODS \
            and opts.method not in SELF_CONSISTENT:
        aligned = colorfit.align_lowfreq(base, mask, sigma=opts.lf_sigma,
                                         strength=0.9, verbose=verbose)
        prep["corr"] = (aligned.astype(np.float32) - base.astype(np.float32)) \
            * weight[..., None]
    if opts.grain > 0:
        hf = base.astype(np.float32) - cv2.GaussianBlur(base.astype(np.float32),
                                                        (0, 0), 22.0)
        ring = colorfit.ring_of(mask, 2, 14)
        if ring.any():
            add = max(0.0, float(hf[ring].std()) - float(hf[mask.astype(bool)].std()))
            if add >= 0.35:
                rng = np.random.RandomState(3)
                noise = rng.normal(0, add * opts.grain, mask.shape).astype(np.float32)
                prep["noise"] = cv2.GaussianBlur(noise, (0, 0), 0.7)
    return prep


def apply_spatial(frame: np.ndarray, mask: np.ndarray, weight: np.ndarray,
                  opts: Options, prep: dict) -> np.ndarray:
    """逐帧修复: 填充 + 叠加预先算好的校准场(便宜)"""
    fixed = methods.fill(frame, mask, method=opts.method, weight=weight,
                         verbose=False, **opts.params).astype(np.float32)
    if prep.get("corr") is not None:
        fixed = fixed + prep["corr"]
    if prep.get("noise") is not None:
        w = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 3.0)
        w = np.clip(w * 1.6, 0, 1)[..., None]
        fixed = fixed + prep["noise"][..., None] * w
    fixed = np.clip(fixed, 0, 255).astype(np.uint8)
    return colorfit.composite(frame, fixed, mask, None, feather=opts.feather)


# -------------------------------------------------------------------- 预览
def preview_frame(path: str, index: int, shapes: Sequence[Shape], opts: Options,
                  verbose: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (原图, 处理后, 掩码) —— 供界面预览/参数试错"""
    if os.path.isfile(path) and path.lower().endswith(tuple(media_img_ext())):
        frame = imread_u(path)
    else:
        info = probe(path)
        with FrameReader(info) as r:
            if index:
                r.cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                r.index = int(index)
            _, frame = next(r)
    H, W = frame.shape[:2]
    mask, weight, _raw = build_mask(shapes, W, H, opts, ref_frames=[frame],
                                    verbose=verbose)
    fixed = _spatial_repair(frame, mask, weight, opts, verbose=verbose)
    return frame, fixed, mask


def _median_frame(info: VideoInfo, idxs: Sequence[int]) -> np.ndarray:
    want = set(int(i) for i in idxs)
    frames = []
    with FrameReader(info) as r:
        for i, f in r:
            if i in want:
                frames.append(f)
            if len(frames) >= len(want):
                break
    if not frames:
        raise RuntimeError("读取视频帧失败")
    if len(frames) == 1:
        return frames[0]
    return np.median(np.stack(frames).astype(np.float32), axis=0).astype(np.uint8)


def _grab_frames(info: VideoInfo, idxs: Sequence[int]) -> list[np.ndarray]:
    want = set(int(i) for i in idxs)
    frames = []
    with FrameReader(info) as r:
        for i, f in r:
            if i in want:
                frames.append(f)
            if len(frames) >= len(want):
                break
    return frames


def _background_motion(info: VideoInfo, mask: np.ndarray, n: int = 10) -> float:
    """背景运动强度: 掩码外区域相对中间帧的平均变化(灰度)"""
    H, W = mask.shape
    idxs = [int(info.n_frames * f) for f in (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95)]
    idxs = [min(info.n_frames - 1, max(0, i)) for i in idxs][:max(3, min(n, len(idxs)))]
    want = set(idxs)
    frames = []
    with FrameReader(info) as r:
        for i, f in r:
            if i in want:
                frames.append(f.astype(np.float32))
            if len(frames) >= len(want):
                break
    if len(frames) < 2:
        return 0.0
    mid = frames[len(frames) // 2]
    outer = ~(cv2.dilate(mask, np.ones((9, 9), np.uint8)) > 0)
    diffs = [float(np.abs(f[outer] - mid[outer]).mean()) for f in frames]
    return float(np.median(diffs))


def _choose_spatial_mode(info: VideoInfo, mask: np.ndarray, opts: Options,
                         verbose: bool = False) -> str:
    if opts.per_frame is True:
        return "perframe"
    motion = _background_motion(info, mask)
    mode = "inject" if motion < 2.5 else "perframe"
    if verbose:
        print(f"[空间] 背景运动强度 {motion:.2f} 灰度 -> "
              f"{'参考帧修复+差值注入' if mode == 'inject' else '逐帧修复'}")
    return mode


def _pick_smooth_fill(ref: np.ndarray, mask: np.ndarray,
                      verbose: bool = False) -> str | None:
    """大掩码 auto 选优: harmonic/side(/texture), 参考帧上都算一次,
    按边界接缝+色差选小者并整段锁定(逐帧双算太慢)。

    harmonic 是四面边界的调和插值, 平滑渐变背景(黑板/墙面)上无缝,
    但在烟花/舞台等高纹理背景上会抹出一块"无纹理的马赛克感"平滑区;
    side 是最均匀单侧带延展, 异质边界不会混出污渍; texture 搬运周边
    细纹理, 高纹理背景上最自然。按参考帧实测定夺。
    """
    try:
        harm = methods.harmonic_fill(ref, mask, verbose=False)
    except Exception:
        return None
    cands = [("harmonic", harm)]
    try:
        side = methods.side_fill(ref, mask)
        if side is not None:
            cands.append(("side", side))
    except Exception:
        pass
    # 周边纹理能量高(烟花/舞台/人群)时 texture 候选才值得评估:
    # 平滑背景上搬运纹理反而显眼。
    md = mask.astype(np.uint8)
    band = cv2.subtract(cv2.dilate(md, np.ones((13, 13), np.uint8)),
                        cv2.dilate(md, np.ones((5, 5), np.uint8))) > 0
    hf = ref.astype(np.float32) - cv2.GaussianBlur(
        ref.astype(np.float32), (0, 0), 6.0)
    tex = float(hf[band].std()) if band.any() else 0.0
    if tex >= 3.5:
        try:
            texf = methods.fill(ref, mask, method="texture", verbose=False)
            cands.append(("texture", texf))
        except Exception:
            pass
    best, best_s = None, None
    for name, out in cands:
        rep = colorfit.report(ref, out, mask)
        s = rep.get("接缝比", 99.0) * 2.0 \
            + rep.get("色差(区域-环)", 99.0) \
            + rep.get("灰度差", 99.0)
        if best_s is None or s < best_s:
            best, best_s = name, s
        if verbose:
            print(f"    [选优] {name}: 接缝比 {rep.get('接缝比', 99.0):.2f} "
                  f"色差 {rep.get('色差(区域-环)', 99.0):.2f}")
    return best


def media_img_ext():
    from .media import IMAGE_EXT
    return tuple(IMAGE_EXT)


# ------------------------------------------------------------------ 图片处理
def repair_image(img: np.ndarray, shapes: Sequence[Shape], opts: Options,
                 verbose: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    """对一张图(ndarray)去水印, 返回 (处理后的图, 掩码, 质量指标)"""
    H, W = img.shape[:2]
    mask, weight, _raw = build_mask(shapes, W, H, opts, ref_frames=[img],
                                    verbose=verbose)
    fixed = _spatial_repair(img, mask, weight, opts, verbose=verbose)
    rep = colorfit.report(img, fixed, mask)
    return fixed, mask, rep


def process_image(src: str, dst: str | None, shapes: Sequence[Shape], opts: Options,
                  verbose: bool = True) -> dict:
    t0 = time.time()
    img = imread_u(src) if isinstance(src, str) else np.asarray(src)
    H, W = img.shape[:2]
    if verbose:
        print(f"[图片] {os.path.basename(src) if isinstance(src, str) else '<内存>'} "
              f"{W}x{H}")
    fixed, mask, rep = repair_image(img, shapes, opts, verbose=verbose)
    if verbose:
        print(f"[质量] 色差 {rep['色差(区域-环)']:.2f}, 灰度差 {rep['灰度差']:.2f}, "
              f"接缝比 {rep['接缝比']:.2f}")
    if dst:
        ext = os.path.splitext(dst)[1].lower()
        if ext in (".jpg", ".jpeg"):
            dst = os.path.splitext(dst)[0] + ".png"     # 避免二次压缩
        imwrite_u(dst, fixed)
        if verbose:
            print(f"[完成] {time.time() - t0:.1f}s -> {dst}")
    return {"out": dst, "report": rep, "mask": mask, "image": fixed}


# ------------------------------------------------------------------ 视频处理
def process_video(src: str, dst: str, shapes: Sequence[Shape], opts: Options,
                  verbose: bool = True,
                  progress: Callable[[int, int], None] | None = None) -> dict:
    t_start = time.time()
    info = probe(src)
    H, W = info.out_height, info.out_width
    if verbose:
        print(f"[视频] {os.path.basename(src)} {W}x{H} {info.fps:.2f}fps "
              f"{info.n_frames}帧 旋转{info.rotation}° 音轨{'有' if info.audio else '无'}")
    idxs = [max(0, info.n_frames * k // 8) for k in (1, 2, 3, 4, 5, 6, 7)]
    # 开头帧必须覆盖: 片头水印(剪映/必剪/UP 主 logo 等)只出现在前几秒,
    # 均匀采样帧大概率全部落在其消失后, 紧致化完全检不出(实测 4.mp4
    # 左上 "Created in updream" 仅前 93 帧, 均匀采样 320,962,1603,2245
    # 全部不在场 → 掩码为空 → 只修掉零星笔画)。
    idxs = [5, min(30, max(0, info.n_frames - 1))] + idxs
    refs = []
    seen = set()
    for i in idxs:
        if i not in seen:
            seen.add(i)
            refs.append(i)
    ref = _median_frame(info, sorted(seen))
    # 紧致掩码用多帧并集: 水印淡入淡出/压缩波动时也能圈全
    tight_refs = _grab_frames(info, sorted(seen)[::2] or [0])
    mask, weight, raw_mask = build_mask(shapes, W, H, opts, ref_frames=tight_refs,
                                        verbose=verbose)

    kind, plan = ("spatial", None)
    if opts.temporal and opts.method == "auto":
        if verbose:
            print("[时序] 分析静态水印…")
        kind, plan = auto_plan(info, mask, allow_fade=opts.fade,
                               allow_unmix=opts.unmix, raw_mask=raw_mask,
                               opts=opts, verbose=verbose)
        if kind == "unmix" and isinstance(plan, UnmixPlan):
            cover = float(((mask > 0) & plan.ok).sum()) / max(1.0, float((mask > 0).sum()))
            if cover < 0.5:                  # 可用像素太少, 直接走空间修复更稳
                if verbose:
                    print(f"[时序] 仿射分离只覆盖 {cover * 100:.0f}% 掩码, 改用空间修复")
                kind, plan = "spatial", None
        if verbose:
            print(f"[时序] 采用方案: {kind}")

    invalid = None
    if kind == "unmix" and isinstance(plan, UnmixPlan):
        invalid = ((mask > 0) & (~plan.ok)).astype(np.uint8) * 255
        if invalid.sum() > 0:
            invalid = dilate(invalid, 2)
            if verbose:
                print(f"[时序] 其中 {int((invalid > 0).sum())}px 无法时域还原, "
                      f"改逐帧空间修复")

    rep_last = None
    # 空间修复路径: 背景基本不动时, 只修一帧参考帧, 再把"修复增量"注入每一帧
    # (速度快一个数量级, 而且没有逐帧闪烁); 背景在动才逐帧修复。
    inject = None
    inject_ring = None
    inject_gref = None
    prep = None
    static_ctx = None
    if kind == "static" and isinstance(plan, StaticPlan):
        # 权重 = "替换掩码"外扩(≈8%√面积, 专用版 1.10 倍椭圆的经验)后羽化:
        # 替换必须完整盖住水印圆盘的最外圈描边, 过渡带落在圆盘外的干净区域,
        # 否则圆盘边缘会留下可见弧线。实测水印超出框选时(plan.repl, 见
        # plan_static_replace 的"水印完整性"), 用扩展掩码 —— 框选偏小也
        # 能完整覆盖水印。
        grow = max(6, int(0.08 * np.sqrt(float((raw_mask > 0).sum()))))
        base_m = plan.repl if plan.repl is not None else raw_mask
        if base_m.dtype != np.uint8:
            base_m = base_m.astype(np.uint8) * 255
        weight = feather_weights(dilate(base_m, grow), max(4.0, opts.feather))
        T_f = plan.T.astype(np.float32)
        # 相位相关用补丁: 掩码外扩 80px 的灰度参考(M), 用于逐帧估计背景平移
        ys_r, xs_r = np.where(base_m > 0)
        pad = 80
        px0 = max(0, int(xs_r.min()) - pad)
        py0 = max(0, int(ys_r.min()) - pad)
        px1 = min(W, int(xs_r.max()) + pad)
        py1 = min(H, int(ys_r.max()) + pad)
        gM_full = cv2.cvtColor(plan.M.astype(np.uint8),
                               cv2.COLOR_BGR2GRAY).astype(np.float32)
        m_big = dilate(base_m, grow) > 0
        static_ctx = {
            "T": T_f,
            "L": plan.L.astype(np.float32),
            "gT": cv2.cvtColor(np.clip(T_f, 0, 255).astype(np.uint8),
                               cv2.COLOR_BGR2GRAY).astype(np.float32),
            "gM": gM_full,
            "protect": plan.protect.astype(np.float32),
            "patch": (px0, py0, px1, py1),
            "ref_patch": np.ascontiguousarray(gM_full[py0:py1, px0:px1]),
            "patch_mask": np.ascontiguousarray(
                (~m_big[py0:py1, px0:px1]).astype(np.uint8) * 255),
            "warp_flags": cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            "warp_full": None,   # 缓存的运动矩阵(每 8 帧重估)
            "ecc_i": -999,
            "ring": (dilate(base_m, grow + 60) > 0) & ~m_big,  # 场景切换检测环带
        }
        # 水印笔画掩码: 水印在所有场景帧恒定, 而画面UI/内容只存在于部分
        # 场景 —— 对多个分散时刻的帧各自做迭代背景检测, 投票取交集, 得到
        # "跨场景恒定"的笔画。场景切换后的逐帧修复只修这些笔画, 周围
        # 背景原样保留(整掩码修复会把切换后的新场景涂花)。
        try:
            n_sc = 7 if info.n_frames >= 120 else max(2, info.n_frames // 24)
            sc_fis = sorted({max(1, min(info.n_frames - 1,
                                        int(info.n_frames * f)))
                             for f in np.linspace(0.05, 0.95, n_sc)})
            sc_frames = grab_frames(src, sc_fis)
            votes = np.zeros((H, W), np.float32)
            n_valid = 0
            reg_px = max(1, int((raw_mask > 0).sum()))
            for f in sc_frames:
                st = detect.refine_mask(f, raw_mask, k=opts.tight_k,
                                        thr=opts.tight_thr, pad=opts.tight_pad)
                if 0 < int((st > 0).sum()) < 0.9 * reg_px:
                    votes += (st > 0)
                    n_valid += 1
            if n_valid >= 2:
                # 0.7 票: 足够剔除"只出现在部分场景"的画面UI; 更高的票数
                # 会把视频固有的恒定装饰(如顶部横梁)也误判成水印抹掉。
                strokes = (votes >= 0.7 * n_valid) & (mask > 0)
                strokes = dilate(strokes.astype(np.uint8) * 255, 2) > 0
                strokes &= (mask > 0)
                if 200 < int(strokes.sum()) < 0.7 * int((mask > 0).sum()):
                    static_ctx["strokes"] = strokes
                    if verbose:
                        print(f"[时序] 水印笔画(跨场景交集): "
                              f"{int(strokes.sum())}px")
        except Exception:
            pass
        if verbose:
            st = plan.stats or {}
            print(f"[时序] 静态层替换: 参考窗 {plan.ref_window}, "
                  f"层强 {st.get('层强', '?')} 灰度, 目标填充 {opts.static_fill}")
    if kind == "spatial":
        # 大掩码 auto: 参考帧上评估 harmonic/side, 选优后锁定具体方法
        # (apply_spatial 每帧调 fill, 双算不可接受)。
        if opts.method == "auto" and int((mask > 0).sum()) >= 8000:
            pick = _pick_smooth_fill(ref, mask, verbose=verbose)
            if pick:
                opts.method = pick
        mode = _choose_spatial_mode(info, mask, opts, verbose=verbose)
        if mode == "inject":
            ref_fixed = _spatial_repair(ref, mask, weight, opts, verbose=verbose)
            inject = (ref_fixed.astype(np.float32) - ref.astype(np.float32)) \
                * weight[..., None]
            # 注入门控预计算: 注入场属于参考帧, 掩码邻域内容变化的帧
            # (场景切换/镜头运动)注入会把旧参考帧的修复色涂到新内容上
            # (实测框选区域白色斑块横跨所有场景帧)。逐帧用掩码外环带
            # 与参考帧的差异检测, 变化帧回退逐帧修复。
            inject_ring = (dilate(mask, 12) > 0) & ~(dilate(mask, 4) > 0)
            inject_gref = cv2.GaussianBlur(
                cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32),
                (0, 0), 3.0)
            if verbose:
                print("[空间] 使用参考帧修复 + 差值注入(整段一次计算, "
                      "内容变化帧自动回退逐帧修复)")
        else:
            prep = prepare_spatial(ref, mask, weight, opts, verbose=verbose)
            if verbose:
                print("[空间] 逐帧修复(颜色校准场已在参考帧算好)")
    # fade 方案的多区域兜底: 层配对法只对"淡入淡出"的水印有效, 同一画面
    # 里时域恒定的贴片(抖音账号信息等)在层中为 0, 减层后原样留下(实测
    # 3.mp4: 右上角淡入水印与左上角账号贴片共存)。按贴片与中值帧的关系
    # 分两种模式兜底:
    #   * ref 含贴片(恒定贴片, 在场帧过半): harmonic(ref) 的差值场即
    #     "去贴片场", 逐帧注入;
    #   * ref 不含贴片(间歇贴片, 如账号信息只显示开头几秒, 采样帧中占比
    #     <50% 被中值滤掉 —— 实测 3.mp4): 逐帧 min/max(frame, ref) ——
    #     贴片在场帧被贴片增亮/压暗, 与无贴片的 ref 取 min/max 即得干净
    #     背景; 贴片不在场时 frame≈ref, 取值不变; 场景切换帧(如结尾
    #     分享页)同样自然回落, 无需门控。方向由采样帧与 ref 的差值符号定。
    fade_sub = None
    fade_inject = None
    fade_fix = None            # None / "min" / "max"
    ref_f = None
    if kind == "fade" and isinstance(plan, FadeLayer):
        lamp = np.abs(plan.layer.astype(np.float32)).max(axis=2)
        fade_sub = ((mask > 0) & (lamp < 2.5)).astype(np.uint8) * 255
        fade_sub = dilate(fade_sub, 2)
        if int((fade_sub > 0).sum()) > 100:
            sub_b = fade_sub > 0
            fixed_ref = methods.fill(ref, sub_b, method="harmonic")
            fixed_ref = colorfit.align_lowfreq(fixed_ref, sub_b,
                                               sigma=opts.lf_sigma)
            fade_inject = (fixed_ref.astype(np.float32)
                           - ref.astype(np.float32))
            amp = float(np.abs(fade_inject)[sub_b].mean())
            if amp > 3.0:
                if verbose:
                    print(f"[时序] 层未覆盖恒定贴片 {int(sub_b.sum())}px, "
                          f"谐波修复场注入兜底")
            else:
                # ref 无贴片 → 间歇贴片: 收集与 ref 差异显著的候选帧,
                # 用候选组的**中值帧**做代表(贴片帧互相相似, 中值天然
                # 排除场景切换帧 —— 实测 3.mp4 结尾的分享页曾被误选为
                # "最清晰帧"污染方向判定), 方向由代表帧与 ref 的差值
                # 符号决定。帧图像在循环里直接留副本 —— 二次 cap.set+
                # read 在某些流上会静默失败, 不要依赖再次 seek。
                gr_ref = cv2.GaussianBlur(
                    cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32),
                    (0, 0), 3.0)
                cand = []
                cap = cv2.VideoCapture(src)
                step = max(1, info.n_frames // 9)
                for si in range(0, info.n_frames, step):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, si)
                    okk, fr = cap.read()
                    if not okk:
                        continue
                    gb = cv2.GaussianBlur(
                        cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                        .astype(np.float32), (0, 0), 3.0)
                    if float(np.abs(gb - gr_ref)[sub_b].mean()) > 6.0:
                        cand.append(fr.copy())
                cap.release()
                if cand:
                    stack = np.stack(
                        [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                         .astype(np.float32) for f in cand])
                    rep_fr = cand[int(np.argmin(
                        np.abs(stack - np.median(stack, axis=0))
                        .mean(axis=(1, 2))))]
                    gb = cv2.cvtColor(rep_fr, cv2.COLOR_BGR2GRAY) \
                        .astype(np.float32)
                    grb = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY) \
                        .astype(np.float32)
                    fade_fix = "min" if float(
                        (gb - grb)[sub_b].mean()) > 0 else "max"
                    ref_f = ref.astype(np.float32)
                    if verbose:
                        print(f"[时序] 层未覆盖间歇贴片 "
                              f"{int(sub_b.sum())}px (候选 {len(cand)} 帧), "
                              f"{fade_fix}(frame, ref) 兜底")
                elif verbose:
                    print(f"[时序] 层未覆盖区域 {int(sub_b.sum())}px "
                          f"与中值帧无显著差异, 无需兜底")
            if fade_inject is not None and amp <= 3.0:
                fade_inject = None
    with VideoWriter(dst, info, crf=opts.crf, preset=opts.preset) as vw:
        for i, frame in FrameReader(info):
            if kind == "fade" and isinstance(plan, FadeLayer):
                k = plan.k_at(i)
                out = frame.astype(np.float32)
                if k > 0.002:
                    corr = (plan.layer * k)[..., :3]
                    we = weight[..., None]
                    out = out - corr * we
                if fade_inject is not None:
                    wsub = np.minimum(
                        weight, (fade_sub > 0).astype(np.float32))[..., None]
                    out = out + fade_inject * wsub
                elif fade_fix == "min":
                    out = np.minimum(out, ref_f)
                elif fade_fix == "max":
                    out = np.maximum(out, ref_f)
                out = np.clip(out, 0, 255).astype(np.uint8)
            elif kind == "unmix" and isinstance(plan, UnmixPlan):
                out = plan.apply(frame, mask, weight)
                if invalid is not None and invalid.sum() > 0:
                    sub = ((mask > 0) & (invalid > 0))
                    if sub.any():
                        fixed = methods.fill(frame, sub, method="telea",
                                             radius=opts.params.get("radius", 3))
                        fixed = colorfit.align_lowfreq(fixed, sub,
                                                       sigma=opts.lf_sigma)
                        wsub = np.minimum(weight, sub.astype(np.float32))
                        w3 = wsub[..., None]
                        out = np.clip(out.astype(np.float32) * (1 - w3) +
                                      fixed.astype(np.float32) * w3, 0, 255
                                      ).astype(np.uint8)
            elif kind == "static" and isinstance(plan, StaticPlan):
                # 运动补偿替换: 手持拍摄时背景在晃, 固定的 T 会留鬼影。
                # 相位相关(快)逐帧估计平移: 位移很小(<0.5px, 固定机位)直接用
                # 未对齐的 T; 位移明显时做 ECC 欧氏对齐(旋转+缩放+平移), 且
                # 每 8 帧才重新估计一次, 中间帧复用上次结果(运动是连续的)。
                g = frame.astype(np.float32)
                gf = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                px0, py0, px1, py1 = static_ctx["patch"]
                d_sh = cv2.phaseCorrelate(
                    static_ctx["ref_patch"], gf[py0:py1, px0:px1])[0]
                moving = abs(d_sh[0]) + abs(d_sh[1]) > 0.5
                if moving and i - static_ctx["ecc_i"] >= 8:
                    warp = ecc_warp(static_ctx["ref_patch"],
                                    gf[py0:py1, px0:px1],
                                    input_mask=static_ctx["patch_mask"])
                    if warp is not None:
                        # 裁剪坐标 → 全帧坐标
                        A = warp[:, :2]
                        o = np.float32([[px0], [py0]])
                        t_full = warp[:, 2:3] + o - A @ o
                        static_ctx["warp_full"] = np.hstack(
                            [A, t_full]).astype(np.float32)
                    static_ctx["ecc_i"] = i
                warp_full = static_ctx["warp_full"]
                if warp_full is not None:
                    inv = static_ctx["warp_flags"]
                    T_w = cv2.warpAffine(static_ctx["T"], warp_full, (W, H),
                                         flags=cv2.INTER_LINEAR | inv,
                                         borderMode=cv2.BORDER_REPLICATE)
                else:
                    T_w = static_ctx["T"]
                # 门控在模糊域计算: 相位相关的亚像素误差在纹理边缘产生的
                # 假差异被平滑掉; 真实遮挡(手/液体等大块变化)仍然保留。
                gf_b = cv2.GaussianBlur(gf, (0, 0), 4.0)
                # 门控参考 = 对齐后的干净背景 T_w + 屏幕固定的水印层 L:
                # 帧 = 当前背景 + 水印, 参考 = 对齐背景 + 水印 → 水印在差值中
                # 完全抵消, d 只反映真实背景变化(若用 warp 后的 M, 水印层被
                # 错误搬移, 会在台标处产生假差异把注入关掉 → 残影)。
                gate_ref = np.clip(T_w + static_ctx["L"], 0, 255).astype(np.uint8)
                gG_b = cv2.GaussianBlur(
                    cv2.cvtColor(gate_ref, cv2.COLOR_BGR2GRAY).astype(np.float32),
                    (0, 0), 4.0)
                d_gate = np.abs(gf_b - gG_b)
                # 软台阶门控: 差异小(良性变化, 如均匀墙面的纹理更替)→全注入;
                # 差异大(真实遮挡/内容替换)→全保留; 中间平滑过渡。
                # 过渡带压窄(0.70~1.15τ): 带内像素 = 半开混合, 会把对齐背景
                # T_w(玻璃色)以 (1-sim) 混入液体等变化内容(实测 f180 液面上
                # 的白色斑块); 带越窄, 半开像素越少, 色斑越轻。液体上涨是慢
                # 过程, 边界逐帧移动小, 窄带不会产生可见跳变。
                tau = max(1.0, opts.gate_tau)
                t0, t1 = 0.70 * tau, 1.15 * tau
                # 场景切换检测: 掩码外环带与对齐参考的差异。环带也大变 →
                # 镜头已切/大幅移动, ECC 对齐不可信, 注入的 T_w 属于旧场景
                # (内容错乱), 保留原帧又会留下台标 → 改逐帧空间修复, 从
                # 当前画面周边重建, 自动适应新场景。环带没变(如 808 液体
                # 涨入, 只有掩码内变化)→ 走正常门控, 内容被保留。
                ring = static_ctx["ring"]
                d_ring = float(d_gate[ring].mean()) if ring.any() else 0.0
                # "环带未变但掩码内持续高差异"——场景颜色巧合相似(环带
                # 差异小)但内容已换: 门控 sim 半开会长期残留水印。掩码内
                # 差异连续多帧居高 → 按场景切换处理; 短暂遮挡仍走门控。
                # 切换过渡带(0.8~1.0·t1)也直接修复: 此处正是硬切边界,
                # 门控半开会产生"两场景混合+水印残留"的闪烁帧。
                mb = mask > 0
                d_in = float(d_gate[mb].mean()) if mb.any() else 0.0
                # "hot"要检测的是场景切换(内容突变)。液体上涨这类缓慢演化
                # 的 d_in 也会持续超阈, 但它是逐帧渐增(每帧增量小): 若照
                # 单纯超阈累计 hot, 涨满后会被误判成切换而走 telea 修复,
                # 在液体上糊出大片色斑(f180 实测)。因此要求"超阈且跳变
                # (或已在 hot 态)"才累计 —— 跳变阈值 8 灰度/帧。
                prev_in = static_ctx.get("d_in_prev", 0.0)
                hot_prev = static_ctx.get("hot", 0)
                if d_in > t1 and (d_in - prev_in > 8.0 or hot_prev > 0):
                    static_ctx["hot"] = hot_prev + 1
                else:
                    static_ctx["hot"] = 0
                static_ctx["d_in_prev"] = d_in
                # 滞回状态机: 场景切换后 d_ring 常在阈值附近波动(灯光频闪/
                # 相机运动), 逐帧独立判断会反复横跳——高差异帧走修复、低谷
                # 帧走半开门控(水印残留 70%+)。检测到切换后强制保持修复
                # 模式数帧; 真正回到原场景时 d_ring 深度回落, 自然恢复门控。
                sw = static_ctx.get("scene_switch", 0)
                if d_ring > 0.8 * t1:
                    sw = 8
                elif sw > 0:
                    sw -= 1
                static_ctx["scene_switch"] = sw
                if verbose and os.environ.get("WM_GATE_LOG"):
                    print(f"    [gate] f{i} d_ring={d_ring:.1f} d_in={d_in:.1f} "
                          f"hot={static_ctx['hot']} sw={sw} t1={t1:.1f}")
                if d_ring > 0.8 * t1 or static_ctx["hot"] >= 12 or sw > 0:
                    # 首选"笔画级修复": 只修跨场景恒定的水印笔画, 切换后
                    # 的新场景背景原样保留(整掩码修复会把新场景涂花)。
                    strokes = static_ctx.get("strokes")
                    w_fix = None
                    if strokes is not None:
                        fixed = methods.fill(frame, strokes, method="telea",
                                             radius=opts.params.get("radius", 3))
                        w_fix = np.minimum(
                            weight, strokes.astype(np.float32))[..., None]
                    else:
                        # 无可靠笔画 → "最均匀侧带延展"(含与所选侧紧邻
                        # 环带的颜色一致性校验); 仍不可行再 telea。telea
                        # 本身在边界连续, 不做低频对齐(异质环带的低频场
                        # 会把修复区拉出色偏)。
                        fixed = _extend_fill(frame, mask > 0)
                        if fixed is None:
                            fixed = methods.fill(
                                frame, mask, method="telea",
                                radius=opts.params.get("radius", 3))
                    ww = w_fix if w_fix is not None else weight[..., None]
                    out = np.clip(g * (1 - ww) +
                                  fixed.astype(np.float32) * ww,
                                  0, 255).astype(np.uint8)
                else:
                    # 笔画区独立判据: 环带可能与新场景巧合相似(d_ring 低,
                    # 误判"同场景"), 但水印笔画区的内容差异骗不了人——
                    # d_st 中位数超 t0 说明笔画区背景已换, 门控半开会让
                    # 水印残留(半透明白字)。此时对笔画区逐帧 telea 重建
                    # (从当前帧周边取材, 自动适配新场景); 掩码其余部分
                    # (羽化边缘, 无水印本体)仍走门控注入, 保留 808 液体
                    # 这类"环带同场景但掩码内有内容变化"的保护语义。
                    strokes = static_ctx.get("strokes")
                    stroke_fix = False
                    if strokes is not None:
                        sm = strokes & mb
                        if sm.any():
                            stroke_fix = \
                                float(np.median(d_gate[sm])) > t0
                    if stroke_fix:
                        fixed = methods.fill(
                            frame, strokes, method="telea",
                            radius=opts.params.get("radius", 3))
                        w_s = np.minimum(
                            weight, strokes.astype(np.float32))[..., None]
                        out = np.clip(g * (1 - w_s) +
                                      fixed.astype(np.float32) * w_s,
                                      0, 255).astype(np.uint8)
                    else:
                        sim = cv2.GaussianBlur(
                            np.clip((t1 - d_gate) / max(1e-3, t1 - t0), 0.0, 1.0)
                            .astype(np.float32), (0, 0), 2.0)
                        we = (weight * sim * static_ctx["protect"])[..., None]
                        out = np.clip(g * (1 - we) + T_w * we, 0, 255
                                      ).astype(np.uint8)
            elif inject is not None:
                if inject_ring is None or inject_gref is None:
                    d_inject = 0.0
                else:
                    g_now = cv2.GaussianBlur(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        .astype(np.float32), (0, 0), 3.0)
                    d_inject = float(
                        np.abs(g_now - inject_gref)[inject_ring].mean())
                if d_inject > 6.0:
                    # 环带已变(场景切换/镜头运动): 注入场不再适用, 该帧
                    # 逐帧修复(从当前画面周边取材, 自动适配新场景)。
                    out = _spatial_repair(frame, mask, weight, opts,
                                          verbose=False)
                else:
                    out = np.clip(frame.astype(np.float32) + inject, 0, 255
                                  ).astype(np.uint8)
            elif prep is not None:
                out = apply_spatial(frame, mask, weight, opts, prep)
                if verbose and (i % 60 == 0):
                    rep_last = colorfit.report(frame, out, mask)
            else:
                out = _spatial_repair(frame, mask, weight, opts, verbose=False)
                if verbose and (i % 60 == 0):
                    rep_last = colorfit.report(frame, out, mask)
            vw.write(out)
            if progress and (i % 5 == 0):
                progress(i + 1, info.n_frames)
    if verbose:
        print(f"[完成] {time.time() - t_start:.1f}s -> {dst}")
        if rep_last:
            print(f"[质量] 色差 {rep_last['色差(区域-环)']:.2f}, "
                  f"灰度差 {rep_last['灰度差']:.2f}, 接缝比 {rep_last['接缝比']:.2f}")
    return {"out": dst, "kind": kind, "report": rep_last, "mask": mask,
            "frames": info.n_frames}


# ------------------------------------------------------------------ 统一入口
def process(src: str, dst: str | None, shapes: Sequence[Shape], opts: Options,
            verbose: bool = True, progress=None) -> dict:
    from .media import IMAGE_EXT, is_image
    if is_image(src):
        return process_image(src, dst, shapes, opts, verbose=verbose)
    return process_video(src, dst, shapes, opts, verbose=verbose, progress=progress)
