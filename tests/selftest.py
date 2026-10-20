# -*- coding: utf-8 -*-
"""合成真值回归测试: python tests/selftest.py [-k 关键字] [--no-video]

生成已知水印的合成图/合成视频, 跑一遍去水印管线, 用 PSNR/边界台阶量化效果。
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wmremove import Options, Shape, engine           # noqa: E402
from wmremove.colorfit import seam_step               # noqa: E402
from wmremove.shapes import feather_weights           # noqa: E402


def psnr(a: np.ndarray, b: np.ndarray, mask=None) -> float:
    d = a.astype(np.float32) - b.astype(np.float32)
    if mask is not None:
        d = d[mask.astype(bool)]
    mse = float(np.mean(d ** 2)) if d.size else 0.0
    return 10 * np.log10(255.0 ** 2 / max(1e-6, mse))


def gradient_bg(h: int = 240, w: int = 420) -> np.ndarray:
    """纯渐变背景"""
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), np.float32)
    for c, k in enumerate((0.5, 0.3, 0.7)):
        img[..., c] = 40 + 120 * (k * xx / w + (1 - k) * yy / h)
    return np.clip(img, 0, 255).astype(np.uint8)


def text_watermark(h: int, w: int, text: str = "watermark", alpha: float = 0.6):
    layer = np.zeros((h, w), np.uint8)
    cv2.putText(layer, text, (w // 2 - 120, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                1.4, 255, 3, cv2.LINE_AA)
    m = layer > 0
    return m, alpha


def case_image_gradient(tmp: str) -> None:
    H, W = 240, 420
    bg = gradient_bg(H, W)
    m, alpha = text_watermark(H, W, "gradient", 0.5)
    src = bg.copy()
    src[m] = (src[m] * (1 - alpha) + 255 * alpha).astype(np.uint8)
    shape = Shape("rect", (W // 2 - 160, H // 2 - 40, 320, 80))
    opts = Options()
    out, _, _ = engine.repair_image(src.copy(), [shape], opts)
    p = psnr(out, bg, m)
    print(f"图片·纯渐变背景: 区域 PSNR {p:.1f} dB")


def case_image_text(tmp: str) -> None:
    rng = np.random.RandomState(3)
    H, W = 240, 420
    bg = (rng.rand(H, W, 3) * 60 + 90).astype(np.uint8)
    m, alpha = text_watermark(H, W, "hello wm", 0.7)
    src = bg.copy()
    src[m] = (src[m] * (1 - alpha) + 255 * alpha).astype(np.uint8)
    shape = Shape("rect", (W // 2 - 150, H // 2 - 40, 300, 80))
    opts = Options()
    out, _, _ = engine.repair_image(src.copy(), [shape], opts)
    p = psnr(out, bg, m)
    print(f"图片·文字水印: 区域 PSNR {p:.1f} dB")


def _write_video(path: str, frames, fps: float = 25.0) -> None:
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(np.clip(f, 0, 255).astype(np.uint8))
    vw.release()


def _read_video(path: str):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def case_video_static(tmp: str) -> None:
    """静止背景 + 常驻水印(半透明白)"""
    H, W = 240, 420
    rng = np.random.RandomState(5)
    bg = (rng.rand(H, W, 3) * 40 + 100).astype(np.float32)
    layer = np.zeros((H, W), np.uint8)
    cv2.putText(layer, "static wm", (90, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                255, 2, cv2.LINE_AA)
    m = layer > 0
    alpha = 0.55
    frames = []
    for k in range(40):
        f = bg + rng.normal(0, 2.0, bg.shape)
        f[m] = f[m] * (1 - alpha) + 255 * alpha
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    src_v = os.path.join(tmp, "static_src.mp4")
    _write_video(src_v, frames)
    shape = Shape("rect", (80, 90, 260, 60))
    opts = Options(temporal=True)
    engine.process(src_v, os.path.join(tmp, "static_out.mp4"), [shape], opts,
                   verbose=False)
    out_frames = _read_video(os.path.join(tmp, "static_out.mp4"))
    mid = out_frames[len(out_frames) // 2]
    p = psnr(mid, frames[len(frames) // 2], m)
    print(f"视频·静止背景常驻水印: 区域 PSNR {p:.1f} dB")


def case_video_fade(tmp: str) -> None:
    """动态背景 + AI 淡入水印"""
    H, W = 240, 420
    rng = np.random.RandomState(9)
    layer = np.zeros((H, W), np.uint8)
    cv2.putText(layer, "AI fade", (130, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                255, 2, cv2.LINE_AA)
    m = layer > 0
    frames = []
    n = 60
    for k in range(n):
        bg = (rng.rand(H, W, 3) * 50 + 90).astype(np.float32)
        a = min(1.0, max(0.0, (k - n * 0.3) / (n * 0.2))) * 0.6
        f = bg
        f[m] = f[m] * (1 - a) + 255 * a
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    src_v = os.path.join(tmp, "fade_src.mp4")
    _write_video(src_v, frames)
    shape = Shape("rect", (120, 90, 190, 60))
    opts = Options(temporal=True)
    engine.process(src_v, os.path.join(tmp, "fade_out.mp4"), [shape], opts,
                   verbose=False)
    out_frames = _read_video(os.path.join(tmp, "fade_out.mp4"))
    mid = out_frames[int(n * 0.8)]
    p = psnr(mid, frames[int(n * 0.8)], m)
    print(f"视频·AI 淡入水印: 区域 PSNR {p:.1f} dB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--filter", default="")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="wmr_selftest_")
    tests = [
        ("gradient", case_image_gradient),
        ("text", case_image_text),
        ("static", case_video_static),
        ("fade", case_video_fade),
    ]
    for name, fn in tests:
        if args.filter and args.filter not in name:
            continue
        if args.no_video and name in ("static", "fade"):
            continue
        try:
            fn(tmp)
        except Exception as e:                                # noqa: BLE001
            print(f"[失败] {name}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
