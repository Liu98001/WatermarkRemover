# -*- coding: utf-8 -*-
"""水印区域形状: 矩形 / 椭圆 / 多边形 / 涂抹(笔刷) / 外部掩码图

统一栅格化成 0/255 的 uint8 掩码, 并提供羽化权重。
所有形状都用"图像像素坐标"描述, 与分辨率无关的写法见 spec_to_pixels。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import cv2
import numpy as np

KINDS = ("rect", "ellipse", "polygon", "brush", "mask")


@dataclass
class Shape:
    """一个待去除区域。

    rect    : data = (x, y, w, h)
    ellipse : data = (cx, cy, rx, ry)
    polygon : data = [(x1, y1), (x2, y2), ...]
    brush   : data = [(x1, y1), (x2, y2), ...]  + radius
    mask    : data = np.ndarray(H, W) uint8 (255=待处理)
    """

    kind: str
    data: object
    radius: float = 12.0        # brush 笔刷半径
    relative: bool = False      # data 是否为相对比例(0~1), 用于跨分辨率复用
    note: str = ""

    # ---------------------------------------------------------------- 栅格化
    def to_mask(self, W: int, H: int) -> np.ndarray:
        m = np.zeros((H, W), np.uint8)
        d = self._pixel_data(W, H)
        if self.kind == "rect":
            x, y, w, h = d
            x0, y0 = int(round(x)), int(round(y))
            x1, y1 = int(round(x + w)), int(round(y + h))
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(W, x1), min(H, y1)
            if x1 > x0 and y1 > y0:
                m[y0:y1, x0:x1] = 255
        elif self.kind == "ellipse":
            cx, cy, rx, ry = d
            cv2.ellipse(m, (int(round(cx)), int(round(cy))),
                        (max(1, int(round(rx))), max(1, int(round(ry)))),
                        0, 0, 360, 255, -1)
        elif self.kind == "polygon":
            pts = np.asarray(d, np.float64)
            if len(pts) >= 3:
                cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 255)
        elif self.kind == "brush":
            pts = np.asarray(d, np.float64)
            r = max(1, int(round(self._scale_radius(W, H))))
            if len(pts) == 1:
                cv2.circle(m, (int(pts[0][0]), int(pts[0][1])), r, 255, -1)
            elif len(pts) > 1:
                cv2.polylines(m, [np.round(pts).astype(np.int32)], False, 255,
                              thickness=2 * r, lineType=cv2.LINE_8)
                cv2.circle(m, (int(pts[0][0]), int(pts[0][1])), r, 255, -1)
                cv2.circle(m, (int(pts[-1][0]), int(pts[-1][1])), r, 255, -1)
        elif self.kind == "mask":
            arr = np.asarray(d)
            if arr.ndim == 3:
                arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            arr = arr.astype(np.uint8)
            if arr.shape != (H, W):
                arr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_NEAREST)
            m = np.where(arr > 127, 255, 0).astype(np.uint8)
        return m

    def _scale_radius(self, W: int, H: int) -> float:
        if self.relative:
            return self.radius * min(W, H)
        return self.radius

    def _pixel_data(self, W: int, H: int):
        if not self.relative:
            if self.kind == "mask":
                return self.data
            return self.data
        if self.kind == "rect":
            x, y, w, h = self.data
            return (x * W, y * H, w * W, h * H)
        if self.kind == "ellipse":
            cx, cy, rx, ry = self.data
            return (cx * W, cy * H, rx * W, ry * H)
        if self.kind in ("polygon", "brush"):
            return [(px * W, py * H) for px, py in self.data]
        return self.data

    def bbox(self, W: int, H: int):
        m = self.to_mask(W, H)
        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    def scale(self, sx: float, sy: float) -> "Shape":
        """按比例缩放(用于把预览窗口上画的框换算回原图坐标)"""
        d = self.data
        if self.kind == "rect":
            x, y, w, h = d
            return Shape("rect", (x * sx, y * sy, w * sx, h * sy), note=self.note)
        if self.kind == "ellipse":
            cx, cy, rx, ry = d
            return Shape("ellipse", (cx * sx, cy * sy, rx * sx, ry * sy), note=self.note)
        if self.kind in ("polygon", "brush"):
            return Shape(self.kind, [(px * sx, py * sy) for px, py in d],
                         radius=self.radius * (sx + sy) / 2, note=self.note)
        return Shape("mask", d, note=self.note)

    # ---------------------------------------------------------------- 序列化
    def to_spec(self) -> str:
        if self.kind == "rect":
            x, y, w, h = self.data
            return f"rect:{_n(x)},{_n(y)},{_n(w)},{_n(h)}"
        if self.kind == "ellipse":
            cx, cy, rx, ry = self.data
            return f"ellipse:{_n(cx)},{_n(cy)},{_n(rx)},{_n(ry)}"
        if self.kind in ("polygon", "brush"):
            pts = ";".join(f"{_n(px)},{_n(py)}" for px, py in self.data)
            if self.kind == "brush":
                return f"brush[{_n(self.radius)}]:{pts}"
            return f"poly:{pts}"
        return "mask:<外部掩码>"

    def to_dict(self) -> dict:
        d = self.data
        if isinstance(d, np.ndarray):
            d = None
        return {"kind": self.kind, "data": d, "radius": self.radius,
                "relative": self.relative, "note": self.note}

    @staticmethod
    def from_dict(o: dict) -> "Shape":
        return Shape(o.get("kind", "rect"), o.get("data"),
                     radius=o.get("radius", 12.0),
                     relative=o.get("relative", False), note=o.get("note", ""))


def _n(v) -> str:
    f = float(v)
    return str(int(round(f))) if abs(f - round(f)) < 1e-6 else f"{f:.3f}"


def parse_shapes(spec: str) -> list[Shape]:
    """解析命令行/界面输入, 支持分号分隔的多个区域。

    例: "100,200,300,400; ellipse:500,300,120,90; poly:10,10;200,20;150,120"
    """
    out: list[Shape] = []
    if not spec:
        return out
    for raw in str(spec).replace("\n", ";").split(";"):
        s = raw.strip()
        if not s:
            continue
        low = s.lower()
        if low.startswith("rect:"):
            vals = _floats(s[5:])
            if len(vals) == 4:
                out.append(Shape("rect", tuple(vals)))
        elif low.startswith("ellipse:") or low.startswith("ell:"):
            vals = _floats(s.split(":", 1)[1])
            if len(vals) == 4:
                out.append(Shape("ellipse", tuple(vals)))
        elif low.startswith("poly:") or low.startswith("polygon:"):
            pts = _points(s.split(":", 1)[1])
            if len(pts) >= 3:
                out.append(Shape("polygon", pts))
        elif low.startswith("brush"):
            r = 12.0
            body = s[s.find("]") + 1:] if "[" in s else s[6:]
            if "[" in s:
                r = float(s[s.find("[") + 1:s.find("]")])
            pts = _points(body)
            if pts:
                out.append(Shape("brush", pts, radius=r))
        else:
            vals = _floats(s)
            if len(vals) == 4:
                out.append(Shape("rect", tuple(vals)))
            else:
                pts = _points(s)
                if len(pts) >= 3:
                    out.append(Shape("polygon", pts))
    return out


def _floats(s: str) -> list[float]:
    return [float(v) for v in s.replace("，", ",").split(",") if v.strip() != ""]


def _points(s: str) -> list[tuple[float, float]]:
    pts = []
    for pair in s.replace("，", ",").split(";"):
        pair = pair.strip()
        if not pair:
            continue
        vals = _floats(pair)
        if len(vals) >= 2:
            pts.append((vals[0], vals[1]))
    return pts


def shapes_to_json(shapes: Sequence[Shape], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in shapes], f, ensure_ascii=False, indent=2)


def shapes_from_json(path: str) -> list[Shape]:
    with open(path, encoding="utf-8") as f:
        return [Shape.from_dict(o) for o in json.load(f)]


# -------------------------------------------------------------------- 掩码工具
def combine_masks(masks: Iterable[np.ndarray]) -> np.ndarray:
    out = None
    for m in masks:
        out = m if out is None else np.maximum(out, m)
    return out


def feather_weights(mask: np.ndarray, sigma: float) -> np.ndarray:
    """0~1 权重: 掩码内≈1, 边缘向外平滑过渡(消除硬边)"""
    if sigma <= 0:
        return (mask > 0).astype(np.float32)
    w = cv2.GaussianBlur((mask > 0).astype(np.float32), (0, 0), float(sigma))
    return np.clip(w, 0, 1)


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask, k)


def erode(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.erode(mask, k)


def ring_mask(mask: np.ndarray, inner: int = 2, outer: int = 14) -> np.ndarray:
    """掩码外侧的环带(用于统计周边颜色/纹理)"""
    a = dilate(mask, inner)
    b = dilate(mask, outer)
    return cv2.subtract(b, a)


def mask_bbox(mask: np.ndarray, pad: int = 0):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    H, W = mask.shape[:2]
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(W, int(xs.max()) + 1 + pad)
    y1 = min(H, int(ys.max()) + 1 + pad)
    return x0, y0, x1, y1
