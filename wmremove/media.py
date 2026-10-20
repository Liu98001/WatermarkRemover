# -*- coding: utf-8 -*-
"""媒体读写: ffmpeg 定位 / 视频探测 / 逐帧管道编码(音频直通) / 图片读写(兼容中文路径)"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

import cv2
import numpy as np

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".flv", ".wmv", ".webm", ".m4v",
             ".mpg", ".mpeg", ".ts", ".3gp", ".rmvb", ".vob"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# --------------------------------------------------------------------- ffmpeg
def find_ffmpeg() -> str:
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    for c in ("ffmpeg.exe", "ffmpeg"):
        cand = os.path.join(here, c)
        if os.path.isfile(cand):
            return cand
    raise RuntimeError("找不到 ffmpeg。请安装: pip install imageio-ffmpeg")


def find_ffprobe() -> str | None:
    return shutil.which("ffprobe")


# Windows 下子进程不弹控制台窗口(ffmpeg/ffprobe 是控制台程序)
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def has_audio(path: str) -> bool:
    ff = find_ffprobe()
    if ff:
        try:
            out = subprocess.run([ff, "-v", "error", "-select_streams", "a",
                                  "-show_entries", "stream=index", "-of", "csv=p=0", path],
                                 capture_output=True, text=True, encoding="utf-8",
                                 errors="replace", timeout=30,
                                 creationflags=CREATE_NO_WINDOW)
            return bool(out.stdout.strip())
        except Exception:
            pass
    ffm = find_ffmpeg()
    try:
        out = subprocess.run([ffm, "-hide_banner", "-i", path],
                             capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=30,
                             creationflags=CREATE_NO_WINDOW)
        return "Audio:" in (out.stderr or "")
    except Exception:
        return False


# ------------------------------------------------------------------ 图片读写
def imread_u(path: str, flags=cv2.IMREAD_COLOR) -> np.ndarray:
    """兼容中文/空格路径的读图"""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise IOError(f"无法读取图片: {path}")
    img = cv2.imdecode(data, flags)
    if img is None:
        raise IOError(f"无法解码图片: {path}")
    return img


def imwrite_u(path: str, img: np.ndarray, quality: int = 95) -> None:
    """兼容中文/空格路径的写图"""
    ext = os.path.splitext(path)[1].lower() or ".png"
    params = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError(f"无法编码图片: {path}")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    buf.tofile(path)


# ------------------------------------------------------------------ 视频探测
@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    n_frames: int
    duration: float
    rotation: int = 0
    audio: bool = False

    @property
    def out_width(self) -> int:
        return self.height if self.rotation in (90, 270) else self.width

    @property
    def out_height(self) -> int:
        return self.width if self.rotation in (90, 270) else self.height


def probe(path: str) -> VideoInfo:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {path}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    rot = 0
    try:
        r = cap.get(getattr(cv2, "CAP_PROP_ORIENTATION_META", 48))
        rot = int(round(r or 0)) % 360
    except Exception:
        rot = 0
    cap.release()
    if fps <= 0:
        fps = 25.0
    if n <= 0:
        n = 0
    dur = (n / fps) if fps else 0.0
    if rot in (90, 270):
        W, H = H, W
    return VideoInfo(path, W, H, fps, n, dur, rot, has_audio(path))


class FrameReader:
    """按序读帧, 自动应用旋转(保证所有帧方向一致)"""

    def __init__(self, info: VideoInfo, start: int = 0, count: int = 0):
        self.info = info
        self.cap = cv2.VideoCapture(info.path)
        if start:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
        self.index = int(start)
        self.count = count

    def __iter__(self):
        return self

    def __next__(self) -> tuple[int, np.ndarray]:
        if self.count and self.index >= self.count:
            raise StopIteration
        ok, f = self.cap.read()
        if not ok or f is None:
            raise StopIteration
        if self.info.rotation in (90, 270):
            f = cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE if self.info.rotation == 90
                           else cv2.ROTATE_90_COUNTERCLOCKWISE)
        i = self.index
        self.index += 1
        return i, f

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.release()


def _encode_args(crf: int, preset: str = "slow", pix_fmt: str = "yuv420p",
                 x264_opts: str = "") -> list[str]:
    args = ["-c:v", "libx264", "-crf", str(int(crf)), "-preset", preset,
            "-pix_fmt", pix_fmt]
    if x264_opts:
        args += ["-x264-params", x264_opts]
    return args


class VideoWriter:
    """把逐帧 numpy(BGR) 写进 FFmpeg, 音频直接从源视频复制"""

    def __init__(self, out_path: str, info: VideoInfo, crf: int = 17,
                 preset: str = "slow", copy_audio: bool = True, verbose: bool = False):
        ff = find_ffmpeg()
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        cmd = [ff, "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{info.out_width}x{info.out_height}",
               "-r", f"{info.fps:.6f}", "-i", "pipe:0"]
        if copy_audio and info.audio:
            cmd += ["-i", info.path, "-map", "0:v:0", "-map", "1:a:0?",
                    "-c:a", "copy", "-shortest"]
        cmd += _encode_args(crf, preset)
        if out_path.lower().endswith((".mp4", ".mov", ".m4v")):
            cmd += ["-movflags", "+faststart"]
        cmd += [out_path]
        self.cmd = cmd
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=None if verbose else subprocess.DEVNULL,
                                     stderr=None if verbose else subprocess.PIPE,
                                     creationflags=CREATE_NO_WINDOW)
        self.out_path = out_path
        self.n = 0

    def write(self, frame: np.ndarray) -> None:
        f = np.ascontiguousarray(frame)
        try:
            self.proc.stdin.write(f.tobytes())
        except BrokenPipeError:
            err = b""
            if self.proc.stderr:
                err = self.proc.stderr.read() or b""
            raise RuntimeError("FFmpeg 写入失败: " + err.decode("utf-8", "ignore")[:500])
        self.n += 1

    def close(self) -> None:
        if self.proc.stdin:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
        rc = self.proc.wait()
        if self.proc.stderr:
            err = self.proc.stderr.read() or b""
            self.proc.stderr.close()
        else:
            err = b""
        if rc != 0:
            raise RuntimeError("FFmpeg 编码失败: " + err.decode("utf-8", "ignore")[:800])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:
            try:
                self.proc.kill()
            except Exception:
                pass


# ------------------------------------------------------------------- 其它工具
def is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXT


def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXT


def collect_media(folder: str, recursive: bool = True) -> list[str]:
    out = []
    if recursive:
        for root, _dirs, files in os.walk(folder):
            for f in files:
                p = os.path.join(root, f)
                if is_video(p) or is_image(p):
                    out.append(p)
    else:
        for f in os.listdir(folder):
            p = os.path.join(folder, f)
            if os.path.isfile(p) and (is_video(p) or is_image(p)):
                out.append(p)
    return sorted(out)


def default_out_path(src: str, out_dir: str | None = None, suffix: str = "_no_watermark") -> str:
    d = out_dir or os.path.join(os.path.dirname(os.path.abspath(src)), "去水印输出")
    stem, ext = os.path.splitext(os.path.basename(src))
    if ext.lower() in (".jpg", ".jpeg"):
        ext = ".png"        # 去水印后重编码, 用无损格式避免二次压缩
        stem = stem + "_no_watermark"
        return os.path.join(d, stem + ext)
    return os.path.join(d, stem + suffix + ext)


def first_frame(path: str, idx: int = 0) -> np.ndarray:
    if is_image(path):
        return imread_u(path)
    info = probe(path)
    with FrameReader(info) as r:
        if idx:
            r.cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            r.index = int(idx)
        _, f = next(r)
        return f


def grab_frames(path: str, idxs) -> list[np.ndarray]:
    if is_image(path):
        img = imread_u(path)
        return [img for _ in idxs]
    info = probe(path)
    out = []
    with FrameReader(info) as r:
        for i, f in r:
            if i in set(idxs):
                out.append(f)
            if out and i > max(idxs):
                break
    return out
