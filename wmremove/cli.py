# -*- coding: utf-8 -*-
"""命令行入口: remove / detect / download / batch / gui / selftest"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import Options, Shape, engine
from .detect import describe_hits, detect_image, detect_video, select_hits
from .media import (collect_media, default_out_path, is_image, is_video,
                    imread_u, probe)
from .shapes import parse_shapes, shapes_from_json, shapes_to_json


def add_common(p: argparse.ArgumentParser) -> None:
    p.set_defaults(func=cmd_remove)
    p.add_argument("-i", "--input", help="本地文件 / 文件夹 / 视频链接")
    p.add_argument("-o", "--output", help="输出文件或目录(默认: 源目录/去水印输出)")
    p.add_argument("--box", action="append", default=[],
                   help="矩形 x,y,w,h (可多次)")
    p.add_argument("--ellipse", action="append", default=[],
                   help="椭圆 cx,cy,rx,ry (可多次)")
    p.add_argument("--polygon", action="append", default=[],
                   help="多边形 x1,y1;x2,y2;x3,y3... (可多次)")
    p.add_argument("--brush", action="append", default=[],
                   help="涂抹 x1,y1;x2,y2... 或 带半径 x1,y1;x2,y2 (半径12)")
    p.add_argument("--mask", action="append", default=[],
                   help="外部掩码图(png, 白=待处理)")
    p.add_argument("--regions-json", help="从 JSON 读入区域(GUI 保存的文件)")
    p.add_argument("--relative", action="store_true",
                   help="坐标按画面比例(0~1)解释, 便于批量套用")
    p.add_argument("--auto", action="store_true", help="自动检测水印区域")
    p.add_argument("--auto-name", default="四角",
                   help="自动检测的位置提示: 四角/豆包AI/抖音/视频号/腾讯视频/即梦AI…")
    p.add_argument("--save-regions", help="把(检测到的)区域存成 JSON, 供界面/批量复用")
    # 方法与质量
    p.add_argument("--method", default="auto",
                   choices=["auto", "telea", "ns", "texture", "synthesis",
                            "harmonic", "vertical", "horizontal", "clone",
                            "blur", "pixelate", "dark"],
                   help="修复方法, 默认 auto")
    p.add_argument("--no-tight", action="store_true",
                   help="关闭紧致掩码(默认开启: 只修复水印笔画, 伤害最小)")
    p.add_argument("--tight-thr", type=float, default=4.0)
    p.add_argument("--tight-pad", type=int, default=3)
    p.add_argument("--dilate", type=int, default=0, help="掩码额外扩张(px)")
    p.add_argument("--erode", type=int, default=0, help="掩码额外收缩(px)")
    p.add_argument("--feather", type=float, default=3.0, help="边缘羽化(px)")
    p.add_argument("--no-colorfit", action="store_true",
                   help="关闭颜色/低频一致性处理")
    p.add_argument("--lf-sigma", type=float, default=25.0)
    p.add_argument("--grain", type=float, default=0.0,
                   help="补颗粒强度(0=自动不加)")
    p.add_argument("--no-temporal", action="store_true",
                   help="关闭时序分析(淡入层/仿射分离/静态层替换)")
    p.add_argument("--static-fill", default="auto",
                   choices=["auto", "vertical", "horizontal", "texture", "harmonic"],
                   help="静态层替换的目标填充: auto=按掩码几何自动选轴(默认), "
                        "vertical=竖向(玻璃/柱面), horizontal=横向(贴边台标), "
                        "texture=纹理搬运, harmonic=平滑")
    p.add_argument("--gate-tau", type=float, default=45.0,
                   help="静态层替换的内容变化门控阈值(灰度, 越小越保守)")
    p.add_argument("--per-frame", action="store_true",
                   help="强制逐帧修复(默认自动: 背景不动整段只算一次)")
    p.add_argument("--crf", type=int, default=17, help="H.264 质量(越小越好)")
    p.add_argument("--strength-params", default="",
                   help="传给底层方法的参数, 如 patch=9,search=60")
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("--verify", action="store_true", help="处理后输出质量指标")


def collect_shapes(args, W: int, H: int, frame=None, verbose=False):
    shapes: list[Shape] = []
    for s in args.box:
        shapes += parse_shapes(f"rect:{s}")
    for s in args.ellipse:
        shapes += parse_shapes(f"ellipse:{s}")
    for s in args.polygon:
        shapes += parse_shapes(f"poly:{s}")
    for s in args.brush:
        shapes += parse_shapes(f"brush:{s}" if "[" in s else f"brush:{s}")
    for mpath in args.mask:
        import cv2
        import numpy as np
        arr = imread_u(mpath, cv2.IMREAD_GRAYSCALE)
        shapes.append(Shape("mask", arr))
    if args.regions_json:
        shapes += shapes_from_json(args.regions_json)
    if args.relative:
        shapes = [Shape(s.kind, s.data, radius=s.radius, relative=True,
                        note=s.note) for s in shapes]
    if args.auto:
        if frame is None:
            frame = imread_u(args.input) if is_image(args.input) else \
                engine._median_frame(probe(args.input),
                                     [probe(args.input).n_frames // 3])
        hits = (detect_image(frame, args.auto_name) if is_image(args.input)
                else detect_video(probe(args.input), args.auto_name))
        if verbose:
            print(f"[检测] {describe_hits(hits)}")
        if not hits:
            print("  未检测到明显水印, 请手动指定区域")
        else:
            # 指定平台: 最多取 2 个(双水印平台如抖音 用户名+logo); 通用扫描取 1 个
            named = args.auto_name not in ("四角", "auto", "")
            for h in select_hits(hits, 2 if named else 1):
                x0, y0, x1, y1 = h["box"]
                shapes.append(Shape("rect", (x0, y0, x1 - x0, y1 - y0),
                                    note=f"自动检测 {h['tag']}"))
    if not shapes:
        return None
    return shapes


def build_opts(args) -> Options:
    params = {}
    for kv in filter(None, args.strength_params.split(",")):
        if "=" in kv:
            k, v = kv.split("=", 1)
            try:
                params[k.strip()] = float(v) if "." in v else int(v)
            except ValueError:
                params[k.strip()] = v
    return Options(method=args.method, feather=args.feather,
                   tight=not args.no_tight, tight_thr=args.tight_thr,
                   tight_pad=args.tight_pad, colorfit=not args.no_colorfit,
                   lf_sigma=args.lf_sigma, grain=args.grain,
                   dilate_px=args.dilate, erode_px=args.erode,
                   temporal=not args.no_temporal,
                   static_fill=args.static_fill, gate_tau=args.gate_tau,
                   per_frame=True if args.per_frame else None,
                   crf=args.crf, params=params)


def cmd_remove(args) -> int:
    opts = build_opts(args)
    verbose = not args.quiet
    src = args.input
    if not src:
        print("请用 -i 指定输入(文件/文件夹/链接)"); return 2
    if re_url(src):
        from .downloader import download
        src = download(src, args.output or "downloads")
    targets = collect_media(src, recursive=False) if os.path.isdir(src) else [src]
    if not targets:
        print(f"没有可处理的媒体文件: {src}"); return 2
    out_dir = args.output if (args.output and not os.path.splitext(args.output)[1]) \
        else None
    rc = 0
    for t in targets:
        if not (is_video(t) or is_image(t)):
            continue
        try:
            info = probe(t) if is_video(t) else None
            W = info.out_width if info else imread_u(t).shape[1]
            H = info.out_height if info else imread_u(t).shape[0]
            frame = engine._median_frame(info, [info.n_frames // 3]) if info else \
                imread_u(t)
            shapes = collect_shapes(args, W, H, frame=frame, verbose=verbose)
            if not shapes:
                print(f"[跳过] {t}: 未指定任何区域"); rc = 2; continue
            if verbose:
                for s in shapes:
                    print(f"  区域: {s.to_spec()} {s.note}")
            if args.save_regions:
                shapes_to_json(shapes, args.save_regions)
                print(f"  区域已保存: {args.save_regions}")
            dst = args.output if (len(targets) == 1 and args.output and
                                  os.path.splitext(args.output)[1]) else \
                default_out_path(t, out_dir)
            engine.process(t, dst, shapes, opts, verbose=verbose)
        except Exception as e:                                # noqa: BLE001
            print(f"[失败] {t}: {e}")
            rc = 1
    return rc


def re_url(s: str) -> bool:
    return bool(s and s.lower().startswith(("http://", "https://")))


def cmd_detect(args) -> int:
    src = args.input
    if re_url(src):
        from .downloader import download
        src = download(src, args.output or "downloads")
    info = probe(src) if is_video(src) else None
    frame = engine._median_frame(info, [info.n_frames // 3]) if info else imread_u(src)
    hits = detect_video(info, args.name) if info else \
        detect_image(frame, args.name)
    print(describe_hits(hits))
    for h in hits:
        print(json.dumps({"box": h["box"], "tag": h["tag"],
                          "score": round(h["score"], 2)}, ensure_ascii=False))
    if args.save_mask:
        import cv2
        import numpy as np
        m = np.zeros(frame.shape[:2], np.uint8)
        for h in hits:
            m = np.maximum(m, h["mask"])
        cv2.imwrite(args.save_mask, m)
        print(f"掩码已保存: {args.save_mask}")
    return 0 if hits else 1


def cmd_download(args) -> int:
    from .downloader import download, download_direct
    if args.direct:
        download_direct(args.url, args.output or "downloads", args.name)
    else:
        download(args.url, args.output or "downloads", args.quality,
                 args.cookies, platform=getattr(args, "platform", None))
    return 0


def cmd_batch(args) -> int:
    """批量: 对文件夹里所有视频/图片套用同一组区域(建议配合 --relative)"""
    opts = build_opts(args)
    files = collect_media(args.input)
    if not files:
        print("没有可处理的文件"); return 2
    print(f"批量处理 {len(files)} 个文件")
    rc = 0
    for f in files:
        try:
            info = probe(f) if is_video(f) else None
            W = info.out_width if info else imread_u(f).shape[1]
            H = info.out_height if info else imread_u(f).shape[0]
            frame = engine._median_frame(info, [info.n_frames // 3]) if info else \
                imread_u(f)
            shapes = collect_shapes(args, W, H, frame=frame)
            if not shapes:
                print(f"[跳过] {f}: 未指定区域"); continue
            engine.process(f, default_out_path(f, args.output), shapes, opts,
                           verbose=not args.quiet)
        except Exception as e:                                # noqa: BLE001
            print(f"[失败] {f}: {e}")
            rc = 1
    return rc


def cmd_gui(_args) -> int:
    from .gui import main as gui_main
    return gui_main()


def cmd_selftest(args) -> int:
    import subprocess
    tests = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "tests", "selftest.py")
    cmd = [sys.executable, tests]
    if args.filter:
        cmd += ["-k", args.filter]
    if args.no_video:
        cmd.append("--no-video")
    return subprocess.call(cmd)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="去水印",
        description="通用去水印工具: 视频/图片, 任意形状框选, 自动检测, 颜色无痕融合")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("remove", help="去水印(视频/图片)")
    add_common(p)

    p = sub.add_parser("detect", help="自动检测水印区域(打印候选框/掩码)")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("--name", default="四角",
                   help="位置提示: 四角/豆包AI/抖音/视频号/腾讯视频/即梦AI…")
    p.add_argument("--save-mask")
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("download", help="下载视频(抖音/小红书/B站/快手/视频号/直链…)")
    p.add_argument("url", help="链接或分享文本")
    p.add_argument("-o", "--output", default="downloads")
    p.add_argument("--quality", default="best",
                   choices=["best", "1080", "720", "480"])
    p.add_argument("--platform", default=None,
                   help="指定平台(跳过自动识别): 抖音/小红书/B站/快手/"
                        "微信视频号/爱奇艺/优酷/腾讯视频/直链")
    p.add_argument("--cookies", help="cookies 文件, 或浏览器名(edge/chrome/firefox)")
    p.add_argument("--direct", action="store_true", help="按直链直接下载")
    p.add_argument("--name", help="直链下载的文件名")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("batch", help="批量处理文件夹(配合 --relative 套用同区域)")
    add_common(p)
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("gui", help="打开框选界面")
    p.set_defaults(func=cmd_gui)

    p = sub.add_parser("selftest", help="运行自检测试")
    p.add_argument("-k", "--filter", default="")
    p.add_argument("--no-video", action="store_true")
    p.set_defaults(func=cmd_selftest)

    args = ap.parse_args(argv)
    if not args.cmd:
        ap.print_help()
        return 0
    if args.cmd in ("remove", "batch") and not getattr(args, "input", None):
        print("请用 -i 指定输入"); return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
