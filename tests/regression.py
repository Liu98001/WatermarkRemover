# -*- coding: utf-8 -*-
"""多平台去水印回归测试

对每个输入视频: 取开头片段 -> 平台先验自动检测水印 -> 处理 -> 抽帧输出
对比图(源 vs 处理后), 供人工核对水印是否去除干净、修复是否自然。

用法:
    python tests/regression.py                    # 跑全部默认用例
    python tests/regression.py --seconds 20       # 每个视频取前 20 秒
    python tests/regression.py --only r5,r6       # 只跑指定用例
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from wmremove import Options, engine
from wmremove.media import probe
from wmremove.shapes import parse_shapes

# 用例: (名称, 视频路径, 平台先验[None=通用四角])
CASES_DIR = r"D:\dsh-data\temp\wmr_plat"
LOCAL_DIR = r"D:\HuaweiMoveData\Users\liuho\Desktop\去水印测试结果"


def default_cases() -> list[tuple[str, str, str | None]]:
    cases = [
        ("r1", os.path.join(LOCAL_DIR, "1.mp4"), None),
        ("r2", os.path.join(LOCAL_DIR, "2.mp4"), None),
        ("r3", os.path.join(LOCAL_DIR, "3.mp4"), "抖音"),
        ("r4", os.path.join(LOCAL_DIR, "4.mp4"), "B站"),
        ("r5", os.path.join(CASES_DIR, "bili",
                            "【官方 MV】Never Gonna Give You Up - Rick "
                            "Astley.mp4"), "B站"),
        ("r6", os.path.join(CASES_DIR, "bili",
                            "全新自助奶茶机，体积小，功能多，操作简单，现货出售.mp4"),
         "B站"),
        ("r7", os.path.join(CASES_DIR, "douyin",
                            "水韵天堂，幸福泰州 #吹爆美好目的地 #有一种幸福叫"
                            "泰州 @泰州文旅 #每一帧都是热爱 #大美中国之江苏"
                            ".mp4"), "抖音"),
        ("r8", os.path.join(CASES_DIR, "weibo",
                            "微短剧《台湾人在大陆阿里山》第1集#原创微短剧##台"
                            "湾人在大陆阿里山#.mp4"), None),
    ]
    return [(n, p, a) for n, p, a in cases if os.path.exists(p)]


def cut_clip(src: str, dst: str, seconds: int) -> bool:
    """取前 N 秒( ffmpeg/opencv 裁剪)"""
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_target = int(fps * seconds)
    vw = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    i = 0
    while i < n_target:
        ok, f = cap.read()
        if not ok:
            break
        vw.write(f)
        i += 1
    cap.release()
    vw.release()
    return i > 0


def run_case(name: str, src: str, auto_name: str | None, seconds: int,
             out_dir: str, extra_box: list[str] | None = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    clip = os.path.join(out_dir, f"{name}_clip.mp4")
    dst = os.path.join(out_dir, f"{name}_out.mp4")
    if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
        if not cut_clip(src, clip, seconds):
            return {"name": name, "ok": False, "msg": "裁剪失败"}
    info = probe(clip)
    W, H = info.out_width, info.out_height
    # 自动检测(平台先验)
    from wmremove.detect import detect_video, describe_hits, select_hits
    frame = engine._median_frame(info, [info.n_frames // 3])
    hits = detect_video(info, auto_name) if auto_name \
        else detect_video(info, "四角")
    shapes = []
    picked = []
    named = auto_name not in (None, "四角", "")
    for hk in select_hits(hits, 2 if named else 1):
        x0, y0, x1, y1 = hk["box"]
        shapes.append(engine.Shape("rect", (x0, y0, x1 - x0, y1 - y0),
                                   note=f"自动检测 {hk['tag']}"))
        picked.append(hk["tag"])
    if extra_box:
        for b in extra_box:
            shapes += parse_shapes(f"rect:{b}")
    if not shapes:
        return {"name": name, "ok": False, "msg": "未检测到水印(无水印流?)"}
    opts = Options(method="auto", temporal=True)
    rep = engine.process_video(clip, dst, shapes, opts, verbose=False)
    # 抽帧对比图: 源 vs 处理后(拼接水印区域)
    cap_s = cv2.VideoCapture(clip)
    cap_o = cv2.VideoCapture(dst)
    ys, xs = np.where(rep["mask"] > 0)
    x0, x1 = max(0, int(xs.min()) - 40), min(W, int(xs.max()) + 40)
    y0, y1 = max(0, int(ys.min()) - 40), min(H, int(ys.max()) + 40)
    for fi in (int(info.n_frames * 0.1), int(info.n_frames * 0.5)):
        cap_s.set(cv2.CAP_PROP_POS_FRAMES, fi)
        cap_o.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok1, fs = cap_s.read()
        ok2, fo = cap_o.read()
        if ok1 and ok2:
            a = fs[y0:y1, x0:x1]
            b = fo[y0:y1, x0:x1]
            img = np.hstack([a, np.full((a.shape[0], 6, 3), 255, np.uint8), b])
            cv2.imwrite(os.path.join(out_dir, f"{name}_cmp_f{fi}.png"), img)
    cap_s.release()
    cap_o.release()
    r = rep.get("report") or {}
    return {"name": name, "ok": True, "检测": picked,
            "色差": round(r.get("色差(区域-环)", -1), 2),
            "接缝": round(r.get("接缝比", -1), 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=20)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=r"D:\dsh-data\temp\wmr_plat\reg")
    args = ap.parse_args()
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    results = []
    for name, src, auto_name in default_cases():
        if only and name not in only:
            continue
        print(f"[回归] {name}: {os.path.basename(src)[:40]} "
              f"(先验={auto_name or '四角'})", flush=True)
        try:
            r = run_case(name, src, auto_name, args.seconds, args.out)
        except Exception as e:                            # noqa: BLE001
            r = {"name": name, "ok": False, "msg": str(e)[:120]}
        results.append(r)
        print(f"  -> {'OK' if r['ok'] else 'FAIL'} {r.get('检测', '')} "
              f"{r.get('msg', '')} 色差={r.get('色差', '-')} "
              f"接缝={r.get('接缝', '-')}", flush=True)
    print("\n==== 回归汇总 ====")
    n_ok = sum(1 for r in results if r["ok"])
    for r in results:
        print(f"{r['name']}: {'OK' if r['ok'] else 'FAIL'} "
              f"{r.get('检测', '')} {r.get('msg', '')} "
              f"色差={r.get('色差', '-')} 接缝={r.get('接缝', '-')}")
    print(f"{n_ok}/{len(results)} 通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
