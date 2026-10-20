# -*- coding: utf-8 -*-
"""wmremove —— 通用去水印工具包

快速上手
--------
    from wmremove import Options, Shape, process

    opts = Options(method="auto")
    shapes = [Shape("rect", (100, 200, 300, 120)),     # 矩形框选
              Shape("ellipse", (500, 300, 120, 90)),   # 椭圆框选
              Shape("polygon", [(10, 10), (200, 20), (150, 120)])]   # 多边形框选
    process("in.mp4", "out.mp4", shapes, opts)

命令行
------
    python main.py --help
    python main.py remove -i in.mp4 -o out.mp4 --box 100,200,300,120
    python main.py gui
    python main.py download "https://v.douyin.com/xxx/" -o downloads
"""

from .engine import Options, process, process_image, process_video, preview_frame
from .shapes import Shape, parse_shapes
from .media import probe, imread_u, imwrite_u, default_out_path

__version__ = "1.0.0"
__all__ = ["Options", "Shape", "parse_shapes", "process", "process_image",
           "process_video", "preview_frame", "probe", "imread_u", "imwrite_u",
           "default_out_path", "__version__"]
