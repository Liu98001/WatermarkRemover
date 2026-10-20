# -*- coding: utf-8 -*-
"""生成应用图标 packaging/icon.ico

设计: 蓝色渐变圆角方块 + 白色播放三角(视频) + 金色星光(魔法去除)
"""
import os

from PIL import Image, ImageDraw

here = os.path.dirname(os.path.abspath(__file__))
S = 512                                   # 高分辨率绘制, 缩到各尺寸抗锯齿
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))

# ---- 1) 对角渐变背景(裁进圆角方块) ----
grad = Image.new("RGBA", (S, S))
c1 = (59, 130, 246)                       # 亮蓝 #3b82f6
c2 = (23, 54, 138)                        # 深蓝 #17368a
px = grad.load()
for y in range(S):
    for x in range(S):
        t = (x + y) / (2 * S)
        px[x, y] = (int(c1[0] + (c2[0] - c1[0]) * t),
                    int(c1[1] + (c2[1] - c1[1]) * t),
                    int(c1[2] + (c2[2] - c1[2]) * t), 255)
mask = Image.new("L", (S, S), 0)
dm = ImageDraw.Draw(mask)
dm.rounded_rectangle((16, 16, S - 16, S - 16), radius=112, fill=255)

# ---- 2) 顶部高光(提升质感): 先合成进背景, 再整体按圆角裁剪贴入 ----
hl = Image.new("RGBA", (S, S), (0, 0, 0, 0))
dh = ImageDraw.Draw(hl)
dh.ellipse((-S * 0.35, -S * 0.55, S * 0.95, S * 0.35), fill=(255, 255, 255, 36))
grad.alpha_composite(hl)

img.paste(grad, (0, 0), mask)

d = ImageDraw.Draw(img)

# ---- 3) 白色播放三角(视频) ----
cx, cy = 216, 268
tri = [(cx - 52, cy - 86), (cx - 52, cy + 86), (cx + 92, cy)]
d.polygon(tri, fill=(255, 255, 255, 255))

# ---- 4) 金色四角星光(魔法去除) ----
def sparkle(x, y, r, color, ratio=0.22):
    k = r * ratio
    d.polygon([(x, y - r), (x + k, y - k), (x + r, y), (x + k, y + k),
               (x, y + r), (x - k, y + k), (x - r, y), (x - k, y - k)],
              fill=color)

sparkle(356, 128, 92, (255, 209, 77, 255))          # 主星光(金)
sparkle(356, 128, 34, (255, 255, 255, 235))         # 星光白芯
sparkle(150, 96, 40, (255, 255, 255, 210))          # 次星光
sparkle(396, 330, 34, (255, 209, 77, 200))          # 点缀

# ---- 5) 底部细线(媒体进度条意象) ----
d.rounded_rectangle((108, 388, S - 108, 412), radius=12,
                    fill=(255, 255, 255, 200))

# ---- 6) 内描边(细节) ----
d.rounded_rectangle((16, 16, S - 16, S - 16), radius=112,
                    outline=(255, 255, 255, 46), width=6)

out = os.path.join(here, "icon.ico")
img.save(out, sizes=[(256, 256), (128, 128), (64, 64), (48, 48),
                     (32, 32), (16, 16)])
png = os.path.join(here, "icon_preview.png")
img.resize((256, 256), Image.LANCZOS).save(png)
print("icon ->", out)
