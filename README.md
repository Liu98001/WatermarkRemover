# 通用去水印工具（视频 / 图片）

框选任意形状的水印区域 → 自动选择最优算法去除 → 周边颜色无痕融合。
支持：抖音 / 腾讯视频 / B站 / 微博等平台下载；豆包AI、即梦、可灵、通义等 AI
生成视频的半透明水印（含淡入淡出）；图片水印；批量处理。

> 请仅处理你拥有权利或已获得授权的内容。

---

## 一、安装

```bat
pip install -r requirements.txt
```

依赖：`numpy`、`opencv-python`、`Pillow`、`imageio-ffmpeg`（自带 ffmpeg，无需另装）、
`yt-dlp`（下载链接用）。Windows 下 tkinter 为 Python 自带。

## 二、快速上手

### 1) 图形界面（推荐，支持框选）

双击 `start_gui.bat`，或：

```bat
python main.py gui
```

* 工具条：**矩形 / 椭圆 / 多边形（逐点单击，双击闭合） / 涂抹（按住拖动）/ 选择**
* 视频：拖动滑杆选参考帧；「预览当前帧」看效果；满意后「处理并保存」
* 「自动检测」：选择平台（抖音/B站/小红书/优酷/快手/腾讯视频/爱奇艺/视频号…）
  即按该平台实测水印位置**自动框选**——内容分析命中时返回紧致精修框，
  未命中时用平台先验框兜底；也可选 四角/全扫描 自由扫描。结果可再手动微调
* 区域可增删、可保存/载入 JSON（配合批量复用）

### 2) 命令行

```bat
:: 图片：两个矩形 + 一个椭圆区域
python main.py remove -i 照片.jpg --box 100,200,300,120 --ellipse 500,300,80,60

:: 视频：多边形框选（自动选算法、自动做颜色融合）
python main.py remove -i in.mp4 --polygon 20,30;200,40;150,150

:: 视频：自动检测豆包AI水印并去除
python main.py remove -i in.mp4 --auto --auto-name 豆包AI

:: 从链接下载并去除（抖音分享文本直接整段粘贴即可）
python main.py remove -i "复制的抖音分享链接…" --box 20,20,300,120

:: 批量：目录内所有视频/图片套用同一组比例区域
python main.py batch -i 文件夹 --relative --box 0.01,0.02,0.3,0.12

:: 只检测，看看找到什么
python main.py detect -i in.mp4 --name 豆包AI
```

区域写法：`--box x,y,w,h`、`--ellipse cx,cy,rx,ry`、`--polygon x1,y1;x2,y2;…`、
`--brush x1,y1;x2,y2`（涂抹）、`--mask 掩码.png`（白=待处理）、`--regions-json 区域.json`；
加 `--relative` 表示 0~1 比例坐标。

### 3) 下载视频

```bat
python main.py download "https://v.douyin.com/xxxx/" -o downloads
python main.py download "https://xhslink.com/xxxx" -o downloads
python main.py download "https://b23.tv/xxxx" -o downloads
python main.py download "https://www.bilibili.com/video/BVxxxx" -o downloads
python main.py download "https://finder.video.qq.com/…mp4" --direct
python main.py download "https://v.qq.com/x/cover/…" --cookies edge
```

各平台策略（专用解析失败会自动降级到下一级）：

| 平台 | 专用解析（无需登录） | 兜底 |
|---|---|---|
| **抖音** | 分享短链 → 新鲜 ttwid Cookie → iesdouyin SSR 数据 → `/play/` **无水印流**，MP4 校验逐候选下载 | 无头浏览器 → 匿名新鲜 Cookie → 浏览器 Cookie → yt-dlp |
| **小红书** | xhslink 短链 → 笔记页 `__INITIAL_STATE__` → 视频流（h264 优先）/图集；需要登录的笔记会给出明确提示 | 浏览器 Cookie → yt-dlp（内置 XiaoHongShu 提取器） |
| **B站** | 官方 web API（b23.tv 短链解析、分 P 支持、buvid3 匿名 Cookie 预热）→ 渐进式 mp4，匿名最高约 480p | yt-dlp（匿名/浏览器 Cookie，可解锁更高清晰度） |
| **快手** | GraphQL `visionVideoDetail`（匿名设备 Cookie）→ 原始流 photoUrl | 浏览器 Cookie → yt-dlp |
| **微信视频号** | `finder.video.qq.com` 直链直接下载；分享页尝试提取媒体地址 | 无公开接口时给出 F12 抓包/另存指引 |
| **爱奇艺 / 优酷 / 腾讯视频** | yt-dlp 内置提取器（免费内容） | 浏览器 Cookie 解锁更高清晰度 |
| 通用直链 / 其他网站 | `.mp4` 等媒体 URL 直接下载 | yt-dlp（800+ 网站） |

注意：
* **VIP/付费/DRM 内容技术上无法下载**（爱奇艺/优酷/腾讯视频的版权保护流），
  工具会在识别到时给出明确提示；免费内容请提供登录态（`--cookies edge/chrome`）
  以获得更好效果。
* B站/小红书/快手作品若开启了登录限制，同样通过 `--cookies` 提供登录态。


---

## 三、算法说明（为什么没有违和感）

1. **紧致掩码**：先在框内检测水印笔画/字形本身，只修复这些像素，
   伤害面积最小（大块水印也只动它该动的地方）。
2. **时序分析（视频）**：
   * **层配对法**——AI 水印常在某段时间淡入/淡出。全片搜索"背景相同、水印不同"
     的帧对，两者之差就是**逐像素精确的水印层**，再对每帧估计系数 k_t 后减掉，
     精度只受编码噪声限制；
   * **仿射分离法**——静态半透明水印满足 `观测 = (1-α)·背景 + α·水印色`，
     用周边干净像素的时域均值做背景代理，逐块线性回归反解出背景；
   * **静态层替换法**——大面积水印（瓶身圆盘 logo、角落台标、用户名/徽标贴片）
     内部没有背景代理，仿射分离失效。改为：自动寻找"掩码内内容最稳定"的帧窗，
     **手持拍摄时先做 ECC 欧氏对齐**（排除水印区、只用周边背景估计运动）再取
     中值，得到清晰的水印参考 M 与清晰背景；用**结构保持填充**构造无水印目标 T
     ——填充轴自动选择（比较上下/左右参考带的一致性：贴边横条台标用横向、
     玻璃/柱面用竖向；带内深色结构自动补齐），T 的低频对齐到紧邻环（对齐量限幅，
     防亮盘/暗盘）；逐帧**运动补偿替换** `out = frame·(1-wg) + T_w·wg`（T 按每帧
     估计的全局运动对齐，手持晃动不留鬼影），权重与目标外扩 ~8%√面积覆盖软边缘；
     带**内容变化门控**（真实内容变化如液体/手自动放弃注入）与暗结构保护
     （只保护延伸到掩码外的真实结构，水印自身的不透明深色图案照常去除）；
   * 以上都不适用（背景乱动 / 水印完全不透明）时自动回退**空间修复**。
3. **空间修复**（图片 / 视频回退）：
   * 细笔画 → `Telea` 修复；光滑背景 → **谐波填充**（多尺度 Laplace 方程，
     边界严格连续）；有纹理背景 → **谐波打底 + 邻域纹理搬运**（从最吻合的几处
     干净区域平移细节纹理，按吻合度加权，边界淡入）。
4. **颜色一致性**（关键）：
   * 低频对齐——把区域内的明暗/色温用"周边谐波延展"校准，校正在边界处自动归零，
     不会产生台阶；
   * 电平项——区域内整体电平对齐到紧邻干净环（内部为 1、边界为 0 的斜坡加权）；
   * 自检保护——若对齐后"区域 vs 周边"色差反而变大或边界台阶变大，自动放弃；
   * 可选颗粒匹配——区域内比周围"太干净"时补同强度细颗粒。

### 质量指标（tests/selftest.py，合成真值自动验证）

| 用例 | 指标 |
|---|---|
| 图片·纯渐变背景 | 区域 PSNR ≈ **50 dB**，边界台阶 0.6 灰度 |
| 图片·文字水印（紧致/整框） | PSNR 30~33 dB，台阶 < 1.3 |
| 图片·圆形角标 | PSNR ≈ 37 dB |
| 视频·AI 淡入水印（层配对法） | 区域 PSNR > 36 dB（逐像素精确） |
| 视频·动态背景静态水印 | PSNR ≈ 40 dB |
| 视频·静止背景常驻水印 | PSNR ≈ 39 dB |

`边界台阶`：跨掩码边界的灰度差，<1 基本不可见，>2 开始显眼。
运行 `python main.py selftest` 可随时回归验证。

---

## 四、参数速查

| 参数 | 说明 |
|---|---|
| `--method auto` | auto/telea/ns/texture/synthesis/harmonic/clone/blur/pixelate/dark |
| `--no-tight` | 关闭紧致掩码（整框修复） |
| `--dilate 3 / --erode 2` | 掩码扩张/收缩 |
| `--feather 3` | 边缘羽化 |
| `--no-colorfit` | 关闭颜色一致性（不建议） |
| `--grain 0.5` | 补颗粒强度 |
| `--no-temporal` | 关闭时序分析（强制空间修复） |
| `--static-fill auto\|vertical\|horizontal\|texture\|harmonic` | 静态层替换的目标填充（默认 auto：按四条参考带一致性自动选轴——贴边横条台标自动用横向，玻璃/柱面用竖向） |
| `--gate-tau 45` | 静态层替换的内容变化门控阈值（灰度，越小越保守） |
| `--per-frame` | 强制逐帧修复（默认自动：背景不动整段只算一次，速度快） |
| `--crf 17` | 输出 H.264 质量（越小越好） |
| `--verify` | 处理后打印质量指标 |

---

## 五、文件结构

```
watermark_remover/            (原"去水印工具")
├─ main.py            统一入口 (python main.py [remove|detect|download|batch|gui|selftest])
├─ start_gui.bat      双击打开 GUI
├─ requirements.txt
├─ regions_808.json   示例: 预置区域文件(界面可载入)
├─ packaging/         打包资源(icon.ico / make_icon.py)
├─ wmremove/
│  ├─ shapes.py       区域形状(矩形/椭圆/多边形/涂抹/掩码)与羽化
│  ├─ media.py        ffmpeg 定位/视频探测/逐帧管道编码(音频直通)/图片读写
│  ├─ methods.py      修复算法(谐波/Telea/NS/竖向结构保持/纹理搬运/逐块合成/克隆/遮盖)
│  ├─ colorfit.py     颜色一致性(低频对齐/颗粒匹配/接缝评估)
│  ├─ temporal.py     时序分析(层配对法/仿射分离法/静态层替换)
│  ├─ detect.py       自动检测(平台位置库+内容分析)与紧致掩码
│  ├─ downloader.py   链接下载(yt-dlp/直链)
│  ├─ engine.py       调度引擎(图片/视频管线)
│  ├─ cli.py          命令行
│  └─ gui.py          框选界面
└─ tests/selftest.py  合成真值回归测试
```

---

## 六、打包成 exe

```bat
pip install pyinstaller
pyinstaller --noconfirm --clean --onedir --windowed --icon packaging/icon.ico ^
    --name WatermarkRemover --paths . ^
    --collect-all imageio_ffmpeg --collect-all yt_dlp main.py
```

产物为 `dist\WatermarkRemover\` 文件夹（内含 `WatermarkRemover.exe`）。
**onedir 模式启动最快**（热启动 <1s）；onefile 模式虽是单文件，但每次启动要解压
150MB，会慢很多，不推荐。不带参数启动时默认进入 GUI。
分发时把整个文件夹打包（zip）发给别人，或给 exe 建桌面快捷方式即可。
