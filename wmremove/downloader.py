# -*- coding: utf-8 -*-
"""视频链接下载(抖音 / 腾讯视频 / B站 / 微博 / 通用直链…)

* 用 yt-dlp 解析各平台页面; 微信视频号若无官方接口支持, 会给出明确指引:
  1) 直接粘贴 .mp4 直链(浏览器 F12 → 网络 → 找到 media 请求), 工具可直接下载;
  2) 或先保存到本地再用本工具处理。
* 支持从"分享文本"里自动抽取链接(例如抖音分享口令里夹带的 https://v.douyin.com/xxx)。
* 请仅处理你拥有权利或已获得授权的内容。
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import sys

from .media import find_ffmpeg


def _force_utf8_stdio() -> None:
    """Windows 控制台默认 GBK, 视频标题里的 emoji(⚡️ 等)一打印就
    UnicodeEncodeError —— 而此时视频往往已经下载完成, 却被当成下载
    失败。统一切到 UTF-8(不可编码字符替换), GUI 管道/重定向不受影响。
    """
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if s is None or not hasattr(s, "buffer"):
            continue
        try:
            enc = (getattr(s, "encoding", "") or "").lower()
            if enc.replace("-", "") not in ("utf8", "utf"):
                setattr(sys, name, io.TextIOWrapper(
                    s.buffer, encoding="utf-8", errors="replace",
                    line_buffering=True))
        except Exception:
            pass


_force_utf8_stdio()

URL_RE = re.compile(r"(https?://[^\s,，。;；\"'<>】」』\)]+)", re.I)

PLATFORMS = [
    ("douyin", r"douyin\.com|iesdouyin", "抖音"),
    ("vqq", r"v\.qq\.com", "腾讯视频"),
    ("channels", r"channels\.weixin|finder\.video\.qq\.com", "微信视频号"),
    ("bili", r"bilibili\.com|b23\.tv", "B站"),
    ("weibo", r"weibo\.com|weibo\.cn", "微博"),
    ("kuaishou", r"kuaishou\.com", "快手"),
    ("xiaohongshu", r"xiaohongshu\.com|xhslink", "小红书"),
    ("iqiyi", r"iqiyi\.com|iq\.com", "爱奇艺"),
    ("youku", r"youku\.com", "优酷"),
    ("tiktok", r"tiktok\.com", "TikTok"),
]


def extract_url(text: str) -> str | None:
    m = URL_RE.search(text or "")
    return m.group(1).rstrip(".,;") if m else None


def detect_platform(url: str) -> str:
    for key, pat, name in PLATFORMS:
        if re.search(pat, url or "", re.I):
            return name
    return "通用网站"


def _ydl_opts(out_dir: str, quality: str, cookies: str | None, info_only=False):
    from .media import IMAGE_EXT
    fmt = {
        "best": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
        "1080": "best[height<=1080]/bv*[height<=1080]+ba/b",
        "720": "best[height<=720]/bv*[height<=720]+ba/b",
        "480": "best[height<=480]/bv*[height<=480]+ba/b",
    }.get(quality, "bv*+ba/b")
    o = {
        "format": fmt,
        "outtmpl": os.path.join(out_dir, "%(title).80s.%(ext)s"),
        "merge_output_format": "mp4",
        "ffmpeg_location": find_ffmpeg(),
        "noplaylist": True,
        "quiet": not info_only,
        "no_warnings": True,
        "retries": 3,
        "socket_timeout": 25,
        "concurrent_fragment_downloads": 4,
        "progress_hooks": [_hook] if not info_only else [],
    }
    if cookies:
        if os.path.isfile(cookies):
            o["cookiefile"] = cookies
        else:                      # 例如 "edge" / "chrome"
            o["cookiesfrombrowser"] = (cookies,)
    return o


def _hook(d):
    if d.get("status") == "downloading":
        p = d.get("_percent_str", "").strip() or f"{d.get('downloaded_bytes', 0)}B"
        spd = d.get("_speed_str", "").strip()
        sys.stdout.write(f"\r    下载中 {p} {spd}   ")
        sys.stdout.flush()
    elif d.get("status") == "finished":
        sys.stdout.write("\r    下载完成，正在合成…            \n")


DOUYIN_MOBILE_UA = ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) "
                    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 "
                    "Mobile/15E148 Safari/604.1")
_douyin_ttwid: str | None = None      # 新鲜匿名 ttwid 缓存

DESKTOP_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36")


def _fresh_douyin_cookiefile(cache_dir: str | None = None) -> str | None:
    """生成"新鲜匿名 Cookie"(抖音现在要求即使是游客也要有新鲜 ttwid)。

    步骤: 访问抖音首页收集 Set-Cookie → 若无 ttwid 再向字节的
    ttwid 注册接口申请 → 写成 Netscape cookies.txt 交给 yt-dlp。
    """
    import http.cookiejar
    import json as _json
    import tempfile
    import urllib.request

    cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "wmremove")
    os.makedirs(cache_dir, exist_ok=True)
    global _douyin_ttwid
    path = os.path.join(cache_dir, "douyin_cookies.txt")
    jar = http.cookiejar.MozillaCookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    opener.addheaders = [("User-Agent", DESKTOP_UA),
                         ("Accept", "text/html,application/xhtml+xml,*/*;q=0.8"),
                         ("Accept-Language", "zh-CN,zh;q=0.9")]
    try:                       # 1) 首页: 拿 __ac_nonce / 基础 cookie
        opener.open("https://www.douyin.com/?recommend=1", timeout=20).read(2048)
    except Exception:
        pass
    have = {c.name for c in jar}
    if "ttwid" not in have:    # 2) ttwid 注册接口
        payload = _json.dumps({
            "region": "cn", "aid": 1768, "needFid": False,
            "service": "www.ixigua.com",
            "migrate_info": {"ticket": "", "source": "node"},
            "cbUrlProtocol": "https", "union": True,
        }).encode()
        try:
            req = urllib.request.Request(
                "https://ttwid.bytedance.com/ttwid/union/register/",
                data=payload, headers={"Content-Type": "application/json",
                                       "User-Agent": DESKTOP_UA})
            resp = urllib.request.urlopen(req, timeout=20)
            for sc in resp.headers.get_all("Set-Cookie") or []:
                if sc.lower().startswith("ttwid="):
                    val = sc.split(";", 1)[0].split("=", 1)[1]
                    ck = http.cookiejar.Cookie(
                        version=0, name="ttwid", value=val, port=None,
                        port_specified=False, domain=".douyin.com",
                        domain_specified=True, domain_initial_dot=True, path="/",
                        path_specified=True, secure=True, expires=None,
                        discard=True, comment=None, comment_url=None, rest={})
                    jar.set_cookie(ck)
        except Exception:
            pass
    if not any(c.name in ("ttwid", "__ac_nonce") for c in jar):
        return None
    for c in jar:
        if c.name == "ttwid":
            _douyin_ttwid = c.value           # 缓存给分享页 SSR 请求使用
    try:
        jar.save(path, ignore_discard=True, ignore_expires=True)
        return path
    except Exception:
        return None


def _douyin_resolve_id(url: str) -> str | None:
    """解析分享短链, 取出视频/图集的 aweme id"""
    import urllib.request
    m = re.search(r"/(?:video|note)/(\d+)", url) or re.search(r"modal_id=(\d+)", url)
    if m:
        return m.group(1)
    if not re.search(r"douyin\.com", url, re.I):
        return None
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) "
                      "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 "
                      "Mobile/15E148 Safari/604.1"})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            final = r.geturl()
            body = r.read(65536).decode("utf-8", "ignore")
    except Exception:
        return None
    m = (re.search(r"/(?:video|note)/(\d+)", final) or re.search(r"modal_id=(\d+)", final)
         or re.search(r"/(?:video|note)/(\d+)", body) or re.search(r"item_ids?=(\d+)", body))
    return m.group(1) if m else None


def _douyin_fetch_item(aweme_id: str, ttwid: str | None = None,
                       retries: int = 2) -> dict:
    """从 iesdouyin 分享页 SSR 数据提取条目(与 f2/TikTokDownload 同源做法)。

    关键点: 请求必须携带新鲜的 ttwid Cookie, 服务端才会渲染出
    `_ROUTER_DATA.loaderData['video_(id)/page'].videoInfoRes.item_list[0]`。
    """
    import json as _json
    import time
    import urllib.request
    ua = ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) "
          "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 "
          "Safari/604.1")
    if not ttwid:
        _fresh_douyin_cookiefile()              # 顺便生成
        ttwid = _douyin_ttwid
    last = ""
    for attempt in range(max(1, retries)):
        try:
            req = urllib.request.Request(
                f"https://www.iesdouyin.com/share/video/{aweme_id}/",
                headers={"User-Agent": ua, "Accept-Language": "zh-CN,zh;q=0.9",
                         "Cookie": f"ttwid={ttwid}" if ttwid else ""})
            with urllib.request.urlopen(req, timeout=25) as r:
                html = r.read().decode("utf-8", "ignore")
            m = re.search(r"window\._ROUTER_DATA\s*=\s*(\{.*?\})\s*</script>",
                          html, re.S)
            if not m:
                last = "分享页没有 _ROUTER_DATA(被风控)"
                time.sleep(1.0)
                continue
            data = _json.loads(m.group(1))
            loader = data.get("loaderData", {})
            page = next((v for k, v in loader.items() if k.endswith("/page")), None)
            if not page:
                last = "分享页数据结构变化"
                time.sleep(1.0)
                continue
            items = (page.get("videoInfoRes") or {}).get("item_list") or []
            if not items:
                last = "分享页未包含条目(视频可能已被删除/私密, 或 Cookie 不新鲜)"
                time.sleep(1.0)
                continue
            return items[0]
        except Exception as e:                          # noqa: BLE001
            last = str(e)[:120]
            time.sleep(1.0)
    raise RuntimeError(f"获取抖音条目失败: {last}")


def _looks_like_mp4(data: bytes) -> bool:
    """严谨校验: MP4 文件头 box size(4B) + 'ftyp'(4B)"""
    return len(data) > 12 and data[4:8] == b"ftyp"


def douyin_direct_download(url: str, out_dir: str = "downloads",
                           no_watermark: bool = True) -> str:
    """抖音专用下载(与 f2 / TikTokDownload 同源思路):

      1) 分享短链(或含 ID 的链接) → aweme_id;
      2) 携带新鲜 ttwid Cookie 请求 iesdouyin 分享页, 服务端 SSR 出
         `_ROUTER_DATA.videoInfoRes.item_list[0]`(标题/作者/播放地址);
      3) `play_addr.url_list` 里 `/playwm/` 替换成 `/play/` 即无水印流;
      4) 逐候选下载(自动跟随 302 到 CDN), MP4 魔数校验, 失败换下一个。
    """
    import time
    aweme = _douyin_resolve_id(url)
    if not aweme:
        raise RuntimeError("无法从链接解析出视频 ID(请确认是抖音分享链接)")
    print(f"[下载] 抖音视频 ID: {aweme}")
    item = _douyin_fetch_item(aweme, ttwid=_douyin_ttwid)
    desc = (item.get("desc") or "douyin").strip()
    author = ((item.get("author") or {}).get("nickname") or "").strip()
    safe = re.sub(r'[\\/:*?"<>|\r\n]+', "_", desc[:60]) or "douyin"
    v = item.get("video") or {}

    if item.get("images"):                     # 图集帖
        os.makedirs(out_dir, exist_ok=True)
        paths = []
        for i, im in enumerate(item["images"][:10], 1):
            urls = list(im.get("url_list") or [])
            if not urls:
                continue
            p = os.path.join(out_dir, f"{safe}_{i:02d}.jpg")
            if _douyin_download_to(urls[0], p, DOUYIN_MOBILE_UA):
                paths.append(p)
        if not paths:
            raise RuntimeError("图集下载失败")
        print(f"[下载] 图集 {len(paths)} 张 -> {out_dir}")
        return paths[0]

    addr = (v.get("play_addr") or v.get("play_addr_lowbr")
            or v.get("download_addr") or {})
    mirrors = list(addr.get("url_list") or [])
    if not mirrors:
        raise RuntimeError("未找到播放地址(视频可能已被删除/私密)")
    # 候选顺序: 无水印流(playwm→play) 的每个镜像, 再退回带水印原始流
    cands: list[str] = []
    for u in mirrors:
        if "/playwm/" in u:
            cands.append(u.replace("/playwm/", "/play/"))
    cands += mirrors
    seen: set[str] = set()
    cands = [u for u in cands if not (u in seen or seen.add(u))]
    print(f"[下载] 标题: {desc[:50]}" + (f"  作者: {author}" if author else ""))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, safe + ".mp4")
    last_err = ""
    for i, u in enumerate(cands):
        tag = "无水印流" if "/play/" in u else "带水印流"
        for _retry in range(2):
            if _douyin_download_to(u, path, DOUYIN_MOBILE_UA,
                                   referer="https://www.douyin.com/"):
                head = open(path, "rb").read(16)
                if not _looks_like_mp4(head):
                    last_err = f"候选{i + 1}返回的不是 MP4"
                    print(f"    {tag}: {last_err}, 换下一个 …")
                    break
                print(f"[下载] 完成({tag}, {os.path.getsize(path) / 1e6:.1f}MB) "
                      f"-> {path}")
                return path
            last_err = "下载失败或内容过小"
            time.sleep(1.0)
        print(f"    {tag}候选{i + 1}: {last_err}, 尝试下一候选 …")
    raise RuntimeError(f"视频流下载失败: {last_err}")


def _douyin_download_to(url: str, path: str, ua: str,
                        referer: str | None = None) -> bool:
    """流式下载到 path(自动跟随 302 到 CDN), 成功且 >50KB 返回 True"""
    return _http_download_to(url, path, ua, referer=referer)


def _http_download_to(url: str, path: str, ua: str,
                      referer: str | None = None) -> bool:
    """流式下载到 path(自动跟随 302 到 CDN), 成功且 >50KB 返回 True"""
    import urllib.request
    headers = {"User-Agent": ua, "Accept": "*/*"}
    if referer:
        headers["Referer"] = referer
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            while True:
                chunk = r.read(1 << 17)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if total:
                    sys.stdout.write(f"\r    {got / total * 100:5.1f}% "
                                     f"({got / 1e6:.1f}MB / {total / 1e6:.1f}MB)")
                    sys.stdout.flush()
        print()
        return os.path.isfile(path) and os.path.getsize(path) > 50 * 1024
    except Exception as e:                                 # noqa: BLE001
        print(f"\n    下载错误: {str(e)[:80]}")
        return False


def _safe_name(text: str, limit: int = 60) -> str:
    return re.sub(r'[\\/:*?"<>|\r\n]+', "_", (text or "").strip()[:limit]) \
        or "video"


# =============================================================== 小红书专用
XHS_UA = ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) "
          "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 "
          "Mobile/15E148 Safari/604.1")


def _follow_redirect(url: str, ua: str, timeout: int = 25) -> str:
    """跟随短链 302/JS 跳转, 返回最终 URL(保留 xsec_token 等查询参数)"""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.geturl()


def xhs_resolve_note(url: str) -> tuple[str, str]:
    """解析小红书分享链接 → (笔记页 URL, 笔记 id)

    兼容三种形态:
      * www.xiaohongshu.com/explore/<id>?xsec_token=…  (标准分享)
      * www.xiaohongshu.com/discovery/item/<id>?…      (app 分享)
      * xhslink.com|xhslink.cn 短链 → 302 到登录页时, 从登录页
        redirectPath 参数里 URL 解码出真实笔记地址(含 xsec_token)。
    """
    from urllib.parse import unquote
    cand = [url]
    if not re.search(r"xiaohongshu\.com", url, re.I):
        try:
            cand.append(_follow_redirect(url, XHS_UA))
        except Exception:                                  # noqa: BLE001
            pass
    # 登录跳转页: redirectPath 里藏着真实地址, 解码后作为候选
    extra = []
    for u in list(cand):
        m = re.search(r"[?&]redirectPath=([^&]+)", u)
        if m:
            extra.append(unquote(m.group(1)))
    cand += extra

    for u in cand:
        m = (re.search(r"/(?:explore|discovery/item)/([0-9a-f]{24})", u, re.I)
             or re.search(r"note_id=([0-9a-f]{24})", u, re.I))
        if m:
            page = u.replace(" ", "%20")       # 分享参数里可能带空格
            if page.startswith("http://"):
                page = "https://" + page[7:]   # 统一走 https
            return page, m.group(1)
    raise RuntimeError("无法从链接解析出小红书笔记 ID(请确认是分享链接)")


def _xhs_initial_state(html: str) -> dict:
    """从小红书页面提取 __INITIAL_STATE__ JSON(容错 undefined/null)"""
    m = re.search(r"window\.__INITIAL_STATE__\s*=\s*(\{.*?\})\s*</script>",
                  html, re.S)
    if not m:
        m = re.search(r"window\.__INITIAL_STATE__\s*=\s*(\{.*?\});?\s*$",
                      html, re.S | re.M)
    if not m:
        raise RuntimeError("页面未包含笔记数据(笔记可能需要登录/已被删除)")
    raw = re.sub(r"\bundefined\b", "null", m.group(1))
    import json as _json
    return _json.loads(raw)


def xhs_fetch_note(final_url: str, note_id: str) -> dict:
    """抓取笔记页, 返回 note 对象(视频流/图集/标题/作者)

    兼容两种页面结构:
      /explore/<id>:     state.note.noteDetailMap.<id>.note      (snake_case 字段)
      /discovery/item/:  state.noteData.data.noteData            (camelCase 字段)
    """
    import urllib.request
    req = urllib.request.Request(final_url, headers={
        "User-Agent": XHS_UA, "Accept-Language": "zh-CN,zh;q=0.9",
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8"})
    with urllib.request.urlopen(req, timeout=25) as r:
        html = r.read().decode("utf-8", "ignore")
    state = _xhs_initial_state(html)
    detail = ((state.get("note") or {}).get("noteDetailMap") or {})
    note = (detail.get(note_id) or next(iter(detail.values()), {})
            ).get("note") or {}
    if not note:
        note = ((state.get("noteData") or {}).get("data") or {}
                ).get("noteData") or {}
    if not note:
        raise RuntimeError("页面数据里没有笔记内容(可能需要登录 Cookie)")
    return note


def xhs_direct_download(url: str, out_dir: str = "downloads") -> str:
    """小红书专用下载(仿抖音):
      1) xhslink 短链 → 笔记页(保留 xsec_token);
      2) 解析 __INITIAL_STATE__ → 视频流(优先 h264 原始流)/图集;
      3) 候选流逐个下载 + MP4 魔数校验。
    图集笔记保存全部图片, 返回第一张。"""
    final, note_id = xhs_resolve_note(url)
    note = xhs_fetch_note(final, note_id)
    # 兼容 snake_case(explore 页) / camelCase(discovery 页) 两种字段风格
    def g(obj, *names):
        for n in names:
            if obj.get(n):
                return obj[n]
        return None

    title = _safe_name(g(note, "title", "desc") or "xhs")
    author = ((g(note, "user") or {}).get("nickname") or
              ((g(note, "user") or {}).get("nickName") or "")).strip()
    print(f"[下载] 标题: {title}" + (f"  作者: {author}" if author else ""))
    os.makedirs(out_dir, exist_ok=True)

    image_list = g(note, "image_list", "imageList") or []
    media = ((g(note, "video") or {}).get("media") or {})
    stream = media.get("stream") or {}
    cands = []
    for kind in ("h264", "h265", "av1", "h266"):
        for entry in stream.get(kind) or []:
            mu = g(entry, "master_url", "masterUrl")
            if mu:
                cands.append(mu)
            cands += [u for u in g(entry, "backup_urls", "backupUrls") or []]
    seen: set[str] = set()
    cands = [u for u in cands if not (u in seen or seen.add(u))]

    # 视频笔记的 imageList 里也有封面图: 有视频流时优先按视频处理
    if cands:
        path = os.path.join(out_dir, title + ".mp4")
        last = ""
        for i, u in enumerate(cands):
            if _http_download_to(u, path, XHS_UA,
                                 referer="https://www.xiaohongshu.com/"):
                head = open(path, "rb").read(16)
                if _looks_like_mp4(head):
                    print(f"[下载] 完成({os.path.getsize(path) / 1e6:.1f}MB) -> {path}")
                    return path
                last = f"候选{i + 1}返回的不是 MP4"
            print(f"    候选{i + 1}: {last or '下载失败'}, 换下一个 …")
        raise RuntimeError(f"视频流下载失败: {last}")

    if image_list:                                  # 图集笔记
        paths = []
        for i, im in enumerate(image_list[:18], 1):
            urls = ([g(im, "url_default", "urlDefault")] +
                    [g(x, "image_url", "imageUrl") or x.get("url")
                     for x in (g(im, "info_list", "infoList") or [])])
            urls = [u for u in urls if u]
            for u in urls:
                p = os.path.join(out_dir, f"{title}_{i:02d}.jpg")
                if _http_download_to(u, p, XHS_UA,
                                     referer="https://www.xiaohongshu.com/"):
                    paths.append(p)
                    break
        if not paths:
            raise RuntimeError("图集下载失败")
        print(f"[下载] 图集 {len(paths)} 张 -> {out_dir}")
        return paths[0]
    raise RuntimeError("未找到视频流(该笔记可能不是视频或需要登录)")


# =============================================================== 视频号专用
def channels_direct_download(url: str, out_dir: str = "downloads") -> str:
    """视频号: 直链(finder.video.qq.com)直接下载; 分享页尝试提取媒体地址。

    视频号接口需登录态且签名加密, 匿名解析的成功率有限;
    失败时给出明确的抓包/另存指引(见 download 的提示)。
    """
    if re.search(r"finder\.video\.qq\.com", url, re.I):
        path = os.path.join(out_dir, _safe_name("视频号_" +
                                                str(abs(hash(url)) % 99999))
                            + ".mp4")
        if _http_download_to(url, path, DESKTOP_UA,
                             referer="https://channels.weixin.qq.com/"):
            head = open(path, "rb").read(16)
            if _looks_like_mp4(head):
                print(f"[下载] 完成({os.path.getsize(path) / 1e6:.1f}MB) -> {path}")
                return path
        raise RuntimeError("视频号直链下载失败(直链可能已过期, 请重新抓包)")

    import urllib.request
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": DESKTOP_UA, "Accept-Language": "zh-CN,zh;q=0.9"})
        with urllib.request.urlopen(req, timeout=25) as r:
            html = r.read().decode("utf-8", "ignore")
    except Exception as e:                                 # noqa: BLE001
        raise RuntimeError(f"分享页打开失败: {str(e)[:80]}")
    m = re.search(r"https?://[^\"'\\s]+finder\.video\.qq\.com[^\"'\\s]+"
                  r"|https?://[^\"'\\s]+\.mp4[^\"'\\s]*", html)
    if not m:
        raise RuntimeError("分享页里没有可直接提取的视频地址")
    return channels_direct_download(m.group(0).replace("\\u0026", "&"),
                                    out_dir)


# =============================================================== 快手专用
KUAISHOU_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36")

_KS_QUERY = ("query visionVideoDetail($photoId: String, $page: String, "
             "$webPage: String) { visionVideoDetail(photoId: $photoId, "
             "page: $page, webPage: $webPage) { status type photo { id "
             "caption width height photoUrl } } }")


def kuaishou_resolve(url: str) -> tuple[str, str]:
    """快手分享链接(短链 v.kuaishou.com / 长链) → (页面 URL, photoId)"""
    if re.search(r"v\.kuaishou\.com", url, re.I):
        url = _follow_redirect(url, KUAISHOU_UA)
    m = (re.search(r"/(?:short-video|photo)/([0-9a-zA-Z]+)", url)
         or re.search(r"modal_id=([0-9a-zA-Z]+)", url)
         or re.search(r"photoId=([0-9a-zA-Z]+)", url))
    if not m:
        raise RuntimeError("无法从链接解析出快手作品 ID(请确认是分享链接)")
    page = "vision" if "short-video" in url else "photoDetail"
    return url, m.group(1)


def kuaishou_direct_download(url: str, out_dir: str = "downloads") -> str:
    """快手专用下载(仿抖音): GraphQL visionVideoDetail 取 photoUrl(原始流)。

    需要匿名设备 Cookie(did=UUID) 与 kpf/kpn 头; 失败时回退 yt-dlp。
    """
    import http.cookiejar
    import json as _json
    import urllib.request
    import uuid

    page_url, photo_id = kuaishou_resolve(url)
    jar = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    did = str(uuid.uuid4())
    for c in (f"did=web_{did}", "kpf=PC_WEB", "kpn=KUAISHOU_VISION"):
        ck = http.cookiejar.Cookie(
            version=0, name=c.split("=")[0], value=c.split("=", 1)[1],
            port=None, port_specified=False, domain=".kuaishou.com",
            domain_specified=True, domain_initial_dot=True, path="/",
            path_specified=True, secure=False, expires=None, discard=True,
            comment=None, comment_url=None, rest={})
        jar.set_cookie(ck)
    payload = _json.dumps({
        "operationName": "visionVideoDetail",
        "variables": {"photoId": photo_id, "page": "vision",
                      "webPage": page_url},
        "query": _KS_QUERY,
    }).encode()
    req = urllib.request.Request(
        "https://www.kuaishou.com/graphql", data=payload, headers={
            "User-Agent": KUAISHOU_UA, "Content-Type": "application/json",
            "Accept": "*/*", "Referer": page_url, "Origin": "https://www.kuaishou.com"})
    with op.open(req, timeout=25) as r:
        data = _json.loads(r.read().decode("utf-8", "ignore"))
    detail = (data.get("data") or {}).get("visionVideoDetail") or {}
    photo = detail.get("photo") or {}
    photo_url = photo.get("photoUrl")
    if detail.get("status") != 1 or not photo_url:
        raise RuntimeError("作品数据不可用(可能需要登录/已被删除/地区限制)")
    title = _safe_name(photo.get("caption") or "kuaishou")
    print(f"[下载] 标题: {title}")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, title + ".mp4")
    if _http_download_to(photo_url, path, KUAISHOU_UA,
                         referer="https://www.kuaishou.com/"):
        head = open(path, "rb").read(16)
        if _looks_like_mp4(head):
            print(f"[下载] 完成({os.path.getsize(path) / 1e6:.1f}MB) -> {path}")
            return path
        raise RuntimeError("下载内容不是 MP4")
    raise RuntimeError("视频流下载失败")


# =============================================================== B站辅助
def bili_resolve(url: str) -> str | None:
    """b23.tv 短链 → B站规范视频页 URL(其余原样返回)"""
    if not re.search(r"b23\.tv", url, re.I):
        return url
    try:
        final = _follow_redirect(url, DESKTOP_UA)
    except Exception:                                      # noqa: BLE001
        return url
    m = re.search(r"(BV[0-9A-Za-z]{10})", final)
    if m:
        return f"https://www.bilibili.com/video/{m.group(1)}"
    return final


_bili_jar = None


def _bili_opener():
    """带匿名 Cookie 的 opener: 先访问首页收集 buvid3, 规避 412 风控"""
    global _bili_jar
    import http.cookiejar
    import urllib.request
    if _bili_jar is None:
        _bili_jar = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(_bili_jar))
    if not any(c.name == "buvid3" for c in _bili_jar):
        try:
            req = urllib.request.Request("https://www.bilibili.com/",
                                         headers={"User-Agent": DESKTOP_UA})
            op.open(req, timeout=15).read(2048)
        except Exception:                                  # noqa: BLE001
            pass
    return op


def _bili_get_json(api_url: str) -> dict:
    import json as _json
    import urllib.request
    req = urllib.request.Request(api_url, headers={
        "User-Agent": DESKTOP_UA, "Referer": "https://www.bilibili.com/",
        "Accept": "application/json"})
    with _bili_opener().open(req, timeout=25) as r:
        data = _json.loads(r.read().decode("utf-8", "ignore"))
    if data.get("code") != 0:
        raise RuntimeError(f"B站接口返回错误: {data.get('message', data.get('code'))}")
    return data.get("data") or {}


def bili_direct_download(url: str, out_dir: str = "downloads") -> str:
    """B站专用下载(仿抖音): 官方 web API 匿名取视频流。

    1) 解析 BV 号(含 b23.tv 短链、?p= 分P);
    2) view API 拿标题与 cid → playurl(platform=html5) 拿渐进式 mp4;
    3) 下载 + MP4 魔数校验。匿名清晰度最高约 480p;
      需要更高清晰度时可给 yt-dlp 提供 Cookie(回退路径自动尝试)。
    """
    u = bili_resolve(url) or url
    m = re.search(r"(BV[0-9A-Za-z]{10})", u, re.I)
    if not m:
        raise RuntimeError("无法从链接解析出 B站 BV 号")
    bvid = m.group(1)
    pm = re.search(r"[?&]p=(\d+)", u)
    page = int(pm.group(1)) if pm else 1

    view = _bili_get_json(
        f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}")
    title = _safe_name(view.get("title") or bvid, 70)
    pages = view.get("pages") or []
    if not pages:
        raise RuntimeError("视频信息里没有分 P(可能已被删除)")
    page_info = pages[min(page, len(pages)) - 1]
    cid = page_info.get("cid")
    if len(pages) > 1:
        title += f"_P{page_info.get('page', page)}"
    print(f"[下载] 标题: {title}")
    play = _bili_get_json(
        f"https://api.bilibili.com/x/player/playurl?bvid={bvid}&cid={cid}"
        f"&qn=64&platform=html5&high_quality=1")
    durl = play.get("durl") or []
    if not durl:
        raise RuntimeError("播放接口没有返回视频流(匿名清晰度受限或被风控)")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, title + ".mp4")
    for i, seg in enumerate(durl):
        u2 = seg.get("url") or (seg.get("backup_url") or [""])[0]
        if not u2:
            continue
        if _http_download_to(u2, path, DESKTOP_UA,
                             referer="https://www.bilibili.com/"):
            head = open(path, "rb").read(16)
            if _looks_like_mp4(head):
                print(f"[下载] 完成({os.path.getsize(path) / 1e6:.1f}MB, "
                      f"{play.get('quality')}清晰度) -> {path}")
                return path
            print(f"    段{i + 1}返回的不是 MP4, 换备用地址 …")
    raise RuntimeError("视频流下载失败")


def _douyin_browser_download(url: str, out_dir: str = "downloads") -> str:
    """用无头浏览器(render playwright)打开分享页, 拦截真实视频流下载。

    这是抖音风控收紧后最稳的方式: 真浏览器环境自动通过验证,
    并能拿到预签名的视频直链(无需登录)。
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise RuntimeError(
            "需要无头浏览器支持: 请运行  pip install playwright  "
            "&&  python -m playwright install chromium")
    aweme = _douyin_resolve_id(url)
    share = (f"https://www.iesdouyin.com/share/video/{aweme}/" if aweme
             else extract_url(url) or url)
    ua = ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) "
          "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 "
          "Safari/604.1")
    hits: list[str] = []
    pages = [share] + ([f"https://www.douyin.com/video/{aweme}"] if aweme else [])
    with sync_playwright() as p:
        # 优先用系统已装的 Edge/Chrome(无需下载 chromium), 再退回内置 chromium
        launches: list[tuple[str, object]] = []
        auto = ["--autoplay-policy=no-user-gesture-required",
                "--mute-audio"]
        for channel in ("msedge", "chrome"):
            try:
                b = p.chromium.launch(channel=channel, headless=True)
                b.close()
                launches.append((f"系统{channel}无头",
                                 lambda ch=channel: p.chromium.launch(
                                     channel=ch, headless=True, args=auto)))
                launches.append((f"系统{channel}有头",
                                 lambda ch=channel: p.chromium.launch(
                                     channel=ch, headless=False, args=auto)))
            except Exception as e:                           # noqa: BLE001
                print(f"    系统浏览器 {channel} 不可用: {str(e)[:70]}")
        launches.append(("内置chromium",
                         lambda: p.chromium.launch(headless=True, args=auto)))
        launches.append(("内置chromium有头",
                         lambda: p.chromium.launch(headless=False, args=auto)))
        errs = []
        for label, launch in launches:
            try:
                browser = launch()
            except Exception as e:                           # noqa: BLE001
                errs.append(f"{label}: {str(e)[:60]}")
                continue
            ctx = browser.new_context(user_agent=ua,
                                      viewport={"width": 390, "height": 844},
                                      locale="zh-CN")
            page = ctx.new_page()

            def on_response(resp):
                u = resp.url
                if ("aweme/v1/play" in u or "douyinvod" in u
                        or ("/play/" in u and "douyin" in u)):
                    if u not in hits:
                        hits.append(u)

            page.on("response", on_response)
            page.on("request", lambda r: on_response(r))
            for target in pages:
                try:
                    page.goto(target, wait_until="domcontentloaded", timeout=60000)
                    page.wait_for_timeout(6000)
                    if not hits:             # 直接读 <video> 元素的 src
                        try:
                            srcs = page.eval_on_selector_all(
                                "video",
                                "els => els.map(e => e.src).filter(Boolean)")
                            for s in srcs:
                                if s not in hits:
                                    hits.append(s)
                        except Exception:
                            pass
                    if not hits:             # 尝试点播放按钮
                        for sel in ("video", ".xgplayer-play",
                                    "[aria-label='Play']", ".play-btn"):
                            el = page.query_selector(sel)
                            if el:
                                try:
                                    el.click(timeout=2500)
                                except Exception:
                                    pass
                                break
                        page.wait_for_timeout(6000)
                except Exception as e:                       # noqa: BLE001
                    errs.append(f"{label}@{target[:40]}: {str(e)[:60]}")
                if hits:
                    break
            # 在浏览器上下文内下载(自带 Cookie/签名), 优先无水印流
            if hits:
                item = None
                try:
                    item = _douyin_fetch_item(aweme) if aweme else None
                except Exception:
                    item = None
                desc = ((item or {}).get("desc") or "douyin").strip()[:60]
                safe = re.sub(r'[\\/:*?"<>|\r\n]+', "_", desc) or "douyin"
                os.makedirs(out_dir, exist_ok=True)
                path = os.path.join(out_dir, safe + ".mp4")
                cands = [hits[-1]]
                if "playwm" in hits[-1]:
                    cands.insert(0, hits[-1].replace("/playwm/", "/play/"))
                print(f"[下载] 标题: {desc}")
                for i, u in enumerate(cands):
                    try:
                        r = ctx.request.get(
                            u, headers={"Referer": "https://www.douyin.com/"})
                        body = r.body()
                        if len(body) > 50 * 1024:
                            with open(path, "wb") as f:
                                f.write(body)
                            if i > 0:
                                print("    (无水印流不可用, 已下载带水印流)")
                            print(f"[下载] 完成 {len(body) / 1e6:.1f}MB -> {path}")
                            browser.close()
                            return path
                        print(f"    流{i + 1}过小({len(body)}B), 换下一个 …")
                    except Exception as e:                   # noqa: BLE001
                        print(f"    流{i + 1}下载失败: {str(e)[:70]}")
                browser.close()
                continue
            browser.close()
    if not hits:
        detail = "; ".join(errs[-3:]) if errs else "所有浏览器均未捕获视频流"
        raise RuntimeError(f"浏览器方案未捕获到视频流({detail})")
    raise RuntimeError("未能下载视频流(所有候选均失败)")


def _browser_cookie_candidates() -> list[str]:
    """按本机实际安装情况给出浏览器 Cookie 候选(edge → chrome → firefox)"""
    out = []
    local = os.environ.get("LOCALAPPDATA", "")
    appdata = os.environ.get("APPDATA", "")
    checks = [
        ("edge", [os.path.join(local, r"Microsoft\Edge\User Data\Default\Network\Cookies"),
                  os.path.join(local, r"Microsoft\Edge\User Data\Default\Cookies")]),
        ("chrome", [os.path.join(local, r"Google\Chrome\User Data\Default\Network\Cookies"),
                    os.path.join(local, r"Google\Chrome\User Data\Default\Cookies")]),
        ("firefox", list(_firefox_cookie_files(appdata))),
    ]
    for name, paths in checks:
        if any(os.path.isfile(p) for p in paths):
            out.append(name)
    return out


def _firefox_cookie_files(appdata: str):
    base = os.path.join(appdata, "Mozilla", "Firefox", "Profiles")
    if os.path.isdir(base):
        for prof in os.listdir(base):
            p = os.path.join(base, prof, "cookies.sqlite")
            if os.path.isfile(p):
                yield p


def _looks_like_media_url(url: str) -> bool:
    """URL 指向媒体文件(.mp4 等) —— 可尝试直链下载"""
    return bool(re.search(r"\.(mp4|m4v|mov|webm|mkv|flv|avi|ts)([?#]|$)",
                          url or "", re.I))


def download(url: str, out_dir: str = "downloads", quality: str = "best",
             cookies: str | None = None,
             platform: str | None = None) -> str:
    """下载视频, 返回本地文件路径。失败时抛出带指引的 RuntimeError。

    platform: 指定平台(跳过自动识别), 如 "抖音"/"B站"/"小红书"/"快手"/
              "微信视频号"/"爱奇艺"/"优酷"/"腾讯视频"/"直链"; None=自动识别。
    """
    os.makedirs(out_dir, exist_ok=True)
    url = extract_url(url) or url
    platform = platform or detect_platform(url)
    if platform == "直链":
        return download_direct(url, out_dir)
    print(f"[下载] 平台: {platform}")
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("缺少 yt-dlp: 请先运行  pip install yt-dlp")

    # 直链(.mp4 等)先走直链下载: 视频号抓包直链、无需登录的 CDN 直链都适用
    if _looks_like_media_url(url) and platform != "微信视频号":
        try:
            return download_direct(url, out_dir)
        except Exception as e:                               # noqa: BLE001
            print(f"    直链下载失败: {str(e)[:90]}, 改用后续方案 …")

    attempts = [(cookies, "指定 Cookie")] if cookies else [(None, "匿名方式")]
    bili_api_fallback = False
    if platform == "抖音" and not cookies:
        try:
            print("[下载] 使用抖音专用解析(无需登录) …")
            return douyin_direct_download(url, out_dir)
        except Exception as e:                               # noqa: BLE001
            last_err_direct = str(e).strip().splitlines()[-1] if str(e) else str(e)
            print(f"    专用解析失败: {last_err_direct[:90]}")
            try:
                print("[下载] 尝试无头浏览器方案 …")
                return _douyin_browser_download(url, out_dir)
            except Exception as e2:                          # noqa: BLE001
                print(f"    无头浏览器失败: {str(e2)[:90]}")
        cf = _fresh_douyin_cookiefile()
        if cf:
            attempts.insert(0, (cf, "匿名新鲜 Cookie"))
        for b in _browser_cookie_candidates():
            attempts.append((b, f"浏览器 {b}"))
    elif platform == "小红书" and not cookies:
        try:
            print("[下载] 使用小红书专用解析(无需登录) …")
            return xhs_direct_download(url, out_dir)
        except Exception as e:                               # noqa: BLE001
            print(f"    专用解析失败: {str(e).strip().splitlines()[-1][:90]}")
        for b in _browser_cookie_candidates():     # 登录态可解锁更多笔记
            attempts.append((b, f"浏览器 {b}"))
    elif platform == "微信视频号":
        try:
            print("[下载] 使用视频号专用解析 …")
            return channels_direct_download(url, out_dir)
        except Exception as e:                               # noqa: BLE001
            raise RuntimeError(f"{e}\n    视频号需登录态, 推荐: ①电脑版微信打开"
                               f"视频号 → F12 抓包复制 .mp4 直链后直接粘贴给本工具;"
                               f" ②或把视频另存到本地后用本地文件处理")
    elif platform == "B站" and not cookies:
        url = bili_resolve(url) or url
        if quality != "best":
            # 明确要低清晰度时直接走专用 API(最稳)
            try:
                print("[下载] 使用B站专用解析(无需登录) …")
                return bili_direct_download(url, out_dir)
            except Exception as e:                           # noqa: BLE001
                print(f"    专用解析失败: {str(e).strip().splitlines()[-1][:90]}")
        for b in _browser_cookie_candidates():     # 登录态可解锁更高清晰度
            attempts.append((b, f"浏览器 {b}"))
        bili_api_fallback = True                    # yt-dlp 全失败后专用 API 兜底(480p)
    elif platform == "快手" and not cookies:
        try:
            print("[下载] 使用快手专用解析(无需登录) …")
            return kuaishou_direct_download(url, out_dir)
        except Exception as e:                               # noqa: BLE001
            print(f"    专用解析失败: {str(e).strip().splitlines()[-1][:90]}")
        for b in _browser_cookie_candidates():
            attempts.append((b, f"浏览器 {b}"))
    elif platform in ("爱奇艺", "优酷", "腾讯视频"):
        for b in _browser_cookie_candidates():     # 登录态解锁更高清晰度/更多内容
            attempts.append((b, f"浏览器 {b}"))
    before = set(os.listdir(out_dir))
    last_err = ""
    all_errs: list[str] = []
    for ck, tag in attempts:
        try:
            print(f"[下载] 使用 {tag} …")
            with yt_dlp.YoutubeDL(_ydl_opts(out_dir, quality, ck)) as ydl:
                ydl.download([url])
            new = [f for f in os.listdir(out_dir) if f not in before]
            vids = [f for f in new if not f.endswith((".part", ".ytdl"))]
            if not vids:
                raise RuntimeError("下载结束但没有生成视频文件")
            path = os.path.join(out_dir, vids[0])
            print(f"[下载] 完成 -> {path}")
            return path
        except Exception as e:                               # noqa: BLE001
            last_err = str(e).strip().splitlines()[-1] if str(e) else str(e)
            all_errs.append(f"{tag}: {last_err[:80]}")
            print(f"    该方式失败: {last_err[:90]}")
    if bili_api_fallback:                        # yt-dlp 全失败 → 专用 API 兜底(480p)
        try:
            print("[下载] 回退 B站专用 API(480p) …")
            return bili_direct_download(url, out_dir)
        except Exception as e:                           # noqa: BLE001
            all_errs.append(f"专用API: {str(e)[:80]}")
            print(f"    专用 API 兜底失败: {str(e)[:90]}")
    msg = last_err or "未知错误"
    detail = ("\n    " + " | ".join(all_errs)) if len(all_errs) > 1 else ""
    hint = ""
    low = msg.lower()
    if "channels.weixin" in url or "视频号" in platform:
        hint = ("\n    微信视频号暂无公开解析接口, 两个办法:\n"
                "    ① 在电脑版微信里打开视频号 → F12 抓包找到 .mp4 直链 → "
                "直接把直链粘给本工具;\n"
                "    ② 把视频另存到本地后, 直接把本地文件路径交给本工具。")
    elif (platform in ("爱奇艺", "优酷", "腾讯视频")
          and ("vip" in low or "drm" in low or "copyright" in low
               or "付费" in msg or "版权" in msg or "试看" in msg)):
        hint = ("\n    该内容为 VIP/付费或受版权保护(DRM), 技术上无法下载。\n"
                "    免费内容请确认链接正确后重试, 并可尝试 --cookies edge/chrome"
                " 提供登录态。")
    elif "cookie" in low or "login" in low or "登录" in low or "403" in low:
        hint = ("\n    建议: ① 安装无头浏览器支持后重试(自动过验证):\n"
                "         pip install playwright && python -m playwright install chromium\n"
                "       ② 或在浏览器打开一次 douyin.com 后重试(工具会自动取新鲜 Cookie);\n"
                "       ③ 或命令行加 --cookies edge / chrome 使用浏览器 Cookie"
                "(需先关闭该浏览器)。")
    raise RuntimeError(f"下载失败: {msg}{detail}{hint}")


def download_direct(url: str, out_dir: str = "downloads", name: str | None = None) -> str:
    """直接下载直链(不经过 yt-dlp), 适合视频号 .mp4 直链等场景"""
    import urllib.request
    os.makedirs(out_dir, exist_ok=True)
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": url,
    })
    path = os.path.join(out_dir, name or ("direct_" + str(abs(hash(url)) % 999999) + ".mp4"))
    print(f"[下载] 直链下载 -> {path}")
    with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if total:
                sys.stdout.write(f"\r    {got / total * 100:.1f}% ({got / 1e6:.1f}MB)")
                sys.stdout.flush()
    print()
    return path
