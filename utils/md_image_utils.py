"""
Markdown 图片引用解析工具
资料包里的图片引用存在两种写法，同一篇文档中也可能混用：
1. Markdown 语法：![说明](images/a.png "title") / ![](./images/a.png)
2. HTML 标签（Typora 缩放、语雀/Word 导出常见）：<img src="images/a.jpg" style="zoom:67%;" />

本模块负责：抽取引用 → 过滤远程/非法 src → 解析为本地图片文件 → 生成替换文本。
"""
import re
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

# 支持的图片后缀（与 node_md_img 上传 MinIO 的范围一致）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

# 远程/内联图片不落本地，跳过处理
REMOTE_PREFIXES = ("http://", "https://", "//", "data:", "blob:", "ftp://")

# Markdown 图片：![alt](src "title")
_MD_IMAGE_RE = re.compile(
    r"!\[(?P<alt>[^\]\n]*)\]\(\s*(?P<src><[^>\n]+>|[^)\s]+)"
    r"(?:\s+(?:\"[^\"\n]*\"|'[^'\n]*'))?\s*\)"
)
# HTML img 标签及 src 属性（双引号/单引号/无引号）
_HTML_IMG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_HTML_SRC_RE = re.compile(
    r"""\bsrc\s*=\s*(?:"(?P<dq>[^"]*)"|'(?P<sq>[^']*)'|(?P<bare>[^\s>]+))""",
    re.IGNORECASE,
)
# 围栏代码块起始（``` / ~~~）
_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
# 行内代码 `code` / ``code``
_INLINE_CODE_RE = re.compile(r"(?<!`)(`{1,2})(?!`)([^\n]+?)(?<!`)\1(?!`)")


@dataclass
class ImageRef:
    """MD 中的一处图片引用（start/end 为原文偏移，用于精确替换）"""
    start: int
    end: int
    raw: str
    src: str
    alt: str
    kind: str  # markdown / html


@dataclass
class ResolvedImage:
    """同一本地图片文件（可能被引用多次）"""
    key: str
    path: Path
    refs: list = field(default_factory=list)
    context: tuple | None = None  # (上文, 下文)，供 VL 识别增强


def is_remote_src(src: str) -> bool:
    """远程 / data URI / 协议相对地址：不参与本地图片处理"""
    return src.strip().lower().startswith(REMOTE_PREFIXES)


def normalize_image_src(src: str) -> str:
    """清洗 src：去尖括号、URL 解码、去 query/fragment、统一分隔符、去 ./ 前缀"""
    text = (src or "").strip()
    if text.startswith("<") and text.endswith(">"):
        text = text[1:-1].strip()
    text = unquote(text)
    text = text.split("#", 1)[0].split("?", 1)[0].strip()
    text = text.replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def _code_block_ranges(md_content: str) -> list[tuple[int, int]]:
    """围栏代码块（``` / ~~~）的字符区间：其中的图片示例只是代码，不能当引用处理"""
    ranges: list[tuple[int, int]] = []
    offset = 0
    fence_char = ""
    fence_start = 0
    for line in md_content.splitlines(keepends=True):
        m = _FENCE_RE.match(line)
        if m:
            marker = m.group(1)
            if not fence_char:
                fence_char, fence_start = marker[0], offset
            elif marker[0] == fence_char:
                ranges.append((fence_start, offset + len(line)))
                fence_char = ""
        offset += len(line)
    if fence_char:
        ranges.append((fence_start, len(md_content)))
    return ranges


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """合并重叠区间（行内代码可能落在代码块内）"""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _protected_ranges(md_content: str) -> list[tuple[int, int]]:
    """不参与图片解析的区间：围栏代码块 + 行内代码"""
    ranges = _code_block_ranges(md_content)
    ranges.extend((m.start(), m.end()) for m in _INLINE_CODE_RE.finditer(md_content))
    return _merge_ranges(ranges)


def _in_protected(pos: int, ranges: list[tuple[int, int]], starts: list[int]) -> bool:
    if not ranges:
        return False
    idx = bisect_right(starts, pos) - 1
    return idx >= 0 and ranges[idx][0] <= pos < ranges[idx][1]


def extract_image_refs(md_content: str) -> list[ImageRef]:
    """
    抽取 MD 中全部图片引用（Markdown + HTML 两种写法），按出现位置排序、去重叠
    代码块 / 行内代码中的示例图片语法会被忽略（文档里讲图片处理时常见）
    :param md_content: Markdown 原文
    :return: ImageRef 列表
    """
    refs: list[ImageRef] = []
    protected = _protected_ranges(md_content)
    starts = [r[0] for r in protected]

    for m in _MD_IMAGE_RE.finditer(md_content):
        if _in_protected(m.start(), protected, starts):
            continue
        refs.append(ImageRef(
            start=m.start(), end=m.end(), raw=m.group(0),
            src=m.group("src"), alt=m.group("alt") or "", kind="markdown",
        ))

    for m in _HTML_IMG_RE.finditer(md_content):
        if _in_protected(m.start(), protected, starts):
            continue
        src_match = _HTML_SRC_RE.search(m.group(0))
        if not src_match:
            continue
        src = next((v for v in src_match.groupdict().values() if v is not None), "")
        alt_match = re.search(r"""\balt\s*=\s*(?:"([^"]*)"|'([^']*)')""", m.group(0), re.IGNORECASE)
        alt = (alt_match.group(1) or alt_match.group(2) or "") if alt_match else ""
        refs.append(ImageRef(
            start=m.start(), end=m.end(), raw=m.group(0),
            src=src, alt=alt, kind="html",
        ))

    refs.sort(key=lambda r: r.start)
    # 去重叠（保序：先出现者优先）
    deduped: list[ImageRef] = []
    last_end = -1
    for ref in refs:
        if ref.start < last_end:
            continue
        deduped.append(ref)
        last_end = ref.end
    return deduped


def resolve_image_path(src: str, md_dir: str | Path, search_root: str | Path | None = None) -> Path | None:
    """
    把 MD 中的图片 src 解析为本地图片文件
    查找顺序：绝对路径 → 相对 MD 目录 → 相对资料包根目录 → 同名文件兜底（MD 目录/资料包根递归）
    :param src: 原始 src
    :param md_dir: MD 所在目录
    :param search_root: 资料包解压根目录（图片集中放在包根 images/ 时使用）
    :return: 图片文件 Path；找不到返回 None
    """
    raw = normalize_image_src(src)
    if not raw:
        return None

    md_dir_obj = Path(md_dir)
    name = PurePosixPath(raw).name
    if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
        return None

    candidates: list[Path] = []
    raw_path = Path(raw)
    if raw_path.is_absolute():
        candidates.append(raw_path)
    candidates.append(md_dir_obj / raw)
    if search_root:
        candidates.append(Path(search_root) / raw)
    candidates.append(md_dir_obj / name)

    for candidate in candidates:
        try:
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
                return candidate
        except OSError:
            continue

    # 兜底：按文件名在 MD 目录 / 资料包根目录递归查找（路径前缀与打包结构不一致时）
    bases = [md_dir_obj] if not search_root else [md_dir_obj, Path(search_root)]
    for base in bases:
        if not base.is_dir():
            continue
        try:
            for found in base.rglob(name):
                if found.is_file() and found.suffix.lower() in IMAGE_EXTENSIONS:
                    return found
        except OSError:
            continue
    return None


def build_image_key(image_path: Path, md_dir: str | Path, search_root: str | Path | None = None) -> str:
    """
    图片唯一键：优先相对资料包根目录的路径（区分不同子目录的同名图片），退化到文件名
    """
    for base in (search_root, md_dir):
        if not base:
            continue
        try:
            return Path(image_path).relative_to(Path(base)).as_posix()
        except ValueError:
            continue
    return Path(image_path).name


def render_image_markdown(summary: str, url: str) -> str:
    """生成替换文本 ![VL摘要](MinIO URL)：摘要去换行、方括号转全角（避免破坏语法）"""
    text = " ".join((summary or "").split())
    text = text.replace("[", "［").replace("]", "］")
    return f"![{text}]({url})"
