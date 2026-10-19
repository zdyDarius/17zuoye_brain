"""
ZIP 资料包解析工具（Markdown 批量导入）
背景：Typora / 语雀 / Word 导出的 Markdown 资料经常把多个 .md 与 images/ 目录打成一个 zip，
      上传 zip 即可一次性导入包内全部文档，图片按 MD 内的相对引用自动定位。

能力：
1. 安全解压：zip-slip 路径逃逸拦截、隐藏文件/__MACOSX 垃圾过滤、总解压体积与条目数上限
2. 中文文件名修复：Windows 打包的 zip 不带 UTF-8 标记，zipfile 默认按 cp437 解码 → 还原字节后按 utf-8/gb18030 重解
3. MD 发现：递归查找 .md/.markdown，过滤 _new.md 备份，自然序排序（01_、02_…10_ 不会错位）
"""
import os
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath

from common.logging.logger import logger

# 压缩包内忽略的目录（macOS 打包垃圾）
_IGNORED_DIR_NAMES = {"__MACOSX"}
# 忽略的系统/编辑器垃圾文件
_SKIP_FILE_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}
# Markdown 后缀（.markdown 兼容 Obsidian / 语雀导出）
MD_SUFFIXES = {".md", ".markdown"}

# 解压总量上限（防 zip 炸弹）、条目数上限、单次导入文档数上限
MAX_UNZIP_TOTAL_SIZE = 2 * 1024 * 1024 * 1024   # 2GB
MAX_ZIP_ENTRIES = 20000
MAX_MD_PER_ZIP = 50

_DIGIT_RE = re.compile(r"(\d+)")


class ZipExtractError(RuntimeError):
    """资料包解析失败（非法压缩包/路径逃逸/超出体积上限）"""


def _decode_name(info: zipfile.ZipInfo) -> str:
    """
    zip 条目文件名解码
    - 带 UTF-8 标记（flag_bits & 0x800）：zipfile 已正确解码
    - 否则 zipfile 按 cp437 解码：还原原始字节后依次尝试 utf-8 / gb18030（Windows 中文压缩包）
    """
    if info.flag_bits & 0x800:
        return info.filename
    raw = info.filename.encode("cp437", errors="ignore")
    for encoding in ("utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return info.filename


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    """Unix 模式高位为 symlink 的条目直接跳过（防解压到包外）"""
    return (info.external_attr >> 16) & 0o170000 == 0o120000


def _is_within(root: Path, target: Path) -> bool:
    """目标路径是否位于 root 之内（软链已展开）"""
    target_real = Path(os.path.realpath(target))
    return target_real == root or root in target_real.parents


def extract_zip(zip_path: str | Path, target_dir: str | Path,
                max_total_size: int = MAX_UNZIP_TOTAL_SIZE) -> list[str]:
    """
    安全解压 zip 到目标目录
    :param zip_path: 压缩包路径
    :param target_dir: 解压目标目录（不存在则创建）
    :param max_total_size: 解压后总字节上限
    :return: 已解压的相对路径列表
    :raise ZipExtractError: 压缩包非法 / 路径逃逸 / 超限
    """
    zip_path_obj = Path(zip_path)
    target_dir_obj = Path(target_dir)
    target_dir_obj.mkdir(parents=True, exist_ok=True)
    root_resolved = Path(os.path.realpath(target_dir_obj))

    try:
        zf = zipfile.ZipFile(zip_path_obj)
    except zipfile.BadZipFile as e:
        raise ZipExtractError(f"不是有效的 zip 压缩包：{e}") from e

    extracted = []
    total_size = 0
    skipped = 0

    with zf:
        infos = zf.infolist()
        if len(infos) > MAX_ZIP_ENTRIES:
            raise ZipExtractError(f"压缩包条目数超限（{len(infos)} > {MAX_ZIP_ENTRIES}）")

        for info in infos:
            if info.is_dir() or _is_symlink(info):
                continue

            name = _decode_name(info).replace("\\", "/")
            pure = PurePosixPath(name)
            parts = pure.parts
            if not parts or pure.is_absolute() or any(p in ("..", "") for p in parts):
                logger.warning(f"跳过非法路径条目：{name}")
                skipped += 1
                continue
            if parts[0] in _IGNORED_DIR_NAMES:
                continue
            filename = parts[-1]
            if filename in _SKIP_FILE_NAMES or filename.startswith("."):
                continue
            # 任意层级隐藏目录（.git/.idea 等）
            if any(p.startswith(".") for p in parts[:-1]):
                continue

            dest = target_dir_obj / Path(*parts)
            if not _is_within(root_resolved, dest):
                logger.warning(f"跳过越权路径条目：{name}")
                skipped += 1
                continue

            total_size += info.file_size
            if total_size > max_total_size:
                raise ZipExtractError(
                    f"解压后体积超过上限（>{max_total_size / 1024 / 1024:.0f}MB）")

            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
            extracted.append(pure.as_posix())

    logger.info(f"zip 解压完成：{zip_path_obj.name} → {len(extracted)} 个文件"
                f"（跳过 {skipped} 个非法条目），总体积 {total_size / 1024 / 1024:.1f}MB")
    return extracted


def _natural_key(text: str):
    """自然序键：01_x < 2_x < 10_x（避免字符串排序把 10 排在 2 前面）"""
    return [int(part) if part.isdigit() else part for part in _DIGIT_RE.split(text)]


def discover_md_files(root: str | Path) -> list[Path]:
    """
    递归发现解压目录下的 Markdown 文档
    过滤：_new.md（图片替换后自动备份）、隐藏文件、系统垃圾
    :return: 按自然序排序的 md 路径列表
    """
    root_obj = Path(root)
    if not root_obj.exists():
        return []

    md_files = []
    for path in root_obj.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in MD_SUFFIXES:
            continue
        rel_parts = path.relative_to(root_obj).parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        if path.name in _SKIP_FILE_NAMES or path.stem.endswith("_new"):
            continue
        md_files.append(path)

    md_files.sort(key=lambda p: _natural_key(p.relative_to(root_obj).as_posix()))
    logger.info(f"资料包内发现 {len(md_files)} 个 Markdown 文档")
    return md_files
