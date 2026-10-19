"""
zip 资料包导入链路离线自测（不触碰 VL 模型 / MinIO / Milvus）
覆盖：解压（含中文名）→ md 发现与自然序 → 图片引用抽取 → 本地文件定位 → 引用替换
用法：
  .venv/bin/python test/test_zip_import.py "<资料目录>" [--keep]
"""
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, "/Users/zdy/PycharmProjects/17zuoye_brain")

from utils.zip_utils import ZipExtractError, discover_md_files, extract_zip
from utils.md_image_utils import (
    build_image_key, extract_image_refs, is_remote_src, resolve_image_path, render_image_markdown,
)

WORK = Path("/tmp/zz_zip_selftest")
SRC = Path(sys.argv[1] if len(sys.argv) > 1 else
           "/Volumes/zdy/大模型260708/尚硅谷大模型项目实战之掌柜智库实战/1.资料/教育/数据/项目文档/掌柜智库")
FAKE_URL = "http://127.0.0.1:9000/bucket/upload-images/doc/__test__.jpg"
FAKE_SUMMARY = "整体架构图：展示导入与查询双链路 [示意]"


def build_zip() -> Path:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    zip_path = WORK / "掌柜智库.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(SRC.rglob("*")):
            if p.is_file() and not p.name.startswith("."):
                zf.write(p, p.relative_to(SRC).as_posix())
    return zip_path


def main():
    zip_path = build_zip()
    print(f"① 压缩包：{zip_path.name} {zip_path.stat().st_size / 1024 / 1024:.1f}MB")

    extract_dir = WORK / "extracted"
    files = extract_zip(zip_path, extract_dir)
    print(f"② 解压 {len(files)} 个文件（含中文名）")

    md_files = discover_md_files(extract_dir)
    print(f"③ 发现 {len(md_files)} 个 md（自然序前 3）：{[p.name for p in md_files[:3]]}")
    assert md_files, "未发现 md"
    assert md_files[0].name.startswith("01_"), f"自然序异常：{md_files[0].name}"

    total_refs = total_hit = total_miss = total_remote = 0
    for md in md_files:
        text = md.read_text(encoding="utf-8", errors="ignore")
        refs = extract_image_refs(text)
        hits = {}
        for ref in refs:
            if ref.src.lower().startswith(("http://", "https://", "data:")):
                total_remote += 1
                continue
            total_refs += 1
            p = resolve_image_path(ref.src, md.parent, extract_dir)
            if p is None:
                total_miss += 1
                print(f"   ✗ 未定位：{md.name} → {ref.src}")
            else:
                total_hit += 1
                hits.setdefault(build_image_key(p, md.parent, extract_dir), p)

        # 模拟替换（用假 URL/摘要验证语法与偏移替换）
        replacements = []
        for ref in refs:
            p = resolve_image_path(ref.src, md.parent, extract_dir)
            if p is not None:
                replacements.append((ref.start, ref.end,
                                     render_image_markdown(FAKE_SUMMARY, FAKE_URL)))
        for start, end, new in sorted(replacements, reverse=True):
            text = text[:start] + new + text[end:]
        # 替换后：原本地引用（含 HTML img）应全部消失，仅剩指向 MinIO 的远程引用
        leftover = [r for r in extract_image_refs(text) if not is_remote_src(r.src)]
        assert not leftover, f"{md.name} 替换后仍残留 {len(leftover)} 处本地图片引用"
        print(f"   ✓ {md.name[:28]:<30} 引用 {len(refs):>3} 处 → 本地图片 {len(hits):>3} 张，替换后残留 0")

    print(f"④ 图片引用统计：本地 {total_refs} 处（命中 {total_hit}，未定位 {total_miss}），"
          f"远程 {total_remote} 处")
    assert total_miss == 0, "存在未定位的本地图片"
    print("✅ zip 导入链路离线自测通过")


if __name__ == "__main__":
    try:
        main()
    except ZipExtractError as e:
        print(f"❌ 解压失败：{e}")
        raise
