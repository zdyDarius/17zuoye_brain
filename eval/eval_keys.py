"""
评估侧编码归一化
================
`doc:` 类 golden / 检索键都形如 `doc:{source_file}|{chapter_name}`。

章节名可能带 MinerU 物化出来的自动编号（`1 LangGraph概述 > 3 状态`，来自 Word 的
numbering.xml），而导入侧的 `node_course_doc_parser` 只对 pdf/doc/docx 剥编号，
手写 md 的 `## 2. 核心概念扫盲` 保留作者编号。于是同一个章节在不同来源/不同导入批次里
可能带或不带编号，评估比对必须先归一化，否则会误判为 0 命中。

只在**评估层**做归一化：不动库里的数据，也不影响生产链路。
"""
import re

# 与 processor/import_processor/nodes/node_course_doc_parser._CHAPTER_NUM_RE 同一规则
_CHAPTER_NUM_RE = re.compile(r"^\d+(?:\.\d+)*[.、]?\s+")


def norm_chapter(name: str) -> str:
    """逐段剥掉自动编号：`1 概述 > 1.1 组成` → `概述 > 组成`"""
    return " > ".join(_CHAPTER_NUM_RE.sub("", p) for p in (name or "").split(" > "))


def normalize_code(code: str) -> str:
    """只归一化 doc: 编码；题目/系列/模块编码原样返回"""
    if not code.startswith("doc:"):
        return code
    src, sep, chapter = code[4:].partition("|")
    if not sep:
        return code
    return f"doc:{src}|{norm_chapter(chapter)}"
