"""
课程文档分块解析节点
对应 PRD V1.3 3.2 course_doc 分支 + 4.3 Collection 3：
- 归一化 MD 后按标题层级分块（## / ### 为章节边界）
- 块过长时按字符上限二次切分（保留换行）
- 字段：doc_kind（lecture/project）/ chapter_name / chunk_index / content
- 章节路径提取（如"LangGraph概述 > 什么是LangGraph"）
"""
import re

from common.logging.logger import logger, node_log

# 分块参数
CHUNK_TARGET_SIZE = 1500   # 目标块大小（字符）：小于此值的相邻段落归入同块
CHUNK_MAX_SIZE = 3000      # 单块硬上限（字符）：超出强制二次切分
MIN_CHUNK_SIZE = 50        # 小于此值的尾块并入前块

# 章节标题行（归一化 MD 后）
_HEADING_RE = re.compile(r"^(#{1,6})\s+(?P<text>.+?)\s*$")


def _split_long_chunk(content: str, max_size: int) -> list:
    """
    超长块二次切分：优先在换行处切，保证每块不超过 max_size
    """
    if len(content) <= max_size:
        return [content]
    parts = []
    current = ""
    for line in content.split("\n"):
        if len(current) + len(line) + 1 > max_size and current:
            parts.append(current)
            current = line
        else:
            current = current + "\n" + line if current else line
    if current:
        parts.append(current)
    return parts


@node_log("node_course_doc_parser")
def node_course_doc_parser(state: dict) -> dict:
    """
    课程文档分块解析节点：
    1. 归一化 MD 按标题行切段落，维护章节路径（一级/二级标题）
    2. 同章节内段落累积至目标块大小（CHUNK_TARGET_SIZE）
    3. 超长块二次切分（CHUNK_MAX_SIZE）
    :param state: 导入图状态（md_text / source_file 等）
    :return: 状态更新（course_doc_records / parse_failed_entities）
    """
    md_text = state.get("md_text", "")
    source_file = state.get("source_file", "")
    source_path = state.get("source_path", "")
    # 上传接口可选元数据（PRD 4.1：course_doc 含 project_name 时为项目文档）
    project_name = state.get("project_name", "")

    if not md_text:
        return {"parse_failed_entities": [{"entity_type": "course_doc", "reason": "归一化后 Markdown 为空"}]}

    doc_kind = "project" if project_name else "lecture"

    lines = md_text.splitlines()
    # ---- 第一遍：按标题切分为（章节路径, 段落列表）节 ----
    sections = []               # [(chapter_path, [段落...])]
    chapter_stack = []          # 当前标题层级栈
    current_paragraphs = []
    in_code_block = False       # 代码块内的 # 不是标题

    def _flush_section():
        if current_paragraphs:
            sections.append(("\n".join(chapter_stack), list(current_paragraphs)))

    for line in lines:
        stripped = line.strip()
        # 代码块围栏切换
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            if current_paragraphs:
                current_paragraphs.append(line)
            continue
        # 代码块内：所有行视为正文
        if in_code_block:
            if current_paragraphs:
                current_paragraphs.append(line)
            continue
        # 转义过的行首 #（\#）：正文，非标题
        if stripped.startswith("\\#"):
            if current_paragraphs:
                current_paragraphs.append(line.replace("\\#", "#", 1))
            continue
        m = _HEADING_RE.match(line)
        if m:
            _flush_section()
            current_paragraphs = []
            level = len(m.group(1))
            title = m.group("text").strip()
            # 维护层级栈（只保留 一级/二级 作为章节路径，更深层级并入正文首行）
            if level <= 2:
                chapter_stack = chapter_stack[: level - 1]
                chapter_stack.append(title)
            else:
                # 三级及以下标题作为正文段落保留（保留原文）
                current_paragraphs.append(line)
        else:
            if stripped:
                current_paragraphs.append(line)
            elif current_paragraphs:
                current_paragraphs.append("")  # 段内空行
    _flush_section()

    # ---- 第二遍：章节内累积成块 ----
    records = []
    for chapter_path, paragraphs in sections:
        buffer = ""
        for p in paragraphs:
            if len(buffer) + len(p) + 1 > CHUNK_TARGET_SIZE and buffer:
                _emit_chunk(records, buffer, chapter_path, source_file, source_path, doc_kind, project_name)
                buffer = p
            else:
                buffer = buffer + "\n" + p if buffer else p
        if buffer:
            _emit_chunk(records, buffer, chapter_path, source_file, source_path, doc_kind, project_name)

    # 尾块合并：过小的块并入前块（同章节内）
    if len(records) >= 2 and len(records[-1]["content"]) < MIN_CHUNK_SIZE \
            and records[-1]["chapter_name"] == records[-2]["chapter_name"]:
        records[-2]["content"] += "\n" + records[-1]["content"]
        records.pop()

    logger.info(f"课程文档分块完成：{len(sections)} 个章节 → {len(records)} 个块（doc_kind={doc_kind}）")
    return {"course_doc_records": records}


def _emit_chunk(records: list, content: str, chapter_path: str,
                source_file: str, source_path: str, doc_kind: str, project_name: str):
    """产出分块记录（含超长二次切分）"""
    chapter_name = chapter_path.replace("\n", " > ") if chapter_path else ""
    for piece in _split_long_chunk(content.strip(), CHUNK_MAX_SIZE):
        if not piece.strip():
            continue
        records.append({
            "doc_kind": doc_kind,
            "project_name": project_name or None,
            "chapter_name": chapter_name or None,
            "chunk_index": len(records),
            "content": piece,
            "source_file": source_file,
            "source_path": source_path,
        })


if __name__ == "__main__":
    # 独立验证：转换并解析 LangGraph docx
    from pathlib import Path
    from processor.import_processor.nodes.node_format_convert import _docx_to_md

    docx_path = Path("/Users/zdy/Downloads/教育/数据/课程文档/尚硅谷大模型技术之LangGraphV1.1.0.docx")
    md = _docx_to_md(docx_path)
    print(f"归一化 MD：{len(md)} 字符")
    result = node_course_doc_parser({"md_text": md, "source_file": docx_path.name, "source_path": str(docx_path)})
    records = result["course_doc_records"]
    print(f"分块数：{len(records)}")
    sizes = [len(r["content"]) for r in records]
    print(f"块大小：min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)//len(sizes)}")
    print("\n首块章节:", records[0].get("chapter_name"))
    print("首块预览:", records[0]["content"][:150].replace("\n", "\\n"))
    print("\n第 5 块章节:", records[4].get("chapter_name"))
    # 章节分布
    from collections import Counter
    chapters = Counter(r.get("chapter_name", "")[:30] for r in records)
    print("\n章节块数分布（前 10）:")
    for ch, c in chapters.most_common(10):
        print(f"  {c} 块 | {ch}")
