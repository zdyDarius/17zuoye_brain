"""
课程文档分块解析节点
对应 PRD V1.3 3.2 course_doc 分支 + 4.3 Collection 3：
- 归一化 MD 后按标题层级分块（## / ### 为章节边界）
- 块过长时按字符上限二次切分（保留换行）
- 字段：doc_kind（lecture/project）/ chapter_name / chunk_index / content
- 章节路径提取（如"LangGraph概述 > 什么是LangGraph"）
"""
import re

from common.logging.logger import logger, node_log, step_log

# 分块参数
CHUNK_TARGET_SIZE = 1500   # 目标块大小（字符）：小于此值的相邻段落归入同块
CHUNK_MAX_SIZE = 3000      # 单块硬上限（字符）：超出强制二次切分
MIN_CHUNK_SIZE = 50        # 小于此值的尾块并入前块

# 标题文本上限（字符）：超过则判定为 MinerU 把整段正文误识别成标题
# 真实标题极少超过 40 字；不拦截会导致 ①章节路径超 VARCHAR 长度 ②该段正文被吞（只进栈不进正文）
MAX_HEADING_CHARS = 60

# 章节标题行（归一化 MD 后）
_HEADING_RE = re.compile(r"^(#{1,6})\s+(?P<text>.+?)\s*$")

# 章节名里的自动编号前缀：`1 ` / `2.2 ` / `3.1.1 ` / `3.1.` / `3、`
# 来源：Word 的自动多级编号（numbering.xml）被 MinerU 物化进了标题文本。
# 只影响 chapter_name（检索过滤/引用展示/评估比对），正文标题保留编号——编号对阅读有用。
_CHAPTER_NUM_RE = re.compile(r"^\d+(?:[.、]\d+)*[.、]?\s+")
# 数字后紧跟这些字时视为真标题（"5 个常见误区" "2024 年展望"），不剥
_NUM_UNIT_CHARS = ("年", "月", "日", "个", "种", "条", "大", "周", "次", "步")
_BARE_INT_RE = re.compile(r"^\d+$")


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
    # 1. 校验并获取归一化 MD、溯源信息与文档类型
    data = step_1_validate_and_get_data(state)
    if data is None:
        return {"parse_failed_entities": [{"entity_type": "course_doc", "reason": "归一化后 Markdown 为空"}]}
    md_text, source_file, source_path, project_name, doc_kind = data

    # 2. 按标题层级切分为「章节路径 + 段落列表」
    sections = step_2_split_sections(md_text)

    # 3. 章节内累积成块（超长块二次切分）
    #    只有 MinerU 转换来的文档（pdf/doc/docx）才剥章节编号：Word 的自动编号会被 MinerU
    #    物化进标题文本；手写 md 的 "## 2. 核心概念扫盲" 是作者本意，必须保留。
    strip_numbering = state.get("file_format") in ("pdf", "doc", "docx")
    records = step_3_build_chunks(sections, source_file, source_path, doc_kind, project_name,
                                  strip_numbering=strip_numbering)

    # 4. 尾块合并（过小的末块并入前块）
    records = step_4_merge_tail_chunk(records)

    logger.info(f"课程文档分块完成：{len(sections)} 个章节 → {len(records)} 个块（doc_kind={doc_kind}）")
    return {"course_doc_records": records}


@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state: dict):
    """
    校验并获取归一化 MD、溯源信息，并推断文档类型
    :return: (md_text, source_file, source_path, project_name, doc_kind) 或 None（MD 为空）
    """
    md_text = state.get("md_text", "")
    if not md_text:
        return None

    source_file = state.get("source_file", "")
    source_path = state.get("source_path", "")
    # 上传接口可选元数据（PRD 4.1：course_doc 含 project_name 时为项目文档）
    project_name = state.get("project_name", "")
    doc_kind = "project" if project_name else "lecture"
    return md_text, source_file, source_path, project_name, doc_kind


@step_log("step_2_split_sections")
def step_2_split_sections(md_text: str):
    """
    按标题层级把 MD 切分为节：维护章节路径栈（一级/二级标题入路径，三级及以下并入正文）
    :return: [(chapter_path, [段落...]), ...]
    """
    sections = []               # [(chapter_path, [段落...])]
    chapter_stack = []          # 当前标题层级栈
    current_paragraphs = []
    in_code_block = False       # 代码块内的 # 不是标题

    def _flush_section():
        if current_paragraphs:
            sections.append(("\n".join(chapter_stack), list(current_paragraphs)))

    for line in md_text.splitlines():
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
            title = m.group("text").strip()
            # MinerU 会把整段正文误识别为标题（如 "# 词表（Vocabulary）是由语料库构建出的……"）：
            # 超长"标题"或句读结尾的"标题"降级为正文段落（去掉 # 前缀），否则 ①章名超出 VARCHAR 限制
            # ②该段只进章节栈、不进正文，内容被静默吞掉
            if len(title) > MAX_HEADING_CHARS or title.endswith(("。", "；")):
                current_paragraphs.append(title)
                continue
            _flush_section()
            current_paragraphs = []
            level = len(m.group(1))
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

    logger.info(f"章节切分完成：{len(sections)} 个章节")
    return sections


@step_log("step_3_build_chunks")
def step_3_build_chunks(sections: list, source_file: str, source_path: str,
                        doc_kind: str, project_name: str,
                        strip_numbering: bool = False) -> list:
    """
    章节内段落累积至目标块大小，超长块二次切分，产出分块记录
    :param strip_numbering: 是否剥离章节名开头的自动编号（仅 MinerU 转换文档）
    :return: 分块记录列表
    """
    records = []
    for chapter_path, paragraphs in sections:
        buffer = ""
        for p in paragraphs:
            if len(buffer) + len(p) + 1 > CHUNK_TARGET_SIZE and buffer:
                _emit_chunk(records, buffer, chapter_path, source_file, source_path, doc_kind,
                            project_name, strip_numbering)
                buffer = p
            else:
                buffer = buffer + "\n" + p if buffer else p
        if buffer:
            _emit_chunk(records, buffer, chapter_path, source_file, source_path, doc_kind,
                        project_name, strip_numbering)
    logger.info(f"分块组装完成：{len(records)} 个块")
    return records


@step_log("step_4_merge_tail_chunk")
def step_4_merge_tail_chunk(records: list) -> list:
    """尾块合并：过小的末块并入前块（同章节内）"""
    if len(records) >= 2 and len(records[-1]["content"]) < MIN_CHUNK_SIZE \
            and records[-1]["chapter_name"] == records[-2]["chapter_name"]:
        records[-2]["content"] += "\n" + records[-1]["content"]
        records.pop()
        logger.info(f"尾块过小，已并入前块，最终块数：{len(records)}")
    return records


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


def _strip_chapter_number(title: str) -> str:
    """
    去掉章节名开头的自动编号（`1 LangGraph概述` → `LangGraph概述`）。
    仅用于 chapter_name；正文标题不动，避免破坏"见 3.2 节"这类引用。
    带点号的层级编号（`5.2`、`3.1.1`）几乎只可能来自自动编号，直接剥；
    纯整数才需要防误伤：`2024 年展望`（>99）、`5 个常见误区`（后接量词）保持原样。
    """
    m = _CHAPTER_NUM_RE.match(title)
    if not m:
        return title
    rest = title[m.end():].strip()
    if not rest:
        return title
    num = title[:m.end()].strip().rstrip(".、")
    if _BARE_INT_RE.match(num):
        # 只有纯整数才可能是"3 条建议"这类真标题；"5.2 条件边"里的"条"是"条件"
        if rest[0] in _NUM_UNIT_CHARS:
            return title
        if int(num) > 99:
            return title
    return rest


def _emit_chunk(records: list, content: str, chapter_path: str,
                source_file: str, source_path: str, doc_kind: str, project_name: str,
                strip_numbering: bool = False):
    """产出分块记录（含超长二次切分）
    语义增强：块首拼章节路径前缀（公式/代码密集的块缺少自然语言语义，
    章节名前缀显著提升其向量可检索性，如"Sigmoid 函数"依赖章节上下文）
    """
    if strip_numbering:
        chapter_name = " > ".join(
            _strip_chapter_number(seg) for seg in chapter_path.split("\n") if seg.strip()
        ) if chapter_path else ""
    else:
        chapter_name = chapter_path.replace("\n", " > ") if chapter_path else ""
    for piece in _split_long_chunk(content.strip(), CHUNK_MAX_SIZE):
        if not piece.strip():
            continue
        # 语义增强：章节路径作为块首前缀（检索命中价值 > 存储冗余）
        enriched = f"【{chapter_name}】\n{piece}" if chapter_name else piece
        records.append({
            "doc_kind": doc_kind,
            "project_name": project_name or None,
            "chapter_name": chapter_name or None,
            "chunk_index": len(records),
            "content": enriched,
            "source_file": source_file,
            "source_path": source_path,
        })


if __name__ == "__main__":
    # 独立验证：先走 node_format_convert（MinerU，需要网络与 MINERU_API_TOKEN）转成 MD，再分块
    from pathlib import Path
    from processor.import_processor.nodes.node_format_convert import node_format_convert

    docx_path = Path("/Users/zdy/Downloads/教育/数据/课程文档/尚硅谷大模型技术之LangGraphV1.1.0.docx")
    conv = node_format_convert({
        "local_file_path": str(docx_path),
        "local_dir": str(docx_path.parent),
        "file_format": docx_path.suffix.lstrip(".").lower(),
        "content_type": "course_doc",
    })
    if conv.get("format_convert_error"):
        raise SystemExit(f"格式转换失败：{conv['format_convert_error']}")
    md = conv["md_text"]
    print(f"归一化 MD：{len(md)} 字符（MinerU）")
    result = node_course_doc_parser({"md_text": md, "source_file": docx_path.name,
                                     "source_path": str(docx_path), "file_format": "docx"})
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
