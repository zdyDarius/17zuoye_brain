"""
格式归一化节点
对应 PRD V1.3 3.2.0 格式解析矩阵：
- .md：直接通过（node_entry 已读取）
- .docx：python-docx 提取段落与标题样式，重组为 MD（Heading 1/2/3 → #/##/###，表格转 MD 表格）
- .pdf：MinerU 云端 API 转换（batch 接口：申请上传链接 → 上传 → 轮询结果 → 下载 MD）
注意：node_entry 已对 md 做直读；本节点处理 pdf/docx 两类需要转换的格式
"""
import time
from pathlib import Path

import requests

from common.logging.logger import logger, node_log
from common.config.mineru_config import mineru_config


# ---- docx 转换 ----
_HEADING_LEVEL_MAP = {
    "Heading 1": "# ", "Heading 2": "## ", "Heading 3": "### ",
    "Heading 4": "#### ", "Heading 5": "#### ", "Heading 6": "#### ",
    # 中文 Word 样式名（尚硅谷讲义实测样式）
    "标题 1": "# ", "标题 2": "## ", "标题 3": "### ",
    "文档主标题": "# ", "一级标题": "## ", "次标题": "### ",
    "三级标题": "#### ",
    # 代码相关样式：转为代码块（不是标题）
    "代码样式": "CODE",
}

# 正文文本中以 # 开头的代码注释行需转义（避免被 MD 标题正则误判）
_LEADING_HASH_RE = __import__("re").compile(r"^(#{1,6})(\s)")


def _escape_leading_hash(text: str) -> str:
    """正文行首 # 转义（如 # pip install xxx → \\# pip install xxx）"""
    if _LEADING_HASH_RE.match(text):
        return "\\" + text
    return text


def _docx_to_md(docx_path: Path) -> str:
    """
    docx → Markdown：段落按标题样式映射 MD 标题，表格转 MD 表格
    :param docx_path: docx 文件路径
    :return: Markdown 文本
    """
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(str(docx_path))

    def _iter_block_items(parent):
        """按文档顺序迭代段落与表格"""
        from docx.oxml.ns import qn
        body = parent.element.body
        for child in body.iterchildren():
            if child.tag == qn("w:p"):
                yield Paragraph(child, parent)
            elif child.tag == qn("w:tbl"):
                yield Table(child, parent)

    md_lines = []
    in_code_block = False
    for block in _iter_block_items(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            style_name = (block.style.name if block.style else "") or ""
            prefix = _HEADING_LEVEL_MAP.get(style_name)
            if prefix == "CODE":
                # 代码样式段落 → 代码块行（独立成块，行首 # 不转义）
                if not in_code_block:
                    md_lines.append("```")
                    in_code_block = True
                md_lines.append(text)
                continue
            elif in_code_block:
                md_lines.append("```")
                in_code_block = False
            if prefix and prefix != "CODE":
                md_lines.append(f"{prefix}{text}")
            elif style_name.startswith("List") or style_name.startswith("参数列表") or text.startswith(("•", "·")):
                md_lines.append(f"- {_escape_leading_hash(text.lstrip('•· '))}")
            else:
                # 正文行：行首 # 转义（避免代码注释被误判为 MD 标题）
                md_lines.append(_escape_leading_hash(text))
        elif isinstance(block, Table):
            if in_code_block:
                md_lines.append("```")
                in_code_block = False
            # 表格 → MD 表格（首行作表头）
            rows = []
            for row in block.rows:
                cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                rows.append(cells)
            if rows:
                md_lines.append("| " + " | ".join(rows[0]) + " |")
                md_lines.append("| " + " | ".join(["---"] * len(rows[0])) + " |")
                for r in rows[1:]:
                    md_lines.append("| " + " | ".join(r) + " |")
                md_lines.append("")
    if in_code_block:
        md_lines.append("```")

    return "\n\n".join(md_lines)


# ---- PDF 转换（MinerU 云端 API）----
def _pdf_to_md_via_mineru(pdf_path: Path, local_dir: Path) -> str:
    """
    PDF → Markdown（MinerU batch API 流程）：
    1. POST 申请批量任务（返回上传链接 + 任务批次）
    2. PUT 上传 PDF 文件
    3. GET 轮询任务结果（分钟级等待）
    4. 下载结果中的 .md 文件
    :param pdf_path: PDF 文件路径
    :param local_dir: 任务工作目录（结果落盘）
    :return: Markdown 文本
    :raise: RuntimeError 转换失败（网络/超时/配额）
    """
    headers = {"Authorization": f"Bearer {mineru_config.api_key}"}

    # 1. 申请上传链接
    apply_url = "https://mineru.net/api/v4/extract/task/batch"
    data_name = pdf_path.name
    apply_resp = requests.post(
        apply_url,
        headers={**headers, "Content-Type": "application/json"},
        json={"enable_formula": True, "enable_table": True,
              "language": "ch", "files": [{"name": data_name, "is_ocr": True}]},
        timeout=30,
    )
    apply_resp.raise_for_status()
    apply_data = apply_resp.json()
    if apply_data.get("code") != 0:
        raise RuntimeError(f"MinerU 申请任务失败：{apply_data.get('msg')}")
    batch_id = apply_data["data"]["batch_id"]
    upload_url = apply_data["data"]["file_urls"][0]

    # 2. 上传文件
    with open(pdf_path, "rb") as f:
        put_resp = requests.put(upload_url, data=f,
                                headers={"Content-Type": "application/octet-stream"}, timeout=300)
    put_resp.raise_for_status()
    logger.info(f"[MinerU] 文件已上传，批次 {batch_id}，等待转换...")

    # 3. 轮询结果（最长 10 分钟）
    result_url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
    deadline = time.time() + 600
    md_zip_url = None
    while time.time() < deadline:
        time.sleep(10)
        poll = requests.get(result_url, headers=headers, timeout=30)
        poll.raise_for_status()
        poll_data = poll.json()
        results = (poll_data.get("data") or {}).get("extract_result") or []
        if results:
            state = results[0].get("state")
            if state == "done":
                full_zip = results[0].get("full_zip_url")
                md_zip_url = full_zip or results[0].get("md_url")
                break
            if state == "failed":
                raise RuntimeError(f"MinerU 转换失败：{results[0].get('err_msg')}")
    if not md_zip_url:
        raise RuntimeError("MinerU 转换超时（>10 分钟）")

    # 4. 下载并解压 MD
    import zipfile
    import io
    dl = requests.get(md_zip_url, timeout=120)
    dl.raise_for_status()
    md_text = ""
    with zipfile.ZipFile(io.BytesIO(dl.content)) as zf:
        for name in zf.namelist():
            if name.endswith(".md"):
                md_text = zf.read(name).decode("utf-8", errors="ignore")
                break
    if not md_text:
        raise RuntimeError("MinerU 结果包中无 .md 文件")
    # 落盘备份
    (local_dir / f"{pdf_path.stem}.md").write_text(md_text, encoding="utf-8")
    return md_text


@node_log("node_format_convert")
def node_format_convert(state: dict) -> dict:
    """
    格式归一化节点：pdf / docx → Markdown（md 已在 node_entry 直读）
    :param state: 导入图状态
    :return: 状态更新（md_text / format_convert_error）
    """
    file_format = state.get("file_format")
    if state.get("md_text"):
        # MD 已直读，无需转换
        return {}

    local_file_path = Path(state.get("local_file_path", ""))
    local_dir = Path(state.get("local_dir", str(local_file_path.parent)))

    try:
        if file_format == "docx":
            md_text = _docx_to_md(local_file_path)
            logger.info(f"docx 归一化完成：{len(md_text)} 字符")
            return {"md_text": md_text}

        if file_format == "pdf":
            md_text = _pdf_to_md_via_mineru(local_file_path, local_dir)
            logger.info(f"PDF 归一化完成：{len(md_text)} 字符")
            return {"md_text": md_text}

        return {"format_convert_error": f"未知格式，无法归一化：{file_format}"}

    except Exception as e:
        logger.exception("格式转换失败")
        return {"format_convert_error": f"格式转换失败：{e}"}
