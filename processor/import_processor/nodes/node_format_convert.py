"""
格式归一化节点
对应 PRD V1.3 3.2.0 格式解析矩阵：
- .md：直接通过（node_entry 已读取）
- .pdf / .docx / .doc：MinerU 云端 API 转换（batch 接口：申请上传链接 → 上传 → 轮询结果 → 下载 MD）
统一走云端的原因：MinerU 会把文档里的**图片完整导出**（结果包含 images/ 与 `![](images/xxx)`
引用），下游 node_md_img 才能做 VL 摘要 + MinIO 上传；本地 python-docx 解析只能拿到纯文本，
图片会静默丢失。
注意：node_entry 已对 md 做直读；本节点处理 pdf / doc / docx 三类需要转换的格式
"""
import re
import time
from pathlib import Path

import requests

from common.logging.logger import logger, node_log, step_log
from common.config.mineru_config import mineru_config

# MinerU v4 支持的格式（单文件 ≤200MB、≤600 页）
_MINERU_SUPPORTED_FORMATS = {"pdf", "doc", "docx"}

# 标题行：捕获 `#{1,6} ` 前缀与标题正文
_HEADING_LINE_RE = re.compile(r"^(#{1,6}\s+)(.*)$")
# 深层编号标题：MinerU 会把 Word 的多级列表编号映射成 5~6 级标题
# （如 `###### 2.2.1.1.1.1 定义状态；`），而 node_course_doc_parser 对任何标题都会切分 section，
# 结果是一堆只有一句话的碎块。降级为列表项，既保留文本也避免误切。
_DEEP_NUMBERED_HEADING_RE = re.compile(r"^#{5,6}\s+(?P<num>\d+(?:\.\d+)+)\s*(?P<text>.*)$")


def _normalize_mineru_markdown(md_text: str) -> str:
    """
    归一化 MinerU 输出的 Markdown，解决两处与下游 parser 的方言差异：
    1. Word 的加粗标题会被导出成 `# 1 **LangGraph概述**`，`**` 会一路带进
       chapter_name 与【】向量前缀，对检索是纯噪音 → 去掉标题行里的 `**`；
    2. Word 多级列表编号被导出成 5~6 级标题 → 降级为列表项，避免被当作章节边界。
    只改标题形态，不动正文；函数幂等（重复调用结果不变）。
    """
    lines = []
    for line in md_text.splitlines():
        deep = _DEEP_NUMBERED_HEADING_RE.match(line)
        if deep:
            lines.append(f"- {deep.group('num')} {deep.group('text')}".rstrip())
            continue
        m = _HEADING_LINE_RE.match(line)
        if m:
            # 只去 `**`：保留 `__`，避免破坏 `__init__` 这类标题文本
            lines.append(f"{m.group(1)}{m.group(2).replace('**', '')}".rstrip())
        else:
            lines.append(line)
    return "\n".join(lines)


@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state: dict):
    """
    校验是否需要转换并提取转换入参
    :return: (file_format, 文件路径, 任务目录) 或 None（MD 已直读，无需转换）
    """
    file_format = state.get("file_format")
    if state.get("md_text"):
        # MD 已直读，无需转换
        return None

    local_file_path = Path(state.get("local_file_path", ""))
    local_dir = Path(state.get("local_dir", str(local_file_path.parent)))
    return file_format, local_file_path, local_dir


# ---- MinerU 云端转换（4 个步骤：申请 → 上传 → 轮询 → 下载解压）----
@step_log("step_2_apply_mineru_upload_url")
def step_2_apply_mineru_upload_url(src_path: Path, is_ocr: bool = False):
    """
    MinerU 步骤1：POST /api/v4/file-urls/batch 申请上传链接（本地文件模式 files 传 name）
    :param src_path: 待转换文件（pdf / doc / docx）
    :param is_ocr: 是否强制 OCR。PDF 可能为扫描件故默认 True；
                   Word 是原生文本，强制 OCR 会把可结构化解析的排版交给识别模型，故传 False
    :return: (batch_id, upload_url, headers)
    """
    headers = {"Authorization": f"Bearer {mineru_config.api_key}"}
    apply_url = "https://mineru.net/api/v4/file-urls/batch"
    data_name = src_path.name
    apply_resp = requests.post(
        apply_url,
        headers={**headers, "Content-Type": "application/json"},
        json={"enable_formula": True, "enable_table": True,
              "language": "ch", "files": [{"name": data_name, "is_ocr": is_ocr}]},
        timeout=30,
    )
    apply_resp.raise_for_status()
    apply_data = apply_resp.json()
    if apply_data.get("code") != 0:
        raise RuntimeError(f"MinerU 申请任务失败：{apply_data.get('msg')}")
    batch_id = apply_data["data"]["batch_id"]
    upload_url = apply_data["data"]["file_urls"][0]
    return batch_id, upload_url, headers


@step_log("step_3_upload_to_mineru")
def step_3_upload_to_mineru(src_path: Path, upload_url: str, batch_id: str):
    """MinerU 步骤2：PUT 直传文件到签名链接（不设置 Content-Type 头）"""
    with open(src_path, "rb") as f:
        put_resp = requests.put(upload_url, data=f, timeout=300)
    put_resp.raise_for_status()
    logger.info(f"[MinerU] 文件已上传（批次 {batch_id}），等待云端解析...")


@step_log("step_4_poll_mineru_result")
def step_4_poll_mineru_result(batch_id: str, headers: dict) -> str:
    """
    MinerU 步骤3：轮询 /api/v4/extract-results/batch/{batch_id}（最长 10 分钟）
    :return: 结果包下载地址（full_zip_url / md_url）
    """
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
    return md_zip_url


@step_log("step_5_download_and_extract_md")
def step_5_download_and_extract_md(md_zip_url: str, local_dir: Path, src_path: Path) -> str:
    """
    MinerU 步骤4：下载结果包，解压取 .md 与 images/（图片后续由 node_md_img 上传 MinIO + VL 识别）
    :return: Markdown 文本
    """
    import zipfile
    import io
    dl = requests.get(md_zip_url, timeout=120)
    dl.raise_for_status()
    md_text = ""
    with zipfile.ZipFile(io.BytesIO(dl.content)) as zf:
        for name in zf.namelist():
            if name.endswith(".md"):
                md_text = zf.read(name).decode("utf-8", errors="ignore")
            elif "/images/" in name or name.startswith("images/"):
                # 图片保存到任务目录 images/（与 MD 同级，node_md_img 按该目录扫描）
                img_name = Path(name).name
                if not img_name:
                    continue
                img_dir = local_dir / "images"
                img_dir.mkdir(parents=True, exist_ok=True)
                (img_dir / img_name).write_bytes(zf.read(name))
    if not md_text:
        raise RuntimeError("MinerU 结果包中无 .md 文件")
    # 归一化 MinerU 与下游 parser 的方言差异（标题加粗标记 / 深层编号标题）
    md_text = _normalize_mineru_markdown(md_text)
    # 落盘备份（MD 本体）
    (local_dir / f"{src_path.stem}.md").write_text(md_text, encoding="utf-8")
    return md_text


@node_log("node_format_convert")
def node_format_convert(state: dict) -> dict:
    """
    格式归一化节点：pdf / doc / docx → Markdown（md 已在 node_entry 直读）
    Word 与 PDF 统一走 MinerU，确保图片随文档一起产出（images/ + 相对引用）
    :param state: 导入图状态
    :return: 状态更新（md_text / normalized_md_path / format_convert_error）
    """
    # 1. 校验并获取转换入参（MD 已直读时无需本节点）
    data = step_1_validate_and_get_data(state)
    if data is None:
        return {}
    file_format, local_file_path, local_dir = data

    try:
        if file_format not in _MINERU_SUPPORTED_FORMATS:
            return {"format_convert_error": f"未知格式，无法归一化：{file_format}"}

        # 2. 申请上传链接（PDF 可能为扫描件 → OCR；Word 原生文本 → 不启用 OCR）
        is_ocr = file_format == "pdf"
        batch_id, upload_url, headers = step_2_apply_mineru_upload_url(local_file_path, is_ocr)

        # 3. 上传 → 轮询 → 下载解压（PDF / Word 共用同一套 MinerU 流程）
        step_3_upload_to_mineru(local_file_path, upload_url, batch_id)
        md_zip_url = step_4_poll_mineru_result(batch_id, headers)
        md_text = step_5_download_and_extract_md(md_zip_url, local_dir, local_file_path)

        # MinerU 转换产物已落盘（MD + images/），记录路径供 node_md_img 处理图片
        md_path = local_dir / f"{local_file_path.stem}.md"
        logger.info(f"{file_format} 归一化完成（MinerU）：{len(md_text)} 字符，"
                    f"图片目录：{local_dir / 'images'}")
        return {"md_text": md_text, "normalized_md_path": str(md_path)}

    except Exception as e:
        logger.exception("格式转换失败")
        return {"format_convert_error": f"格式转换失败：{e}"}
