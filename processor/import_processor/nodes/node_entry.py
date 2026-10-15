"""
导入图入口节点
对应 PRD V1.3 3.2：
- 入口校验（content_type 枚举校验、文件白名单、体积上限）
- 格式归一化：md 直接读取 / pdf（MinerU 转换，预留）/ docx（python-docx 转换，预留）
- 内容类型路由结果写入状态（course_catalog / question_bank / course_doc）
"""
from pathlib import Path

from common.logging.logger import logger, node_log

# 上传内容类型合法枚举（PRD 4.1：接口层枚举）
VALID_CONTENT_TYPES = {"course_catalog", "question_bank", "course_doc"}
# 文件格式白名单（PRD 3.2：.md / .pdf / .docx；.doc 老格式拒绝）
VALID_EXTENSIONS = {".md", ".pdf", ".docx"}
# 单文件体积上限（PRD 3.2 上传校验规则）
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


@node_log("node_entry")
def node_entry(state: dict) -> dict:
    """
    导入图入口节点：校验 + 格式归一化 + 路由标记
    校验失败不抛异常中断服务，写入 format_convert_error 由图路由直接终止该任务
    :param state: 导入图状态
    :return: 状态更新（file_format / normalized_md_path / md_text / format_convert_error）
    """
    local_file_path = state.get("local_file_path", "")
    content_type = state.get("content_type", "")

    # ---- 1. content_type 枚举校验 ----
    if content_type not in VALID_CONTENT_TYPES:
        return {"format_convert_error": f"非法 content_type：{content_type}（合法值：{sorted(VALID_CONTENT_TYPES)}）"}

    # ---- 2. 文件存在性与白名单校验 ----
    path_obj = Path(local_file_path)
    if not path_obj.exists():
        return {"format_convert_error": f"文件不存在：{local_file_path}"}
    ext = path_obj.suffix.lower()
    if ext not in VALID_EXTENSIONS:
        return {"format_convert_error": f"不支持的文件类型 {ext}（白名单：.md/.pdf/.docx；.doc 请另存为 .docx）"}

    # ---- 3. 体积校验 ----
    file_size = path_obj.stat().st_size
    if file_size > MAX_FILE_SIZE:
        return {"format_convert_error": f"文件超过 50MB 上限：{file_size / 1024 / 1024:.1f}MB"}

    # ---- 4. 格式归一化 ----
    if ext == ".md":
        # MD 直接读取
        try:
            md_text = path_obj.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # 尝试 GBK 兜底（Windows 导出的中文文件常见）
            try:
                md_text = path_obj.read_text(encoding="gbk")
            except Exception as e:
                return {"format_convert_error": f"MD 文件编码无法解析（utf-8/gbk 均失败）：{e}"}
        return {
            "file_format": "md",
            "normalized_md_path": str(path_obj),
            "md_text": md_text,
        }

    if ext == ".pdf":
        # PDF → node_format_convert（MinerU 云端 API 转换）
        return {"file_format": "pdf"}

    if ext == ".docx":
        # docx → node_format_convert（python-docx 转换）
        return {"file_format": "docx"}

    return {"format_convert_error": f"未知的文件格式：{ext}"}
