"""
导入图入口节点
对应 PRD V1.3 3.2：
- 入口校验（content_type 枚举校验、文件白名单、体积上限）
- 格式归一化路由：md 直接读取 / pdf、doc、docx 交由 node_format_convert（MinerU 云端转换）
- 内容类型路由结果写入状态（course_catalog / question_bank / course_doc）
"""
from pathlib import Path

from common.logging.logger import logger, node_log, step_log

# 上传内容类型合法枚举（PRD 4.1：接口层枚举）
VALID_CONTENT_TYPES = {"course_catalog", "question_bank", "course_doc"}
# 文件格式白名单（PRD 3.2：.md / .pdf / .docx；.doc 由 MinerU 直接解析）
VALID_EXTENSIONS = {".md", ".pdf", ".docx", ".doc"}
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
    content_type = state.get("content_type", "")
    local_file_path = state.get("local_file_path", "")

    # ---- 1. content_type 枚举校验 ----
    error = step_1_validate_content_type(content_type)
    if error:
        return {"format_convert_error": error}

    # ---- 2. 文件存在性与白名单校验 ----
    path_obj, ext, error = step_2_validate_file(local_file_path)
    if error:
        return {"format_convert_error": error}

    # ---- 3. 体积校验 ----
    error = step_3_validate_file_size(path_obj)
    if error:
        return {"format_convert_error": error}

    # ---- 4. 格式归一化（md 直读 / pdf、docx 交由下游节点转换）----
    return step_4_normalize_by_format(path_obj, ext)


@step_log("step_1_validate_content_type")
def step_1_validate_content_type(content_type: str):
    """
    校验 content_type 是否为合法枚举
    :return: 错误信息 or None（None 表示校验通过）
    """
    if content_type not in VALID_CONTENT_TYPES:
        return f"非法 content_type：{content_type}（合法值：{sorted(VALID_CONTENT_TYPES)}）"
    return None


@step_log("step_2_validate_file")
def step_2_validate_file(local_file_path: str):
    """
    校验文件存在性与格式白名单
    :return: (Path 对象, 小写后缀, 错误信息 or None)
    """
    path_obj = Path(local_file_path)
    if not path_obj.exists():
        return path_obj, "", f"文件不存在：{local_file_path}"
    ext = path_obj.suffix.lower()
    if ext not in VALID_EXTENSIONS:
        return path_obj, ext, f"不支持的文件类型 {ext}（白名单：.md/.pdf/.doc/.docx）"
    return path_obj, ext, None


@step_log("step_3_validate_file_size")
def step_3_validate_file_size(path_obj: Path):
    """
    校验文件体积是否超过上限
    :return: 错误信息 or None（None 表示校验通过）
    """
    file_size = path_obj.stat().st_size
    if file_size > MAX_FILE_SIZE:
        return f"文件超过 50MB 上限：{file_size / 1024 / 1024:.1f}MB"
    return None


@step_log("step_4_normalize_by_format")
def step_4_normalize_by_format(path_obj: Path, ext: str) -> dict:
    """
    按文件格式决定归一化路径：md 直读，pdf/docx 标记格式交由 node_format_convert
    :return: 状态更新字典
    """
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
        # docx → node_format_convert（MinerU 云端 API 转换）
        return {"file_format": "docx"}

    if ext == ".doc":
        # doc（老二进制格式）→ node_format_convert（MinerU 云端 API 转换）
        return {"file_format": "doc"}

    return {"format_convert_error": f"未知的文件格式：{ext}"}
