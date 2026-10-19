"""
导入图状态定义
对应 PRD V1.3 3.2：解析分支按「内容类型」路由，格式归一化前置
"""
from typing import TypedDict, List, Dict, Any, Optional


class ImportGraphState(TypedDict, total=False):
    # ---- 任务基础信息 ----
    task_id: str                       # 任务唯一ID（uuid）
    local_dir: str                     # 任务工作目录（上传文件所在）
    local_file_path: str               # 上传文件的本地绝对路径
    source_file: str                   # 来源文件名（幂等键组成部分）
    source_path: str                   # 来源路径/资源链接
    content_type: str                  # 上传内容类型：course_catalog / question_bank / course_doc
    project_name: str                  # 项目名称（course_doc 可选：有值则 doc_kind=project；zip 导入默认取包名）
    file_format: str                   # 归一化后识别的文件格式：md / pdf / docx

    # ---- 格式归一化（node_entry）----
    normalized_md_path: str            # 归一化后的 Markdown 文件路径（md 直接引用原文件）
    md_text: str                       # 归一化后的 Markdown 全文
    format_convert_error: Optional[str]  # 格式转换失败原因（失败则流程终止）

    # ---- 解析结果（course_parser / question_parser）----
    # course_catalog 解析产出：系列记录列表（PRD 4.3 Collection 1）
    course_series_records: List[Dict[str, Any]]
    # course_catalog 解析产出：课程模块记录列表（PRD 4.3 Collection 2）
    course_module_records: List[Dict[str, Any]]
    # question_bank 解析产出：题目记录列表（PRD 4.3 Collection 4，含 answer 归一化）
    question_records: List[Dict[str, Any]]
    # 题库元信息（bank_code -> bank_name），供题目记录冗余
    bank_name_map: Dict[str, str]
    # course_doc 解析产出：文档分块记录列表（PRD 4.3 Collection 3）
    course_doc_records: List[Dict[str, Any]]
    # 解析失败的实体清单（编码 + 原因），单实体失败不影响其他实体
    parse_failed_entities: List[Dict[str, Any]]

    # ---- 向量化（node_bge_embedding）----
    # 待向量化记录集合：{collection_name: [record, ...]}
    pending_collections: Dict[str, List[Dict[str, Any]]]
    embedding_error: Optional[str]

    # ---- 入库（node_import_milvus）----
    import_result: Dict[str, Any]      # 各 Collection 插入/更新/跳过统计
    import_error: Optional[str]


def create_import_default_state(
        task_id: str,
        local_dir: str,
        local_file_path: str,
        content_type: str,
        project_name: str = "",
        source_file: str = "",
) -> Dict[str, Any]:
    """
    创建导入图初始状态
    :param task_id: 任务ID
    :param local_dir: 任务工作目录
    :param local_file_path: 上传文件绝对路径
    :param content_type: 内容类型（course_catalog / question_bank / course_doc）
    :param project_name: 项目名称（course_doc 可选，zip 导入时默认取压缩包名）
    :param source_file: 来源文件名（zip 导入时传包内相对路径，便于溯源；默认取文件名）
    :return: 初始状态字典
    """
    import os
    return {
        "task_id": task_id,
        "local_dir": local_dir,
        "local_file_path": local_file_path,
        "source_file": source_file or os.path.basename(local_file_path),
        "source_path": local_file_path,
        "content_type": content_type,
        "project_name": project_name or "",
        "file_format": None,
        "normalized_md_path": None,
        "md_text": "",
        "format_convert_error": None,
        "course_series_records": [],
        "course_module_records": [],
        "question_records": [],
        "course_doc_records": [],
        "bank_name_map": {},
        "parse_failed_entities": [],
        "pending_collections": {},
        "import_result": {},
        "import_error": None,
    }
