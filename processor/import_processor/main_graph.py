"""
导入主图
对应 PRD V1.3 3.2 导入流程：
node_entry（校验 + MD 直读 + 内容类型路由）
  → node_format_convert（pdf/docx → MD 归一化；md 跳过）
  ├─ course_catalog → node_course_parser
  ├─ question_bank → node_question_parser
  └─ course_doc → 预留（3.2.0 通用分块）
→ node_bge_embedding → node_import_milvus
校验/转换失败 → 直接 END（错误信息在状态中，任务标记 failed）
"""
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from common.logging.logger import logger
from processor.import_processor.state import ImportGraphState
from processor.import_processor.nodes.node_entry import node_entry
from processor.import_processor.nodes.node_format_convert import node_format_convert
from processor.import_processor.nodes.node_course_parser import node_course_parser
from processor.import_processor.nodes.node_question_parser import node_question_parser
from processor.import_processor.nodes.node_course_doc_parser import node_course_doc_parser
from processor.import_processor.nodes.node_md_img import node_md_img
from processor.import_processor.nodes.node_bge_embedding import node_bge_embedding
from processor.import_processor.nodes.node_import_milvus import node_import_milvus

# 环境变量初始化（必须在配置读取前执行）
load_dotenv()

# ===================== 1. 初始化状态图 =====================
workflow = StateGraph(ImportGraphState)

# ===================== 2. 注册业务节点 =====================
workflow.add_node("node_entry", node_entry)
workflow.add_node("node_format_convert", node_format_convert)
workflow.add_node("node_md_img", node_md_img)
workflow.add_node("node_course_parser", node_course_parser)
workflow.add_node("node_question_parser", node_question_parser)
workflow.add_node("node_course_doc_parser", node_course_doc_parser)
workflow.add_node("node_bge_embedding", node_bge_embedding)
workflow.add_node("node_import_milvus", node_import_milvus)

# ===================== 3. 入口与条件路由 =====================
workflow.set_entry_point("node_entry")
workflow.add_edge("node_entry", "node_format_convert")
# 图片处理：VL 识别 + MinIO 上传 + MD 引用替换（仅 PDF 链路有本地图片，其余格式自动跳过）
workflow.add_edge("node_format_convert", "node_md_img")


def route_after_convert(state: ImportGraphState) -> str:
    """图片处理后按 content_type 路由解析分支；转换失败直接终止"""
    if state.get("format_convert_error"):
        return END
    content_type = state.get("content_type")
    if content_type == "course_catalog":
        return "node_course_parser"
    if content_type == "question_bank":
        return "node_question_parser"
    if content_type == "course_doc":
        return "node_course_doc_parser"
    return END


workflow.add_conditional_edges(
    "node_md_img",
    route_after_convert,
    {
        "node_course_parser": "node_course_parser",
        "node_question_parser": "node_question_parser",
        "node_course_doc_parser": "node_course_doc_parser",
        END: END,
    },
)

# ===================== 4. 顺序边 =====================
workflow.add_edge("node_course_parser", "node_bge_embedding")
workflow.add_edge("node_question_parser", "node_bge_embedding")
workflow.add_edge("node_course_doc_parser", "node_bge_embedding")
workflow.add_edge("node_bge_embedding", "node_import_milvus")
workflow.add_edge("node_import_milvus", END)

# ===================== 5. 编译 =====================
kb_import_app = workflow.compile()


if __name__ == "__main__":
    import json
    import uuid

    logger.info("===== 导入图冒烟测试（course_catalog） =====")
    init_state = {
        "task_id": str(uuid.uuid4()),
        "local_dir": "/Users/zdy/Downloads/教育/数据",
        "local_file_path": "/Users/zdy/Downloads/教育/数据/课程介绍.md",
        "source_file": "课程介绍.md",
        "source_path": "/Users/zdy/Downloads/教育/数据/课程介绍.md",
        "content_type": "course_catalog",
    }
    final_state = kb_import_app.invoke(init_state)
    logger.info(
        f"最终结果：系列 {len(final_state.get('course_series_records', []))} 条，"
        f"模块 {len(final_state.get('course_module_records', []))} 条，"
        f"入库结果 {json.dumps(final_state.get('import_result', {}), ensure_ascii=False)}"
    )
    logger.info("===== 测试结束 =====")
