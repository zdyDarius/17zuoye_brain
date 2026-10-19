"""
查询主图
对应 PRD V1.3 3.6 双通道架构：
node_query_rewrite（多轮指代消解）
  ↓
node_intent_route（意图识别 + 实体抽取）
  ├─ ambiguous  → node_answer_output（透传反问）
  ├─ catalog    → node_structured_query → node_answer_output
  ├─ aggregate  → node_aggregate（自取命中集做 count/group-by）→ node_answer_output
  └─ content    → node_semantic_fanout（扇出）
                    ├─ node_search_embedding（原查询混合召回）
                    ├─ node_search_embedding_hyde（HyDE 假设答案召回）
                    └─ node_question_filter（实体预过滤召回）
                          ↓ 三路并行完成后汇聚
                       node_rrf（RRF 融合）→ node_rerank（精排）→ node_answer_output

说明：三路召回必须从同一个扇出节点并行进入，并用「多源 add_edge」让 node_rrf
等待全部上游完成；若只给 hyde/filter 配出边而没有入边，节点将永远不会被调度。
"""
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from common.logging.logger import logger
from processor.query_processor.state import QueryGraphState
from processor.query_processor.nodes.node_query_rewrite import node_query_rewrite
from processor.query_processor.nodes.node_intent_route import node_intent_route
from processor.query_processor.nodes.node_structured_query import node_structured_query
from processor.query_processor.nodes.node_aggregate import node_aggregate
from processor.query_processor.nodes.node_search_embedding import (
    node_search_embedding, node_search_embedding_hyde, node_question_filter, node_rrf,
)
from processor.query_processor.nodes.node_rerank import node_rerank
from processor.query_processor.nodes.node_answer_output import node_answer_output

# 环境变量初始化（必须在配置读取前执行）
load_dotenv()


def node_semantic_fanout(state: QueryGraphState) -> dict:
    """
    语义通道扇出节点（空操作）：
    条件路由只能命中单个目标，用本节点把「content」分流成三路并行分支，
    自身不读写任何状态字段。
    """
    return {}


# ===================== 1. 初始化状态图 =====================
workflow = StateGraph(QueryGraphState)

# ===================== 2. 注册业务节点 =====================
workflow.add_node("node_query_rewrite", node_query_rewrite)
workflow.add_node("node_intent_route", node_intent_route)
workflow.add_node("node_structured_query", node_structured_query)
workflow.add_node("node_aggregate", node_aggregate)
workflow.add_node("node_semantic_fanout", node_semantic_fanout)
workflow.add_node("node_search_embedding", node_search_embedding)
workflow.add_node("node_search_embedding_hyde", node_search_embedding_hyde)
workflow.add_node("node_question_filter", node_question_filter)
workflow.add_node("node_rrf", node_rrf)
workflow.add_node("node_rerank", node_rerank)
workflow.add_node("node_answer_output", node_answer_output)

# ===================== 3. 入口与条件路由 =====================
# 入口：先改写（多轮指代消解）再意图路由
workflow.set_entry_point("node_query_rewrite")
workflow.add_edge("node_query_rewrite", "node_intent_route")


def route_after_intent(state: QueryGraphState) -> str:
    """意图分流（PRD 3.6）：
    - ambiguous（含反问 answer）→ 直接输出
    - catalog → 结构化通道
    - aggregate → 聚合通道（节点内部自取命中集）
    - content → 语义通道扇出（三路并行召回）
    """
    if state.get("intent") == "ambiguous":
        return "node_answer_output"
    if state.get("intent") == "catalog":
        return "node_structured_query"
    if state.get("intent") == "aggregate":
        return "node_aggregate"
    return "node_semantic_fanout"


workflow.add_conditional_edges(
    "node_intent_route",
    route_after_intent,
    {
        "node_answer_output": "node_answer_output",
        "node_structured_query": "node_structured_query",
        "node_aggregate": "node_aggregate",
        "node_semantic_fanout": "node_semantic_fanout",
    },
)

# ===================== 4. 通道内顺序边 =====================
# 结构化通道
workflow.add_edge("node_structured_query", "node_answer_output")

# 聚合通道（白名单反问与统计结果都由 answer_output 统一透传/组装）
workflow.add_edge("node_aggregate", "node_answer_output")

# 语义通道：扇出 → 三路并行 → 汇聚到 RRF → 重排 → 输出
for _semantic_node in ("node_search_embedding", "node_search_embedding_hyde", "node_question_filter"):
    workflow.add_edge("node_semantic_fanout", _semantic_node)
# 多源汇聚：node_rrf 必须等三路全部完成后再执行
workflow.add_edge(
    ["node_search_embedding", "node_search_embedding_hyde", "node_question_filter"],
    "node_rrf",
)
workflow.add_edge("node_rrf", "node_rerank")
workflow.add_edge("node_rerank", "node_answer_output")

# 输出 → 结束
workflow.add_edge("node_answer_output", END)

# ===================== 5. 编译 =====================
query_app = workflow.compile()


if __name__ == "__main__":
    import uuid

    logger.info("===== 查询图冒烟测试 =====")
    test_query = "通用程序设计题库有哪些单选题"
    from processor.query_processor.state import create_query_default_state

    init_state = create_query_default_state(
        session_id=str(uuid.uuid4()),
        original_query=test_query,
    )
    final_state = query_app.invoke(init_state)
    print(f"问题：{test_query}")
    print(f"意图：{final_state.get('intent')}，实体：{final_state.get('entities')}")
    print(f"答案：{final_state.get('answer', '')[:500]}")
    print(f"引用：{len(final_state.get('references', []))} 条")
    logger.info("===== 测试结束 =====")
