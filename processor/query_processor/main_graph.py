"""
查询主图
对应 PRD V1.3 3.6 双通道架构：
node_intent_route（意图识别 + 实体抽取）
  ├─ 目录型意图 → node_structured_query → node_answer_output
  ├─ 聚合统计型 → node_structured_query（拉取命中集）→ node_aggregate → node_answer_output
  ├─ 内容型意图 → 并行三路召回 → node_rrf → node_rerank → node_answer_output
  └─ 意图不明（含反问 answer）→ node_answer_output（透传反问）
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

# ===================== 1. 初始化状态图 =====================
workflow = StateGraph(QueryGraphState)

# ===================== 2. 注册业务节点 =====================
workflow.add_node("node_query_rewrite", node_query_rewrite)
workflow.add_node("node_intent_route", node_intent_route)
workflow.add_node("node_structured_query", node_structured_query)
workflow.add_node("node_aggregate", node_aggregate)
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
    - aggregate → 结构化查询拉取命中集 → 聚合
    - content → 语义通道（三路并行召回）
    """
    if state.get("intent") == "ambiguous":
        return "node_answer_output"
    if state.get("intent") == "catalog":
        return "node_structured_query"
    if state.get("intent") == "aggregate":
        return "node_aggregate"
    return "node_search_embedding"


workflow.add_conditional_edges(
    "node_intent_route",
    route_after_intent,
    {
        "node_answer_output": "node_answer_output",
        "node_structured_query": "node_structured_query",
        "node_aggregate": "node_aggregate",
        "node_search_embedding": "node_search_embedding",
    },
)


def route_after_aggregate(state: QueryGraphState) -> str:
    """聚合节点后的路由：白名单反问（已有 answer）直接输出，否则输出统计结果"""
    if state.get("answer"):
        return "node_answer_output"
    return "node_answer_output"


workflow.add_conditional_edges(
    "node_aggregate",
    route_after_aggregate,
    {"node_answer_output": "node_answer_output"},
)

# ===================== 4. 通道内顺序边 =====================
# 结构化通道
workflow.add_edge("node_structured_query", "node_answer_output")

# 语义通道：三路并行 → RRF 融合 → 重排 → 输出
workflow.add_edge("node_search_embedding", "node_rrf")
workflow.add_edge("node_search_embedding_hyde", "node_rrf")
workflow.add_edge("node_question_filter", "node_rrf")
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
