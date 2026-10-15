"""
查询图状态定义
对应 PRD V1.3 3.6：双通道架构（结构化通道 + 语义通道 + 聚合通道），意图路由分流
"""
from typing import TypedDict, List, Dict, Any, Optional


class QueryGraphState(TypedDict, total=False):
    # ---- 请求基础信息 ----
    session_id: str                  # 会话ID（多轮对话与历史记录关联）
    original_query: str              # 用户原始问题
    rewritten_query: str             # 结合历史改写后的问题（指代消解）
    is_stream: bool                  # 是否流式返回
    history: List[Dict[str, Any]]    # 最近 6 轮（12 条）历史消息（改写输入）

    # ---- 意图路由（node_intent_route）----
    intent: str                      # catalog / aggregate / content / ambiguous
    entities: Dict[str, Any]         # 抽取实体：question_type_cn/bank_name/course_name/series_name/
                                     # course_category/target_audience/suitable_grade/learning_goal
                                     # question_code/course_code（编码精确查询）
    route_reason: str                # 路由决策说明（日志与调试用）

    # ---- 结构化通道（node_structured_query）----
    structured_results: List[Dict[str, Any]]   # 命中记录列表（可穷举）
    structured_total: int                      # 命中总数（截断前）
    structured_truncated: bool                # 是否因规模保护截断

    # ---- 聚合通道（node_aggregate）----
    aggregate_result: Dict[str, Any]           # {value: 数值, groups: [{key, count}]}
    aggregate_field: str                       # 实际聚合字段（白名单校验后）

    # ---- 语义通道（node_search_embedding / hyde / rrf / rerank）----
    embedding_hits: List[Dict[str, Any]]       # 原查询混合向量召回
    hyde_hits: List[Dict[str, Any]]            # HyDE 假设答案召回
    fused_results: List[Dict[str, Any]]        # RRF 融合后结果
    reranked_results: List[Dict[str, Any]]     # 重排后 Top-K
    search_collections: List[str]              # 语义检索的 Collection 范围（按实体过滤）

    # ---- 答案输出（node_answer_output）----
    answer: str                      # 最终答案文本
    references: List[Dict[str, Any]] # 引用列表（编码级来源）
    degraded: bool                   # 降级标记（LLM 不可用等场景）
    error: Optional[str]             # 全局错误信息


def create_query_default_state(
        session_id: str,
        original_query: str,
        is_stream: bool = False,
        history: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    创建查询图初始状态
    :param session_id: 会话ID
    :param original_query: 用户原始问题
    :param is_stream: 是否流式
    :param history: 历史消息（多轮改写输入，可空）
    :return: 初始状态字典
    """
    return {
        "session_id": session_id,
        "original_query": original_query,
        "rewritten_query": original_query,
        "is_stream": is_stream,
        "history": history or [],
        "intent": None,
        "entities": {},
        "structured_results": [],
        "structured_total": 0,
        "structured_truncated": False,
        "aggregate_result": {},
        "aggregate_field": None,
        "embedding_hits": [],
        "hyde_hits": [],
        "fused_results": [],
        "reranked_results": [],
        "search_collections": ["edu_course_series", "edu_course_module", "edu_question", "edu_course_doc"],
        "answer": "",
        "references": [],
        "degraded": False,
        "error": None,
    }
