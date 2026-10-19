"""
结构化查询节点
对应 PRD V1.3 3.6 目录型意图：
- 编码精确过滤（question_code / course_code / series_code）
- 名称模糊召回（Milvus like 查询）
- 标量过滤（question_type / bank_code / course_category / suitable_grade / target_audience / learning_goal）
- 结果规模保护（PRD：分组明细/列表超过 100 条截断并提示）
"""
from common.logging.logger import logger, node_log
from utils.clients.milvus_utils import get_milvus_client, count_rows
from utils.escape_milvus_string_utils import escape_like_term
from processor.import_processor.nodes.node_question_parser import QUESTION_TYPE_MAP

# 结果展示上限（超出的返回前 N 条 + 总数提示）
RESULT_LIMIT = 100

# 语义实体字段 -> 集合与过滤字段映射
_Q_TYPE_FIELD = "question_type"


def _escape_like(term: str) -> str:
    """Milvus like 通配符转义（% 与 _），公共实现见 escape_milvus_string_utils"""
    return escape_like_term(term)


def _build_question_filter(entities: dict) -> str:
    """构造 question 集合的标量过滤表达式"""
    conds = []
    if entities.get("question_type_cn"):
        type_en = QUESTION_TYPE_MAP.get(entities["question_type_cn"])
        if type_en:
            conds.append(f'{_Q_TYPE_FIELD} == "{type_en}"')
    if entities.get("bank_name"):
        conds.append(f'bank_name like "%{_escape_like(entities["bank_name"])}%"')
    if entities.get("bank_code"):
        conds.append(f'bank_code == "{_escape_like(entities["bank_code"])}"')
    if entities.get("question_code"):
        conds.append(f'question_code == "{entities["question_code"]}"')
    return " and ".join(conds)


def _build_series_filter(entities: dict) -> str:
    """构造 course_series 集合的过滤表达式"""
    conds = []
    if entities.get("series_name"):
        conds.append(f'series_name like "%{_escape_like(entities["series_name"])}%"')
    if entities.get("course_category"):
        conds.append(f'course_category like "%{_escape_like(entities["course_category"])}%"')
    if entities.get("target_audience"):
        conds.append(f'target_audience like "%{_escape_like(entities["target_audience"])}%"')
    if entities.get("suitable_grade"):
        conds.append(f'suitable_grade like "%{_escape_like(entities["suitable_grade"])}%"')
    if entities.get("learning_goal"):
        conds.append(f'learning_goal like "%{_escape_like(entities["learning_goal"])}%"')
    return " and ".join(conds)


def _build_module_filter(entities: dict) -> str:
    """构造 course_module 集合的过滤表达式"""
    conds = []
    if entities.get("course_name"):
        conds.append(f'course_name like "%{_escape_like(entities["course_name"])}%"')
    if entities.get("course_code"):
        conds.append(f'course_code == "{entities["course_code"]}"')
    if entities.get("series_name"):
        conds.append(f'series_name like "%{_escape_like(entities["series_name"])}%"')
    return " and ".join(conds)


# 各集合的返回字段
_OUTPUT_FIELDS = {
    "edu_question": ["question_code", "bank_name", "bank_code", "question_type",
                     "content", "answer", "answer_raw", "analysis", "source_file"],
    "edu_course_series": ["series_code", "series_name", "course_category", "description",
                          "target_audience", "learning_goal", "suitable_grade", "content", "source_file"],
    "edu_course_module": ["course_code", "course_name", "series_code", "series_name",
                          "duration", "credit_hours", "description", "content", "source_file"],
}


def _count_hits(client, collection: str, expr: str):
    """
    取真实命中总数（公共实现见 milvus_utils.count_rows）
    :return: int 命中总数；失败返回 None（调用方回退到按拉取行数估算）
    """
    return count_rows(client, collection, expr)


@node_log("node_structured_query")
def node_structured_query(state: dict) -> dict:
    """
    结构化查询节点：目录型意图的标量过滤查询（可穷举，不依赖向量）
    按实体字段决定查询目标集合：
    - question_code/题型/题库实体 → edu_question
    - course_code/course_name → edu_course_module
    - series_name/分类/人群/年级/目标 → edu_course_series
    :param state: 查询图状态（intent=catalog）
    :return: 状态更新（structured_results / structured_total / structured_truncated）
    """
    entities = state.get("entities", {})
    client = get_milvus_client()
    if client is None:
        return {"error": "Milvus 客户端不可用"}

    # ---- 目标集合路由 ----
    query_entity = entities.get("question_code") or entities.get("question_type_cn") or entities.get("bank_name")
    module_entity = entities.get("course_code") or entities.get("course_name")
    series_entity = entities.get("series_name") or entities.get("course_category") \
        or entities.get("target_audience") or entities.get("suitable_grade") or entities.get("learning_goal")

    plans = []  # (collection, filter_expr)
    if query_entity:
        plans.append(("edu_question", _build_question_filter(entities)))
    if module_entity:
        plans.append(("edu_course_module", _build_module_filter(entities)))
    if series_entity and not query_entity and not module_entity:
        plans.append(("edu_course_series", _build_series_filter(entities)))
        # 系列查询附带模块列表（详情场景：series_name 命中时也查模块）
        if entities.get("series_name"):
            plans.append(("edu_course_module", _build_module_filter(entities)))

    if not plans:
        logger.warning("结构化查询：无可识别实体，返回空")
        return {"structured_results": [], "structured_total": 0, "structured_truncated": False}

    # ---- 执行查询 ----
    results = []
    total_count = 0          # 真实命中总数（各 plan 真实总数之和，count(*) 失败时按拉取行数估算）
    truncated = False        # 任一 plan 命中数超过展示上限即视为截断
    for collection, expr in plans:
        if not expr:
            continue
        try:
            # 真实总数：count(*) 聚合（不受 limit 截断影响）
            plan_total = _count_hits(client, collection, expr)
            rows = client.query(
                collection_name=collection,
                filter=expr,
                output_fields=_OUTPUT_FIELDS[collection],
                limit=RESULT_LIMIT,
            )
            if plan_total is None:
                # 回退口径：拿到的就是全部（无法判断是否还有更多）
                plan_total = len(rows)
            total_count += plan_total
            plan_truncated = plan_total > RESULT_LIMIT
            truncated = truncated or plan_truncated
            for r in rows:
                r["_collection"] = collection
            results.extend(rows)
            logger.info(f"结构化查询[{collection}] filter={expr} → 命中 {plan_total} 条，"
                        f"返回 {len(rows)} 条{'（已截断）' if plan_truncated else ''}")
        except Exception as e:
            logger.error(f"结构化查询失败[{collection}]：{e}")

    return {
        "structured_results": results,
        "structured_total": total_count,
        "structured_truncated": truncated,
    }
