"""
聚合统计节点
对应 PRD V1.3 3.6（复审 N3 新增）：
- 实现口径：Milvus 标量过滤拉取命中集 → 应用层 count / group-by（不依赖 Milvus 原生聚合）
- 可聚合字段白名单：question_type / bank_code / course_category / suitable_grade / target_audience / learning_goal / answer
- 聚合键不在白名单 → 反问澄清
- 结果规模保护：分组明细超 100 行截断
- 判断题答案分布依赖答案归一化（answer ∈ true/false）
"""
from collections import Counter

from common.logging.logger import logger, node_log, step_log
from utils.clients.milvus_utils import get_milvus_client, count_rows
from utils.escape_milvus_string_utils import escape_like_term
from processor.import_processor.nodes.node_question_parser import QUESTION_TYPE_MAP

# 可聚合字段白名单（PRD 3.6 N3：聚合键必须在白名单内）
AGGREGATABLE_FIELDS = {
    "question_type", "bank_code", "course_category", "suitable_grade",
    "target_audience", "learning_goal", "answer",
}

# 聚合字段 -> 目标集合
_AGG_TARGET = {
    "question_type": ("edu_question", "question_type", "question_type"),
    "bank_code": ("edu_question", "bank_code", "bank_name"),
    "answer": ("edu_question", "answer", "answer"),
    "course_category": ("edu_course_series", "course_category", "series_name"),
    "suitable_grade": ("edu_course_series", "suitable_grade", "series_name"),
    "target_audience": ("edu_course_series", "target_audience", "series_name"),
    "learning_goal": ("edu_course_series", "learning_goal", "series_name"),
}

# 分组明细展示上限（PRD：超过 100 行截断）
GROUP_LIMIT = 100

# 分组统计时的拉取上限（分组需要逐行拆多值字段，无法用 count(*) 替代）
GROUP_FETCH_LIMIT = 16384

# 课程系列集合的实体过滤字段（与 node_structured_query._build_series_filter 口径一致）
_SERIES_ENTITY_FIELDS = ("series_name", "course_category", "target_audience",
                         "suitable_grade", "learning_goal")


def _build_series_filter(entities: dict) -> str:
    """构造 course_series 集合的实体过滤表达式（分组统计时复用，避免统计全库）"""
    conds = []
    for key in _SERIES_ENTITY_FIELDS:
        val = entities.get(key)
        if val:
            conds.append(f'{key} like "%{escape_like_term(val)}%"')
    return " and ".join(conds)


@node_log("node_aggregate")
def node_aggregate(state: dict) -> dict:
    """
    聚合统计节点：count / group_by / distribution
    - aggregate_op=count：按 entities 过滤后计数（如"通用程序设计题库有多少道单选题"）
    - aggregate_op=group_by/distribution：按 aggregate_field 分组统计（如"题库的题量分布"）
    :param state: 查询图状态（intent=aggregate）
    :return: 状态更新（aggregate_result / aggregate_field，非法字段时 answer 反问）
    """
    # 1. 校验并获取聚合参数与客户端
    data = step_1_validate_and_get_data(state)
    if data is None:
        return {"error": "Milvus 客户端不可用"}
    entities, agg_field, agg_op, client = data

    # 2. 白名单校验（N3：不在白名单 → 反问澄清）
    whitelist_reply = step_2_check_field_whitelist(agg_field)
    if whitelist_reply:
        return whitelist_reply

    # 3. 组装过滤条件（count 场景：复用题型/题库等实体过滤）
    filter_expr = step_3_build_filter_expr(entities, agg_field)

    # 4. 课程数量类问题纠正（PRD 场景二：系列详情含模块清单）
    course_reply = step_4_try_course_count(client, entities, state)
    if course_reply:
        return course_reply

    # 5. 纯计数场景：count(*) 精确统计
    if agg_op == "count" or not agg_field:
        return step_5_count_questions(client, filter_expr, entities)

    # 6. 分组/分布场景：拉取后应用层 group-by
    return step_6_group_by_field(client, agg_field, filter_expr, entities)


@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state: dict):
    """
    获取聚合实体、聚合字段/操作（缺省推断）并校验客户端
    :return: (entities, agg_field, agg_op, client) 或 None（客户端不可用）
    """
    entities = state.get("entities", {})
    agg_field = entities.get("aggregate_field")
    # 缺省操作推断：给了 aggregate_field 说明用户要按维度看，应走分组；
    # 只有"有多少/几道"这类纯计数才默认 count。否则 LLM 漏填 aggregate_op 时
    # 会退化成只回一个总数，答案生成阶段容易编造分布明细。
    agg_op = (entities.get("aggregate_op") or ("group_by" if agg_field else "count")).lower()

    client = get_milvus_client()
    if client is None:
        logger.error("Milvus 客户端不可用，聚合统计终止")
        return None

    logger.info(f"聚合统计入参：field={agg_field}，op={agg_op}，entities={entities}")
    return entities, agg_field, agg_op, client


@step_log("step_2_check_field_whitelist")
def step_2_check_field_whitelist(agg_field: str):
    """
    聚合字段白名单校验
    :return: 澄清话术字典（非法字段）或 None（校验通过）
    """
    if agg_field and agg_field not in AGGREGATABLE_FIELDS:
        logger.warning(f"聚合字段 {agg_field} 不在白名单，返回澄清话术")
        return {
            "aggregate_field": agg_field,
            "answer": (
                f"暂不支持按「{agg_field}」统计。可统计的维度包括："
                f"题型、题库、课程分类、适合年级、适合人群、学习目标"
                f"（判断题还可统计答案分布）。您想按哪个维度统计？"
            ),
        }
    return None


@step_log("step_3_build_filter_expr")
def step_3_build_filter_expr(entities: dict, agg_field: str) -> str:
    """
    组装聚合统计的过滤表达式
    :return: 过滤表达式（空串表示无实体过滤）
    """
    conds = []
    if entities.get("question_type_cn"):
        type_en = QUESTION_TYPE_MAP.get(entities["question_type_cn"])
        if type_en:
            conds.append(f'question_type == "{type_en}"')
    if entities.get("bank_name"):
        conds.append(f'bank_name like "%{escape_like_term(entities["bank_name"])}%"')
    if entities.get("question_code"):
        conds.append(f'question_code == "{entities["question_code"]}"')
    # 答案分布只对判断题有业务意义：answer 归一化为 true/false 后才可统计。
    # 未显式指定题型时强制限定 true_false，避免把单选/多选的答案字母（A/B/C/D）混入分布。
    if agg_field == "answer" and not entities.get("question_type_cn"):
        conds.append('question_type == "true_false"')
    filter_expr = " and ".join(conds)
    logger.info(f"聚合过滤表达式：{filter_expr or '（无，统计全库）'}")
    return filter_expr


@step_log("step_4_try_course_count")
def step_4_try_course_count(client, entities: dict, state: dict):
    """
    课程数量类问题误入聚合通道的纠正（PRD 场景二：系列详情含模块清单）
    "某系列里有几门课/包含哪些模块"本质是 catalog 列举；此处若无题型/题库实体
    且原问题含课程词 → 反查课程集合给出数量级答案（模块清单由答案组装输出）
    :return: 课程计数结果字典 或 None（不适用/查询失败，继续走原聚合逻辑）
    """
    original_query = state.get("rewritten_query") or state.get("original_query", "")
    question_words = any(w in original_query for w in ("系列", "课程", "门课", "模块"))
    has_question_entity = bool(entities.get("question_type_cn") or entities.get("bank_name")
                               or entities.get("question_code"))
    if not question_words or has_question_entity:
        return None

    try:
        # 命中总数用 count(*) 精确统计；清单仍取前 50 条用于展示
        if entities.get("series_name"):
            course_filter = f'series_name like "%{escape_like_term(entities["series_name"])}%"'
            course_total = count_rows(client, "edu_course_module", course_filter)
            rows = client.query(
                collection_name="edu_course_module",
                filter=course_filter,
                output_fields=["course_code", "course_name", "series_name"],
                limit=50,
            )
        else:
            course_filter = _build_series_filter(entities) or 'series_code != ""'
            course_total = count_rows(client, "edu_course_series", course_filter)
            rows = client.query(
                collection_name="edu_course_series",
                filter=course_filter,
                output_fields=["series_code", "series_name"],
                limit=50,
            )
        logger.info(f"课程数量类查询命中：{course_total if course_total is not None else len(rows)} 条")
        return {
            "aggregate_field": "course_count",
            "aggregate_result": {
                "value": course_total if course_total is not None else len(rows),
                "course_rows": [{k: r.get(k) for k in ("course_code", "course_name", "series_name") if r.get(k) is not None} for r in rows],
                "entities": entities,
            },
        }
    except Exception as e:
        logger.error(f"课程计数查询失败：{e}")
        return None


@step_log("step_5_count_questions")
def step_5_count_questions(client, filter_expr: str, entities: dict) -> dict:
    """
    纯计数：对 edu_question 过滤计数（count(*) 不受数据量上限影响）
    :return: 状态更新字典
    """
    count_filter = filter_expr or 'question_code != ""'
    try:
        total = count_rows(client, "edu_question", count_filter)
        if total is None:
            # 回退：聚合查询不可用时按行数估算（受拉取上限约束）
            rows = client.query(
                collection_name="edu_question",
                filter=count_filter,
                output_fields=["question_code"],
                limit=GROUP_FETCH_LIMIT,
            )
            total = len(rows)
        logger.info(f"聚合计数完成：{count_filter} → {total}")
        return {
            "aggregate_field": "count",
            "aggregate_result": {
                "value": total,
                "filter": filter_expr or "全部题目",
                "entities": entities,
            },
        }
    except Exception as e:
        logger.error(f"聚合计数失败：{e}")
        return {"error": f"聚合统计失败：{e}"}


@step_log("step_6_group_by_field")
def step_6_group_by_field(client, agg_field: str, filter_expr: str, entities: dict) -> dict:
    """
    分组/分布统计：拉取分组字段值后应用层 group-by（多值字段按逗号拆分归组）
    :return: 状态更新字典
    """
    collection, group_field, label_field = _AGG_TARGET[agg_field]
    try:
        # 分组必须逐行取字段值，但过滤条件不能丢：question 复用题类实体过滤，
        # course_series 复用系列/分类/人群/年级/目标实体过滤（此前只写 field != "" 会统计全库）
        if collection == "edu_question":
            expr = filter_expr or 'question_code != ""'
        else:
            entity_filter = _build_series_filter(entities)
            expr = f'{entity_filter} and {group_field} != ""' if entity_filter else f'{group_field} != ""'
        rows = client.query(
            collection_name=collection,
            filter=expr,
            output_fields=[group_field, label_field],
            limit=GROUP_FETCH_LIMIT,
        )
        # 应用层分组统计（多值字段按逗号/顿号拆分后归组）
        counter = Counter()
        for r in rows:
            val = (r.get(group_field) or "").strip()
            if not val:
                continue
            # 多值字段（适合人群/学习目标/适合年级/课程分类为逗号分隔多值）
            for part in [p.strip() for p in val.replace("，", ",").split(",") if p.strip()]:
                counter[part] += 1

        groups = [{"key": k, "count": v} for k, v in counter.most_common()]
        truncated = len(groups) > GROUP_LIMIT
        logger.info(f"分组统计完成：{agg_field} 命中 {len(rows)} 行 → {len(groups)} 组"
                    f"{'（已截断）' if truncated else ''}")
        return {
            "aggregate_field": agg_field,
            "aggregate_result": {
                "value": len(rows),
                "groups": groups[:GROUP_LIMIT],
                "group_total": len(groups),
                "truncated": truncated,
                "entities": entities,
            },
        }
    except Exception as e:
        logger.error(f"分组统计失败：{e}")
        return {"error": f"聚合统计失败：{e}"}
