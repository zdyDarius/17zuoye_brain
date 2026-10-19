"""
意图路由节点
对应 PRD V1.3 3.6：node_intent_route
- LLM 结构化输出：intent ∈ {catalog, aggregate, content, ambiguous} + entities 实体抽取
- 题型枚举校验（15 种，PRD 4.2）：LLM 输出的题型必须合法，否则置空（触发反问/澄清）
- 意图不明（ambiguous）时直接生成反问 answer，跳过检索
"""
import json
import re

from common.logging.logger import logger, node_log, step_log
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt

# 合法题型中文枚举（与 node_question_parser.QUESTION_TYPE_MAP 的中文键一致，PRD 4.2）
VALID_QUESTION_TYPES_CN = {
    "单选题", "多选题", "判断题", "简答题", "计算题", "编程题",
    "案例分析题", "情境分析题", "公式推导题", "证明题", "阅读理解题",
    "数据分析题", "完形填空题", "翻译题", "材料分析题",
}

# 实体字段白名单（超集校验：LLM 输出只允许这些键）
VALID_ENTITY_KEYS = {
    "question_type_cn", "bank_name", "bank_code", "question_code", "course_code",
    "series_name", "course_name", "course_category", "target_audience",
    "suitable_grade", "learning_goal", "aggregate_field", "aggregate_op",
}


@node_log("node_intent_route")
def node_intent_route(state: dict) -> dict:
    """
    意图路由节点：LLM 抽取意图与实体，写入状态供条件路由分流
    :param state: 查询图状态
    :return: 状态更新（intent / entities / route_reason / [ambiguous 时直接 answer]）
    """
    # 1. 校验并获取改写后问题与历史上下文
    query, original_query, history_text = step_1_validate_and_get_data(state)

    try:
        # 2. 调用 LLM 做意图识别 + 实体抽取（JSON 结构化输出）
        parsed = step_2_call_llm_intent(query, history_text)

        # 3. 意图枚举与实体白名单/题型枚举基础清洗
        intent, entities = step_3_clean_entities(parsed)

        # 4. 实体过抽取与推断防护（题型推断/课程维度噪声/非法聚合字段/序数指代）
        intent, entities, search_collections = step_4_guard_entities(
            intent, entities, original_query, query
        )

        # 5. catalog 意图的系列名实体兜底（按库内系列名反查）
        step_5_fallback_series_entity(intent, entities, query)

        route_reason = parsed.get("route_reason", "")
        logger.info(f"意图路由：intent={intent}，entities={entities}，理由={route_reason}")

        # 6. 组装路由结果（意图不明/题型非法时直接生成澄清话术）
        return step_6_build_route_result(intent, entities, route_reason, search_collections, original_query)

    except Exception as e:
        logger.exception("意图路由节点执行失败")
        # LLM 失败降级：catalog 类可由后续规则判断，此处先按 ambiguous 反问
        return {
            "intent": "ambiguous",
            "entities": {},
            "route_reason": f"意图识别失败，降级为反问：{e}",
            "answer": "抱歉，我暂时无法理解您的问题，请换个说法再试一次。",
        }


@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state: dict):
    """
    获取路由所需入参：改写后问题（回退原问题）+ 最近历史
    :return: (query, original_query, history_text)
    """
    original_query = state.get("original_query", "")
    # 意图识别必须基于**改写后**的问题：多轮里"第一个系列里有几门课"这类序数指代
    # 只有在改写为具体名称后才能正确抽实体；改写未生效时才回退原问题。
    query = state.get("rewritten_query") or original_query
    history = state.get("history", [])

    history_text = ""
    if history:
        recent = history[-4:]  # 最近 2 轮（4 条消息，history 已按时间正序）
        history_text = "\n".join(
            f"{'用户' if m.get('role') == 'user' else '助手'}: {m.get('text', '')[:100]}"
            for m in recent
        )
    logger.info(f"意图路由入参：query={query}，历史轮次={len(history)}")
    return query, original_query, history_text


@step_log("step_2_call_llm_intent")
def step_2_call_llm_intent(query: str, history_text: str) -> dict:
    """
    调用 LLM 做意图识别与实体抽取，解析 JSON 输出
    :return: LLM 输出的解析结果
    """
    prompt = load_prompt("intent_route", query=query, history=history_text)
    llm = get_llm_client(json_mode=True)
    response = llm.invoke(prompt)
    raw_content = response.content if hasattr(response, "content") else str(response)

    # 解析 JSON（容错：剥离可能的 markdown 代码块包裹）
    json_match = re.search(r"\{[\s\S]*\}", raw_content)
    if not json_match:
        raise ValueError(f"LLM 输出无法解析为 JSON：{raw_content[:200]}")
    parsed = json.loads(json_match.group())
    logger.info(f"LLM 意图原始输出：intent={parsed.get('intent')}，entities={parsed.get('entities')}")
    return parsed


@step_log("step_3_clean_entities")
def step_3_clean_entities(parsed: dict):
    """
    意图枚举校验 + 实体白名单过滤 + 题型枚举校验
    :return: (intent, entities)
    """
    intent = parsed.get("intent", "ambiguous")
    if intent not in {"catalog", "aggregate", "content", "ambiguous"}:
        logger.warning(f"非法意图枚举 {intent}，降级为 ambiguous")
        intent = "ambiguous"

    # ---- 实体清洗 ----
    entities = parsed.get("entities") or {}
    entities = {k: v for k, v in entities.items() if k in VALID_ENTITY_KEYS and v}

    # 题型枚举校验（PRD 4.2 开放集 + 评测用例 A3-24：不存在的题型必须澄清而非误匹配）
    if "question_type_cn" in entities and entities["question_type_cn"] not in VALID_QUESTION_TYPES_CN:
        invalid_type = entities.pop("question_type_cn")
        entities["_invalid_question_type"] = invalid_type  # 保留原值供澄清话术使用
        logger.warning(f"非法题型枚举：{invalid_type}，转为澄清场景")
    return intent, entities


@step_log("step_4_guard_entities")
def step_4_guard_entities(intent: str, entities: dict, original_query: str, query: str):
    """
    实体过抽取/误推断防护：
    1. 题型自纠偏与推断防护（必须由用户明说才保留题型实体）
    2. content 意图丢弃课程维度噪声实体，并定向检索集合
    3. 非法聚合字段丢弃 + 答案分布兜底
    4. 序数指代实体丢弃
    :return: (intent, entities, search_collections)
    """
    # LLM 自纠偏防护（A3-24）：用户原文含"填空题"但知识库无此题型时，
    # LLM 可能擅自映射为"完形填空题"——需以用户原词为准触发澄清
    if "question_type_cn" in entities:
        asked_type = entities["question_type_cn"]
        # 用户问题中出现过的题型词才是"真的被问"；若题型词从未出现在原问题中且原问题含"填空"字样 → 澄清
        if asked_type not in original_query and "填空" in original_query:
            entities.pop("question_type_cn")
            entities["_invalid_question_type"] = "填空题"

    # 题型推断防护：LLM 倾向把「有什么假设 / 什么是张量 / 哪个概念体现…」这类概念类问题
    # 推断成「简答题」，一旦带上非法推断的题型，content 会被错误收窄到 edu_question 并加
    # 错误的 question_type 过滤，直接丢掉课程文档召回。
    # 规则：题型必须由用户明说（题型原名，或「选择题」这类口语泛称），否则丢弃该实体。
    _TYPE_ALIASES = {
        "单选题": ("单选题", "选择题", "单选"),
        "多选题": ("多选题", "选择题", "多选"),
        "判断题": ("判断题", "判断"),
        "完形填空题": ("完形填空题", "完形填空"),
    }
    if "question_type_cn" in entities:
        asked_type = entities["question_type_cn"]
        aliases = _TYPE_ALIASES.get(asked_type, (asked_type,))
        said = f"{original_query}\n{query}"
        if not any(a in said for a in aliases):
            logger.info(f"题型推断防护：问题未提及「{asked_type}」，丢弃该实体（避免错误收窄召回路）")
            entities.pop("question_type_cn")

    # 非题型实体过抽取防护：语义查找类问题（content）只保留题型/题库过滤实体，
    # 抽出的课程维度实体（如 course_category）不用于预过滤（避免错误过滤缩小召回）
    if intent == "content":
        for noise_key in ("course_category", "series_name", "course_name", "target_audience", "suitable_grade", "learning_goal", "course_code"):
            entities.pop(noise_key, None)

    # 内容型意图的检索集合定向（PRD 3.6 语义通道）：
    # 含题型/题库实体（题目类语义查找）→ 仅 edu_question，避免课程系列召回稀释题目结果
    search_collections = ["edu_course_series", "edu_course_module", "edu_question", "edu_course_doc"]
    if intent == "content" and (entities.get("question_type_cn") or entities.get("bank_name")):
        search_collections = ["edu_question"]

    # aggregate_field 防护（代码级兜底）：非法聚合字段直接丢弃（count 场景可不依赖该字段）
    if entities.get("aggregate_field") and entities["aggregate_field"] not in {
        "question_type", "bank_code", "answer", "course_category",
        "suitable_grade", "target_audience", "learning_goal",
    }:
        logger.warning(f"非法聚合字段 {entities['aggregate_field']}，已丢弃")
        entities.pop("aggregate_field")

    # 答案分布兜底（业务确定性映射）：
    # "答案分布/正确率/各占多少"在业务上只有判断题的 answer（true/false）可统计，
    # LLM 常漏给 aggregate_field/aggregate_op，导致退化成只回总数、答案阶段编造明细。
    if intent == "aggregate" and "答案" in query and any(
            w in query for w in ("分布", "占比", "比例", "各多少", "分别", "正确率")
    ):
        entities["aggregate_field"] = "answer"
        entities["aggregate_op"] = "distribution"
        if not entities.get("question_type_cn"):
            entities["question_type_cn"] = "判断题"

    # 序数字面值防护：实体值是"第一个系列/第一道题"这类未消解的序数指代时，
    # 说明改写未生效——丢弃该实体，避免 like 查询无意义匹配
    for key in ("series_name", "course_name", "bank_name"):
        val = entities.get(key, "")
        if val and any(w in val for w in ("第一", "第二", "第三", "这个", "那个", "它")):
            logger.info(f"实体 {key}={val} 为未消解的序数指代，已丢弃")
            entities.pop(key)

    return intent, entities, search_collections


@step_log("step_5_fallback_series_entity")
def step_5_fallback_series_entity(intent: str, entities: dict, query: str):
    """
    代码级实体兜底：改写后的问题文本含明确系列名时直接提取
    （LLM 有时判对意图但漏抽实体；series_name 是 catalog 通道的核心过滤条件）
    """
    if intent != "catalog" or entities.get("series_name") or entities.get("course_name"):
        return

    from utils.clients.milvus_utils import get_milvus_client
    try:
        client = get_milvus_client()
        if client is not None:
            candidates = client.query(
                collection_name="edu_course_series",
                filter='series_code != ""',
                output_fields=["series_name", "course_category"],
                limit=300,
            )
            for c in candidates:
                name = c.get("series_name") or ""
                if name and len(name) >= 4 and name in query:
                    entities["series_name"] = name
                    logger.info(f"实体兜底命中：series_name={name}")
                    break
    except Exception as e:
        logger.debug(f"实体兜底查询失败（忽略）：{e}")


@step_log("step_6_build_route_result")
def step_6_build_route_result(intent: str, entities: dict, route_reason: str,
                              search_collections: list, original_query: str) -> dict:
    """
    组装路由结果；意图不明或题型非法时直接生成澄清话术（跳过检索）
    :return: 状态更新字典
    """
    # ---- 意图不明：直接生成反问（PRD 3.6 反问/拒答机制）----
    if intent == "ambiguous":
        invalid_type = entities.get("_invalid_question_type")
        # 兜底：用户问句含题型字样但知识库无该题型（LLM 可能直接判 ambiguous 而不标 invalid）
        if not invalid_type and "填空" in original_query and "完形" not in original_query:
            invalid_type = "填空题"
        if invalid_type:
            # 题型不存在（A3-24 场景）：澄清而非误匹配
            answer = (
                f"知识库中没有「{invalid_type}」这种题型哦。现有题型包括：单选题、多选题、判断题、"
                f"简答题、计算题、编程题等 15 种。您是想查某一类的题目吗？比如相近的「完形填空题」？"
            )
            logger.info(f"意图不明且题型不存在（{invalid_type}），返回澄清话术")
        else:
            answer = (
                "抱歉，我不太确定您想查询什么。您可以：\n"
                "1. 查课程：如「有哪些适合在校生的课程系列」\n"
                "2. 查题目：如「通用程序设计题库有哪些单选题」\n"
                "3. 按知识点搜题：如「函数参数传递相关的练习题」"
            )
            logger.info("意图不明，返回反问引导话术")
        return {"intent": intent, "entities": entities, "route_reason": route_reason, "answer": answer}

    # 意图明确但题型非法（如 catalog + 填空题）：也走澄清
    if entities.get("_invalid_question_type"):
        invalid_type = entities["_invalid_question_type"]
        answer = (
            f"知识库中没有「{invalid_type}」这种题型哦。现有题型包括：单选题、多选题、判断题、"
            f"简答题、计算题、编程题等 15 种。您是想查相近的「完形填空题」吗？"
        )
        logger.info(f"题型非法（{invalid_type}），返回澄清话术")
        return {"intent": "ambiguous", "entities": entities, "route_reason": f"题型不存在：{invalid_type}",
                "answer": answer}

    return {"intent": intent, "entities": entities, "route_reason": route_reason,
            "search_collections": search_collections}
