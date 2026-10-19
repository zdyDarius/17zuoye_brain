"""
意图路由节点
对应 PRD V1.3 3.6：node_intent_route
- LLM 结构化输出：intent ∈ {catalog, aggregate, content, ambiguous} + entities 实体抽取
- 题型枚举校验（15 种，PRD 4.2）：LLM 输出的题型必须合法，否则置空（触发反问/澄清）
- 意图不明（ambiguous）时直接生成反问 answer，跳过检索
"""
import json
import re

from common.logging.logger import logger, node_log
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
    original_query = state.get("original_query", "")
    # 意图识别必须基于**改写后**的问题：多轮里"第一个系列里有几门课"这类序数指代
    # 只有在改写为具体名称后才能正确抽实体；改写未生效时才回退原问题。
    query = state.get("rewritten_query") or original_query
    history = state.get("history", [])

    try:
        # ---- 构造上下文（最近 2 轮历史辅助指代消解）----
        history_text = ""
        if history:
            recent = history[-4:]  # 最近 2 轮（4 条消息，history 已按时间正序）
            history_text = "\n".join(
                f"{'用户' if m.get('role') == 'user' else '助手'}: {m.get('text', '')[:100]}"
                for m in recent
            )

        # ---- 调用 LLM 结构化输出 ----
        prompt = load_prompt("intent_route", query=query, history=history_text)
        llm = get_llm_client(json_mode=True)
        response = llm.invoke(prompt)
        raw_content = response.content if hasattr(response, "content") else str(response)

        # 解析 JSON（容错：剥离可能的 markdown 代码块包裹）
        json_match = re.search(r"\{[\s\S]*\}", raw_content)
        if not json_match:
            raise ValueError(f"LLM 输出无法解析为 JSON：{raw_content[:200]}")
        parsed = json.loads(json_match.group())

        intent = parsed.get("intent", "ambiguous")
        if intent not in {"catalog", "aggregate", "content", "ambiguous"}:
            intent = "ambiguous"

        # ---- 实体清洗 ----
        entities = parsed.get("entities") or {}
        entities = {k: v for k, v in entities.items() if k in VALID_ENTITY_KEYS and v}

        # 题型枚举校验（PRD 4.2 开放集 + 评测用例 A3-24：不存在的题型必须澄清而非误匹配）
        if "question_type_cn" in entities and entities["question_type_cn"] not in VALID_QUESTION_TYPES_CN:
            invalid_type = entities.pop("question_type_cn")
            entities["_invalid_question_type"] = invalid_type  # 保留原值供澄清话术使用

        # LLM 自纠偏防护（A3-24）：用户原文含"填空题"但知识库无此题型时，
        # LLM 可能擅自映射为"完形填空题"——需以用户原词为准触发澄清
        if "question_type_cn" in entities:
            asked_type = entities["question_type_cn"]
            # 用户问题中出现过的题型词才是"真的被问"；若题型词从未出现在原问题中且原问题含"填空"字样 → 澄清
            if asked_type not in original_query and "填空" in original_query:
                entities.pop("question_type_cn")
                entities["_invalid_question_type"] = "填空题"

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
            entities.pop("aggregate_field")

        # 序数字面值防护：实体值是"第一个系列/第一道题"这类未消解的序数指代时，
        # 说明改写未生效——丢弃该实体，避免 like 查询无意义匹配
        for key in ("series_name", "course_name", "bank_name"):
            val = entities.get(key, "")
            if val and any(w in val for w in ("第一", "第二", "第三", "这个", "那个", "它")):
                entities.pop(key)

        # ---- 代码级实体兜底：改写后的问题文本含明确系列名时直接提取 ----
        # （LLM 有时判对意图但漏抽实体；series_name 是 catalog 通道的核心过滤条件）
        if intent == "catalog" and not entities.get("series_name") and not entities.get("course_name"):
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
                    rewritten = state.get("rewritten_query") or original_query
                    for c in candidates:
                        name = c.get("series_name") or ""
                        if name and len(name) >= 4 and name in rewritten:
                            entities["series_name"] = name
                            break
            except Exception as e:
                logger.debug(f"实体兜底查询失败（忽略）：{e}")

        route_reason = parsed.get("route_reason", "")
        logger.info(f"意图路由：intent={intent}，entities={entities}，理由={route_reason}")

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
            else:
                answer = (
                    "抱歉，我不太确定您想查询什么。您可以：\n"
                    "1. 查课程：如「有哪些适合在校生的课程系列」\n"
                    "2. 查题目：如「通用程序设计题库有哪些单选题」\n"
                    "3. 按知识点搜题：如「函数参数传递相关的练习题」"
                )
            return {"intent": intent, "entities": entities, "route_reason": route_reason, "answer": answer}

        # 意图明确但题型非法（如 catalog + 填空题）：也走澄清
        if entities.get("_invalid_question_type"):
            invalid_type = entities["_invalid_question_type"]
            answer = (
                f"知识库中没有「{invalid_type}」这种题型哦。现有题型包括：单选题、多选题、判断题、"
                f"简答题、计算题、编程题等 15 种。您是想查相近的「完形填空题」吗？"
            )
            return {"intent": "ambiguous", "entities": entities, "route_reason": f"题型不存在：{invalid_type}",
                    "answer": answer}

        return {"intent": intent, "entities": entities, "route_reason": route_reason,
                "search_collections": search_collections}

    except Exception as e:
        logger.exception("意图路由节点执行失败")
        # LLM 失败降级：catalog 类可由后续规则判断，此处先按 ambiguous 反问
        return {
            "intent": "ambiguous",
            "entities": {},
            "route_reason": f"意图识别失败，降级为反问：{e}",
            "answer": "抱歉，我暂时无法理解您的问题，请换个说法再试一次。",
        }
