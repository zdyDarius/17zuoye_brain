"""
查询改写节点
对应 PRD V1.3 3.7 多轮对话：查询改写结合最近历史做指代消解
- 有历史且问题含指代词（它/这个/第一个/上述等）→ LLM 改写为完整独立问题
- 无历史或改写失败 → 原样透传（降级保底）
"""
import re

from common.logging.logger import logger, node_log, step_log
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt

# 指代词特征（命中才触发改写，避免每轮都调 LLM）
# 注意：不要把「相关」放进正则——它是普通问句的高频词（"XX相关的练习题"），
# 会把单轮语义查询误判为指代，并可能把意图改写窄（实测曾把"知识"改写成"课程"）。
_ANAPHORA_PATTERN = re.compile(
    r"它|这个|这些|这种|这里|那个|那些|上述|刚才|前面|上一|第一|第二|第三|换个|换成"
)


@node_log("node_query_rewrite")
def node_query_rewrite(state: dict) -> dict:
    """
    查询改写节点：
    - 无历史或问题不含指代词 → 原样透传（省 LLM 调用）
    - 含指代词 → LLM 结合历史改写为独立完整问题
    """
    # 1. 校验是否需要改写（无历史 / 无指代词 → 透传）
    data = step_1_validate_and_get_data(state)
    if data is None:
        return {"rewritten_query": state.get("original_query", "")}
    original_query, history = data

    # 2. 拼接最近历史文本
    history_text = step_2_build_history_text(history)

    # 3. LLM 改写（失败/异常膨胀 → 回退原问题）
    return {"rewritten_query": step_3_rewrite_by_llm(original_query, history_text)}


@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state: dict):
    """
    校验改写前置条件
    :return: (original_query, history) 或 None（无需改写，原样透传）
    """
    original_query = state.get("original_query", "")
    history = state.get("history", [])

    # 无历史：单轮场景，无需改写
    if not history:
        logger.info("无对话历史，单轮场景无需改写")
        return None

    # 不含指代词：直接透传
    if not _ANAPHORA_PATTERN.search(original_query):
        logger.info("问题不含指代词，无需改写")
        return None

    return original_query, history


@step_log("step_2_build_history_text")
def step_2_build_history_text(history: list) -> str:
    """拼接最近历史文本（history 由 API 侧统一转为时间正序，取尾部即最近的对话）"""
    history_text = "\n".join(
        f"{'用户' if m.get('role') == 'user' else '助手'}: {m.get('text', '')[:300]}"
        for m in history[-6:]  # 最近 3 轮（6 条消息）
    )
    logger.info(f"改写上下文历史：{len(history_text)} 字符")
    return history_text


@step_log("step_3_rewrite_by_llm")
def step_3_rewrite_by_llm(original_query: str, history_text: str) -> str:
    """
    LLM 结合历史改写为独立完整问题
    :return: 改写后问题（异常或结果异常时返回原问题）
    """
    try:
        prompt = load_prompt("rewrite", history=history_text, query=original_query)
        llm = get_llm_client()
        response = llm.invoke(prompt)
        rewritten = (response.content if hasattr(response, "content") else str(response)).strip()
        # 长度保护：改写异常膨胀时回退原问题
        if rewritten and len(rewritten) < len(original_query) * 10:
            logger.info(f"查询改写：{original_query} → {rewritten}")
            return rewritten
        logger.warning("改写结果异常，回退原问题")
        return original_query

    except Exception as e:
        logger.warning(f"查询改写失败，回退原问题：{e}")
        return original_query
