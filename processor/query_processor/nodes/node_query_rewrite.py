"""
查询改写节点
对应 PRD V1.3 3.7 多轮对话：查询改写结合最近 6 轮（12 条）历史做指代消解
- 有历史且问题含指代词（它/这个/第一个/上述等）→ LLM 改写为完整独立问题
- 无历史或改写失败 → 原样透传（降级保底）
"""
import re

from common.logging.logger import logger, node_log
from utils.lm.lm_utils import get_llm_client

# 指代词特征（命中才触发改写，避免每轮都调 LLM）
_ANAPHORA_PATTERN = re.compile(r"它|这|那个|上述|刚才|前面|第一|上一|换个|换成|相关")


@node_log("node_query_rewrite")
def node_query_rewrite(state: dict) -> dict:
    """
    查询改写节点：
    - 无历史或问题不含指代词 → 原样透传（省 LLM 调用）
    - 含指代词 → LLM 结合历史改写为独立完整问题
    """
    original_query = state.get("original_query", "")
    history = state.get("history", [])

    # 无历史：单轮场景，无需改写
    if not history:
        return {"rewritten_query": original_query}

    # 不含指代词：直接透传
    if not _ANAPHORA_PATTERN.search(original_query):
        return {"rewritten_query": original_query}

    # ---- LLM 改写 ----
    try:
        history_text = "\n".join(
            f"{'用户' if m.get('role') == 'user' else '助手'}: {m.get('text', '')[:300]}"
            for m in history[-6:]  # 最近 3 轮
        )
        prompt = (
            "你是对话查询改写器。根据【历史对话】把【用户当前问题】中省略或指代的部分补全，"
            "改写为一个不依赖上下文也能理解的完整独立问题。\n"
            "要求：\n"
            "1. 只输出改写后的问题，不要任何解释\n"
            "2. 保留原问题的查询意图（课程查询/题目查询/统计）\n"
            "3. 当前问题与历史无关时，原样输出当前问题\n"
            "4. 序数指代（如'第一个''第一道''它'）必须替换为历史中实际出现的名称/编码\n"
            "    例：历史回答列出了'1. 通用编程入门班 2. 通用编程项目班'，问'第一个系列里有几门课'"
            " → '通用编程入门班包含哪些课程模块'\n"
            "5. 换条件指代（如'换成编程题呢'）需继承原问题结构并替换条件\n\n"
            f"【历史对话】\n{history_text}\n\n"
            f"【用户当前问题】\n{original_query}\n\n"
            "改写后的问题："
        )
        llm = get_llm_client()
        response = llm.invoke(prompt)
        rewritten = (response.content if hasattr(response, "content") else str(response)).strip()
        # 长度保护：改写异常膨胀时回退原问题
        if rewritten and len(rewritten) < len(original_query) * 10:
            logger.info(f"查询改写：{original_query} → {rewritten}")
            return {"rewritten_query": rewritten}
        logger.warning("改写结果异常，回退原问题")
        return {"rewritten_query": original_query}

    except Exception as e:
        logger.warning(f"查询改写失败，回退原问题：{e}")
        return {"rewritten_query": original_query}
