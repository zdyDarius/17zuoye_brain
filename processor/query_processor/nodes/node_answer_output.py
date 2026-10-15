"""
答案输出节点
对应 PRD V1.3 3.6：
- 各通道结果（structured/aggregate/reranked）组装为上下文
- LLM 生成最终答案 + 引用列表
- LLM 失败降级：结构化/聚合结果以模板化文本直接输出（PRD 3.8 降级矩阵）
- 无有效结果：明确回复"未在知识库中找到相关内容"
"""
import json

from common.logging.logger import logger, node_log
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt


def _format_record(record: dict) -> str:
    """将单条检索记录格式化为上下文条目"""
    collection = record.get("_collection", "")
    if collection == "edu_question":
        options = ""
        if record.get("content"):
            options = f"（内容：{record['content'][:200]}）"
        return (f"[题目] 编码:{record.get('question_code')} 题库:{record.get('bank_name')} "
                f"题型:{record.get('question_type')} 答案:{record.get('answer')} "
                f"解析:{(record.get('analysis') or '')[:100]}{options}")
    if collection == "edu_course_series":
        return (f"[课程系列] 编码:{record.get('series_code')} 名称:{record.get('series_name')} "
                f"分类:{record.get('course_category')} 适合人群:{record.get('target_audience')} "
                f"学习目标:{record.get('learning_goal')} 适合年级:{record.get('suitable_grade')}")
    if collection == "edu_course_module":
        return (f"[课程模块] 编码:{record.get('course_code')} 名称:{record.get('course_name')} "
                f"所属系列:{record.get('series_name')} 课时:{record.get('duration')} "
                f"学时:{record.get('credit_hours')} 描述:{record.get('description')}")
    if collection == "edu_course_doc":
        content = (record.get("content") or "")[:300]
        return (f"[课程文档] 章节:{record.get('chapter_name')} "
                f"来源文件:{record.get('source_file')} 内容:{content}")
    return json.dumps({k: v for k, v in record.items() if not k.startswith("_")},
                      ensure_ascii=False, default=str)[:300]


def _build_context(state: dict) -> str:
    """按通道组装检索上下文"""
    parts = []

    # 结构化通道结果
    structured = state.get("structured_results") or []
    if structured:
        truncated = state.get("structured_truncated")
        head = "【结构化查询结果】" + ("（结果较多，已截断）" if truncated else "")
        body = "\n".join(f"{i + 1}. {_format_record(r)}" for i, r in enumerate(structured[:20]))
        if len(structured) > 20:
            body += f"\n（共 {len(structured)} 条，其余略）"
        parts.append(head + "\n" + body)

    # 聚合结果
    agg = state.get("aggregate_result") or {}
    if agg:
        lines = ["【统计结果】"]
        if agg.get("filter"):
            lines.append(f"过滤条件：{agg['filter']}")
        if "value" in agg and "groups" not in agg:
            lines.append(f"统计值：{agg['value']}")
        # 课程数量类（course_count）：附带模块/系列清单
        if agg.get("course_rows"):
            lines.append(f"清单（共 {len(agg['course_rows'])} 条）：")
            for r in agg["course_rows"][:20]:
                lines.append(f"  {r}")
        if "groups" in agg:
            lines.append(f"总数：{agg.get('value', 0)}，分组数：{agg.get('group_total', len(agg['groups']))}")
            for g in agg["groups"][:20]:
                lines.append(f"  {g['key']}: {g['count']}")
            if agg.get("truncated"):
                lines.append("  （分组较多，已截断展示前 20 组）")
        parts.append("\n".join(lines))

    # 语义通道结果
    reranked = state.get("reranked_results") or []
    if reranked:
        body = "\n".join(f"{i + 1}. {_format_record(r)}" for i, r in enumerate(reranked))
        parts.append("【语义检索结果（按相关性排序）】\n" + body)

    return "\n\n".join(parts)


def _build_references(state: dict) -> list:
    """组装编码级引用列表（PRD 3.6 引用组装）"""
    refs = []
    seen = set()
    for record in (state.get("structured_results") or []) + (state.get("reranked_results") or []):
        collection = record.get("_collection", "")
        if collection == "edu_question":
            key, ref = record.get("question_code"), {
                "type": "question", "code": record.get("question_code"),
                "bank": record.get("bank_name"), "source_file": record.get("source_file"),
            }
        elif collection == "edu_course_series":
            key, ref = record.get("series_code"), {
                "type": "course_series", "code": record.get("series_code"),
                "name": record.get("series_name"), "source_file": record.get("source_file"),
            }
        elif collection == "edu_course_module":
            key, ref = record.get("course_code"), {
                "type": "course_module", "code": record.get("course_code"),
                "name": record.get("course_name"), "source_file": record.get("source_file"),
            }
        elif collection == "edu_course_doc":
            key, ref = f"doc_{record.get('chunk_index')}_{record.get('chapter_name')}", {
                "type": "course_doc", "chapter": record.get("chapter_name"),
                "source_file": record.get("source_file"),
            }
        else:
            continue
        if key and key not in seen:
            seen.add(key)
            refs.append(ref)
    return refs


@node_log("node_answer_output")
def node_answer_output(state: dict) -> dict:
    """
    答案输出节点：
    1. 反问/澄清类（intent=ambiguous 已有 answer）直接透传
    2. 组装上下文 → LLM 生成答案（含引用标注）
    3. LLM 失败降级：模板化输出结构化/聚合结果（PRD 3.8）
    4. 无结果：明确"未找到"话术
    """
    # 反问/澄清场景：answer 已在意图路由生成
    if state.get("intent") == "ambiguous" and state.get("answer"):
        return {"answer": state["answer"], "references": []}

    # 聚合通道的白名单反问场景（answer 已生成）
    if state.get("intent") == "aggregate" and state.get("answer"):
        return {"answer": state["answer"], "references": []}

    context = _build_context(state)
    question = state.get("original_query", "")

    # 无有效结果（PRD：禁止编造）
    if not context:
        answer = ("未在知识库中找到相关内容。您可以：\n"
                  "1. 换个关键词试试（如具体题型名、课程系列名）\n"
                  "2. 查询课程介绍：如「有哪些适合在校生的课程」\n"
                  "3. 查询题目：如「通用程序设计题库有哪些单选题」")
        return {"answer": answer, "references": []}

    # ---- LLM 生成答案 ----
    try:
        prompt = load_prompt("edu_answer", context=context, question=question)
        llm = get_llm_client()
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)
        return {"answer": answer, "references": _build_references(state)}

    except Exception as e:
        # ---- 降级：模板化输出（PRD 3.8 LLM 故障行）----
        logger.warning(f"LLM 生成失败，降级为模板化输出：{e}")
        degraded_answer = f"智能生成暂不可用，以下为检索到的原始结果：\n\n{context[:2000]}"
        return {"answer": degraded_answer, "references": _build_references(state), "degraded": True}
