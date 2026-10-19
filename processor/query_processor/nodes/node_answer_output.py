"""
答案输出节点
对应 PRD V1.3 3.6：
- 各通道结果（structured/aggregate/reranked）组装为上下文
- LLM 生成最终答案 + 引用列表 + 配图列表（image_urls）
- LLM 失败降级：结构化/聚合结果以模板化文本直接输出（PRD 3.8 降级矩阵）
- 无有效结果：明确回复"未在知识库中找到相关内容"
"""
import json
import re
from urllib.parse import urlsplit

from common.logging.logger import logger, node_log, step_log
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt
from processor.import_processor.nodes.node_question_parser import QUESTION_TYPE_MAP

# 英文题型枚举 → 中文（喂给 LLM 前转换，避免答案里出现 single_choice 这类字面）
_QUESTION_TYPE_CN = {v: k for k, v in QUESTION_TYPE_MAP.items()}

# 切片里的图片引用：![VL摘要](URL "可选 title")，只取括号里的第一个 URL
_IMAGE_MD_RE = re.compile(r"!\[[^\]]*\]\(\s*(?P<url>[^)\s]+)")
# 只收常见图片后缀（MinIO 上的对象名以 .jpg/.png 结尾）
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg")
# 单次答案最多展示的配图数（防止一个块里十几张图刷屏）
MAX_ANSWER_IMAGES = 6


def _is_image_url(url: str) -> bool:
    """只接受 http(s) 且路径以图片后缀结尾的 URL（挡掉裸文本/相对路径/非法串）"""
    if not url.lower().startswith(("http://", "https://")):
        return False
    return urlsplit(url).path.lower().endswith(_IMAGE_EXTS)


@step_log("step_6_extract_image_urls")
def step_6_extract_image_urls(state: dict) -> list:
    """
    从检索到的切片里提取配图 URL（结构化 + 语义通道都算），去重后限量返回。
    必须读**原始 content**：`_format_record` 会把 doc 截到 300 字、题目截到 200 字，
    而 MinIO 的 hash URL 有 90+ 字符，从截断文本里提会切掉半截链接。
    """
    urls, seen = [], set()
    records = (state.get("structured_results") or []) + (state.get("reranked_results") or [])
    for record in records:
        content = record.get("content") or ""
        for url in _IMAGE_MD_RE.findall(content):
            if url in seen or not _is_image_url(url):
                continue
            seen.add(url)
            urls.append(url)
            if len(urls) >= MAX_ANSWER_IMAGES:
                logger.info(f"配图提取完成：达到上限 {MAX_ANSWER_IMAGES} 张")
                return urls
    logger.info(f"配图提取完成：{len(urls)} 张")
    return urls


def _with_images(state: dict, payload: dict) -> dict:
    """统一补充配图字段（所有出口都走它，避免某个分支漏传）"""
    payload["image_urls"] = step_6_extract_image_urls(state)
    return payload


@node_log("node_answer_output")
def node_answer_output(state: dict) -> dict:
    """
    答案输出节点：
    1. 反问/澄清类（intent=ambiguous 已有 answer）直接透传
    2. 组装上下文 → LLM 生成答案（含引用标注）
    3. LLM 失败降级：模板化输出结构化/聚合结果（PRD 3.8）
    4. 无结果：明确"未找到"话术
    5. 无论哪个出口，都补充 image_urls（从切片 content 里提取的配图）
    """
    # 1. 反问/澄清/服务异常等直通场景
    passthrough = step_1_check_passthrough(state)
    if passthrough:
        return _with_images(state, passthrough)

    # 2. 按通道组装检索上下文
    context = step_2_build_context(state)
    # 作答基于改写后的问题：多轮里"第一个系列里有几门课"需要完整语境才能正确表述
    question = state.get("rewritten_query") or state.get("original_query", "")

    # 3. 无有效结果（PRD：禁止编造）
    if not context:
        return _with_images(state, step_3_build_no_result_answer())

    # 4. LLM 生成答案（失败降级为模板化输出）
    return _with_images(state, step_4_generate_answer(state, context, question))


@step_log("step_1_check_passthrough")
def step_1_check_passthrough(state: dict):
    """
    检查是否需要直通返回（已有 answer / 检索服务不可用）
    :return: 状态更新字典 或 None（需要走 LLM 生成）
    """
    # 反问/澄清场景：answer 已在意图路由生成
    if state.get("intent") == "ambiguous" and state.get("answer"):
        logger.info("反问/澄清场景，直接透传已有 answer")
        return {"answer": state["answer"], "references": []}

    # 聚合通道的白名单反问场景（answer 已生成）
    if state.get("intent") == "aggregate" and state.get("answer"):
        logger.info("聚合字段非法场景，直接透传已有 answer")
        return {"answer": state["answer"], "references": []}

    # 检索通道整体不可用（Milvus 连接失败等）：明确提示服务异常，不能伪装成"知识库没有"
    has_any_result = bool(
        state.get("structured_results") or state.get("aggregate_result") or state.get("reranked_results")
    )
    if state.get("error") and not has_any_result:
        logger.warning(f"检索通道不可用，返回服务异常提示：{state['error']}")
        return {
            "answer": "检索服务暂时不可用，请稍后重试。",
            "references": [],
            "degraded": True,
        }
    return None


@step_log("step_2_build_context")
def step_2_build_context(state: dict) -> str:
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

    context = "\n\n".join(parts)
    logger.info(f"检索上下文组装完成：{len(parts)} 个通道，{len(context)} 字符")
    return context


@step_log("step_3_build_no_result_answer")
def step_3_build_no_result_answer() -> dict:
    """无有效检索结果时的引导话术（禁止编造）"""
    logger.info("无有效检索结果，返回未找到引导话术")
    answer = ("未在知识库中找到相关内容。您可以：\n"
              "1. 换个关键词试试（如具体题型名、课程系列名）\n"
              "2. 查询课程介绍：如「有哪些适合在校生的课程」\n"
              "3. 查询题目：如「通用程序设计题库有哪些单选题」")
    return {"answer": answer, "references": []}


@step_log("step_4_generate_answer")
def step_4_generate_answer(state: dict, context: str, question: str) -> dict:
    """
    LLM 生成最终答案（含引用组装）；LLM 失败降级为模板化输出（PRD 3.8 LLM 故障行）
    """
    try:
        prompt = load_prompt("edu_answer", context=context, question=question)
        llm = get_llm_client()
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)
        logger.info(f"LLM 答案生成完成：{len(answer)} 字符")
        return {"answer": answer, "references": step_5_build_references(state)}

    except Exception as e:
        # ---- 降级：模板化输出（PRD 3.8 LLM 故障行）----
        logger.warning(f"LLM 生成失败，降级为模板化输出：{e}")
        degraded_answer = f"智能生成暂不可用，以下为检索到的原始结果：\n\n{context[:2000]}"
        return {"answer": degraded_answer, "references": step_5_build_references(state), "degraded": True}


@step_log("step_5_build_references")
def step_5_build_references(state: dict) -> list:
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
    logger.info(f"引用组装完成：{len(refs)} 条")
    return refs


def _format_record(record: dict) -> str:
    """将单条检索记录格式化为上下文条目"""
    collection = record.get("_collection", "")
    if collection == "edu_question":
        options = ""
        if record.get("content"):
            options = f"（内容：{record['content'][:200]}）"
        type_cn = _QUESTION_TYPE_CN.get(record.get("question_type"), record.get("question_type"))
        return (f"[题目] 编码:{record.get('question_code')} 题库:{record.get('bank_name')} "
                f"题型:{type_cn} 答案:{record.get('answer')} "
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
