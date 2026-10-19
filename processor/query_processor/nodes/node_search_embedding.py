"""
语义检索通道节点
对应 PRD V1.3 3.6 内容型意图：
- node_search_embedding：原查询 BGE-M3 混合向量检索（多集合）
- node_search_embedding_hyde：HyDE 假设答案向量检索
- node_question_filter：按实体做标量预过滤的向量检索（题库/题型过滤）
三路并行执行，结果经 node_rrf 融合 → node_rerank 精排

性能约定：查询向量在**每个节点内只编码一次**，循环各集合时复用（BGE-M3 前向是主要耗时项）。
"""
from common.logging.logger import logger, node_log, step_log
from utils.clients.milvus_utils import get_milvus_client, hybrid_search
from utils.lm.embedding_utils import generate_query_embeddings
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt
from pymilvus import AnnSearchRequest

from processor.import_processor.nodes.node_question_parser import QUESTION_TYPE_MAP

# 语义检索各集合的返回字段（source_file 用于答案引用溯源）
# 注意：edu_course_doc 无业务编码，必须返回主键 id，否则 RRF 去重键退化为
# (collection, None)，同集合的所有文档块会被当成同一条记录合并（只剩 1 块）。
_OUTPUT_FIELDS = {
    "edu_question": ["question_code", "bank_name", "bank_code", "question_type",
                     "content", "answer", "analysis", "source_file"],
    "edu_course_series": ["series_code", "series_name", "course_category",
                          "target_audience", "suitable_grade", "content", "source_file"],
    "edu_course_module": ["course_code", "course_name", "series_name",
                          "duration", "credit_hours", "content", "source_file"],
    "edu_course_doc": ["id", "doc_kind", "chapter_name", "chunk_index",
                       "content", "source_file"],
}

# 每集合召回条数
TOP_K_PER_COLLECTION = 5


def _hybrid_search_collection(client, collection: str, emb: dict, expr: str = None):
    """
    对单个集合做稠密+稀疏混合检索（向量由调用方统一编码后传入，避免重复前向）
    :param emb: generate_query_embeddings 的返回结果 {"dense": [[..]], "sparse": [{..}]}
    :param expr: 标量预过滤表达式（题库/题型实体过滤）
    :return: 命中列表（带 _collection 标记），失败返回 []
    """
    try:
        # 与建表索引一致：dense=IP, sparse=IP（含标量预过滤）
        dense_req = AnnSearchRequest(
            data=[emb["dense"][0]], anns_field="dense_vector",
            param={"metric_type": "IP"}, expr=expr, limit=TOP_K_PER_COLLECTION,
        )
        sparse_req = AnnSearchRequest(
            data=[emb["sparse"][0]], anns_field="sparse_vector",
            param={"metric_type": "IP"}, expr=expr, limit=TOP_K_PER_COLLECTION,
        )
        res = hybrid_search(client, collection, [dense_req, sparse_req],
                            limit=TOP_K_PER_COLLECTION, output_fields=_OUTPUT_FIELDS[collection])
        hits = []
        if res and res[0]:
            for hit in res[0]:
                entity = hit.get("entity", {})
                entity["_collection"] = collection
                entity["_score"] = hit.get("distance", 0)
                hits.append(entity)
        return hits
    except Exception as e:
        logger.error(f"混合检索失败[{collection}]：{e}")
        return []


@node_log("node_search_embedding")
def node_search_embedding(state: dict) -> dict:
    """
    原查询混合向量检索：编码一次，对 search_collections 内各集合检索（无标量预过滤）
    """
    # 1. 校验并获取查询文本、客户端与目标集合
    data = step_1_validate_and_get_data(state)
    if data is None:
        return {"error": "Milvus 客户端不可用"}
    query, client, collections = data

    # 2. 编码查询向量（稠密 + 稀疏）
    emb, error = step_2_encode_query(query, "原查询")
    if emb is None:
        return {"error": error}

    # 3. 逐集合混合检索
    hits = step_3_search_collections(client, collections, emb, "原查询")
    return {"embedding_hits": hits}


@node_log("node_search_embedding_hyde")
def node_search_embedding_hyde(state: dict) -> dict:
    """
    HyDE 检索：LLM 先生成假设答案，再用假设答案做向量检索（提升语义命中率）
    LLM 失败时降级为原查询检索（保证通道可用）
    """
    # 1. 校验并获取查询文本、客户端与目标集合
    data = step_1_validate_and_get_data(state)
    if data is None:
        return {"error": "Milvus 客户端不可用"}
    query, client, collections = data

    # 2. 生成假设答案（失败降级用原查询）
    hyde_text = step_2_generate_hyde_answer(query)

    # 3. 编码假设答案向量
    emb, _ = step_2_encode_query(hyde_text, "HyDE")
    if emb is None:
        return {"hyde_hits": []}

    # 4. 逐集合混合检索
    hits = step_3_search_collections(client, collections, emb, "HyDE")
    return {"hyde_hits": hits}


@node_log("node_question_filter")
def node_question_filter(state: dict) -> dict:
    """
    实体预过滤的向量检索：按抽取实体（题型/题库）构造标量过滤表达式，
    仅对 edu_question 做过滤向量检索（如"函数相关的选择题"→ question_type==single_choice）
    """
    # 1. 校验并获取实体、客户端与查询文本
    data = step_1_filter_validate_and_get_data(state)
    if data is None:
        return {"error": "Milvus 客户端不可用"}
    entities, client, query = data

    # 2. 构造标量过滤表达式（无实体过滤时本通道退化为普通检索，跳过）
    expr = step_2_build_filter_expr(entities)
    if not expr:
        logger.info("实体预过滤通道：无可过滤实体，跳过")
        return {"question_filter_hits": []}

    # 3. 编码查询向量
    emb, _ = step_2_encode_query(query, "实体预过滤")
    if emb is None:
        return {"question_filter_hits": []}

    # 4. 仅对 edu_question 做过滤混合检索
    hits = step_3_search_collections(client, ["edu_question"], emb, "实体预过滤", expr=expr)
    return {"question_filter_hits": hits}


@node_log("node_rrf")
def node_rrf(state: dict) -> dict:
    """
    RRF 融合：三路召回结果（embedding / hyde / question_filter）按 RRF 公式融合排序
    RRF：score = Σ 1/(k + rank)，k=60（标准参数）
    """
    # 1. 收集三路召回通道
    channels, active_channels, total_input = step_1_collect_channels(state)

    # 2. RRF 融合去重排序
    fused_records = step_2_fuse_by_rrf(channels)
    logger.info(f"RRF 融合：{active_channels} 路有效通道、{total_input} 条输入 "
                f"→ {len(fused_records)} 条去重融合")
    return {"fused_results": fused_records}


@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state: dict):
    """
    获取查询文本、Milvus 客户端与目标集合
    :return: (query, client, collections) 或 None（客户端不可用）
    """
    query = state.get("rewritten_query") or state.get("original_query", "")
    client = get_milvus_client()
    if client is None:
        logger.error("Milvus 客户端不可用，语义检索终止")
        return None
    collections = state.get("search_collections", [])
    logger.info(f"语义检索入参：query={query}，目标集合={collections}")
    return query, client, collections


@step_log("step_2_generate_hyde_answer")
def step_2_generate_hyde_answer(query: str) -> str:
    """
    LLM 生成 HyDE 假设答案
    :return: 假设答案（生成失败时返回原查询，降级保证通道可用）
    """
    hyde_text = query
    try:
        prompt = load_prompt("hyde_prompt", rewritten_query=query)
        llm = get_llm_client()
        response = llm.invoke(prompt)
        hyde_text = response.content if hasattr(response, "content") else str(response)
        logger.info(f"HyDE 假设答案生成成功（{len(hyde_text)} 字）")
    except Exception as e:
        logger.warning(f"HyDE 生成失败，降级用原查询检索：{e}")
    return hyde_text


@step_log("step_2_encode_query")
def step_2_encode_query(text: str, channel_name: str):
    """
    编码查询向量（稠密 + 稀疏）
    :return: (emb, 错误信息)；emb 为 None 表示编码失败
    """
    try:
        emb = generate_query_embeddings([text])
        return emb, None
    except Exception as e:
        logger.error(f"{channel_name}编码失败：{e}")
        return None, f"{channel_name}向量生成失败：{e}"


@step_log("step_3_search_collections")
def step_3_search_collections(client, collections: list, emb: dict,
                              channel_name: str, expr: str = None) -> list:
    """
    逐集合做稠密+稀疏混合检索（向量复用，避免重复前向）
    :return: 命中列表
    """
    hits = []
    for collection in collections:
        hits.extend(_hybrid_search_collection(client, collection, emb, expr=expr))
    logger.info(f"{channel_name}语义召回：{len(hits)} 条（集合 {collections}）")
    return hits


@step_log("step_1_filter_validate_and_get_data")
def step_1_filter_validate_and_get_data(state: dict):
    """
    实体预过滤通道入参校验
    :return: (entities, client, query) 或 None（客户端不可用）
    """
    entities = state.get("entities", {})
    client = get_milvus_client()
    if client is None:
        logger.error("Milvus 客户端不可用，实体预过滤通道终止")
        return None
    query = state.get("rewritten_query") or state.get("original_query", "")
    return entities, client, query


@step_log("step_2_build_filter_expr")
def step_2_build_filter_expr(entities: dict) -> str:
    """
    按实体构造 edu_question 的标量过滤表达式
    :return: 过滤表达式（空串表示无可过滤实体）
    """
    conds = []
    if entities.get("question_type_cn"):
        type_en = QUESTION_TYPE_MAP.get(entities["question_type_cn"])
        if type_en:
            conds.append(f'question_type == "{type_en}"')
    if entities.get("bank_name"):
        conds.append(f'bank_name like "%{entities["bank_name"]}%"')
    expr = " and ".join(conds)
    logger.info(f"实体预过滤表达式：{expr or '（空）'}")
    return expr


@step_log("step_1_collect_channels")
def step_1_collect_channels(state: dict):
    """
    收集三路召回结果
    :return: (channels, 有效通道数, 输入总条数)
    """
    channels = [
        state.get("embedding_hits") or [],
        state.get("hyde_hits") or [],
        state.get("question_filter_hits") or [],
    ]
    active_channels = sum(1 for c in channels if c)
    total_input = sum(len(c) for c in channels)
    return channels, active_channels, total_input


@step_log("step_2_fuse_by_rrf")
def step_2_fuse_by_rrf(channels: list) -> list:
    """
    RRF 融合：score = Σ 1/(k + rank)，k=60
    :return: 融合后记录列表（带 _rrf_score）
    """
    K = 60
    ranked = {}  # 唯一键 -> {record, score}

    for channel_hits in channels:
        for rank, hit in enumerate(channel_hits):
            # 唯一键：集合 + 业务编码（跨集合不冲突）
            key = (
                hit.get("_collection"),
                hit.get("question_code") or hit.get("series_code")
                or hit.get("course_code") or hit.get("id"),
            )
            if key not in ranked:
                ranked[key] = {"record": hit, "score": 0.0}
            ranked[key]["score"] += 1.0 / (K + rank + 1)

    fused = sorted(ranked.values(), key=lambda x: -x["score"])
    fused_records = []
    for item in fused:
        # 回写融合分，便于评估与调试观察 RRF 排序依据
        item["record"]["_rrf_score"] = round(item["score"], 6)
        fused_records.append(item["record"])
    return fused_records
