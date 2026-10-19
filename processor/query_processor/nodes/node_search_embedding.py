"""
语义检索通道节点
对应 PRD V1.3 3.6 内容型意图：
- node_search_embedding：原查询 BGE-M3 混合向量检索（多集合）
- node_search_embedding_hyde：HyDE 假设答案向量检索
- node_question_filter：按实体做标量预过滤的向量检索（题库/题型过滤）
三路并行执行，结果经 node_rrf 融合 → node_rerank 精排

性能约定：查询向量在**每个节点内只编码一次**，循环各集合时复用（BGE-M3 前向是主要耗时项）。
"""
from common.logging.logger import logger, node_log
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
    query = state.get("rewritten_query") or state.get("original_query", "")
    client = get_milvus_client()
    if client is None:
        return {"error": "Milvus 客户端不可用"}

    collections = state.get("search_collections", [])
    try:
        emb = generate_query_embeddings([query])
    except Exception as e:
        logger.error(f"原查询编码失败：{e}")
        return {"error": f"查询向量生成失败：{e}"}

    hits = []
    for collection in collections:
        hits.extend(_hybrid_search_collection(client, collection, emb))
    logger.info(f"原查询语义召回：{len(hits)} 条（集合 {collections}）")
    return {"embedding_hits": hits}


@node_log("node_search_embedding_hyde")
def node_search_embedding_hyde(state: dict) -> dict:
    """
    HyDE 检索：LLM 先生成假设答案，再用假设答案做向量检索（提升语义命中率）
    LLM 失败时降级为原查询检索（保证通道可用）
    """
    query = state.get("rewritten_query") or state.get("original_query", "")
    client = get_milvus_client()
    if client is None:
        return {"error": "Milvus 客户端不可用"}

    # ---- 生成假设答案 ----
    hyde_text = query
    try:
        prompt = load_prompt("hyde_prompt", rewritten_query=query)
        llm = get_llm_client()
        response = llm.invoke(prompt)
        hyde_text = response.content if hasattr(response, "content") else str(response)
        logger.info(f"HyDE 假设答案生成成功（{len(hyde_text)} 字）")
    except Exception as e:
        logger.warning(f"HyDE 生成失败，降级用原查询检索：{e}")

    try:
        emb = generate_query_embeddings([hyde_text])
    except Exception as e:
        logger.error(f"HyDE 编码失败：{e}")
        return {"hyde_hits": []}

    hits = []
    for collection in state.get("search_collections", []):
        hits.extend(_hybrid_search_collection(client, collection, emb))
    logger.info(f"HyDE 语义召回：{len(hits)} 条")
    return {"hyde_hits": hits}


@node_log("node_question_filter")
def node_question_filter(state: dict) -> dict:
    """
    实体预过滤的向量检索：按抽取实体（题型/题库）构造标量过滤表达式，
    仅对 edu_question 做过滤向量检索（如"函数相关的选择题"→ question_type==single_choice）
    """
    entities = state.get("entities", {})
    client = get_milvus_client()
    if client is None:
        return {"error": "Milvus 客户端不可用"}

    # 构造过滤表达式
    conds = []
    if entities.get("question_type_cn"):
        type_en = QUESTION_TYPE_MAP.get(entities["question_type_cn"])
        if type_en:
            conds.append(f'question_type == "{type_en}"')
    if entities.get("bank_name"):
        conds.append(f'bank_name like "%{entities["bank_name"]}%"')
    expr = " and ".join(conds)

    # 无实体过滤时本通道退化为普通检索（与 embedding 通道重复），跳过
    if not expr:
        logger.info("实体预过滤通道：无可过滤实体，跳过")
        return {"question_filter_hits": []}

    query = state.get("rewritten_query") or state.get("original_query", "")
    try:
        emb = generate_query_embeddings([query])
    except Exception as e:
        logger.error(f"实体预过滤编码失败：{e}")
        return {"question_filter_hits": []}

    hits = _hybrid_search_collection(client, "edu_question", emb, expr=expr)
    logger.info(f"实体预过滤召回：{len(hits)} 条（filter={expr}）")
    return {"question_filter_hits": hits}


@node_log("node_rrf")
def node_rrf(state: dict) -> dict:
    """
    RRF 融合：三路召回结果（embedding / hyde / question_filter）按 RRF 公式融合排序
    RRF：score = Σ 1/(k + rank)，k=60（标准参数）
    """
    K = 60
    ranked = {}  # 唯一键 -> {record, score}

    channels = [
        state.get("embedding_hits") or [],
        state.get("hyde_hits") or [],
        state.get("question_filter_hits") or [],
    ]
    active_channels = sum(1 for c in channels if c)
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
    logger.info(f"RRF 融合：{active_channels} 路有效通道、{sum(len(c) for c in channels)} 条输入 "
                f"→ {len(fused_records)} 条去重融合")
    return {"fused_results": fused_records}
