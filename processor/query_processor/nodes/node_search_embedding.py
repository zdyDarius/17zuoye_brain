"""
语义检索通道节点
对应 PRD V1.3 3.6 内容型意图：
- node_search_embedding：原查询 BGE-M3 混合向量检索（多集合）
- node_search_embedding_hyde：HyDE 假设答案向量检索
- node_question_filter：按实体做标量预过滤的向量检索（题库/题型过滤）
三路结果经 node_rrf 融合 → node_rerank 精排
"""
from common.logging.logger import logger, node_log
from utils.clients.milvus_utils import get_milvus_client, hybrid_search
from utils.lm.embedding_utils import generate_embeddings
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt
from pymilvus import AnnSearchRequest

from processor.import_processor.nodes.node_question_parser import QUESTION_TYPE_MAP

# 语义检索各集合的返回字段
_OUTPUT_FIELDS = {
    "edu_question": ["question_code", "bank_name", "bank_code", "question_type",
                     "content", "answer", "analysis"],
    "edu_course_series": ["series_code", "series_name", "course_category",
                          "target_audience", "suitable_grade", "content"],
    "edu_course_module": ["course_code", "course_name", "series_name",
                          "duration", "credit_hours", "content"],
}

# 每集合召回条数
TOP_K_PER_COLLECTION = 5


def _hybrid_search_collection(client, collection: str, query_text: str, expr: str = None):
    """
    对单个集合做稠密+稀疏混合检索
    :param expr: 标量预过滤表达式（题库/题型实体过滤）
    :return: 命中列表（带 _collection 标记），失败返回 []
    """
    try:
        emb = generate_embeddings([query_text])
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
    原查询混合向量检索：对 3 个集合做混合检索（无标量预过滤）
    """
    query = state.get("rewritten_query") or state.get("original_query", "")
    client = get_milvus_client()
    if client is None:
        return {"error": "Milvus 客户端不可用"}

    hits = []
    for collection in state.get("search_collections", []):
        hits.extend(_hybrid_search_collection(client, collection, query))
    logger.info(f"原查询语义召回：{len(hits)} 条")
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

    hits = []
    for collection in state.get("search_collections", []):
        hits.extend(_hybrid_search_collection(client, collection, hyde_text))
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
        return {}

    query = state.get("rewritten_query") or state.get("original_query", "")
    hits = _hybrid_search_collection(client, "edu_question", query, expr=expr)
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
    fused_records = [item["record"] for item in fused]
    logger.info(f"RRF 融合：{sum(len(c) for c in channels)} 条输入 → {len(fused_records)} 条去重融合")
    return {"fused_results": fused_records}
