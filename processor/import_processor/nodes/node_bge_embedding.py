"""
BGE-M3 向量化节点
对应 PRD V1.3 3.2：将解析出的各类型记录的 content 字段向量化（稠密+稀疏）
向量化前先把各类记录按目标 Collection 分组，写入 pending_collections
"""
from common.logging.logger import logger, node_log, step_log

from utils.lm.embedding_utils import generate_embeddings

# 批量向量化大小（BGE-M3 批量推理，2,628 条实体分批处理）
EMBED_BATCH_SIZE = 32


@node_log("node_bge_embedding")
def node_bge_embedding(state: dict) -> dict:
    """
    向量化节点：
    1. 汇总各内容类型记录到 pending_collections（collection_name -> records）
    2. 对每条记录的 content 字段生成 BGE-M3 混合向量（dense/sparse）
    :param state: 导入图状态
    :return: 状态更新（pending_collections 含向量结果）
    """
    # 1. 按目标 Collection 分组（PRD 4.3：4 个 Collection）
    pending, total = step_1_group_records_by_collection(state)
    if total == 0:
        logger.warning("无可向量化的记录（解析结果为空或已失败）")
        return {"pending_collections": pending, "embedding_error": "无可向量化的记录"}

    # 2. 分批向量化（按集合轮询，单批 EMBED_BATCH_SIZE 条）
    try:
        step_2_batch_generate_vectors(pending, total)
        logger.success(f"全量向量化完成：{total} 条记录")
        return {"pending_collections": pending}

    except Exception as e:
        logger.exception("向量化节点执行失败")
        return {"pending_collections": pending, "embedding_error": str(e)}


@step_log("step_1_group_records_by_collection")
def step_1_group_records_by_collection(state: dict):
    """
    把各内容类型的解析记录按目标 Collection 分组
    :return: (pending, 记录总数)
    """
    pending = {}

    if state.get("course_series_records"):
        pending["edu_course_series"] = state["course_series_records"]
    if state.get("course_module_records"):
        pending["edu_course_module"] = state["course_module_records"]
    if state.get("question_records"):
        pending["edu_question"] = state["question_records"]

    # course_doc 分块记录（V1 预留：course_doc 解析产出）
    if state.get("course_doc_records"):
        pending["edu_course_doc"] = state["course_doc_records"]

    total = sum(len(records) for records in pending.values())
    logger.info(f"向量化分组完成：{list(pending.keys())}，共 {total} 条记录")
    return pending, total


@step_log("step_2_batch_generate_vectors")
def step_2_batch_generate_vectors(pending: dict, total: int):
    """
    分批向量化：逐集合按 EMBED_BATCH_SIZE 切批调用 BGE-M3，向量回写到记录
    """
    done_count = 0
    for collection_name, records in pending.items():
        texts = [r["content"] for r in records]
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch_texts = texts[i:i + EMBED_BATCH_SIZE]
            batch_records = records[i:i + EMBED_BATCH_SIZE]
            embeddings = generate_embeddings(batch_texts)
            for j, record in enumerate(batch_records):
                record["dense_vector"] = embeddings["dense"][j]
                record["sparse_vector"] = embeddings["sparse"][j]
            done_count += len(batch_texts)
            logger.info(f"向量化进度：{done_count}/{total}（当前集合 {collection_name}）")
