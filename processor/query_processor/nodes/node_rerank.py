"""
重排节点
对应 PRD V1.3 3.6 语义通道：BGE-Reranker 对 RRF 融合结果精排，取 Top-K
"""
from common.logging.logger import logger, node_log, step_log
from utils.lm.reranker_utils import get_reranker_model

# 重排输入上限（控制重排计算量）与最终输出条数
# 候选池规模：三路召回（HyDE 开启）约 30~40 条；HyDE 关闭（默认）约 20~25 条。
# 早先 20 会截掉 RRF 排名 21+ 的记录，文档类 golden 块常落在 25~35 名（doc 只有 2 路命中），故放宽到 30。
RERANK_INPUT_LIMIT = 30
FINAL_TOP_K = 5


@node_log("node_rerank")
def node_rerank(state: dict) -> dict:
    """
    重排节点：BGE-Reranker 对融合结果精排
    - 输入：query + 各候选的 content
    - 输出：按相关性分数降序的 Top-K
    - 模型失败降级：直接返回 RRF 融合序（标记 degraded）
    """
    # 1. 校验并截取重排候选
    data = step_1_validate_and_get_candidates(state)
    if data is None:
        return {"reranked_results": []}
    query, candidates = data

    # 2. 逐候选打分并排序
    scored = step_2_score_candidates(query, candidates)
    if scored is None:
        logger.warning("重排失败，降级使用 RRF 融合序")
        return {"reranked_results": candidates[:FINAL_TOP_K], "degraded": True}

    # 3. 取 Top-K 并回写重排分
    top_k = step_3_build_top_k(scored)
    if scored:
        logger.info(f"重排完成：{len(candidates)} 条候选 → Top {len(top_k)}，最高分 {scored[0][1]:.4f}")
    return {"reranked_results": top_k}


@step_log("step_1_validate_and_get_candidates")
def step_1_validate_and_get_candidates(state: dict):
    """
    校验并截取重排候选集
    :return: (query, candidates) 或 None（无融合结果）
    """
    query = state.get("rewritten_query") or state.get("original_query", "")
    fused = state.get("fused_results") or []

    if not fused:
        logger.info("无融合结果需要重排")
        return None

    candidates = fused[:RERANK_INPUT_LIMIT]
    logger.info(f"重排候选：{len(fused)} 条融合结果 → 取前 {len(candidates)} 条")
    return query, candidates


@step_log("step_2_score_candidates")
def step_2_score_candidates(query: str, candidates: list):
    """
    BGE-Reranker 逐候选打分（模型失败抛异常由上层降级）
    :return: [(record, score), ...] 已按分数降序；None 表示打分失败
    """
    try:
        reranker = get_reranker_model()
        pairs = [[query, c.get("content", "") or ""] for c in candidates]
        scores = reranker.compute_score(pairs, normalize=True)
        # 单条结果时 compute_score 返回标量
        if not isinstance(scores, list):
            scores = [scores]

        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: -(x[1] if x[1] is not None else -1e9))
        return scored

    except Exception as e:
        logger.warning(f"重排打分失败：{e}")
        return None


@step_log("step_3_build_top_k")
def step_3_build_top_k(scored: list) -> list:
    """
    取 Top-K 并回写重排分（便于评估脚本读取真实 rerank 排序依据）
    """
    for record, score in scored:
        record["rerank_score"] = None if score is None else round(float(score), 6)
    return [c for c, s in scored[:FINAL_TOP_K]]
