# 环境配置与依赖导入
import threading

from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from common.logging.logger import logger
from common.config.embedding_config import embedding_config

# 模型单例对象，避免重复初始化
_bge_m3_ef = None
# 模型初始化/推理锁：防止多任务并发初始化（meta tensor 竞态）与并发推理（MPS 设备非线程安全）
# RLock（可重入）：generate_embeddings 持锁后调用 get_bge_m3_ef() 需再次进入同一把锁
_bge_m3_lock = threading.RLock()

def get_bge_m3_ef():
    """
    获取BGE-M3模型单例对象，自动加载环境变量配置
    加锁初始化：防止多线程并发创建（meta tensor 竞态）
    :return: 初始化完成的BGEM3EmbeddingFunction实例
    """
    global _bge_m3_ef
    # 快路径：已初始化直接返回
    if _bge_m3_ef is not None:
        return _bge_m3_ef
    # 慢路径：加锁双重检查后初始化
    with _bge_m3_lock:
        if _bge_m3_ef is not None:
            return _bge_m3_ef
        # 从环境变量加载配置，无配置则使用默认值
        model_name = embedding_config.bge_m3_path or "BAAI/bge-m3"
        device = embedding_config.bge_device or "cpu"
        use_fp16 = embedding_config.bge_fp16 or False

        logger.info(
            "开始初始化BGE-M3模型",
            extra={"model_name": model_name, "device": device, "use_fp16": use_fp16}
        )

        try:
            # 初始化BGE-M3模型（首次调用时触发真实权重加载，确保模型就绪）
            _bge_m3_ef = BGEM3EmbeddingFunction(
                model_name=model_name,
                device=device,
                use_fp16=use_fp16,
                normalize_embeddings=True
            )
            # 触发一次预热推理（实例化真实 tensor，避免后续并发时 meta tensor 问题）
            _bge_m3_ef.encode_queries(["预热"])
            logger.success("BGE-M3模型初始化成功（含预热推理）")
            return _bge_m3_ef
        except Exception as e:
            logger.error(f"BGE-M3模型初始化失败：{str(e)}", exc_info=True)
            raise  # 向上抛出异常，由调用方处理


def _parse_bgem3_output(embeddings, text_count):
    """
    将 BGEM3EmbeddingFunction 的原始输出解析为可序列化结构
    :param embeddings: 模型输出 {"dense": np.ndarray, "sparse": csr_matrix}
    :param text_count: 输入文本条数（用于按 indptr 切分稀疏向量）
    :return: {"dense": [[float]], "sparse": [{index: weight}]}
    """
    # 初始化稀疏向量处理结果，解析为字典格式（适配序列化/存储）
    processed_sparse = []
    for i in range(text_count):
        # 提取第i个文本的稀疏向量索引：np.int64 → Python int（满足字典key可哈希要求）
        sparse_indices = embeddings["sparse"].indices[
            embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]
        ].tolist()
        # 提取第i个文本的稀疏向量权重：np.float32 → Python float（适配JSON序列化/接口返回）
        sparse_data = embeddings["sparse"].data[
            embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]
        ].tolist()
        # 构造{特征索引: 归一化权重}的稀疏向量字典
        sparse_dict = {k: v for k, v in zip(sparse_indices, sparse_data)}
        processed_sparse.append(sparse_dict)

    # 构造最终返回结果，稠密向量转列表（解决numpy数组不可序列化问题）
    return {
        "dense": [emb.tolist() for emb in embeddings["dense"]],  # 嵌套列表，与输入文本一一对应
        "sparse": processed_sparse  # 字典列表，模型已做L2归一化
    }


def generate_embeddings(texts):
    """
    为文本列表生成稠密+稀疏混合向量嵌入（**文档侧**：入库/建索引使用，模型原生L2归一化）
    :param texts: 要生成嵌入的文本列表，单文本也需封装为列表
    :return: 字典格式的向量结果，key为dense/sparse，对应嵌套列表/字典列表
    :raise: 向量生成过程中的异常，由调用方捕获处理
    """
    # 入参合法性校验
    if not isinstance(texts, list) or len(texts) == 0:
        logger.warning("生成向量入参不合法，texts必须为非空列表")
        raise ValueError("参数texts必须是包含文本的非空列表")

    logger.info(f"开始为{len(texts)}条文本生成混合向量嵌入")
    try:
        # 加锁推理：MPS 设备非线程安全，串行化避免并发计算竞态
        with _bge_m3_lock:
            # 加载BGE-M3模型单例
            model = get_bge_m3_ef()
            # 模型编码生成向量，返回dense（稠密向量）+sparse（CSR格式稀疏向量）
            embeddings = model.encode_documents(texts)
        logger.debug(f"模型编码完成，开始解析稀疏向量格式，共{len(texts)}条")

        result = _parse_bgem3_output(embeddings, len(texts))
        logger.success(f"{len(texts)}条文本向量生成完成，格式已适配工业级使用")
        return result

    except Exception as e:
        logger.error(f"文本向量生成失败：{str(e)}", exc_info=True)
        raise  # 不吞异常，向上传递让调用方做重试/降级处理


def generate_query_embeddings(texts):
    """
    为**查询**文本生成稠密+稀疏混合向量（检索侧专用）
    BGE-M3 是非对称检索模型：查询应走 encode_queries（内部附加 query instruction），
    与入库时的 encode_documents 分布对齐；若查询误用文档编码会削弱召回质量。
    :param texts: 查询文本列表，单条也需封装为列表
    :return: {"dense": [[float]], "sparse": [{index: weight}]}
    :raise: 编码异常向上抛出，由调用方降级
    """
    if not isinstance(texts, list) or len(texts) == 0:
        logger.warning("生成查询向量入参不合法，texts必须为非空列表")
        raise ValueError("参数texts必须是包含文本的非空列表")

    logger.debug(f"开始为{len(texts)}条查询生成混合向量嵌入")
    try:
        with _bge_m3_lock:
            model = get_bge_m3_ef()
            embeddings = model.encode_queries(texts)
        return _parse_bgem3_output(embeddings, len(texts))
    except Exception as e:
        logger.error(f"查询向量生成失败：{str(e)}", exc_info=True)
        raise


"""
核心设计亮点&适配说明：
1. 模型原生归一化：开启normalize_embeddings = True，自动对稠密+稀疏向量做L2归一化，完美适配Milvus IP内积检索（单位化后IP等价于余弦，计算更快）；
2. 彻底解决NumPy类型做key问题：sparse_indices加.tolist()，将np.int64转为Python原生int，满足字典key的可哈希要求，无报错风险；
3. 稀疏值适配序列化：sparse_data加.tolist()，将np.float32转为Python原生float，支持JSON写入/接口返回/Milvus入库等所有场景；
4. 单例模式优化：模型仅初始化一次，避免重复加载耗时耗资源，提升批量处理效率；
5. 格式匹配业务调用：返回dense嵌套列表、sparse字典列表，与vector_result["dense"][0]/sparse_vector["sparse"][0]取值逻辑完美契合；
6. 分级日志覆盖：从模型初始化、向量生成到异常报错，全流程日志记录，便于生产环境问题排查；
7. 入参合法性校验：防止空列表/非列表入参导致的内部报错，提升工具类健壮性。
"""