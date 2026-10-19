"""
Milvus 入库节点
对应 PRD V1.3 4.3：4 个 Collection 独立 Schema + 倒排索引 + HNSW/稀疏索引
对应 PRD 3.2 幂等策略：按「业务主键」upsert，禁止按文件名批量先删
"""
import time

from pymilvus import DataType

from common.logging.logger import logger, node_log, step_log
from utils.clients.milvus_utils import get_milvus_client

# ---- 各 Collection 的业务主键字段（幂等 upsert 依据，PRD 3.2）----
COLLECTION_PRIMARY_KEYS = {
    "edu_course_series": "series_code",
    "edu_course_module": "course_code",
    "edu_course_doc": "id",
    "edu_question": "question_code",
}

# ---- 各 Collection 的标量字段定义（PRD 4.3 字段清单表）----
# 格式：(字段名, 数据类型, 最大长度或None)
_SCALAR_FIELDS = {
    "edu_course_series": [
        ("id", DataType.VARCHAR, 64),
        ("series_code", DataType.VARCHAR, 64),
        ("series_name", DataType.VARCHAR, 128),
        ("description", DataType.VARCHAR, 512),
        ("course_category", DataType.VARCHAR, 256),
        ("target_audience", DataType.VARCHAR, 256),
        ("learning_goal", DataType.VARCHAR, 256),
        ("suitable_grade", DataType.VARCHAR, 128),
        ("content", DataType.VARCHAR, 2048),
        ("source_file", DataType.VARCHAR, 256),
        ("source_path", DataType.VARCHAR, 512),
        ("created_at", DataType.INT64, None),
    ],
    "edu_course_module": [
        ("id", DataType.VARCHAR, 64),
        ("course_code", DataType.VARCHAR, 64),
        ("course_name", DataType.VARCHAR, 128),
        ("series_code", DataType.VARCHAR, 64),
        ("series_name", DataType.VARCHAR, 128),
        ("duration", DataType.INT32, None),
        ("credit_hours", DataType.FLOAT, None),
        ("description", DataType.VARCHAR, 512),
        ("content", DataType.VARCHAR, 2048),
        ("source_file", DataType.VARCHAR, 256),
        ("source_path", DataType.VARCHAR, 512),
        ("created_at", DataType.INT64, None),
    ],
    "edu_course_doc": [
        ("id", DataType.VARCHAR, 64),
        ("doc_kind", DataType.VARCHAR, 16),
        ("course_code", DataType.VARCHAR, 64),
        ("series_code", DataType.VARCHAR, 64),
        ("project_name", DataType.VARCHAR, 256),
        # 注意：VARCHAR max_length 按 UTF-8 **字节** 计，中文约 3 字节/字
        # chapter_name 是「一级 > 二级」章节路径，256 字节约 85 汉字
        ("chapter_name", DataType.VARCHAR, 256),
        ("chunk_index", DataType.INT32, None),
        # 分块器 CHUNK_MAX_SIZE=3000（字符）→ 中文最坏 9000 字节，故取 16384
        ("content", DataType.VARCHAR, 16384),
        ("source_file", DataType.VARCHAR, 256),
        ("source_path", DataType.VARCHAR, 512),
        ("created_at", DataType.INT64, None),
    ],
    "edu_question": [
        ("id", DataType.VARCHAR, 64),
        ("question_code", DataType.VARCHAR, 64),
        ("bank_code", DataType.VARCHAR, 64),
        ("bank_name", DataType.VARCHAR, 128),
        ("question_type", DataType.VARCHAR, 32),
        ("group_id", DataType.VARCHAR, 64),
        ("content", DataType.VARCHAR, 8192),
        ("options", DataType.JSON, None),
        ("answer", DataType.VARCHAR, 2048),
        ("answer_raw", DataType.VARCHAR, 2048),
        ("analysis", DataType.VARCHAR, 4096),
        ("source_file", DataType.VARCHAR, 256),
        ("source_path", DataType.VARCHAR, 512),
        ("created_at", DataType.INT64, None),
    ],
}

# 建倒排索引的标量字段（PRD 4.3 标注「倒排」的字段）
_INVERTED_INDEX_FIELDS = {
    "edu_course_series": ["series_code", "series_name", "course_category", "target_audience", "suitable_grade"],
    "edu_course_module": ["course_code", "course_name", "series_code"],
    "edu_course_doc": ["doc_kind", "course_code", "series_code", "project_name", "chapter_name"],
    "edu_question": ["question_code", "bank_code", "bank_name", "question_type", "group_id"],
}

# 可空字段（None 允许；Milvus 标量字段插入 None 即跳过）
_NULLABLE_FIELDS = {
    "edu_course_series": [],
    "edu_course_module": ["duration", "credit_hours"],
    "edu_course_doc": ["course_code", "series_code", "project_name", "chapter_name", "chunk_index"],
    "edu_question": ["group_id", "options", "analysis"],
}


def _ensure_collection(client, collection_name: str):
    """
    确保 Collection 存在（幂等建表：不存在则按 Schema 创建）
    :param client: MilvusClient 实例
    :param collection_name: 集合名
    :return: True=已就绪 / False=创建失败
    """
    if client.has_collection(collection_name):
        return True

    from pymilvus import MilvusClient

    schema = MilvusClient.create_schema(api_version="V2", auto_id=False, enable_dynamic_field=False)
    nullable_fields = set(_NULLABLE_FIELDS.get(collection_name, []))
    for field_name, data_type, max_len in _SCALAR_FIELDS[collection_name]:
        kwargs = {
            "datatype": data_type,
            "is_primary": field_name == "id",
            # 可空字段显式声明 nullable（Milvus 要求：省略插入的字段必须 nullable 或有默认值）
            "nullable": field_name in nullable_fields,
        }
        if max_len is not None:
            kwargs["max_length"] = max_len
        schema.add_field(field_name=field_name, **kwargs)

    # 双向量字段：稠密 1024 维（BGE-M3）+ 稀疏
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    # 索引参数：稠密 HNSW + 稀疏 SPARSE_INVERTED_INDEX（PRD 4.3 索引规则）
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        metric_type="IP",
        params={"M": 16, "efConstruction": 200},
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
    )
    # 标量倒排索引
    for field_name in _INVERTED_INDEX_FIELDS[collection_name]:
        index_params.add_index(field_name=field_name, index_type="INVERTED")

    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
    logger.info(f"创建 Collection[{collection_name}]（含 HNSW/稀疏/倒排索引）")
    return True


def _clip_varchar(row: dict, collection_name: str) -> dict:
    """
    按 Schema 的 max_length（UTF-8 字节）截断超长 VARCHAR 字段。

    背景：Milvus 的 VARCHAR 上限按字节计，而分块/解析参数普遍按字符计，
    中文 3 字节/字 → 3000 字的上限会被算成 9000 字节。不截断时 Milvus 会
    整条记录报错（code=1100），当前实现只记日志跳过 → 分块静默丢失。
    """
    limits = {
        name: max_len
        for name, dtype, max_len in _SCALAR_FIELDS.get(collection_name, [])
        if dtype == DataType.VARCHAR and max_len
    }
    for field, max_bytes in limits.items():
        value = row.get(field)
        if not isinstance(value, str):
            continue
        raw = value.encode("utf-8")
        if len(raw) <= max_bytes:
            continue
        # errors="ignore" 防止切断多字节字符产生非法 UTF-8
        clipped = raw[:max_bytes].decode("utf-8", "ignore")
        logger.warning(
            f"[{collection_name}] 字段 {field} 超长 {len(raw)}B > {max_bytes}B，已截断至 {len(clipped.encode('utf-8'))}B"
        )
        row[field] = clipped
    return row


def _to_row(record: dict, collection_name: str) -> dict:
    """
    将解析记录转为 Milvus 行数据（补 id/created_at，剔除 parser 内部字段）
    注意：值为 None 的标量字段直接省略（Milvus VARCHAR 不接受显式 None，省略即存默认空值）
    """
    import uuid
    scalar_names = {f[0] for f in _SCALAR_FIELDS[collection_name]}
    row = {}
    for k, v in record.items():
        if k not in scalar_names:
            continue
        if v is None:
            continue  # 省略空值字段（group_id/options/analysis 等可空字段）
        row[k] = v
    row.setdefault("id", str(uuid.uuid4()))
    row["created_at"] = int(time.time())
    row["dense_vector"] = record.get("dense_vector")
    row["sparse_vector"] = record.get("sparse_vector")
    return _clip_varchar(row, collection_name)


@node_log("node_import_milvus")
def node_import_milvus(state: dict) -> dict:
    """
    入库节点：按幂等键（业务主键）upsert 各 Collection
    - 已存在同主键记录 → 先按主键精确删除该条，再插入（upsert 语义，PRD P1-2）
    - 禁止按文件名批量删除
    :param state: 导入图状态（pending_collections 含向量结果）
    :return: 状态更新（import_result 统计）
    """
    # 1. 校验待入库数据
    pending = step_1_validate_and_get_data(state)
    if pending is None:
        return {"import_result": {"inserted": 0, "updated": 0, "collections": {}},
                "import_error": "无可入库记录"}

    # 2. 获取 Milvus 客户端
    client = step_2_get_milvus_client()
    if client is None:
        return {"import_result": {"inserted": 0, "updated": 0, "collections": {}},
                "import_error": "Milvus 客户端不可用"}

    # 3. 逐集合 upsert
    inserted, updated, collection_stats = step_3_upsert_collections(client, pending)
    return {"import_result": {"inserted": inserted, "updated": updated, "collections": collection_stats}}


@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state: dict):
    """
    校验并获取待入库记录集合
    :return: {collection_name: records} 或 None（无可入库记录）
    """
    pending = state.get("pending_collections") or {}
    if not pending:
        logger.warning("pending_collections 为空，无可入库记录")
        return None
    logger.info(f"待入库集合：{ {name: len(rows) for name, rows in pending.items()} }")
    return pending


@step_log("step_2_get_milvus_client")
def step_2_get_milvus_client():
    """获取 Milvus 客户端"""
    client = get_milvus_client()
    if client is None:
        logger.error("Milvus 客户端不可用，无法入库")
    return client


@step_log("step_3_upsert_collections")
def step_3_upsert_collections(client, pending: dict):
    """
    逐集合按业务主键 upsert（存在则先删后插，不存在直接插入）
    :return: (inserted, updated, collection_stats)
    """
    inserted = updated = 0
    collection_stats = {}

    for collection_name, records in pending.items():
        if not _ensure_collection(client, collection_name):
            collection_stats[collection_name] = {"error": "Collection 创建失败"}
            continue

        pk_field = COLLECTION_PRIMARY_KEYS[collection_name]
        ins_cnt = upd_cnt = 0

        for record in records:
            if not record.get("dense_vector"):
                logger.warning(f"记录缺少向量，跳过：{record.get(pk_field)}")
                continue
            row = _to_row(record, collection_name)
            try:
                # 幂等 upsert：按业务主键查已有记录，存在则先删后插
                pk_value = row.get(pk_field)
                if pk_field == "id" or not pk_value:
                    # course_doc：id 为自动生成的 uuid，无业务主键 → 直接插入
                    client.insert(collection_name=collection_name, data=[row])
                    ins_cnt += 1
                    continue
                expr = f'{pk_field} == "{pk_value}"'
                existing = client.query(collection_name=collection_name, filter=expr, output_fields=["id"])
                if existing:
                    client.delete(collection_name=collection_name, ids=[e["id"] for e in existing])
                    upd_cnt += 1
                else:
                    ins_cnt += 1
                client.insert(collection_name=collection_name, data=[row])
            except Exception as e:
                logger.error(f"记录入库失败 [{collection_name}] {record.get(pk_field)}：{e}")

        collection_stats[collection_name] = {"inserted": ins_cnt, "updated": upd_cnt}
        inserted += ins_cnt
        updated += upd_cnt
        logger.info(f"Collection[{collection_name}] 入库完成：新增 {ins_cnt}，更新 {upd_cnt}")

    return inserted, updated, collection_stats
