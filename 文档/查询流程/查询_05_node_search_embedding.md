# 查询_05 node_search_embedding —— 语义通道：扇出 + 三路召回

> 文件：`processor/query_processor/nodes/node_search_embedding.py`（一个文件里放了 4 个节点）
> 职责：content 意图下，用三种不同角度召回候选，交给 RRF 融合。

## 一、为什么要三路

单一路径的召回有结构性缺陷，三路是互补的：

| 路 | 节点 | 思路 | 补什么短板 |
|---|---|---|---|
| ① 原查询 | `node_search_embedding` | query 直接编码检索 | 基线，稳定 |
| ② HyDE | `node_search_embedding_hyde` | 先让 LLM 写一段"假设答案"，用答案去检索 | **问句与文档文体不对齐**：用户问"状态管理怎么做"，文档写的是陈述句"State 是共享数据结构…"，问句向量和陈述句向量有距离；假设答案是陈述句，距离更近 |
| ③ 实体过滤 | `node_question_filter` | query 编码 + 标量预过滤（题型/题库） | 纯向量检索会被"语义相近但题型不对"的题占满 Top-5 |

三路并行跑，然后 RRF 融合（`查询_06`）。

### 1.1 现状：HyDE 默认关闭（A/B 结论）

2026-10-19 用 32 条黄金集做了 A/B（同一数据集、同一进程参数，只切 `ENABLE_HYDE`）：

| 指标 | HyDE 开 | HyDE 关 | 差值 |
|---|---|---|---|
| 题目通道 Hit@5 / MRR | 1.0 / 0.9583 | 1.0 / 0.9583 | 0 |
| 全量 Hit@5 | 0.375 | 0.375 | 0 |
| content 平均耗时 | 7.23s | **5.02s** | **−2.21s（−31%）** |
| 32 条总耗时 | 217.5s | 154.3s | −63s |

题目通道 12 条用例的 top5 有 7 条排序发生变化，但 **Hit/MRR/Recall 完全一致**——HyDE 没有改变任何一个用例的命中结果；而它每次查询要花约 2 秒（LLM 生成假设答案 + 编码 + 一路检索）。

**文档通道这一轮无法评估**：黄金集的 `doc:` 键与当前库不同步（详见本节末尾的"注意"），两边都是 0 命中。所以结论只覆盖题目通道，文档通道等黄金集重建后再复核。

因此代码里默认 `ENABLE_HYDE=0`：

```python
# 需要时 ENABLE_HYDE=1 打开（重启进程生效）
ENABLE_HYDE = os.getenv("ENABLE_HYDE", "0") == "1"
```

关闭时 `node_search_embedding_hyde` 在函数入口直接返回 `{"hyde_hits": []}`，**不调 LLM、不编码、不检索**——不是"跑了但丢弃"，那省不下 2 秒。图结构不变，RRF 自动少一路。

> **注意（黄金集漂移）**：Word 改走 MinerU 后，`chapter_name` 从 `LangGraph概述 > 状态` 变成 `1 LangGraph概述 > 3 状态`（带章节编号）；同时库里新增了掌柜智库文档、部分讲义版本也变了（`深度学习2.0.0` vs 黄金集的 `2.2.3`）。结果是所有 `doc:` 类型的 golden 都对不上，文档通道指标在**任何**实验里都会显示 0。要恢复可测性，需要二选一：
> 1. 在 `node_course_doc_parser._emit_chunk` 里剥掉章节编号（`^\d+(\.\d+)*[.、]?\s*`），再重导文档集合；
> 2. 或者按当前 `chapter_name` 重建黄金集。

### 1.1 扇出节点

`node_semantic_fanout` 定义在 `main_graph.py` 里：

```python
def node_semantic_fanout(state: QueryGraphState) -> dict:
    """
    语义通道扇出节点（空操作）：
    条件路由只能命中单个目标，用本节点把「content」分流成三路并行分支，
    自身不读写任何状态字段。
    """
    return {}
```

它存在的唯一原因是 LangGraph 的限制（条件边只能返回一个目标）。返回 `{}` 表示不改状态。

## 二、公共检索函数

三个节点共用 `_hybrid_search_collection`：

```python
def _hybrid_search_collection(client, collection: str, emb: dict, expr: str = None):
    try:
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
```

### 2.1 为什么是 `hybrid_search` 而不是两次 `search`

Milvus 的 `hybrid_search` 接收多个 `AnnSearchRequest` 和一个 `ranker`，在服务端完成"两路结果融合 + 排序"，只返回一份 Top-K。

如果用两次 `search` 再在 Python 里合并，就得自己实现分数归一化和去重——而两路分数尺度不同（dense 是归一化内积，sparse 是稀疏权重和），自己加权很容易失衡。交给 `WeightedRanker` 更稳。

`utils/clients/milvus_utils.py` 里的实现：

```python
rerank = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)   # 默认 (0.5, 0.5)
res = client.hybrid_search(collection_name=..., reqs=reqs, ranker=rerank, limit=limit, ...)
```

### 2.2 `metric_type="IP"` 必须与建库一致

导入时 HNSW 索引建的是 `IP`（内积），检索也必须用 `IP`。因为 BGE-M3 开了 `normalize_embeddings=True`，内积等价于余弦（见 `导入_07` 5.2）。

用错 metric 不会报错，但排序结果会变得没有意义——这是"静默降级"型故障。

### 2.3 `expr` 同时作用于两路

`AnnSearchRequest` 的 `expr` 是**标量预过滤**，Milvus 先在标量索引上筛出候选，再在候选里做向量检索。

必须给**两个 request 都传**：只给 dense 传的话，稀疏那一路会检索全库，返回的候选和 dense 的过滤条件不一致，融合后混进不该出现的记录。

### 2.4 `_score` 的来源

```python
entity["_score"] = hit.get("distance", 0)
```

`hybrid_search` 返回的 `distance` 是 **ranker 融合后的分数**（不是原始 cosine）。它只用于日志和调试，RRF 融合**不看这个分**（RRF 只看排名）。

### 2.5 异常吞掉的代价

```python
except Exception as e:
    logger.error(...)
    return []
```

单个集合检索失败 → 该集合 0 条，其他集合照常。好处是"一个集合的索引坏了不至于整条链路挂掉"；代价是**部分召回缺失不会体现在 `error` 字段里**，答案看起来正常但召回少了一块。

排查"某类内容总是搜不到"时，第一步应该是搜日志里的"混合检索失败"。

## 三、`_OUTPUT_FIELDS`：一个字段缺失导致的召回塌陷

```python
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
```

注释特意写了 `edu_course_doc` 为什么要带 `id`：

> 注意：`edu_course_doc` 无业务编码，必须返回主键 id，否则 RRF 去重键退化为 `(collection, None)`，同集合的所有文档块会被当成同一条记录合并（只剩 1 块）。

这是修过的一个真实 bug：文档类问题本来命中 8 个块，融合后只剩 1 条，因为 8 个块的 `question_code/series_code/course_code` 全是 `None`，去重键都一样。

**教训**：加集合、改字段时，必须同步检查 RRF 的去重键能不能取到值（见 `查询_06` 第三节）。

## 四、① `node_search_embedding`：原查询检索

```python
@node_log("node_search_embedding")
def node_search_embedding(state: dict) -> dict:
    data = step_1_validate_and_get_data(state)
    if data is None:
        return {"error": "Milvus 客户端不可用"}
    query, client, collections = data

    emb, error = step_2_encode_query(query, "原查询")
    if emb is None:
        return {"error": error}

    hits = step_3_search_collections(client, collections, emb, "原查询")
    return {"embedding_hits": hits}
```

三步：校验 → 编码一次 → 逐集合检索。

### 4.1 编码一次，循环复用（性能关键）

```python
@step_log("step_3_search_collections")
def step_3_search_collections(client, collections: list, emb: dict, channel_name: str, expr: str = None) -> list:
    hits = []
    for collection in collections:
        hits.extend(_hybrid_search_collection(client, collection, emb, expr=expr))
    logger.info(f"{channel_name}语义召回：{len(hits)} 条（集合 {collections}）")
    return hits
```

`emb` 是**参数传入**的，不是每个集合内部编码。BGE-M3 前向是这条链路上最贵的 CPU 操作（百毫秒级），4 个集合各编一次就是 4 倍浪费。

文件头部的注释把这条写成了约定：

> 性能约定：查询向量在**每个节点内只编码一次**，循环各集合时复用（BGE-M3 前向是主要耗时项）。

之前是每个集合单独调用 `generate_query_embeddings`，同一个 query 编了 4 次——纯粹浪费。

### 4.2 用 `encode_queries` 而不是 `encode_documents`

```python
emb = generate_query_embeddings([text])       # 内部调 model.encode_queries
```

BGE-M3 是**非对称检索模型**：文档侧用 `encode_documents`，查询侧用 `encode_queries`（内部附加 query instruction）。

用反了不会报错，但**向量分布不对齐，召回率明显下降**——这是本项目历史上出现过的问题（查询侧误用 `encode_documents`）。

### 4.3 `error` 的语义

```python
if data is None:
    return {"error": "Milvus 客户端不可用"}
...
if emb is None:
    return {"error": error}       # "原查询向量生成失败：..."
```

写 `error` 是为了让 `node_answer_output` 能短路成"检索服务暂时不可用"，而不是"未找到相关内容"。

但注意：**只有原查询这一路写 `error`**。HyDE 和 filter 两路失败时只返回空列表，不写 `error`。这是合理的——只要有一路成功，就不算"服务不可用"。

## 五、② `node_search_embedding_hyde`：假设答案检索

```python
hyde_text = step_2_generate_hyde_answer(query)
emb, _ = step_2_encode_query(hyde_text, "HyDE")
if emb is None:
    return {"hyde_hits": []}
hits = step_3_search_collections(client, collections, emb, "HyDE")
```

### 5.1 生成假设答案

```python
hyde_text = query
try:
    prompt = load_prompt("hyde_prompt", rewritten_query=query)
    response = get_llm_client().invoke(prompt)
    hyde_text = response.content if hasattr(response, "content") else str(response)
    logger.info(f"HyDE 假设答案生成成功（{len(hyde_text)} 字）")
except Exception as e:
    logger.warning(f"HyDE 生成失败，降级用原查询检索：{e}")
return hyde_text
```

**降级点是 `hyde_text = query` 这个初值**：LLM 失败时函数返回原查询，后续编码的就是原 query——这一路退化成"和第二路一样的检索"，虽然冗余，但保证通道可用。

Prompt（`hyde_prompt.prompt`）：

```
请基于以下用户查询生成一个简洁的回答范文。
用户查询: {rewritten_query}
要求：
1. 回答要简洁明了，包含核心信息即可
2. 假设你是该领域的专家，提供专业的解释
3. 不要使用"假设"、"可能"等不确定的词汇
4. 保持回答与查询主题高度相关
5. 使用中文回答且不超过300字
```

第 3 条很重要：HyDE 生成的文本要尽量像"真实文档"，出现"这可能是因为…"这种不确定表述会引入噪音。第 5 条控制在 300 字，避免向量被泛泛而谈稀释。

### 5.2 编码失败为什么只返回空

```python
emb, _ = step_2_encode_query(hyde_text, "HyDE")   # 错误信息被忽略
if emb is None:
    return {"hyde_hits": []}
```

因为这一路是**增强项**：HyDE 挂了，原查询那一路还在，RRF 照样能融合。写 `error` 反而会让 `node_answer_output` 误判"服务不可用"。

### 5.3 HyDE 的代价

多一次 LLM 调用（约 1~2 秒）。但因为它和另外两路**并行**，实际只增加 `max(三路)` 的耗时——这就是为什么要把它放进扇出而不是串在原查询后面。

## 六、③ `node_question_filter`：实体预过滤检索

```python
@node_log("node_question_filter")
def node_question_filter(state: dict) -> dict:
    data = step_1_filter_validate_and_get_data(state)
    if data is None:
        return {"error": "Milvus 客户端不可用"}
    entities, client, query = data

    expr = step_2_build_filter_expr(entities)
    if not expr:
        logger.info("实体预过滤通道：无可过滤实体，跳过")
        return {"question_filter_hits": []}

    emb, _ = step_2_encode_query(query, "实体预过滤")
    if emb is None:
        return {"question_filter_hits": []}

    hits = step_3_search_collections(client, ["edu_question"], emb, "实体预过滤", expr=expr)
    return {"question_filter_hits": hits}
```

### 6.1 无实体就跳过（省一次编码 + 一次检索）

```python
expr = step_2_build_filter_expr(entities)
if not expr:
    return {"question_filter_hits": []}
```

判断放在**编码之前**——没有过滤条件时这一路就是"原查询检索"的重复，跑它纯浪费。

### 6.2 过滤表达式

```python
def step_2_build_filter_expr(entities: dict) -> str:
    conds = []
    if entities.get("question_type_cn"):
        type_en = QUESTION_TYPE_MAP.get(entities["question_type_cn"])
        if type_en:
            conds.append(f'question_type == "{type_en}"')
    if entities.get("bank_name"):
        conds.append(f'bank_name like "%{entities["bank_name"]}%"')
    expr = " and ".join(conds)
    return expr
```

只支持两个维度：题型（英文枚举，`==`）和题库名（`like`）。没有题码——题码是精确查询，走 structured 通道更合适。

### 6.3 ⚠️ 这里没有转义 `like` 通配符

对比 `node_structured_query`：

```python
# structured
conds.append(f'bank_name like "%{_escape_like(entities["bank_name"])}%"')
# question_filter
conds.append(f'bank_name like "%{entities["bank_name"]}%"')
```

`node_question_filter` **没做 `escape_like_term`**。影响：

- 题库名里出现 `%` 或 `_` 时，会被当成通配符（题库名实际很少含这两个字符，风险低）；
- 更重要的是**与 structured 通道口径不一致**——同一实体在两个通道的匹配行为不同。

这是一处应该统一的技术债。修法很简单：import `escape_like_term` 后包一层。

### 6.4 只查 `edu_question`

```python
hits = step_3_search_collections(client, ["edu_question"], emb, "实体预过滤", expr=expr)
```

写死单个集合，不读 `search_collections`。因为题型/题库过滤条件只存在于 `edu_question` 的 schema 里，对课程集合用这些条件是无效的（字段不存在会报错）。

## 七、三路的产出规模

```
每路：len(search_collections) × 5 条
默认 4 个集合：
  ① 原查询     → 最多 20 条
  ② HyDE       → 最多 20 条（默认关闭，实际 0 条）
  ③ 实体过滤   → 最多 5 条（仅 question）；无实体时 0 条

总候选池：HyDE 关闭时约 20~25 条；开启时约 40~45 条（去重前）
```

这个量级直接决定了 `node_rerank` 的 `RERANK_INPUT_LIMIT`（30）和 RRF 的排序特性（见 `查询_07`）。

**为什么每集合只取 5 条**：`TOP_K_PER_COLLECTION=5` 是精度与成本的折中。取 10 条会让候选池到 80+，重排成本翻倍，而 RRF 之后进 Top-5 的基本还是前几条。

## 八、调试手册

| 现象 | 原因 | 排查 |
|---|---|---|
| 某类内容永远搜不到 | 某集合检索一直失败被吞 | 搜日志"混合检索失败\[集合名\]" |
| 文档类只召回 1 条 | `_OUTPUT_FIELDS["edu_course_doc"]` 少了 `id` | 见第三节 |
| 检索结果和问题无关 | 用了 `encode_documents` 而不是 `encode_queries` | 见 4.2 |
| 题型过滤不生效 | 中文题型没转英文 | 查 `QUESTION_TYPE_MAP` |
| `question_filter_hits` 永远空 | 实体被防护逻辑删光了 | 看 intent 节点日志的 `entities` |
| HyDE 日志显示"降级用原查询" | LLM 超时/额度 | 正常降级，两路变一路 |
| 检索整体很慢 | 每集合重复编码（约定被破坏） | 确认 `emb` 是参数传入 |
| 报 `error` 且答案"服务不可用" | Milvus 客户端不可用 | 检查 `MILVUS_URL` |

## 九、设计取舍

| 决定 | 替代方案 | 为什么选现在这个 |
|---|---|---|
| 三路并行 | 串行叠加 | 耗时取 max 而非 sum |
| HyDE 用 LLM 写答案 | 只做 query 扩展 | 答案文体更接近文档 |
| 实体过滤只查题目库 | 查全部集合 | 题型/题库字段只存在于题目 schema |
| 每集合 5 条 | 10~20 条 | 候选池太大拖慢重排 |
| 编码一次复用 | 每集合编码 | BGE-M3 前向是主要成本 |
| 单集合失败返回空 | 整体失败 | 部分召回优于全挂 |
