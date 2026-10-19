# 查询_06 node_rrf —— RRF 融合

> 文件：`processor/query_processor/nodes/node_search_embedding.py`（`node_rrf` 与三路召回同文件）
> 职责：把三路召回结果按**名次**融合成一份有序列表。不看重排分，只看"排第几"。

## 一、为什么用 RRF 而不是加权求和

三路召回的分数**不可比**：

| 路 | 分数来源 | 取值范围 |
|---|---|---|
| 原查询 | `hybrid_search` 的 WeightedRanker 输出（dense+sparse 加权） | 约 0.2 ~ 1.0 |
| HyDE | 同上，但 query 是假设答案 | 约 0.2 ~ 1.0，但分布不同 |
| 实体过滤 | 同上，且候选集被标量过滤缩窄 | 分布又不同 |

直接加权求和需要先归一化，而归一化的方式（min-max / z-score / softmax）会显著影响结果，且没有客观依据。

**RRF（Reciprocal Rank Fusion）绕开了这个问题**：只看名次，不看分数。

$$\text{score}(d) = \sum_{c \in C} \frac{1}{k + \text{rank}_c(d)}$$

名次是天然可比的——"在这路里排第 3"在任何一路里含义一致。

## 二、实现

```python
@node_log("node_rrf")
def node_rrf(state: dict) -> dict:
    channels, active_channels, total_input = step_1_collect_channels(state)
    fused_records = step_2_fuse_by_rrf(channels)
    logger.info(f"RRF 融合：{active_channels} 路有效通道、{total_input} 条输入 "
                f"→ {len(fused_records)} 条去重融合")
    return {"fused_results": fused_records}
```

```python
@step_log("step_2_fuse_by_rrf")
def step_2_fuse_by_rrf(channels: list) -> list:
    K = 60
    ranked = {}  # 唯一键 -> {record, score}

    for channel_hits in channels:
        for rank, hit in enumerate(channel_hits):
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
        item["record"]["_rrf_score"] = round(item["score"], 6)
        fused_records.append(item["record"])
    return fused_records
```

### 2.1 `k=60` 的作用

`k` 是**平滑常数**，控制"名次差异"的权重：

| k | 第 1 名 | 第 2 名 | 第 10 名 | 第 1 名/第 10 名 |
|---|---|---|---|---|
| 0 | 1.0 | 0.5 | 0.1 | 10× |
| 60 | 0.01639 | 0.01613 | 0.01408 | 1.16× |
| 1000 | 0.000999 | 0.000998 | 0.000990 | 1.01× |

k 越大，头部名次之间的差距越小，**"多路都出现"比"单路排第一"更重要**。

60 是 RRF 原始论文（Cormack et al. 2009）推荐值，也是业界默认。这里取 60 表示：**跨路共识的权重远大于单路内部的名次微调**。

### 2.2 为什么是 `K + rank + 1`

`rank` 从 0 开始，`+1` 让第一名进分母是 61（而不是 60）。这是为了让"第 0 名"也有意义，避免 k=0 时分母为 0。等价于"名次从 1 开始计"。

## 三、去重键：`(collection, 业务编码)`

```python
key = (
    hit.get("_collection"),
    hit.get("question_code") or hit.get("series_code")
    or hit.get("course_code") or hit.get("id"),
)
```

### 3.1 为什么必须带 `_collection`

不同集合的编码可能撞车（虽然是不同命名空间，但字符串层面可能相同的值），而且"同一条内容"在不同集合里就是不同实体。

带集合前缀后，`("edu_question", "q001")` 和 `("edu_course_module", "q001")` 不会被合并。

### 3.2 编码的优先级链

```
question_code → series_code → course_code → id
```

对应四个集合的业务主键（见 `导入_08`）。走的是"**哪个有值用哪个**"：

- 题目记录有 `question_code`；
- 系列有 `series_code`；
- 模块有 `course_code`；
- 文档块**没有业务编码**，只能退到主键 `id`。

### 3.3 这里有个必须记住的教训

如果 `edu_course_doc` 的查询没返回 `id`（`_OUTPUT_FIELDS` 里漏了），key 会变成：

```python
("edu_course_doc", None)
```

所有文档块的 key 完全一样 → **第一块进 `ranked`，后面 7 块全被当成重复而丢弃**。

表现：文档类问题的 `fused_results` 里只有 1 条 doc 记录，重排后自然也只有 1 条，答案只能引用一段。

修法就在 `node_search_embedding._OUTPUT_FIELDS["edu_course_doc"]` 里加 `id`（已在 `查询_05` 强调）。

## 四、一个可验证的算例

三路输入（简化为题目 + 文档）：

```python
a = [q1, q2]          # 原查询：q1 第 1、q2 第 2
b = [q2, d1]          # HyDE：  q2 第 1、d1 第 2
c = [q1]              # 实体过滤：q1 第 1
```

计算：

| 记录 | 参与的路与名次 | 分数 |
|---|---|---|
| `q1` | a 第 0（1/61）+ c 第 0（1/61） | **0.032787** |
| `q2` | a 第 1（1/62）+ b 第 0（1/61） | **0.032522** |
| `d1` | b 第 1（1/62） | **0.016129** |

实测输出：

```
q1 0.032787
q2 0.032522
d1 0.016129
```

**这个例子的意义**：`q1` 在两路都是第一，`q2` 在一路第一、一路第二。差距只有 0.000265——**非常小**。这就是 k=60 的效果：名次微调几乎不影响，**"被几路同时召回"才是决定性的**。

`d1` 只被一路召回，分数直接是别人一半。**跨路共识是 RRF 的核心信号**。

## 五、⚠️ 名次是"通道内的全局位置"，不是"集合内名次"

这是本节点最容易被忽略、影响却最大的一个细节。

```python
for collection in collections:
    hits.extend(_hybrid_search_collection(client, collection, emb, expr=expr))
```

`hits` 是**逐集合拼接**的列表。然后：

```python
for rank, hit in enumerate(channel_hits):     # rank 是拼接后的全局位置
```

也就是说，`rank` 是"这条记录在这条通道**整个候选列表**里的位置"，不是"它在自己集合里排第几"。

默认 `search_collections` 顺序：

```python
["edu_course_series", "edu_course_module", "edu_question", "edu_course_doc"]
```

每集合最多 5 条，所以各集合的 `rank` 区间是：

| 集合 | rank 区间 | 单路最高分 |
|---|---|---|
| `edu_course_series` | 0~4 | 1/61 = 0.01639 |
| `edu_course_module` | 5~9 | 1/66 = 0.01515 |
| `edu_question` | 10~14 | 1/71 = 0.01408 |
| `edu_course_doc` | 15~19 | 1/76 = 0.01316 |

**后果**：一个在 `edu_course_doc` 里排第 1 的块（rank=15），RRF 分数只有 0.01316，**低于**一个在 `edu_course_series` 里排第 5 的记录（0.01639）。

而文档只有两路召回（原查询 + HyDE，`question_filter` 只查 `edu_question`），所以：

```
文档块理论最高分 = 2 × 1/(60+15) ≈ 0.0267
题目理论最高分   = 3 × 1/(60+10) ≈ 0.0429
```

**文档类候选在 RRF 阶段就系统性吃亏**。这与重排节点注释里的观察完全吻合：

> 三路召回融合后候选池约 30~40 条（4 集合 × 2~3 路），20 会截掉 RRF 排名 21+ 的记录，**文档类 golden 块常落在 25~35 名**（RRF 对"多路共识"加权，doc 只有 2 路命中）

所以 `RERANK_INPUT_LIMIT` 从 20 放宽到 30——不是因为重排本身需要 30 条，而是**必须把 RRF 排名靠后的文档块捞进重排**，否则它们连被精排的机会都没有。

### 5.1 如果想从根本上改善

三个方向（当前都没做）：

1. **按集合内名次融合**：`key` 里带上集合内 rank，让每个集合的第一名都从 61 起算。实现上要在 `_hybrid_search_collection` 里记录 `_rank_in_collection`；
2. **调换集合顺序**：把 `edu_course_doc` 放前面。这是最省事的 hack，但会反过来压制题目召回；
3. **给 doc 路加权**：RRF 变体 `score = Σ w_c / (k + rank)`，给 doc 两路更高权重。

方案 1 最"正确"，但要改融合逻辑；方案 2 是零成本试错。

## 六、融合后还会发生什么

```python
fused = sorted(ranked.values(), key=lambda x: -x["score"])
```

- 排序后**长度 = 去重后的记录数**（约 30~40 条）；
- 每条回写 `_rrf_score`（保留 6 位小数），供评估脚本和调试观察排序依据；
- **不截断**——截断发生在 `node_rerank`（取前 30 进重排）。

`_rrf_score` 是一个"附加字段"，会跟着记录一路传到 `references` 之前的环节，但不会进 Milvus、也不会进答案（`_format_record` 只取特定字段）。

## 七、三路都没结果时

```python
active_channels = sum(1 for c in channels if c)     # 有几路非空
```

如果三路都是空列表：

- `channels = [[], [], []]`，`ranked = {}`，`fused_records = []`；
- 日志会显示"0 路有效通道、0 条输入 → 0 条去重融合"；
- 下游 `node_rerank` 的 `step_1` 发现 `fused` 为空，直接返回 `{"reranked_results": []}`；
- `node_answer_output` 的 `context` 为空 → "未在知识库中找到相关内容"。

**注意**：这条路径说明"检索正常但没命中"，与"Milvus 挂了"是两回事。后者会先被 `error` 短路（见 `查询_08`）。

## 八、调试手册

| 现象 | 原因 | 排查 |
|---|---|---|
| 文档块只剩 1 条 | 去重键退化为 `(集合, None)` | 检查 `edu_course_doc` 是否返回 `id` |
| 题目总是压过文档 | 集合顺序 + 3 路 vs 2 路（第五节） | 看 `_rrf_score` 分布 |
| 融合后条数等于输入总数 | 去重键没能命中相同记录 | 打印各路的 `_collection` + 编码 |
| 同一记录出现两次 | `_collection` 不同 / 编码字段不同 | 检查是否同一实体写入两个集合 |
| `0 路有效通道` | 三路都失败 | 看上游"混合检索失败"日志 |
| `_rrf_score` 都一样 | 每路只有 1 条且名次相同 | 正常（k=60 平滑了差异） |
| 想让文档排名更靠前 | 见 5.1 | 优先试"集合内名次"改造 |

## 九、设计取舍

| 决定 | 替代方案 | 为什么选现在这个 |
|---|---|---|
| RRF（看名次） | 加权分数求和 | 三路分数尺度不可比 |
| `k=60` | 更小/更大 | 论文默认，共识优先 |
| 不截断 | 直接取 Top-30 | 截断交给重排节点，职责清晰 |
| 带 `_collection` 去重 | 只用编码 | 跨集合编码可能重名 |
| 保留 `_rrf_score` | 丢弃 | 评估与调试需要看排序依据 |
| 通道内全局名次 | 集合内名次 | 实现简单；代价是 doc 系统性吃亏（第五节） |
