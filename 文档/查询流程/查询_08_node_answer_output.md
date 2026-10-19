# 查询_08 node_answer_output —— 上下文组装与答案生成

> 文件：`processor/query_processor/nodes/node_answer_output.py`（Prompt：`common/prompt/edu_answer.prompt`）
> 职责：四条通道的终点。把检索结果拼成上下文，交给 LLM 写成答案；写不出来就降级成模板。

## 一、它是"四条路的汇合点"

```
ambiguous（intent 已生成反问）      ─┐
catalog（structured_results）      ─┤
aggregate（aggregate_result）      ─┼──→ node_answer_output ──→ answer + references
content（reranked_results）        ─┘
Milvus 不可用（error）             ─┘
```

这个节点要处理五种输入形态，所以它的主体是一串**提前返回**：

```python
passthrough = step_1_check_passthrough(state)
if passthrough:
    return passthrough

context = step_2_build_context(state)
question = state.get("rewritten_query") or state.get("original_query", "")

if not context:
    return step_3_build_no_result_answer()

return step_4_generate_answer(state, context, question)
```

四条出口，按顺序判断：

| 顺序 | 条件 | 出口 | 是否调 LLM |
|---|---|---|---|
| 1 | 已有 `answer`（反问/澄清/白名单提示） | 直通 | ❌ |
| 2 | `error` 且无任何结果 | 服务异常提示 | ❌ |
| 3 | 上下文为空 | "未找到"引导 | ❌ |
| 4 | 有上下文 | LLM 生成 / 模板降级 | ✅ |

## 二、直通路径（三条）

```python
# 反问/澄清场景：answer 已在意图路由生成
if state.get("intent") == "ambiguous" and state.get("answer"):
    return {"answer": state["answer"], "references": []}

# 聚合通道的白名单反问场景（answer 已生成）
if state.get("intent") == "aggregate" and state.get("answer"):
    return {"answer": state["answer"], "references": []}

# 检索通道整体不可用
has_any_result = bool(
    state.get("structured_results") or state.get("aggregate_result") or state.get("reranked_results"))
if state.get("error") and not has_any_result:
    return {"answer": "检索服务暂时不可用，请稍后重试。", "references": [], "degraded": True}
```

### 2.1 为什么反问要"直通"而不是交给 LLM 润色

反问话术是**精确设计的**：

```
知识库中没有「填空题」这种题型哦。现有题型包括：单选题、多选题、判断题、
简答题、计算题、编程题等 15 种。您是想查某一类的题目吗？比如相近的「完形填空题」？
```

如果交给 LLM 改写，可能变成"很抱歉，目前系统暂未收录该题型…"——语气变了、枚举丢了、引导语没了。这类"产品文案"应该原样输出。

### 2.2 为什么 `error` 要单独成一个分支

这是修过的一个 bug 的痕迹：

```
Milvus 挂了 → node_search_embedding 返回 {"error": "Milvus 客户端不可用"}
→ 早期 answer_output 不看 error → context 为空 → "未在知识库中找到相关内容"
→ 用户以为知识库没有这个知识，实际上系统坏了
```

**把系统故障伪装成"知识库没有"是最坏的一种失败**——用户会去改问题、换关键词，永远不会重试。

现在的判断是 `error and not has_any_result`：只有"出错且没有任何结果"才判定服务不可用。如果出错但部分通道有结果（比如一个集合失败、其他正常），仍然正常作答。

### 2.3 三种 `answer` 来源的优先级

| 来源 | 谁写的 | 触发条件 |
|---|---|---|
| `intent=ambiguous` 的澄清 | `node_intent_route` step6 | 意图不明 / 题型不存在 |
| `intent=aggregate` 的白名单提示 | `node_aggregate` step2 | `aggregate_field` 非法 |
| "检索服务暂时不可用" | 本节点 | `error` 且无结果 |

注意 aggregate 的直通**不看 `intent` 是否仍为 aggregate**——只要 `intent=aggregate` 且有 `answer` 就直通。目前只有白名单场景会同时满足这两个条件。

## 三、上下文组装

```python
parts = []
```

三段拼接，顺序固定：**结构化 → 聚合 → 语义**。

### 3.1 结构化结果

```python
structured = state.get("structured_results") or []
if structured:
    truncated = state.get("structured_truncated")
    head = "【结构化查询结果】" + ("（结果较多，已截断）" if truncated else "")
    body = "\n".join(f"{i + 1}. {_format_record(r)}" for i, r in enumerate(structured[:20]))
    if len(structured) > 20:
        body += f"\n（共 {len(structured)} 条，其余略）"
    parts.append(head + "\n" + body)
```

三个细节：

- **只取前 20 条** —— 100 条记录拼起来上万字，既贵又会让模型抓不住重点；
- **`truncated` 提示** —— 告诉模型"这不是全部"，避免它说"一共只有这些"；
- **`（共 N 条，其余略）`** —— 这里的 `N` 是 `len(structured)`（最多 100），**不是 `structured_total`**。

### 3.2 ⚠️ 一个已知瑕疵：总数说小了

`structured_total` 是 `count(*)` 的真实命中数（比如 213），`structured_results` 只有 100 条。但上面那句写的是：

```python
body += f"\n（共 {len(structured)} 条，其余略）"     # len = 100
```

于是模型看到的是"共 100 条"，可能回答"共 100 道单选题"——而实际 213 道。

**修法**（一行）：

```python
total = state.get("structured_total") or len(structured)
body += f"\n（共 {total} 条，其余略）"
```

当前没修，是因为"列举类"问题很少精确报数；但"有哪些 + 多少道"这种混合问法会踩到。

### 3.3 聚合结果

```python
agg = state.get("aggregate_result") or {}
if agg:
    lines = ["【统计结果】"]
    if agg.get("filter"):
        lines.append(f"过滤条件：{agg['filter']}")
    if "value" in agg and "groups" not in agg:
        lines.append(f"统计值：{agg['value']}")
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
```

三种聚合形态对应三块逻辑：

| 形态 | 判断 | 输出 |
|---|---|---|
| 纯计数 | 有 `value` 无 `groups` | `统计值：213` |
| 课程数量 | 有 `course_rows` | `清单（共 50 条）：{...}` |
| 分组/分布 | 有 `groups` | `总数：235，分组数：2` + `true: 145` |

**`过滤条件` 这一行很重要**：它让答案能复述"（筛选条件：题型=单选题）"，用户才知道数字的口径。没有这一行，模型可能把"单选题数"说成"题库总数"。

分组只展示前 20 组（虽然 `node_aggregate` 最多给 100 组），同样是上下文预算的考虑。

### 3.4 语义结果

```python
reranked = state.get("reranked_results") or []
if reranked:
    body = "\n".join(f"{i + 1}. {_format_record(r)}" for i, r in enumerate(reranked))
    parts.append("【语义检索结果（按相关性排序）】\n" + body)
```

`reranked` 最多 5 条（`FINAL_TOP_K`），全部拼进去，不截断。

标题里特意写了"按相关性排序"——告诉模型**第一条最相关**，避免它从后往前引用。

## 四、`_format_record`：每条记录怎么变成文本

四条分支，按 `_collection` 分发：

```python
def _format_record(record: dict) -> str:
    collection = record.get("_collection", "")
    if collection == "edu_question":
        options = f"（内容：{record['content'][:200]}）" if record.get("content") else ""
        type_cn = _QUESTION_TYPE_CN.get(record.get("question_type"), record.get("question_type"))
        return (f"[题目] 编码:{record.get('question_code')} 题库:{record.get('bank_name')} "
                f"题型:{type_cn} 答案:{record.get('answer')} "
                f"解析:{(record.get('analysis') or '')[:100]}{options}")
    ...
```

三个设计点：

1. **`[题目]` / `[课程系列]` / `[课程模块]` / `[课程文档]` 前缀** —— 让模型明确知道这是什么实体，避免把课程模块当成题目；
2. **题型英文转中文**：
   ```python
   _QUESTION_TYPE_CN = {v: k for k, v in QUESTION_TYPE_MAP.items()}
   ```
   库里存 `single_choice`，直接展示会变成"题型:single_choice"，答案里就会出现英文枚举。反转映射表复用导入侧的字典，保证两边一致；
3. **字段级截断**：题干 200 字、解析 100 字、文档正文 300 字。一条记录太长会挤占其他候选的预算。

**兜底分支**：

```python
return json.dumps({k: v for k, v in record.items() if not k.startswith("_")},
                  ensure_ascii=False, default=str)[:300]
```

遇到未知 `_collection`（将来新增集合）时，把记录整个序列化。`_` 开头的内部字段（`_collection` / `_score` / `_rrf_score`）被过滤掉——它们是流程元数据，不该出现在答案上下文里。

## 五、无结果话术

```python
answer = ("未在知识库中找到相关内容。您可以：\n"
          "1. 换个关键词试试（如具体题型名、课程系列名）\n"
          "2. 查询课程介绍：如「有哪些适合在校生的课程」\n"
          "3. 查询题目：如「通用程序设计题库有哪些单选题」")
```

**这属于"拒答"，不是"生成"**。PRD 明确要求禁止编造，所以没有上下文时**不调 LLM**——调了大概率会编。

给出三条具体示例，是因为"未找到"本身没有行动价值，用户需要知道"怎么问才对"。

## 六、LLM 生成

```python
try:
    prompt = load_prompt("edu_answer", context=context, question=question)
    response = get_llm_client().invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)
    return {"answer": answer, "references": step_5_build_references(state)}
except Exception as e:
    logger.warning(f"LLM 生成失败，降级为模板化输出：{e}")
    degraded_answer = f"智能生成暂不可用，以下为检索到的原始结果：\n\n{context[:2000]}"
    return {"answer": degraded_answer, "references": step_5_build_references(state), "degraded": True}
```

### 6.1 Prompt 的关键约束

`edu_answer.prompt` 有 8 条要求，其中三条直接决定了输出质量：

```
1. 严格基于【检索结果】作答，禁止编造知识库中不存在的内容
3. 引用来源：答案中涉及具体数据时，在句末标注来源，格式为[来源:编码]
7. **题目检索类问题**：语义检索结果中的题目（含题干、选项、答案）就是用户要找的内容，
   即使题面措辞与用户提问不完全一致（语义检索天然如此），也必须列出这些题目并说明
   "为您找到以下相关题目"；只要检索结果中存在题目记录，就不得回答"未找到"
```

**第 7 条是针对一类真实失败加的**：语义检索返回了 5 道相关题目，但模型觉得"题面和问题不完全一样"，于是回答"未找到相关题目"。用户明明看到了 5 条引用，答案却说没有。

这句话强制模型把"有记录"和"完全匹配"区分开——前者必须列出。

### 6.2 用 `rewritten_query` 而不是原问题

```python
question = state.get("rewritten_query") or state.get("original_query", "")
```

注释：

> 作答基于改写后的问题：多轮里"第一个系列里有几门课"需要完整语境才能正确表述

如果拿"它有几门课"去生成答案，模型不知道"它"是什么，写出来的句子会很别扭（"它有 12 门课"），用户也无法确认指的是什么。

### 6.3 降级：模板化输出

```python
degraded_answer = f"智能生成暂不可用，以下为检索到的原始结果：\n\n{context[:2000]}"
```

- 明确告知"智能生成不可用"，不伪装成正常答案；
- 直接把上下文（带 `【结构化查询结果】` 等标题）贴出来，信息不丢；
- `[:2000]` 截断，避免超长响应；
- `degraded=True` 让接口层能标记。

### 6.4 一个没处理的情况

如果 LLM 返回**空字符串**（`""`），代码不会走 except，直接返回空答案。前端会渲染一个空白气泡。

理论上少见（模型通常至少输出一句话），但可以加一层：

```python
if not answer.strip():
    raise ValueError("LLM 返回空答案")
```

## 七、引用组装

```python
def step_5_build_references(state: dict) -> list:
    refs, seen = [], set()
    for record in (state.get("structured_results") or []) + (state.get("reranked_results") or []):
        collection = record.get("_collection", "")
        if collection == "edu_question":
            key, ref = record.get("question_code"), {
                "type": "question", "code": record.get("question_code"),
                "bank": record.get("bank_name"), "source_file": record.get("source_file")}
        elif collection == "edu_course_series":
            key, ref = record.get("series_code"), {...}
        elif collection == "edu_course_module":
            key, ref = record.get("course_code"), {...}
        elif collection == "edu_course_doc":
            key, ref = f"doc_{record.get('chunk_index')}_{record.get('chapter_name')}", {
                "type": "course_doc", "chapter": record.get("chapter_name"),
                "source_file": record.get("source_file")}
        else:
            continue
        if key and key not in seen:
            seen.add(key)
            refs.append(ref)
    return refs
```

### 7.1 只取 structured + reranked

**聚合通道没有引用**。这是合理的——"1752 道题"这个数字来自 `count(*)`，没有具体记录可以指向。前端对空引用有兜底文案（"引用: 统计结果"）。

### 7.2 去重键的选择

| 集合 | key | 说明 |
|---|---|---|
| question | `question_code` | 业务主键 |
| series | `series_code` | 业务主键 |
| module | `course_code` | 业务主键 |
| doc | `doc_{chunk_index}_{chapter_name}` | **没有业务主键**，用"块序号 + 章节名"组合 |

doc 的 key 里带 `chunk_index`（全局递增，见 `导入_06`）——同一个章节的不同块不会被误合并。

注意这里**没有用 `id`**（虽然 doc 记录里有）。用 `chunk_index + chapter` 更可读，且对前端展示更有意义。

### 7.3 引用结构

```python
{"type": "question",   "code": "...", "bank": "...", "source_file": "..."}
{"type": "course_series", "code": "...", "name": "...", "source_file": "..."}
{"type": "course_module", "code": "...", "name": "...", "source_file": "..."}
{"type": "course_doc", "chapter": "...", "source_file": "..."}
```

前端渲染成小标签：

```javascript
`${r.type === 'question' ? '题目' : r.type === 'course_series' ? '系列' : '模块'}: ${(r.code || r.name || '').slice(0, 30)}`
```

注意前端只区分了三种类型，`course_doc` 会落到"模块"分支（因为三元表达式只有两个条件）——显示成"模块: "。这是前端的小瑕疵。

## 八、配图提取（step_6）

答案除了文字，还会带一份 `image_urls`。图片不是"再检索一次"，而是**从已命中的切片里捞**——`node_md_img` 在入库前已经把图片替换成了 `![VL摘要](http://minio/...)`，所以 URL 本来就在 `content` 里。

```python
@step_log("step_6_extract_image_urls")
def step_6_extract_image_urls(state: dict) -> list:
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
                return urls
    return urls
```

三个约束：

| 约束 | 值 | 原因 |
|---|---|---|
| 来源 | `structured_results` + `reranked_results` | 两条通道都可能带图；聚合通道没有记录可捞 |
| 过滤 | `http(s)` + 路径后缀 ∈ `.png/.jpg/.jpeg/.gif/.webp/.bmp/.svg` | 挡掉普通链接（`.pdf` 文档、网页）和相对路径 |
| 上限 | `MAX_ANSWER_IMAGES = 6` | 一个 doc 块可能塞了十几张图，全渲染会刷屏 |

### 8.1 必须读原始 `content`，不能用 `_format_record` 的输出

这是实现上最容易踩的坑：`_format_record` 会把 doc 正文截到 300 字、题目截到 200 字，而 MinIO 的 hash 文件名 URL 有 90+ 字符：

```
http://minio:9000/kb/img/doc/1aae03bc92257395585188ceb7600e3481b65aaa26dd6defd9544676c3d9ac4c.jpg
```

从截断文本里提，经常只剩半截链接（`...jpg` 被切掉），前端 `isImageUrl` 判不过，图就丢了。所以提取走的是 `record["content"]` 全量。

### 8.2 `_with_images`：所有出口统一补字段

节点有 4 个出口（直通 / 服务不可用 / 无结果 / LLM 生成），任何一条漏了 `image_urls` 都会让前端拿不到字段：

```python
def _with_images(state: dict, payload: dict) -> dict:
    """统一补充配图字段（所有出口都走它，避免某个分支漏传）"""
    payload["image_urls"] = step_6_extract_image_urls(state)
    return payload
```

节点主体因此变成：

```python
if passthrough:
    return _with_images(state, passthrough)
...
if not context:
    return _with_images(state, step_3_build_no_result_answer())
return _with_images(state, step_4_generate_answer(state, context, question))
```

### 8.3 为什么不让 LLM 输出图片区块

shopkeeper 的做法是 prompt 要求模型在末尾追加 `【图片】` 区块，前端再把区块剥离。我们**没有**采用：

- 图片本来就在切片里，代码提取是确定性的，不需要模型"判断哪张相关"；
- 让模型输出 URL 会诱导它在答案正文里复述链接，反而需要前端剥离；
- 我们的 `content` 里图片是 `![VL摘要](URL)`，摘要已经在向量里起作用，URL 对模型没有信息价值。

前端仍然保留了对 `【图片】` 区块的兼容剥离（万一模型自己输出），但主路径不依赖它。

## 九、四条出口的产出对照

| 场景 | `answer` 来源 | `references` | `image_urls` | `degraded` |
|---|---|---|---|---|
| 意图不明 | intent 节点的澄清话术 | `[]` | 通常 `[]` | 未设置 |
| 聚合字段非法 | aggregate 节点的提示 | `[]` | 通常 `[]` | 未设置 |
| Milvus 不可用 | 本节点固定话术 | `[]` | `[]` | `True` |
| 检索无结果 | 本节点引导话术 | `[]` | `[]` | 未设置 |
| 正常（LLM 成功） | LLM | 组装 | 提取（≤6） | 未设置 |
| 正常（LLM 失败） | 模板 + 上下文 | 组装 | 提取（≤6） | `True` |

## 十、调试手册

| 现象 | 原因 | 排查 |
|---|---|---|
| 答案说"共 100 条"但实际更多 | 用了 `len(structured)`（3.2） | 改成 `structured_total` |
| 答案出现 `single_choice` | 题型反查映射缺项 | 检查 `QUESTION_TYPE_MAP` |
| 有引用却说"未找到" | prompt 第 7 条未生效 | 看 `context` 是否为空 |
| 答案编造了库里没有的内容 | context 太短 / prompt 第 1 条失守 | 看 `context` 长度与记录条数 |
| 空白气泡 | LLM 返回空串（6.4） | 加空值校验 |
| `degraded=true` 但答案正常 | 走了模板降级 | 搜 warning "LLM 生成失败" |
| 引用是空的但答案有数据 | 走的是聚合通道 | 正常（聚合无引用） |
| 引用显示成"模块: undefined" | `course_doc` 类型前端未处理 | 前端映射问题 |
| `image_urls` 为空但切片里有图 | 从截断文本里提（8.1）/ 后缀不在白名单 | 看日志"配图提取完成：N 张" |
| 图片只在部分答案出现 | 命中的切片本身没有图 | 正常；图片只跟随命中的块 |

## 十一、设计取舍

| 决定 | 替代方案 | 为什么选现在这个 |
|---|---|---|
| 反问直通不润色 | 交给 LLM 改写 | 产品文案要稳定、可控 |
| `error` 单独短路 | 统一走"未找到" | 不能把故障伪装成"没有" |
| 上下文只取 20 条 | 全量 | 预算与注意力 |
| 无结果不调 LLM | 让 LLM 兜底回答 | 会编造 |
| 降级贴原始上下文 | 返回错误信息 | 至少信息不丢 |
| 引用按业务编码去重 | 按记录 id | 同一个业务实体只该出现一次 |
