# 查询_01 node_query_rewrite —— 多轮指代消解

> 文件：`processor/query_processor/nodes/node_query_rewrite.py`（Prompt：`common/prompt/rewrite.prompt`）
> 职责：把"那第一个呢"这种**依赖上下文才成立**的问题，改写成独立完整的问题。

## 一、为什么需要改写

多轮对话里用户的追问通常省略了大量信息：

```
用户：有哪些适合理科生的课程系列？
助手：1. 通用编程入门班  2. 算法与数据结构班 ...
用户：第一个系列里有几门课？        ← 单独看这句话，没人知道"第一个"指什么
```

如果直接把这句话丢给意图识别：

- LLM 可能抽出 `series_name="第一个系列"` → 标量查询 `series_name like "%第一个系列%"` → **0 条命中**；
- 也可能判成 `ambiguous` → 反问用户，体验断裂。

改写的目标就是把第二句变成：

```
通用编程入门班包含哪些课程模块
```

这样无论意图识别还是后续过滤，都能拿到干净输入。

## 二、触发条件：两道门

改写要调 LLM，不能让每轮都调。所以设了两道门：

```python
def step_1_validate_and_get_data(state: dict):
    original_query = state.get("original_query", "")
    history = state.get("history", [])

    # 无历史：单轮场景，无需改写
    if not history:
        logger.info("无对话历史，单轮场景无需改写")
        return None

    # 不含指代词：直接透传
    if not _ANAPHORA_PATTERN.search(original_query):
        logger.info("问题不含指代词，无需改写")
        return None

    return original_query, history
```

```
无历史 ──────────────┐
                     ├─→ 透传 original_query（省一次 LLM 调用）
不含指代词 ──────────┘
两者都满足 ─→ 调用 LLM 改写
```

### 2.1 指代词正则的取舍（有个坑写进了注释）

```python
_ANAPHORA_PATTERN = re.compile(
    r"它|这个|这些|这种|这里|那个|那些|上述|刚才|前面|上一|第一|第二|第三|换个|换成"
)
```

覆盖四类：

| 类型 | 词 | 例子 |
|---|---|---|
| 代词 | 它 / 这个 / 这些 / 那个 | "它有几门课" |
| 指代性指示 | 这种 / 这里 / 上述 / 刚才 / 前面 / 上一 | "上面说的那个" |
| 序数 | 第一 / 第二 / 第三 | "第一个系列" |
| 换条件 | 换个 / 换成 | "换成多选题呢" |

**「相关」被刻意排除**，注释里写了原因：

> 不要把「相关」放进正则——它是普通问句的高频词（"XX相关的练习题"），会把单轮语义查询误判为指代，并可能把意图改写窄（实测曾把"知识"改写成"课程"）

这是一个"看起来该加、实际有害"的词。把它放进去的后果不只是多调一次 LLM，而是**改写会悄悄改变意图**：用户问"知识相关的题"，LLM 可能改写成"课程相关的题"，后面的检索范围直接跑偏。

正则用 `search` 而不是 `match`——指代词出现在句中任何位置都算。

### 2.2 无历史时为什么不改写

单轮问题没有可消解的对象，改写只会引入幻觉。而且 `history` 初始值就是 `[]`，第一轮调 LLM 是纯浪费。

## 三、拼接历史

```python
def step_2_build_history_text(history: list) -> str:
    """拼接最近历史文本（history 由 API 侧统一转为时间正序，取尾部即最近的对话）"""
    history_text = "\n".join(
        f"{'用户' if m.get('role') == 'user' else '助手'}: {m.get('text', '')[:300]}"
        for m in history[-6:]  # 最近 3 轮（6 条消息）
    )
```

四个细节：

1. **`history[-6:]`** —— API 层已经按时间正序传进来，取尾部就是最近的 3 轮（一问一答为 2 条）；
2. **每条截断 300 字** —— 助手回答可能很长（列了 20 个系列），不截断会把 prompt 撑爆；300 字足够保留"列出了哪些名字"；
3. **role 转中文** —— "用户/助手"比 "user/assistant" 让模型更容易分清对话角色；
4. **`m.get('text','')`** —— Mongo 文档字段名是 `text`，不是 `content`。

**为什么只取 3 轮**：指代通常指向前 1~2 轮的内容。取太多会让 prompt 变长、干扰变多，且"第一个"这种序数指代如果跨了 5 轮，本身就说明用户表述有问题。

## 四、LLM 改写

```python
prompt = load_prompt("rewrite", history=history_text, query=original_query)
llm = get_llm_client()
response = llm.invoke(prompt)
rewritten = (response.content if hasattr(response, "content") else str(response)).strip()
```

`get_llm_client()` 返回缓存的 `ChatOpenAI` 实例（`temperature=0.1`，低温度保证改写稳定）。没用 `json_mode`，因为输出就是一句话，不是结构化数据。

### 4.1 Prompt 的三个关键约束

`rewrite.prompt` 里最重要的是第 2、4、5 条：

```
2. 保留原问题的查询意图（课程查询/题目查询/统计）与关键限定词，**不要扩大或收窄意图**
4. 序数指代（如"第一个""第一道""它"）必须替换为历史中实际出现的名称/编码
   例：历史回答列出了"1. 通用编程入门班 2. 通用编程项目班"，问"第一个系列里有几门课"
   → "通用编程入门班包含哪些课程模块"
5. 换条件指代（如"换成编程题呢"）需继承原问题结构并替换条件
   例：上文问"单选题有多少道"，问"换成多选题呢" → "多选题有多少道"
```

- **第 2 条防"意图漂移"**：改写模型很容易顺手把"有哪些"改成"有多少"（列表→统计），一旦改了就走了错误的通道；
- **第 4 条给了 few-shot 例子**，因为"序数替换"是最容易做错的动作——模型可能输出"第一个系列"这种换汤不换药的结果；
- **第 5 条处理"换条件"**：这是多轮里很常见的模式，需要继承上一问的结构（"有多少道"）只替换条件（单选题→多选题）。

### 4.2 长度保护

```python
if rewritten and len(rewritten) < len(original_query) * 10:
    logger.info(f"查询改写：{original_query} → {rewritten}")
    return rewritten
logger.warning("改写结果异常，回退原问题")
return original_query
```

**`< 10 倍`** 是一个宽松但有意义的护栏。正常情况下改写后长度和原问题相当（补几个名字，可能长 2~3 倍）；如果长 10 倍以上，说明模型在"解释"而不是"改写"，比如：

```
输入：它呢
输出：您好，您提到的"它"可能指代…（省略 500 字分析）
```

这种输出如果被当成 query 送进意图识别，会彻底跑偏。所以宁可回退。

注意条件是 `and`：`rewritten` 为空也算异常（空串是 falsy），一并回退。

### 4.3 异常兜底

```python
except Exception as e:
    logger.warning(f"查询改写失败，回退原问题：{e}")
    return original_query
```

LLM 超时、API 报错、额度用尽——全部回退原问题。**改写是增强项，不是必需项**：单轮问题本来就能答。这就是为什么这里只 `warning` 不 `exception`（不需要堆栈）。

## 五、改写的下游影响（重要）

改写结果写进 `rewritten_query`，而**下游几乎所有节点都用它**：

```python
query = state.get("rewritten_query") or state.get("original_query")
```

| 节点 | 用 `rewritten_query` 做什么 |
|---|---|
| `node_intent_route` | 意图识别 + 实体抽取的输入 |
| `node_search_embedding` | 向量检索的 query |
| `node_search_embedding_hyde` | 生成假设答案的输入 |
| `node_question_filter` | 向量检索的 query |
| `node_rerank` | 与原文档配对打分的 query |
| `node_answer_output` | 生成答案时的问题描述 |

所以改写错一步，**整条链路都错**。这也是为什么它放在最前面，且加了长度护栏。

### 5.1 但有两个地方必须用 `original_query`

改写是"为了检索更准"，不是"为了替换用户的话"。有些判断必须以用户**原话**为准：

**① 题型是否"明说"**（`node_intent_route` 的推断防护）：

```python
said = f"{original_query}\n{query}"
if not any(a in said for a in aliases):
    entities.pop("question_type_cn")
```

这里把两者拼起来判断，是**放宽**而不是收紧：只要用户原话或改写后任一处提到题型，就认为"用户真的问了"。因为改写可能补全题型词（"换成多选题呢" → "多选题有多少道"），此时用户原话里确实有"多选题"。

**② 填空类澄清**：

```python
if asked_type not in original_query and "填空" in original_query:
    entities.pop("question_type_cn")
    entities["_invalid_question_type"] = "填空题"
```

用 `original_query` 判断"用户是不是说了填空题"，避免改写把"填空题"润色成"完形填空题"后丢失澄清的依据。

**③ 答案生成** 用的是 `rewritten_query`（需要完整语境才能正确表述），注释里写明：

```python
# 作答基于改写后的问题：多轮里"第一个系列里有几门课"需要完整语境才能正确表述
```

## 六、与 intent_route 的历史窗口不一致

一个容易困惑的细节：两个节点取的历史长度不同。

| 节点 | 取多少 | 每条截断 | 理由 |
|---|---|---|---|
| `node_query_rewrite` | `history[-6:]`（3 轮） | 300 字 | 指代消解需要足够的上下文，尤其是"第一个/第二个"要靠列表内容定位 |
| `node_intent_route` | `history[-4:]`（2 轮） | 100 字 | 只是辅助判断（比如"换成多选题"要知道上文在问什么），不需要长文本 |

不是 bug，是**按用途分配预算**：改写要做"文本替换"，需要看到候选名字；意图识别只做"分类"，扫一眼就够。

## 七、实测表现

`rewrite.prompt` 里的两个例子对应两类真实问法，改写效果（参考）：

| 原始问题（含历史） | 改写结果 |
|---|---|
| "第一个系列里有几门课"（上文列了 2 个系列） | "通用编程入门班包含哪些课程模块" |
| "换成多选题呢"（上文问"单选题有多少道"） | "多选题有多少道" |
| "它有多少道题"（上文提到题库） | "通用程序设计题库有多少道题" |
| "函数相关的练习题"（有历史但无指代） | 不改写（正则不命中） |

## 八、调试手册

| 现象 | 原因 | 排查 |
|---|---|---|
| 该改写时没改写 | 指代词不在正则里（如"前者/后者"） | 看日志"问题不含指代词"，缺词就补正则 |
| 不该改写时改写了 | 正则命中了普通词 | 检查是否误加了高频词（如"相关"） |
| 改写后意图变了 | Prompt 第 2 条未生效 / 温度过高 | 对比原始与改写结果，必要时加 few-shot |
| 改写结果是一段解释 | 模型没遵守"只输出问题" | 长度保护会兜住（>10 倍回退），但应调 prompt |
| 多轮完全失效 | `history` 为空 | 检查 Mongo 拉取与 `reversed()`（见 `查询_09`） |
| 指代被保留成"第一个系列" | Prompt 第 4 条未生效 | 实体层还有一层序数防护（`node_intent_route` 会丢弃该实体），但会退化成反问 |

## 九、设计取舍

| 决定 | 替代方案 | 为什么选现在这个 |
|---|---|---|
| 正则预筛再调 LLM | 每轮都调 LLM 改写 | 单轮占多数，每轮调用浪费且可能引入漂移 |
| 只取最近 3 轮 | 全量历史 | 指代通常近邻，"第一个"跨 5 轮说明表述本身有问题 |
| 长度 10 倍护栏 | 不限制 | 防模型输出"解释"而不是"改写" |
| 失败回退原问题 | 报错中断 | 改写是增强项，不该阻断问答 |
| 改写结果只写 `rewritten_query` | 覆盖 `original_query` | 原话在题型澄清、溯源时需要保留 |
