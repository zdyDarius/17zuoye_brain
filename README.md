# education-brain · 教育知识库系统

面向教育培训场景的智能教育知识库系统：将课程介绍、题库题目、课程文档（讲义）三类内容结构化导入 Milvus，提供双通道（结构化 + 语义）自然语言问答能力，答案可解释、可追溯（编码级引用）。

- 技术方案基线：`PRD.md` V1.3（教育知识库系统产品需求文档）
- 数据基线：219 课程系列 / 657 课程模块 / 73 题库 / 1,752 道题目（15 种题型）/ 5 份课程文档（357 分块）/ 1 份 PDF 讲义（23 分块，12 张图片经 VL 语义识别 + MinIO 托管）
- 质量基线：RAGAS Context Recall 0.9464 / Precision 0.9675，题目检索 Hit@5 = 1.0（32 条黄金集，详见 `eval/metrics_report.md`）

---

## 1. 总体架构

```
                    ┌────────────────────────────────────────────────┐
                    │              基础设施层（本地 Docker）           │
                    │  Milvus :19530 │ MongoDB :27017 │ MinIO :9000  │
                    └────────────────────────────────────────────────┘
                                     ▲
        ┌────────────────────────────┴─────────────────────────────┐
        │                                                          │
┌───────┴────────────┐                                 ┌───────────┴──────────┐
│  导入服务 :8000     │                                 │   问答服务 :8001      │
│  file_import_service│                                 │    query_service      │
│                    │                                 │                      │
│  POST /upload       │                                 │  POST /query          │
│  GET  /status/{id}  │                                 │  GET  /stream/{sid}  │
│  GET  /import.html  │                                 │  GET  /chat.html      │
└───────┬────────────┘                                 │  GET/DELETE /history  │
        │                                              └───────┬──────────────┘
        ▼                                                      ▲
┌─────────────────────┐    LangGraph 导入图                    │      LangGraph 查询图
│  kb_import_app      │                                       │      query_app
│                     │                                       │
│ node_entry          │                              node_query_rewrite │
│   ↓                 │                                       ↓        │
│ node_format_convert │                              node_intent_route  │──┐
│   ↓                 │                     ┌─────────────┬───────────┬───┘
│ node_md_img         │                     │catalog      │aggregate  │content
│  (VL识别+MinIO)     │                     ↓             ↓           ↓
│   ↓ (按类型路由)    │             node_structured  node_aggregate  三路并行召回
│ ├─ course_parser    │                 _query       (count/group-by) ├ search_embedding
│ ├─ question_parser  │                     │             │          ├ search_hyde
│ └─ course_doc_parser│                     └──────┬──────┘           └ question_filter
│        ↓            │                            ↓                          ↓
│ node_bge_embedding  │                    node_answer_output ← node_rrf ← node_rerank
│        ↓            │                                                │
│ node_import_milvus  │                                                │
└─────────────────────┘                                                │
```

### 分层说明

| 层 | 位置 | 职责 |
|---|---|---|
| API 层 | `api/` | 导入服务（:8000）与问答服务（:8001），FastAPI + 后台任务 + SSE |
| 编排层 | `processor/` | LangGraph 状态图：导入图（8 节点）+ 查询图（10 节点） |
| 工具层 | `utils/` | Milvus/MongoDB/MinIO 客户端、BGE-M3 向量、Reranker、LLM、SSE、任务追踪 |
| 配置层 | `common/` | 环境配置（.env 驱动）、loguru 日志（含 node_log 装饰器）、prompt 模板 |

---

## 2. 数据模型（Milvus 4 Collection）

| Collection | 内容 | 业务主键 | 关键标量字段（倒排索引） |
|---|---|---|---|
| `edu_course_series` | 课程系列 | series_code | series_name / course_category / target_audience / suitable_grade |
| `edu_course_module` | 课程模块 | course_code | course_name / series_code |
| `edu_course_doc` | 课程文档分块 | id（自动 uuid） | doc_kind / chapter_name / source_file |
| `edu_question` | 题目 | question_code | bank_code / bank_name / question_type / group_id |

统一向量字段：`dense_vector`（BGE-M3，1024 维，HNSW + IP）+ `sparse_vector`（稀疏，SPARSE_INVERTED_INDEX + IP）。

### 关键设计决策（对应 PRD V1.3）

- **答案归一化**（parser 阶段落库）：判断题 `A/对/正确→true`、`B/错/错误→false`；多选题统一半角逗号 + 字母序；全题型另存 `answer_raw` 原文可回溯。归一化后统计类问题（如"判断题答案分布"）才可执行
- **题型开放集**：15 种题型存英文枚举码（`single_choice` 等），中文→英文映射集中在 `QUESTION_TYPE_MAP`；新增题型先插配置表
- **幂等 upsert**：按「来源文件名 + content_type + 业务主键」识别重复导入，先按主键精确删除再插入；禁止按文件名批量先删
- **组题模型**：V1 每题独立存全文（材料在前问题在后），`group_id` 保留置空；共享材料组题为前瞻设计
- **可空字段**：Milvus Schema 显式声明 `nullable=True`，行数据省略空值字段

---

## 3. 导入链路（file_import_service :8000）

### 流程

```
上传（.md/.pdf/.docx ≤50MB，单次 ≤20 个）
  → node_entry        白名单/枚举/体积校验；MD 直接读取（utf-8/gbk 兜底）
  → node_format_convert  docx→MD（python-docx：标题样式→# 标题、表格→MD 表格、代码样式→``` 围栏、
  │                      行首 # 转义防误判）；PDF→MD（MinerU 云端 API：申请上传链接→PUT 直传→
  │                      轮询→下载解压，保留 .md 与 images/ 图片到任务目录）
  → node_md_img       图片处理（仅 PDF 链路，无图片自动跳过）：
  │                      ① 扫描 MD 引用的本地图片 + 提取上下文（上文/下文各 100 字符）
  │                      ② VL 多模态模型（qwen3-vl-flash，base64 内联 + 上下文提示词）
  │                         生成 50 字内中文摘要（图片语义 → 文本）
  │                      ③ 上传 MinIO 公开读桶（/upload-images/{文件名}/，上传前清旧对象幂等）
  │                      ④ MD 引用替换：![](images/死链.jpg) → ![VL摘要](MinIO URL)
  │                      ⑤ 新 MD 备份落盘（xxx_new.md）
  ├─ course_catalog → node_course_parser      系列/模块解析（## 系列边界 + 字段锚点 + 课程小节）
  ├─ question_bank  → node_question_parser    题库/题目解析 + 答案归一化 + 实体级失败隔离
  └─ course_doc     → node_course_doc_parser   按标题层级分块（目标 1500 字符、上限 3000、
                                               超长按换行二次切分、尾块合并、章节路径前缀增强
                                               ——块首拼【章节>子节】提升公式密集块可检索性）
  → node_bge_embedding   BGE-M3 混合向量化（分批 32 条）
  → node_import_milvus   幂等 upsert 入库
```

### 图片处理设计（node_md_img）

图片的价值通过「双轨」实现：

| 轨道 | 位置 | 作用 |
|---|---|---|
| **语义轨** | VL 摘要写入图片 alt 文本（`![摘要](url)`） | 摘要随块内容参与向量化——图片内容从此可被语义检索（如查"高斯分布"可命中误差分布图） |
| **展示轨** | MinIO 公开 URL（桶已设公开读策略） | 前端直接展示原图，回答引用中可回链 |

- **上下文增强识别**：VL 识别时附带图片在 MD 中的上文/下文（各 100 字符），摘要更贴合文档语境而非孤立描述
- **幂等重导**：MinIO 上传前清理同前缀（`/upload-images/{文件stem}/`）旧对象，避免存储垃圾累积
- **API 限速保护**：滑动窗口限速器（3000 次/60 秒），防触发 VL 模型 API 限流
- **实测**：回归算法.pdf 12 张图全链路 2 分钟——VL 识别（如"标准高斯分布曲线—解释误差项假设"）+ MinIO 上传（URL 200 可访问）+ Milvus 块内 12 处引用全替换、死链 0

### 容错与安全

- **实体级失败隔离**：单实体解析失败（如缺业务主键）只跳过该实体，不影响其他实体；失败清单随任务状态返回
- **宽容归一化**：源数据脏点（如标注单选但答案多字母）按实际形态归一化入库，同时标记 `data_anomaly` 留痕
- **并发向量化保护**：BGE-M3 模型初始化与推理经 RLock 串行化（MPS 设备非线程安全），初始化含预热推理，避免 meta tensor 竞态
- **任务进度**：stream 模式执行导入图，`/status/{task_id}` 返回全局状态 + 已完成/进行中节点（中文）
- **course_doc 重导注意**：course_doc 无业务主键（自动 uuid），重导入同一文件需先按 `source_file` 删除旧块（当前由运维侧处理，后续可在导入图加"按 source_file 先删"步骤）

---

## 4. 查询链路（query_service :8001）

### 双通道架构与意图分流

```
node_query_rewrite   多轮指代消解（"第一个系列"→实际系列名；序数/承前/换条件指代改写；无指代词时直通省 LLM）
        ↓
node_intent_route    LLM 结构化输出 intent ∈ {catalog, aggregate, content, ambiguous} + 实体抽取
        ↓ 4 路分流
catalog ────→ node_structured_query   编码精确过滤 + 名称模糊召回 + 标量过滤（可穷举，100 条截断保护）
aggregate ──→ node_aggregate         应用层 count / group-by（可聚合字段白名单；多值字段拆分归组；
                                     白名单外字段反问澄清；课程数量类问题兜底返回清单）
content ────→ 三路并行召回             ① 原查询混合向量 ② HyDE 假设答案向量 ③ 实体预过滤向量
                ↓                        （含题型/题库实体时检索集合定向为 edu_question，防课程召回稀释）
             node_rrf                RRF 融合（k=60，三路去重）
                ↓
             node_rerank             BGE-Reranker-large 精排 Top-5（模型故障降级 RRF 序）
        ↓
node_answer_output   LLM 生成答案（含编码级引用标注）；LLM 故障降级模板化输出；
                     无结果明确"未找到"话术（禁止编造）
```

- **ambiguous → 直接反问**：意图不明、或题型不存在（"填空题"→ 澄清而非误匹配"完形填空题"）
- **代码级兜底**：LLM 实体抽取不稳定（漏抽/字面序数），intent 正确但实体缺失时从 Milvus 拉系列名做文本匹配补抽

### 交互能力

| 能力 | 实现 |
|---|---|
| 非流式 | `POST /query`（is_stream=false）同步返回 |
| 流式 | `is_stream=true` 后台执行 + `GET /stream/{session_id}` SSE：step（节点进度）→ final（答案+引用）；**事件自增 id + 重放缓冲 + Last-Event-ID 断线重连精确补发** |
| 多轮对话 | 查询前拉取最近 12 条（6 轮）历史供改写；"1 轮 = user + assistant 两条" |
| 历史管理 | MongoDB `chat_message` 集合；assistant 消息持久化编码级引用 `references`（历史回放可还原引用来源）；`GET/DELETE /history/{session_id}` |
| 前端页面 | `chat.html`（流式对话/节点进度/引用标签/快捷问题）、`import.html`（拖拽上传/任务进度轮询） |

### 降级矩阵（PRD 3.8）

| 故障 | 行为 |
|---|---|
| Milvus 不可用 | 结构化/语义通道返回明确错误，服务不崩溃 |
| LLM 不可用 | 意图路由降级反问；答案生成降级模板化输出结构化/聚合结果（degraded=true） |
| Reranker 故障 | 直接使用 RRF 融合序 |
| HyDE 生成失败 | 降级用原查询检索 |

---

## 5. 目录结构

```
17zuoye_brain/
├── api/
│   ├── file_import_service.py      # 导入服务 :8000（上传/任务进度/导入页）
│   └── query_service.py            # 问答服务 :8001（query/SSE/历史/chat 页）
├── processor/
│   ├── import_processor/           # 导入图
│   │   ├── main_graph.py           # 状态图定义与条件路由
│   │   ├── state.py                # ImportGraphState
│   │   └── nodes/
│   │       ├── node_entry.py           # 校验 + MD 直读
│   │       ├── node_format_convert.py   # docx/PDF → MD 归一化（MinerU 结果含图片）
│   │       ├── node_md_img.py           # 图片处理：VL 识别 + MinIO 上传 + MD 引用替换
│   │       ├── node_course_parser.py    # 课程目录解析（系列+模块）
│   │       ├── node_question_parser.py  # 题库解析 + 答案归一化
│   │       ├── node_course_doc_parser.py# 课程文档按标题分块
│   │       ├── node_bge_embedding.py    # BGE-M3 向量化
│   │       └── node_import_milvus.py    # 幂等 upsert（4 Collection Schema）
│   └── query_processor/            # 查询图
│       ├── main_graph.py
│       ├── state.py
│       └── nodes/
│           ├── node_query_rewrite.py    # 多轮指代消解改写
│           ├── node_intent_route.py     # 意图识别 + 实体抽取
│           ├── node_structured_query.py  # 结构化通道
│           ├── node_aggregate.py        # 聚合通道
│           ├── node_search_embedding.py # 语义通道（embedding/hyde/filter/rrf）
│           ├── node_rerank.py           # 重排
│           └── node_answer_output.py     # 答案组装 + 引用
├── common/
│   ├── config/                     # .env 配置类（milvus/embedding/lm/mineru/minio/reranker/bailian_mcp）
│   ├── logging/logger.py           # loguru + node_log/step_log 装饰器
│   └── prompt/                     # intent_route / edu_answer / hyde_prompt / image_summary / answer_out 等
├── utils/
│   ├── clients/                    # milvus_utils / mongo_history_utils（含 references 持久化）/ minio_utils
│   ├── lm/                         # embedding_utils（RLock 单例）/ lm_utils / reranker_utils
│   ├── download_model/             # ModelScope 模型下载脚本（BGE-M3 / BGE-Reranker）
│   ├── sse_utils.py                # SSE 事件缓冲 + Last-Event-ID 重连
│   ├── rate_limit_utils.py         # 滑动窗口 API 限速器（VL 调用防护）
│   ├── task_utils.py               # 导入任务状态追踪（中文节点名）
│   └── ...
├── eval/                           # 检索评估体系（双口径：RAGAS + 编码级确定性指标）
│   ├── golden_dataset.json         # 黄金集 32 条（12 题目 + 16 文档 + 4 课程结构化）
│   ├── run_retrieval.py            # 检索收集：进程内跑查询图，落盘全量检索明细
│   ├── ragas_eval.py               # RAGAS 评判 + Hit@K/Recall@K/Precision@K/MRR
│   ├── metrics_report.md           # 评估报告（指标解读/低分案例/改进建议）
│   └── eval_runs/                  # 评估产物（results/ragas_scores/metrics_summary）
├── page/
│   ├── chat.html                   # 流式对话页面
│   └── import.html                 # 导入操作页面
├── test/
│   └── test_query_graph.py         # 冒烟测试集（12 条：目录/聚合/语义/澄清/文档检索）
├── .env                            # 环境配置
└── pyproject.toml                  # uv 依赖管理
```

---

## 6. 快速开始

### 环境要求

- Python 3.12+（uv 管理依赖）
- 本地 Docker：Milvus（:19530）、MongoDB（:27017）、MinIO（:9000，图片存储）
- BGE-M3 模型本地路径（`BGE_M3_PATH`）、BGE-Reranker-large 本地路径
- LLM：OpenAI 兼容 API（默认通义 qwen-flash）
- VL 多模态模型：`VL_MODEL`（默认 qwen3-vl-flash，PDF 图片语义识别用）
- MinerU API Token（`MINERU_API_TOKEN`，PDF 云端转换用）
- 每日额度注意：MinerU 每账号每天 1000 页最高优先级解析额度

### 启动

```bash
# 1. 安装依赖
uv sync

# 2. 配置 .env（MILVUS_URL / MONGO_URL / BGE_M3_PATH / OPENAI_API_KEY 等）

# 2.1 下载本地模型（已有 BGE-M3 / BGE-Reranker 本地路径可跳过；
#     脚本从 ModelScope 拉取，注意按本机情况修改 cache_dir）
.venv/bin/python utils/download_model/download_bgem3.py
.venv/bin/python utils/download_model/download_reranker.py

# 3. 启动问答服务
PYTHONPATH=. .venv/bin/python api/query_service.py      # :8001

# 4. 启动导入服务
PYTHONPATH=. .venv/bin/python api/file_import_service.py # :8000

# 5. 导入数据（或访问 http://127.0.0.1:8000/import.html 页面操作）
curl -X POST http://127.0.0.1:8000/upload \
  -F "files=@数据/课程介绍.md" -F "content_type=course_catalog"
curl -X POST http://127.0.0.1:8000/upload \
  -F "files=@数据/题目资料.md" -F "content_type=question_bank"
curl -X POST http://127.0.0.1:8000/upload \
  -F "files=@数据/课程文档/xxx.docx" -F "content_type=course_doc"

# 6. 问答（或访问 http://127.0.0.1:8001/chat.html）
curl -X POST http://127.0.0.1:8001/query \
  -H "Content-Type: application/json" \
  -d '{"query": "通用程序设计题库有多少道单选题？"}'
```

### 自测

```bash
PYTHONPATH=. .venv/bin/python test/test_query_graph.py
# 预期：12 通过 / 0 失败（目录/聚合/语义/澄清/文档检索五类通道）
```

---

## 7. 检索评估体系（eval/）

双口径互相印证：**RAGAS LLM 评判**（信息覆盖视角）+ **编码级确定性指标**（实体定位视角，口径更严）。

| 组件 | 作用 |
|---|---|
| `golden_dataset.json` | 黄金集 32 条（12 题目检索 + 16 课程文档检索 + 4 课程结构化），golden 实体全部核对自 Milvus 实际入库数据 |
| `run_retrieval.py` | 进程内调用 `query_app` 逐条执行黄金集，落盘全量检索明细（answer / references / reranked / fused / 各召回路 hits） |
| `ragas_eval.py` | RAGAS 0.4.3（评判模型复用项目 LLM 配置）+ Hit@K / Recall@K / Precision@K / MRR |
| `metrics_report.md` | 评估报告：指标口径解读、低分案例分析、P0–P2 改进建议 |

### 最新结果（2026-10-17，32 条全量）

| 指标 | 数值 | 说明 |
|---|---|---|
| RAGAS Context Recall | **0.9464** | golden 答案信息点 94.6% 可被检索上下文覆盖 |
| RAGAS Context Precision | **0.9675** | 相关上下文以 96.8% AP 精度排在前列 |
| 编码级 Hit@5 — 题目检索 | **1.0000** | 题目通道接近满分（MRR 0.958） |
| 编码级 Hit@5 — 文档检索 | 0.7500 | 短板：概述章节高频词压过具体章节块（章节定位偏移） |
| 编码级 MRR — 全量语义 | 0.8073 | golden 首次命中排名倒数均值 |
| 结构化通道引用命中 | 4/4 | catalog 通道 references 编码级全命中 |
| 意图路由正确率 | 32/32 | 无误入 aggregate/ambiguous；全链路平均 6.3s/条，无降级 |

### 复现

```bash
.venv/bin/python eval/run_retrieval.py   # 收集检索结果（需 Milvus + LLM，约 3.5 分钟）
.venv/bin/python eval/ragas_eval.py      # RAGAS + 确定性指标（约 12 分钟，并发压到 2 防限流）
```

---

## 8. 验收基线（2026-10-15 实测）

| 项 | 结果 |
|---|---|
| 全量导入 | 219 系列 + 657 模块 + 1,752 题 + 357 文档块 + 1 份 PDF（23 块），实体失败 0（1 处源数据脏点宽容入库留痕） |
| 幂等重导入 | 0 inserted / 876 updated，总数不变 |
| 答案归一化 | 判断题 true=145 / false=90；多选题分隔符统一 0 异常 |
| 评测集 | 12/12 通过（含 F3 文档检索 3 条） |
| 多轮对话 | 序数指代/换条件指代均正确消解 |
| SSE | 节点级 step 事件 + final 事件；Last-Event-ID 重连精确补发 |
| 向量化吞吐 | MPS 设备，2,628 实体分钟级；5 份文档 357 块 16 秒 |
| PDF 转换 | MinerU 云端 23 秒/份，标题/表格/LaTeX 公式还原良好 |
| PDF 图片 | 12 张全链路：VL 语义识别 + MinIO 上传（URL 200 可访问）+ MD 引用全替换、死链 0 |

## 9. 已知限制与后续方向

- `doc_kind=project`（项目文档）字段预留，当前导入均为讲义（lecture）
- course_doc 无业务主键，重导入同一文件需先按 `source_file` 删旧块（未自动化）
- 题库→课程映射缺失，题目返回"题库归属"而非"课程归属"（映射表补录后升级）
- VL 图片识别依赖云端 API（qwen3-vl-flash），大批量图片导入受 API 限速（已内置滑动窗口限速器）
- **文档检索章节级定位偏差**（评估实测 doc Hit@5=0.75）：概述章节高频词在向量空间中压过具体章节块；改进方案（章节级查询增强 / 概述块降权 / Top-K 自适应）见 `eval/metrics_report.md` 第 5 节
- **Milvus 存在历史测试残留数据**（如 `test_docx_series_full`，系列/模块实存 220/658 vs 业务 219/657），需按编码清理，避免污染课程检索
- V2 方向（PRD 第 8 节）：文档增量更新（hash 比对）、权限与多租户、学习路径推荐
