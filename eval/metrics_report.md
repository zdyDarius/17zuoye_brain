# 17zuoye_brain 检索性能评估报告（召回率 / 准确率）

- 评估日期：2026-10-17
- 评估对象：`17zuoye_brain` 教育知识库系统（双通道 RAG：结构化 + 语义三路召回 → RRF → BGE-Reranker Top-5）
- 评估工具：**RAGAS 0.4.3**（LLM-as-Judge，评判模型复用项目配置的 qwen-flash）+ 编码级确定性指标（Hit@K / Recall@K / Precision@K / MRR）
- 评测集：`eval/golden_dataset.json`，32 条（12 题目检索 + 16 课程文档检索 + 4 课程结构化检索），golden 实体全部核对自 Milvus 实际入库数据（219 系列 / 657 模块 / 1752 题 / 426 文档块）

---

## 1. 核心结论

| 指标 | 数值 | 说明 |
|---|---|---|
| **RAGAS Context Recall（召回率）** | **0.9464** | golden 答案信息点 94.6% 可被检索上下文覆盖 |
| **RAGAS Context Precision（准确率）** | **0.9675** | 相关上下文以 96.8% 的 AP 精度排在前列 |
| 编码级 Hit@5 | 0.75 | 28 条语义用例中 golden 实体进入 Top-5 的比例 |
| 编码级 Recall@5 | 0.7188 | Top-5 覆盖 golden 实体的比例 |
| 编码级 MRR | 0.8073 | golden 首次命中排名倒数均值 |
| 结构化通道引用命中 | 4/4 = 100% | catalog 通道 references 编码级命中 |

**总体判断：题目检索接近满分（Hit@5 = 100%，MRR = 0.958），文档检索存在章节粒度偏差（编码级 Hit@5 = 75%），课程结构化检索全命中。系统检索质量整体优良，短板集中在课程文档的章节级定位。**

## 2. RAGAS 指标（LLM 评判，语义通道 28 条）

| 通道 | Context Recall | Context Precision | 样本数 |
|---|---|---|---|
| 题目检索 question | 0.9583 | 0.9542 | 12 |
| 文档检索 doc | 0.9375 | **0.9774** | 16 |
| **合计** | **0.9464** | **0.9675** | 28 |

### 与编码级指标的关系

两套指标口径不同、互相印证：

- **RAGAS context_recall** 评"信息覆盖"：只要 Top-5 中任一块承载了 golden 答案的信息点（哪怕是同文档相邻章节），即计为覆盖。文档检索语义分散但信息相近，故 RAGAS 召回率高（0.9375）而编码级章节命中偏低（0.6875）。
- **编码级指标**评"实体定位"：要求精确召回 golden 章节/题目实体。它是上限更严的口径，两者差值（0.94 vs 0.72）恰好量化了"信息覆盖没问题、精确章节定位有偏差"这一事实。

### 低分案例分析（RAGAS recall < 1 的 3 条）

| 用例 | 问题 | recall | 分析 |
|---|---|---|---|
| Q03 | 面向对象设计中哪个概念体现隐藏内部实现？ | 0.5 | golden 题目（q007，封装）排名第 1，但 ground_truth 中"降低耦合"等表述未被 Top-5 完整覆盖——上下文偏短所致，答案本身正确 |
| D15 | 过拟合怎么解决？ | 0.5 | 召回的是深度学习"学习的技巧（优化）"章节邻块，正则化/Dropout/早停等信息分散在相邻分块，Top-5 只覆盖一半 |
| D16 | 预训练模型怎么微调？ | 0.5 | NLP"预训练模型"章节分块后，微调细节分散多块，且混入题库实体稀释了 Top-5 |

## 3. 编码级确定性指标（Rerank Top-5 精确匹配）

| 通道 | Hit@5 | Recall@5 | Precision@5 | MRR | 样本数 |
|---|---|---|---|---|---|
| 题目检索 | **1.0000** | **1.0000** | 0.2000 | **0.9583** | 12 |
| 文档检索 | 0.7500 | 0.6875 | 0.1500 | 0.6562 | 16 |
| 课程结构化 | —（references 命中 4/4） | — | — | — | 4 |
| 全量语义 | 0.8929 | 0.7188 | 0.1786 | 0.8073 | 28 |

> Precision@5 偏低是设计使然：Top-5 中通常只含 1 个 golden 实体（题目类 golden 唯一），故 12/28 条的 precision 天然为 0.2 左右；这不代表检索质量差，重点看 Hit/Recall/MRR。

### 文档检索未命中案例（4/16）

| 用例 | 问题 | 实际 Top-1 | 偏差类型 |
|---|---|---|---|
| D01 | LangGraph 的状态管理是怎么做的？ | LangGraph 概述章节 | **章节定位偏移**：召回了正确文档，但"概述"而非"状态"章节 |
| D05 | 线性回归的误差项有什么假设？ | 回归算法.pdf 线性回归首章 | **章节定位偏移**：文档命中，误差小节未进 Top-5 |
| D06 | 梯度下降算法的思想是什么？ | 深度学习"神经网络的学习"章节 | **跨文档混淆**：深度学习文档的梯度下降块压过了回归算法.pdf 的目标块 |
| D13 | LangChain 的 Retrieval 组件能做什么？ | LangChain 概述章节 | **章节定位偏移**：同 D01 模式 |

共性根因：概述章节因高频关键词（LangGraph/LangChain/线性回归）在向量空间中与几乎所有针对该文档的问题都相似，rerank 后总是压过具体章节块。

## 4. 其他发现

1. **Milvus 混入测试数据**：C04（"有哪些适合在校生的课程"）的引用中出现 `test_docx_series_full`——为历史导入测试残留，建议按该编码清理，否则会污染线上课程检索结果。
2. **性能**：单条全链路（意图路由 + 三路召回 + RRF + rerank + LLM 生成）平均 **6.3 秒**，其中 LLM 生成占大头；结构化通道仅 3-4 秒。无降级发生（degraded = 0/32）。
3. **意图路由**：32 条全部路由正确（content 28 / catalog 4，无误入 aggregate/ambiguous）。
4. **RAGAS 兼容性**：ragas 0.4.3 与项目 langchain-community 0.4 不兼容（vertexai 模块移除），评估脚本内置 stub 已解决；RAGAS 评判走云端 API，并发需压到 2（RunConfig）避免限流。

## 5. 改进建议

| 优先级 | 建议 | 预期收益 |
|---|---|---|
| P0 | **章节级查询增强**：content 意图下若实体抽取能识别文档名，在 rerank 阶段给"文档命中且章节名与问题关键词重叠"的块加权；或在分块时把章节路径前缀权重提高 | 直接修复 D01/D05/D13 类章节偏移，doc Hit@5 预计 0.75 → 0.9+ |
| P1 | **清理 Milvus 测试残留**（`test_docx_series_full` 等） | 消除课程检索污染 |
| P1 | **概述章节去重降权**：同一文档的概述块与具体章节块同时进 Top-5 时，若概述块仅为目录性内容，可在 rerank 后过滤 | 提升 doc precision 与上下文信噪比 |
| P2 | **Top-K 自适应**：文档类问题（检索集合含 edu_course_doc）可将 rerank Top-5 放宽到 Top-8，缓解信息分散（D15/D16） | RAGAS recall 0.94 → 0.97+ |
| P2 | **评测集扩充**：32 条 → 100+ 条并纳入多轮对话用例；把本评测挂入 CI，导入数据变更后自动回归 | 度量可持续 |

## 6. 复现方式

```bash
cd /Users/zdy/PycharmProjects/17zuoye_brain

# 1. 收集检索结果（需 Milvus + LLM API 可用，约 3.5 分钟）
.venv/bin/python eval/run_retrieval.py

# 2. RAGAS 评估（需 LLM API，约 12 分钟）
.venv/bin/python eval/ragas_eval.py

# 产物
# eval/eval_runs/results.json          全量检索明细（32 条 × 5 层级）
# eval/eval_runs/ragas_scores.json     RAGAS 逐条 + 汇总分数
# eval/eval_runs/metrics_summary.json  确定性指标 + RAGAS 汇总
# eval/metrics_report.md               本报告
```

---

### 附录：32 条用例逐条结果

| 用例 | 通道 | intent | 编码命中排名 | RAGAS recall | RAGAS precision |
|---|---|---|---|---|---|
| Q01 | question | content | 1 | 1.0 | 1.0 |
| Q02 | question | content | 1 | 1.0 | 1.0 |
| Q03 | question | content | 1 | 0.5 | 1.0 |
| Q04 | question | content | 1 | 1.0 | 0.7 |
| Q05 | question | content | 1 | 1.0 | 1.0 |
| Q06 | question | content | 1 | 1.0 | 1.0 |
| Q07 | question | content | 2 | 1.0 | 1.0 |
| Q08 | question | content | 1 | 1.0 | 0.75 |
| Q09 | question | content | 1 | 1.0 | 1.0 |
| Q10 | question | content | 1 | 1.0 | 1.0 |
| Q11 | question | content | 1 | 1.0 | 1.0 |
| Q12 | question | content | 1 | 1.0 | 1.0 |
| D01 | doc | content | 未进 Top-5 | 1.0 | 1.0 |
| D02 | doc | content | 1 | 1.0 | 1.0 |
| D03 | doc | content | 1 | 1.0 | 1.0 |
| D04 | doc | content | 1 | 1.0 | 1.0 |
| D05 | doc | content | 未进 Top-5 | 1.0 | 1.0 |
| D06 | doc | content | 未进 Top-5 | 1.0 | 0.8333 |
| D07 | doc | content | 1 | 1.0 | 1.0 |
| D08 | doc | content | 1 | 1.0 | 1.0 |
| D09 | doc | content | 2 | 1.0 | 1.0 |
| D10 | doc | content | 1 | 1.0 | 1.0 |
| D11 | doc | content | 1 | 1.0 | 1.0 |
| D12 | doc | content | 2 | 1.0 | 1.0 |
| D13 | doc | content | 未进 Top-5 | 1.0 | 1.0 |
| D14 | doc | content | 1 | 1.0 | 1.0 |
| D15 | doc | content | 2 | 0.5 | 1.0 |
| D16 | doc | content | 1 | 0.5 | 0.8056 |
| C01 | course | catalog | references 命中 | — | — |
| C02 | course | catalog | references 命中 | — | — |
| C03 | course | catalog | references 命中 | — | — |
| C04 | course | catalog | references 命中（含测试残留实体） | — | — |
