# 项目长期记忆

## 文档组织约定

- 项目文档统一放在 `文档/` 下，按链路分目录：
  - `文档/导入流程/` —— 导入链路（8 个图节点各一篇 + ZIP 资料包通道 + `导入_00_流程总览.md`）
  - `文档/查询流程/` —— 查询链路
- 单篇文件名格式：`导入_<两位序号>_<节点名或主题>.md`，如 `导入_03_node_md_img.md`、`导入_09_zip资料包导入.md`；序号与数据流顺序一致，`00` 是总览
- 每篇节点文档的结构约定：流程位置/上下游 → mermaid 图 → 逐 step 代码实现 → 输入输出字段表 → 边界与坑 → 调试手册 → 设计取舍

## 导入链路关键事实（写文档/改代码前先看）

- 8 个节点：`node_entry` → `node_format_convert` → `node_md_img` → 三个 parser 之一 → `node_bge_embedding` → `node_import_milvus`
- 失败不抛异常：`format_convert_error`（终止）/ `parse_failed_entities`（不阻断）/ `embedding_error` / `import_error`
- 格式归一化：**pdf / doc / docx 统一走 MinerU**（本地 python-docx 解析已移除，无降级）；`md` 在 `node_entry` 直读
- `node_format_convert._normalize_mineru_markdown()` 做两件事：去标题行 `**`（Word 加粗）、把 5~6 级点分编号标题降为列表项（MinerU 把 Word 多级列表导成了标题，会让分块器切碎）
- 指标基线：课程目录 219 系列 + 657 模块；题库 73 库 + 1752 题（1 条源数据异常）；LangGraph 讲义（docx→MinerU）26 章节 → 44 块 / avg 994 字符 / 9 张图全保留
- `node_course_doc_parser` 的 `_flush_section()` 对**所有**标题（含 `###`）生效 → 三级标题会切 section 但不改 `chapter_name`
- 章节名剥自动编号：`_emit_chunk` 按 `state["file_format"]` 决定（`pdf/doc/docx` 剥、`md` 保留），只清 `chapter_name`，正文标题保留编号；`_strip_chapter_number` 用「数字>99 / 后接量词」防误伤
- `node_md_img` 两处清洗：`_clean_image_summary` 剥 VL 的 prompt 前导语与 `**`（上限 200 字）；`img_dir = minio_img_dir.strip("/")` 避免 `bucket//upload-images` 双斜杠
- MinerU 对 Word 的已知短板：**代码块围栏丢失**（代码变普通段落）、页眉页脚不导出

## 查询链路关键事实

- 图：`query_rewrite` → `intent_route` →（catalog/aggregate/content/ambiguous）→ `structured_query` / `aggregate` / `semantic_fanout`（三路召回）→ `rrf` → `rerank` → `answer_output`
- **图片链路**：导入侧 `node_md_img` 把图写成 `![VL摘要](MinIO URL)` 存在 `content` 里（无独立字段）→ 查询 `node_answer_output.step_6_extract_image_urls` 从**原始 content**（不能用 `_format_record` 的截断文本）提取 → `image_urls`（去重、≤6）→ API 响应 / SSE final / Mongo / 前端缩略图
- **HyDE 默认关闭**（`ENABLE_HYDE=0`）：A/B 实测题目通道指标零差异、content 7.23s→5.02s；开启用 `ENABLE_HYDE=1`
- **黄金集 v1.1（33 条）**：doc 17 条按真实 `(source_file, chapter_name)` 重写，Q12/C4 沿用；`eval/eval_keys.py` 统一剥章节编号后比对，库内带不带编号都能匹配
- 基线（`eval/eval_runs/results_v11.json`）：question Hit@5 1.0 / doc Hit@5 0.9412（D14 未命中）/ course reference_hit 4/4
- RRF 的 rank 是"通道内拼接后的全局位置"，集合按 series→module→question→doc 拼接，doc 系统性靠后（`RERANK_INPUT_LIMIT=30` 就是为捞回 doc 块）
