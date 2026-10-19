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
- MinerU 对 Word 的已知短板：**代码块围栏丢失**（代码变普通段落）、页眉页脚不导出
