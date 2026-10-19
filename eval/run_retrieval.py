"""
检索收集脚本：对黄金集逐条执行查询图，落盘全量检索结果供 RAGAS 与确定性指标评估
- 进程内调用 query_app.invoke（不依赖 HTTP 服务）
- 收集：answer / references / reranked_results / fused_results / embedding_hits / hyde_hits / structured_results
- 每条记录附编码级标识（question_code / course_code / series_code / doc chunk 键），供 Hit@K 等确定性指标使用
- 输出：eval/eval_runs/results.json
"""
import json
import sys
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from processor.query_processor.main_graph import query_app          # noqa: E402
from processor.query_processor.state import create_query_default_state  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "eval"
RESULTS_PATH = EVAL_DIR / "eval_runs" / "results.json"


def record_key(record: dict) -> str:
    """提取检索记录的编码级唯一标识（与黄金集 golden_codes 同一命名空间）
    - edu_question → question_code
    - edu_course_series → series_code
    - edu_course_module → course_code
    - edu_course_doc → doc:{source_file}|{chapter_name}
    """
    coll = record.get("_collection", "")
    if coll == "edu_question":
        return record.get("question_code") or ""
    if coll == "edu_course_series":
        return record.get("series_code") or ""
    if coll == "edu_course_module":
        return record.get("course_code") or ""
    if coll == "edu_course_doc":
        return f"doc:{record.get('source_file')}|{record.get('chapter_name')}"
    return ""


def record_text(record: dict) -> str:
    """将检索记录转为上下文文本（与 node_answer_output._format_record 对齐，含正文内容）"""
    coll = record.get("_collection", "")
    if coll == "edu_question":
        parts = [f"[题目] 编码:{record.get('question_code')} 题库:{record.get('bank_name')} "
                 f"题型:{record.get('question_type')} 答案:{record.get('answer')}"]
        if record.get("content"):
            parts.append(f"题干：{record['content'][:500]}")
        if record.get("analysis"):
            parts.append(f"解析：{record['analysis'][:300]}")
        return "\n".join(parts)
    if coll == "edu_course_series":
        return (f"[课程系列] 编码:{record.get('series_code')} 名称:{record.get('series_name')} "
                f"分类:{record.get('course_category')} 适合人群:{record.get('target_audience')} "
                f"学习目标:{record.get('learning_goal')} 适合年级:{record.get('suitable_grade')}")
    if coll == "edu_course_module":
        return (f"[课程模块] 编码:{record.get('course_code')} 名称:{record.get('course_name')} "
                f"所属系列:{record.get('series_name')} 课时:{record.get('duration')} "
                f"学时:{record.get('credit_hours')} 描述:{record.get('description')}")
    if coll == "edu_course_doc":
        return (f"[课程文档] 章节:{record.get('chapter_name')} "
                f"来源文件:{record.get('source_file')} 内容:{(record.get('content') or '')[:800]}")
    return json.dumps({k: v for k, v in record.items() if not k.startswith("_")},
                      ensure_ascii=False, default=str)[:300]


def to_view(records: list) -> list:
    """检索记录 → (编码, 排序分数, 文本) 视图"""
    out = []
    for r in records or []:
        key = record_key(r)
        score = r.get("rerank_score", r.get("_rrf_score", r.get("score")))
        out.append({"key": key, "score": score, "text": record_text(r)})
    return out


def main(limit: int = None, case_ids: list = None,
         dataset_path: str = None, out_path: str = None):
    src = Path(dataset_path) if dataset_path else EVAL_DIR / "golden_dataset.json"
    dataset = json.loads(src.read_text(encoding="utf-8"))
    out_file = Path(out_path) if out_path else RESULTS_PATH
    print(f"数据集：{src}")
    cases = dataset["cases"]
    if case_ids:
        cases = [c for c in cases if c["id"] in case_ids]
    if limit:
        cases = cases[:limit]

    print(f"开始收集：{len(cases)} 条评测用例")
    results = []
    t0 = time.time()
    for i, case in enumerate(cases, 1):
        qid, query = case["id"], case["query"]
        t1 = time.time()
        try:
            state = create_query_default_state(
                session_id=f"eval-{uuid.uuid4().hex[:8]}",
                original_query=query,
            )
            final = query_app.invoke(state)
            elapsed = round(time.time() - t1, 2)
            row = {
                "id": qid,
                "channel": case["channel"],
                "query": query,
                "golden_codes": case["golden_codes"],
                "ground_truth": case["ground_truth"],
                "intent": final.get("intent"),
                "answer": final.get("answer", ""),
                "degraded": bool(final.get("degraded")),
                "elapsed_s": elapsed,
                "references": final.get("references", []),
                # 全量检索层级（编码 + 分数 + 文本）
                "reranked": to_view(final.get("reranked_results")),
                "fused": to_view(final.get("fused_results")),
                "embedding_hits": to_view(final.get("embedding_hits")),
                "hyde_hits": to_view(final.get("hyde_hits")),
                "structured": to_view(final.get("structured_results")),
            }
            top_keys = [r["key"] for r in row["reranked"][:5]]
            print(f"[{i}/{len(cases)}] {qid} ({elapsed}s) intent={row['intent']} top5={top_keys}")
        except Exception as e:
            elapsed = round(time.time() - t1, 2)
            row = {"id": qid, "channel": case["channel"], "query": query,
                   "golden_codes": case["golden_codes"], "ground_truth": case["ground_truth"],
                   "error": str(e), "elapsed_s": elapsed}
            print(f"[{i}/{len(cases)}] {qid} ERROR: {e}")
        results.append(row)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps({"collected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "dataset": str(src), "results": results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n完成：{len(results)} 条，总耗时 {round(time.time() - t0, 1)}s")
    print(f"结果已写入 {out_file}")


if __name__ == "__main__":
    args = sys.argv[1:]
    kv = {a.split("=", 1)[0]: a.split("=", 1)[1] for a in args if a.startswith("--") and "=" in a}
    limit = int(next((a for a in args if a.isdigit()), 0)) or None
    ids = [a for a in args if not a.isdigit() and not a.startswith("--")] or None
    main(limit=limit, case_ids=ids,
         dataset_path=kv.get("--dataset"), out_path=kv.get("--out"))
