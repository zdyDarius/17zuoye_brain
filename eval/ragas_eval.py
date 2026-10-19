"""
RAGAS 性能评估脚本：召回率（context_recall）+ 准确率（context_precision）
================================================================
评估对象：eval/eval_runs/results.json（由 run_retrieval.py 生成）

指标体系（两层互补）：
1. RAGAS LLM 评判指标（ragas 0.4.3）
   - context_recall        召回率：golden 答案中的信息点能否被检索上下文覆盖
   - context_precision     准确率：相关上下文是否排在前列（AP 式加权）
   评判模型复用项目 .env 配置的 LLM（经 lm_config 注入，脚本不接触明文 key）
2. 编码级确定性指标（与 RAGAS 交叉验证，对 golden_codes 精确匹配）
   - Hit@K       前 K 条是否含 golden 实体
   - Recall@K    前 K 条覆盖 golden 实体的比例
   - Precision@K 前 K 条中 golden 实体占比
   - MRR         golden 首次出现排名的倒数均值

输出：
- eval/eval_runs/ragas_scores.json   （逐条 + 汇总）
- eval/eval_runs/metrics_summary.json（确定性指标 + RAGAS 汇总，供报告生成）
"""
import json
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ---- ragas 0.4.3 兼容 stub：langchain-community 0.4 移除了 vertexai 模块 ----
_mod = types.ModuleType("langchain_community.chat_models.vertexai")
class _ChatVertexAI:  # 仅满足 ragas import/isinstance 需求的桩
    pass
_mod.ChatVertexAI = _ChatVertexAI
sys.modules.setdefault("langchain_community.chat_models.vertexai", _mod)

from ragas import evaluate                                   # noqa: E402
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset  # noqa: E402
from ragas.llms import LangchainLLMWrapper                   # noqa: E402
from ragas.metrics import LLMContextRecall, ContextPrecision  # noqa: E402
from ragas.run_config import RunConfig                       # noqa: E402

from utils.lm.lm_utils import get_llm_client                 # noqa: E402
from eval.eval_keys import normalize_code                    # noqa: E402

EVAL_DIR = PROJECT_ROOT / "eval"
RESULTS_PATH = EVAL_DIR / "eval_runs" / "results.json"
RAGAS_PATH = EVAL_DIR / "eval_runs" / "ragas_scores.json"
SUMMARY_PATH = EVAL_DIR / "eval_runs" / "metrics_summary.json"

TOP_K = 5  # 语义通道 rerank 输出的 Top-K，与线上 node_rerank 一致


# ======================== 编码级确定性指标 ========================

def deterministic_metrics(results: list, top_k: int = TOP_K) -> dict:
    """对 reranked Top-K 做编码级精确匹配（结构化通道命中计入 references）"""
    per_case, by_channel = [], {}

    def upd(store, key, val):
        store.setdefault(key, []).append(val)

    for r in results:
        if r.get("error"):
            continue
        # 归一化 doc: 编码（剥章节自动编号），避免导入批次不同导致同名章节对不上
        golden = {normalize_code(x) for x in (r.get("golden_codes") or [])}
        ranked = [normalize_code(x["key"]) for x in (r.get("reranked") or []) if x["key"]]
        top = ranked[:top_k]
        refs = set()
        for ref in r.get("references") or []:
            if ref.get("code"):
                refs.add(normalize_code(ref["code"]))
            if ref.get("chapter"):
                src = ref.get("source_file") or ""
                refs.add(normalize_code(f"doc:{src}|{ref['chapter']}"))
        # 结构化/聚合通道：references 即命中集
        ref_hit = len(golden & refs) > 0 if refs else False

        hit_k = int(bool(golden & set(top)))
        recall_k = len(golden & set(top)) / len(golden) if golden else 0.0
        prec_k = len(golden & set(top)) / len(top) if top else 0.0
        mrr = 0.0
        for rank, key in enumerate(top, 1):
            if key in golden:
                mrr = 1.0 / rank
                break

        row = {
            "id": r["id"], "channel": r.get("channel"),
            "intent": r.get("intent"),
            "hit@K": hit_k, "recall@K": round(recall_k, 4),
            "precision@K": round(prec_k, 4), "mrr": round(mrr, 4),
            "reference_hit": ref_hit,
            "top_keys": ranked[:top_k],
        }
        per_case.append(row)
        ch = r.get("channel") or "?"
        for k in ("hit@K", "recall@K", "precision@K", "mrr"):
            upd(by_channel, f"{ch}:{k}", row[k])

    def avg(vals):
        return round(sum(vals) / len(vals), 4) if vals else None

    overall = {}
    for key, vals in by_channel.items():
        overall[key] = avg(vals)
    # 全量均值
    for k in ("hit@K", "recall@K", "precision@K", "mrr"):
        overall[f"ALL:{k}"] = avg([row[k] for row in per_case])
    return {"top_k": top_k, "overall": overall, "per_case": per_case}


# ======================== RAGAS 评估 ========================

def ragas_eval(results: list, top_k: int = TOP_K) -> dict:
    """RAGAS context_recall / context_precision（语义通道用例）"""
    samples, meta = [], []
    for r in results:
        if r.get("error") or r.get("channel") == "course":
            continue  # 结构化通道可穷举，确定性指标已覆盖；RAGAS 评语义通道
        contexts = [x["text"] for x in (r.get("reranked") or [])][:top_k]
        if not contexts:
            continue
        samples.append(SingleTurnSample(
            user_input=r["query"],
            retrieved_contexts=contexts,
            response=r.get("answer", ""),
            reference=r["ground_truth"],
        ))
        meta.append({"id": r["id"], "channel": r["channel"]})

    print(f"RAGAS 评估样本：{len(samples)} 条（语义通道：question/doc）")
    llm = LangchainLLMWrapper(get_llm_client())
    dataset = EvaluationDataset(samples)

    # 评判 LLM 云端限流：低并发 + 长超时 + 失败重试
    run_config = RunConfig(max_workers=2, max_wait=120, timeout=180, max_retries=5)
    scores = evaluate(
        dataset=dataset,
        metrics=[LLMContextRecall(llm=llm), ContextPrecision(llm=llm)],
        run_config=run_config,
        show_progress=True,
    )

    df = scores.to_pandas()
    per_case = []
    for i, m in enumerate(meta):
        cr = df.iloc[i].get("context_recall")
        cp = df.iloc[i].get("context_precision")

        def _f(x):
            return None if x is None or (isinstance(x, float) and x != x) else round(float(x), 4)

        per_case.append({
            "id": m["id"], "channel": m["channel"],
            "context_recall": _f(cr),
            "context_precision": _f(cp),
        })

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    summary = {
        "context_recall": _mean([c["context_recall"] for c in per_case]),
        "context_precision": _mean([c["context_precision"] for c in per_case]),
    }
    # 分通道均值
    for ch in ("question", "doc"):
        sub = [c for c in per_case if c["channel"] == ch]
        if sub:
            summary[f"{ch}:context_recall"] = _mean([c["context_recall"] for c in sub])
            summary[f"{ch}:context_precision"] = _mean([c["context_precision"] for c in sub])
    return {"summary": summary, "per_case": per_case}


def main(input_path: str = None, suffix: str = ""):
    src = Path(input_path) if input_path else RESULTS_PATH
    data = json.loads(src.read_text(encoding="utf-8"))
    ragas_path = EVAL_DIR / "eval_runs" / f"ragas_scores{suffix}.json"
    summary_path = EVAL_DIR / "eval_runs" / f"metrics_summary{suffix}.json"
    print(f"载入检索结果：{src}")
    results = data["results"]
    ok = [r for r in results if not r.get("error")]
    print(f"载入 {len(results)} 条结果（{len(ok)} 正常 / {len(results) - len(ok)} 异常）")

    # 1. 确定性指标
    det = deterministic_metrics(ok)
    print("\n== 编码级确定性指标（rerank Top-5）==")
    for k in sorted(det["overall"]):
        print(f"  {k}: {det['overall'][k]}")

    # 2. RAGAS 指标
    rag = ragas_eval(ok)
    print("\n== RAGAS 指标 ==")
    for k, v in rag["summary"].items():
        print(f"  {k}: {v}")

    # 落盘
    ragas_path.write_text(json.dumps(rag, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps({
        "evaluated_at": data.get("collected_at"),
        "top_k": TOP_K,
        "deterministic": det["overall"],
        "deterministic_per_case": det["per_case"],
        "ragas": rag["summary"],
        "ragas_per_case": rag["per_case"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {ragas_path}")
    print(f"已写入 {summary_path}")


if __name__ == "__main__":
    # 用法：ragas_eval.py [--in=results.json] [--suffix=_draft]
    kv = {a.split("=", 1)[0]: a.split("=", 1)[1]
          for a in sys.argv[1:] if a.startswith("--") and "=" in a}
    main(input_path=kv.get("--in"), suffix=kv.get("--suffix", ""))
