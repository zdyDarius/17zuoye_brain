"""golden ground_truth 泄漏检测（评估防作弊检查）

为什么需要：如果出题人（或 LLM）是看着检索块写问题和答案，问题、答案、被检索内容
三者同源，RAGAS Context Recall 会虚高到失去参考价值。本脚本用「逐字重合度」量化
这种泄漏，把"感觉像抄的"变成可执行的阈值判断。

指标：
- gt_lcs : ground_truth 与 golden 块的最长**连续**公共子串字符数
- q_lcs  : query 与 golden 块的最长连续公共子串字符数（问题在抄教材原句）
- cov5   : ground_truth 的 5-gram 被块覆盖的比例（整体措辞相似度）

判定阈值（技术文本必然有术语重合，故不看短片段）：
- gt_lcs >= 50                → 严重抄袭（成段逐字复制）
- 30 <= gt_lcs < 50 或 cov5>=0.55 → 可疑
- 20 <= gt_lcs < 30           → 轻度重合（可接受，建议人工扫一眼）
- 其余                        → OK

用法：
    PYTHONPATH=. .venv/bin/python eval/check_gt_leakage.py                    # 检查 golden_dataset.json
    PYTHONPATH=. .venv/bin/python eval/check_gt_leakage.py eval/xxx_draft.json
    # 存在严重抄袭时退出码为 1，可接入 CI 阻断
"""
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.clients.milvus_utils import get_milvus_client  # noqa: E402

GT_FATAL = 50       # 连续相同字符数：严重抄袭
GT_SUSPECT = 30     # 连续相同字符数：可疑
GT_MINOR = 20       # 连续相同字符数：轻度
COV_SUSPECT = 0.55  # 5-gram 覆盖率：可疑

_NG = 5


def _norm(text: str) -> str:
    """去掉图片与 markdown 标记，只留文字，避免 URL 干扰重合度计算"""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text or "")
    text = re.sub(r"[#*`>|\-\[\]()]", "", text)
    return re.sub(r"\s+", "", text)


def _lcs(a: str, b: str) -> str:
    """最长连续公共子串"""
    m = SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
    return a[m.a:m.a + m.size]


def _coverage(gt: str, body: str, n: int = _NG) -> float:
    """gt 的 n-gram 有多少比例能在块里找到"""
    grams = {gt[i:i + n] for i in range(len(gt) - n + 1)}
    if not grams:
        return 0.0
    return sum(1 for g in grams if g in body) / len(grams)


def _fetch_block(client, source_file: str, chapter: str) -> str:
    rows = client.query(
        collection_name="edu_course_doc",
        filter=f'source_file == "{source_file}"',
        output_fields=["chapter_name", "content"],
        limit=16384,
    )
    return "\n".join(r["content"] for r in rows if (r.get("chapter_name") or "") == chapter)


def main(path: str):
    dataset = json.load(open(path, encoding="utf-8"))
    client = get_milvus_client()

    print(f"检查 {path}（{len(dataset['cases'])} 条）")
    print(f"{'ID':<6}{'gt_lcs':>7}{'q_lcs':>7}{'cov5':>7}  判定")
    print("-" * 60)

    rows, missing = [], []
    for case in dataset["cases"]:
        if case["channel"] != "doc":
            continue
        body = ""
        for code in case["golden_codes"]:
            src, _, chapter = code[4:].partition("|")
            part = _fetch_block(client, src, chapter)
            if not part:
                missing.append((case["id"], code))
            body += _norm(part)
        if not body:
            continue

        gt, q = _norm(case["ground_truth"]), _norm(case["query"])
        seg_gt, seg_q = _lcs(gt, body), _lcs(q, body)
        cov = _coverage(gt, body)

        if len(seg_gt) >= GT_FATAL:
            verdict = "严重抄袭"
        elif len(seg_gt) >= GT_SUSPECT or cov >= COV_SUSPECT:
            verdict = "可疑"
        elif len(seg_gt) >= GT_MINOR:
            verdict = "轻度重合"
        else:
            verdict = "OK"

        rows.append((case["id"], len(seg_gt), len(seg_q), cov, verdict, seg_gt))
        print(f"{case['id']:<6}{len(seg_gt):>7}{len(seg_q):>7}{cov:>6.0%}  {verdict}")

    print("\n最长的逐字重复片段：")
    for cid, n, _, _, _, seg in sorted(rows, key=lambda x: -x[1])[:5]:
        print(f"  {cid} {n:>3} 字：{seg[:90]}")

    if missing:
        print(f"\n[警告] {len(missing)} 个 golden_code 在库中不存在：")
        for cid, code in missing:
            print(f"  {cid} {code}")

    bad = [r for r in rows if r[4] in ("严重抄袭", "可疑")]
    print(f"\n需要重写：{len(bad)}/{len(rows)} 条" + (f" → {[r[0] for r in bad]}" if bad else ""))
    return 1 if bad or missing else 0


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).resolve().parent / "golden_dataset.json")
    sys.exit(main(target))
