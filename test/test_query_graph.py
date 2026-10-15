"""
查询图评测脚本：附录 A 核心用例验证（目录/聚合/语义/反问四通道）
"""
import sys

sys.path.insert(0, "/Users/zdy/PycharmProjects/17zuoye_brain")

import uuid
from processor.query_processor.main_graph import query_app
from processor.query_processor.state import create_query_default_state

TEST_CASES = [
    # (用例编号, 问题, 期望通道, 快速断言)
    ("A3-16", "通用程序设计题库有多少道单选题？", "aggregate", "8"),
    ("A3-20", "完形填空题有几道？", "aggregate", "14"),
    ("A3-26", "判断题的正确答案分布大概什么样？", "aggregate", "145"),
    ("A3-25", "题库的题量分布是怎样的？", "aggregate", "24"),
    ("A1-6", "通用编程入门班包含哪几个课程模块？", "catalog", "3"),
    ("A1-8", "综合项目开发与讲评属于哪个系列？", "catalog", "系列"),
    ("A3-24", "有没有填空题？", "ambiguous", "没有|题型|完形"),
    ("A4-31", "有没有关于变量和常量的练习题？", "content", "q001|变量"),
    ("A4-32", "帮我找函数参数传递相关的选择题", "content", "q004|参数"),
]


def run():
    passed = failed = 0
    for case_id, question, expected_channel, assertion in TEST_CASES:
        try:
            state = create_query_default_state(
                session_id=str(uuid.uuid4()),
                original_query=question,
            )
            final = query_app.invoke(state)
            intent = final.get("intent") or "?"
            answer = final.get("answer", "")
            refs = len(final.get("references", []))
            # 断言：期望通道 + 答案含关键词
            channel_ok = (intent == expected_channel) if expected_channel != "content" else True
            answer_ok = any(kw in answer for kw in assertion.split("|"))
            status = "PASS" if (channel_ok and answer_ok) else "FAIL"
            if status == "PASS":
                passed += 1
            else:
                failed += 1
            print(f"[{status}] {case_id} | {question}")
            print(f"       intent={intent}(期望 {expected_channel}) 引用={refs}条")
            print(f"       答案摘要: {answer[:150].replace(chr(10), ' ')}")
        except Exception as e:
            failed += 1
            print(f"[ERROR] {case_id} | {question} | {e}")
    print(f"\n== 评测汇总：{passed} 通过 / {failed} 失败 / 共 {len(TEST_CASES)} 条 ==")


if __name__ == "__main__":
    run()
