"""
题库 Parser 节点
对应 PRD V1.3 3.2「Markdown 题库解析规则」：
- 题库边界：## {题库名称}，其下 - 题库编码: {bank_code}
- 题目边界：### {question_code}
- 字段锚点：- **题型**: / - **题干**: / - **选项**: / - **答案**: / - **解析**:
- content 入库保留 \n 与缩进（题干与选项不粘连）
- 无「选项」字段的题型 options 可空
对应 PRD 4.3.5「答案归一化规则」：
- 判断题：A/对/正确 → true；B/错/错误 → false
- 单选题：保留大写单字母
- 多选题：半角逗号分隔并按字母序排列
- 长文本题型：保留原文
- 全题型另存 answer_raw 原文
产出：question_records（PRD 4.3 Collection 4）
"""
import re
from pathlib import Path

from common.logging.logger import logger, node_log, step_log


# 题库标题行：## 通用程序设计题库
_BANK_HEADER_RE = re.compile(r"^##\s+(?P<name>.+?)\s*$")
# 题库编码行：- 题库编码: general_purpose_programming_bank
_BANK_CODE_RE = re.compile(r"^-\s*题库编码\s*[:：]\s*(?P<value>.+?)\s*$")
# 题目标题行：### general_purpose_programming_bank_q001
_QUESTION_HEADER_RE = re.compile(r"^###\s+(?P<code>[A-Za-z0-9_\-]+)\s*$")
# 题目字段锚点：- **题型**: 单选题
_FIELD_TYPE_RE = re.compile(r"^-\s*\*\*题型\*\*\s*[:：]\s*(?P<value>.*?)\s*$")
_FIELD_STEM_RE = re.compile(r"^-\s*\*\*题干\*\*\s*[:：]\s*(?P<value>.*?)\s*$")
_FIELD_OPTIONS_RE = re.compile(r"^-\s*\*\*选项\*\*\s*[:：]\s*(?P<value>.*?)\s*$")
_FIELD_ANSWER_RE = re.compile(r"^-\s*\*\*答案\*\*\s*[:：]\s*(?P<value>.*?)\s*$")
_FIELD_ANALYSIS_RE = re.compile(r"^-\s*\*\*解析\*\*\s*[:：]\s*(?P<value>.*?)\s*$")
# 选项行：A. xxx / A、xxx（多行文本，直至下一个字段锚点或空行段落结束）
_OPTION_ITEM_RE = re.compile(r"^(?P<key>[A-Za-z])\s*[.、]\s*(?P<value>.*)$")
# 多选题答案分隔：半角逗号 / 顿号 / 无分隔（如 AB）
_MULTI_ANSWER_SPLIT_RE = re.compile(r"[,，、\s]+")

# 中文题型 → 英文枚举码（PRD 4.2 配置表）
QUESTION_TYPE_MAP = {
    "单选题": "single_choice",
    "多选题": "multiple_choice",
    "判断题": "true_false",
    "案例分析题": "case_analysis",
    "公式推导题": "formula_derivation",
    "阅读理解题": "reading_comprehension",
    "完形填空题": "cloze",
    "材料分析题": "material_analysis",
    "简答题": "short_answer",
    "计算题": "calculation",
    "编程题": "programming",
    "情境分析题": "situation_analysis",
    "证明题": "proof",
    "数据分析题": "data_analysis",
    "翻译题": "translation",
}

# 判断题答案归一化映射（PRD 4.3.5）
_TRUE_FALSE_MAP = {
    "A": "true", "对": "true", "正确": "true",
    "B": "false", "错": "false", "错误": "false",
}


def normalize_answer(answer_raw: str, question_type_en: str):
    """
    答案归一化（PRD 4.3.5，parser 阶段执行）
    :param answer_raw: 原始答案文本
    :param question_type_en: 英文题型枚举码
    :return: (归一化答案 or None, 失败原因 or None, 数据异常标记 or None)
             数据异常标记：非 None 表示发现源数据矛盾（如单选题标注但答案为多字母），已宽容处理但需人工核对
    """
    raw = (answer_raw or "").strip()
    if not raw:
        return None, "答案为空", None

    if question_type_en == "true_false":
        normalized = _TRUE_FALSE_MAP.get(raw)
        if normalized is None:
            return None, f"判断题答案无法归一化：{raw}", None
        return normalized, None, None

    if question_type_en == "single_choice":
        if re.fullmatch(r"[A-Za-z]", raw):
            return raw.upper(), None, None
        # 宽容处理：标注单选但答案为多字母（源数据题型标注错误，实测 legal_certificate_bank_q007）
        # 按多选题归一化逻辑处理，标记数据异常供运营修正源数据
        letters = _MULTI_ANSWER_SPLIT_RE.split(raw) if _MULTI_ANSWER_SPLIT_RE.search(raw) else list(raw)
        letters = [l.strip().upper() for l in letters if l.strip()]
        if letters and all(re.fullmatch(r"[A-Z]", l) for l in letters):
            return ",".join(sorted(set(letters))), None, \
                f"题型标注为单选题但答案为多字母（{raw}），按多选题答案归一化，请核对源数据"
        return None, f"单选题答案无法归一化：{raw}", None

    if question_type_en == "multiple_choice":
        # 分隔符统一半角逗号 + 字母序排列；兼容无分隔的连写形式（如 AB）
        letters = _MULTI_ANSWER_SPLIT_RE.split(raw) if _MULTI_ANSWER_SPLIT_RE.search(raw) else list(raw)
        letters = [l.strip().upper() for l in letters if l.strip()]
        if not letters or not all(re.fullmatch(r"[A-Z]", l) for l in letters):
            return None, f"多选题答案含非字母项：{raw}", None
        return ",".join(sorted(set(letters))), None, None

    # 长文本题型（简答/编程/证明/计算/案例分析等）：保留原文
    return raw, None, None


def _parse_question_block(block_lines: list, bank_code: str, bank_name: str,
                          source_file: str, source_path: str):
    """
    解析单个题目块，组装题目记录
    :return: (record or None, failure dict or None)
    """
    fields = {}          # 字段名 -> 首行值
    options_lines = []   # 选项多行文本缓冲
    current_field = None # 当前正在收集的字段名（多行续行归属）

    for line in block_lines:
        # 字段锚点切换
        for field_name, field_re in [
            ("question_type", _FIELD_TYPE_RE),
            ("stem", _FIELD_STEM_RE),
            ("options", _FIELD_OPTIONS_RE),
            ("answer", _FIELD_ANSWER_RE),
            ("analysis", _FIELD_ANALYSIS_RE),
        ]:
            m = field_re.match(line)
            if m:
                current_field = field_name
                value = m.group("value")
                if field_name == "options":
                    # 选项字段的值可能是首行（A. xxx 同行）或空
                    options_lines = [value] if value else []
                else:
                    fields[field_name] = value
                break
        else:
            # 非锚点行：作为当前字段的续行（题干/解析/选项均为多行文本，保留换行）
            if current_field == "options":
                options_lines.append(line)
            elif current_field in ("stem", "analysis", "answer"):
                fields[current_field] = (fields.get(current_field, "") + "\n" + line).strip("\n")

    # ---- 校验业务主键与必填字段 ----
    question_code = None
    # question_code 由调用方从 ### 标题提取，此处校验其余字段
    question_type_cn = (fields.get("question_type") or "").strip()
    stem = (fields.get("stem") or "").strip()
    if not question_type_cn:
        return None, {"reason": "缺少题型字段"}
    if not stem:
        return None, {"reason": "缺少题干字段"}
    question_type_en = QUESTION_TYPE_MAP.get(question_type_cn)
    if question_type_en is None:
        # 开放集：未知题型记录失败（对应 PRD 4.2 新增题型需先插配置表）
        return None, {"reason": f"未知题型（需先在配置表登记）：{question_type_cn}"}

    # ---- 选项解析：A. xxx 多行 → options 字典 ----
    options = {}
    option_buffer = {}  # key -> [行列表]
    current_option_key = None
    for line in options_lines:
        m = _OPTION_ITEM_RE.match(line.strip())
        if m:
            current_option_key = m.group("key").upper()
            option_buffer[current_option_key] = [m.group("value")]
        elif current_option_key is not None:
            # 选项续行（多行选项文本）
            option_buffer[current_option_key].append(line.rstrip())

    for key, lines_val in option_buffer.items():
        options[key] = "\n".join(lines_val).strip()

    # ---- content 拼接：题干 + 选项，保留 \n（PRD 3.2 入库形态示例） ----
    content_parts = [stem]
    if options:
        content_parts.append("")
        for key in sorted(options.keys()):
            content_parts.append(f"{key}. {options[key]}")
    content = "\n".join(content_parts)

    # ---- 答案归一化（PRD 4.3.5） ----
    answer_raw = (fields.get("answer") or "").strip()
    answer, normalize_error, data_anomaly = normalize_answer(answer_raw, question_type_en)
    if normalize_error:
        # 归一化失败：answer 置空 + 记入失败清单，answer_raw 保留
        return None, {"reason": normalize_error}
    if data_anomaly:
        # 源数据矛盾但已宽容归一化：入库成功 + 记入异常清单（parse_failed_entities 中的 anomaly 类型）
        logger.warning(f"源数据异常（已宽容处理）：{data_anomaly}")

    record = {
        "question_code": None,  # 由调用方回填
        "bank_code": bank_code,
        "bank_name": bank_name,
        "question_type": question_type_en,
        "question_type_cn": question_type_cn,
        "group_id": None,  # V1 每题独立存全文，group_id 置空（PRD 3.2.1，复审 N8）
        "content": content,
        "stem": stem,
        "options": options if options else None,
        "answer": answer,
        "answer_raw": answer_raw,
        "analysis": (fields.get("analysis") or "").strip(),
        "source_file": source_file,
        "source_path": source_path,
        "data_anomaly": data_anomaly,  # 源数据异常说明（已宽容处理，需人工核对），无异常为 None
    }
    return record, None


@node_log("node_question_parser")
def node_question_parser(state: dict) -> dict:
    """
    题库解析节点
    读取归一化后的 Markdown 全文，解析出题目记录（含答案归一化）
    单实体失败（缺题型/题干/答案归一化失败）仅跳过该实体并记录失败清单
    :param state: 导入图状态
    :return: 状态更新（question_records / bank_name_map / parse_failed_entities）
    """
    # 1. 校验并获取归一化 MD 与溯源信息
    data = step_1_validate_and_get_data(state)
    if data is None:
        return {"parse_failed_entities": [{"reason": "归一化后 Markdown 为空"}]}
    md_text, source_file, source_path = data

    # 2. 按「## 题库 / ### 题目」边界切分题目块
    blocks, bank_name_map = step_2_split_question_blocks(md_text)

    # 3. 逐题目块解析 + 答案归一化
    question_records, failed = step_3_parse_question_records(blocks, source_file, source_path)

    logger.info(
        f"题库解析完成：题库 {len(bank_name_map)} 个，题目 {len(question_records)} 条，失败 {len(failed)} 条"
    )
    return {
        "question_records": question_records,
        "bank_name_map": bank_name_map,
        "parse_failed_entities": failed,
    }


@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state: dict):
    """
    校验并获取归一化 MD 与溯源信息
    :return: (md_text, source_file, source_path) 或 None（MD 为空）
    """
    md_text = state.get("md_text", "")
    source_file = state.get("source_file", "")
    source_path = state.get("source_path", "")
    if not md_text:
        return None
    return md_text, source_file, source_path


@step_log("step_2_split_question_blocks")
def step_2_split_question_blocks(md_text: str):
    """
    按题库/题目边界把 Markdown 切分为题目块
    :return: (题目块列表, bank_code -> bank_name 映射)
             题目块结构：{"question_code", "bank_code", "bank_name", "lines"}
    """
    blocks = []
    bank_name_map = {}

    current_bank_name = None
    current_bank_code = None
    current_question_code = None
    question_block = []

    def _flush_question():
        """结束当前题目块：把已收集的行归档为一块"""
        nonlocal current_question_code, question_block
        if current_question_code is None:
            return
        blocks.append({
            "question_code": current_question_code,
            "bank_code": current_bank_code,
            "bank_name": current_bank_name,
            "lines": question_block,
        })
        current_question_code = None
        question_block = []

    for line in md_text.splitlines():
        # 题库边界：## {题库名称}
        m = _BANK_HEADER_RE.match(line)
        if m:
            _flush_question()
            current_bank_name = m.group("name").strip()
            current_bank_code = None
            continue
        # 题库编码行
        m = _BANK_CODE_RE.match(line)
        if m and current_bank_name is not None:
            current_bank_code = m.group("value").strip()
            bank_name_map[current_bank_code] = current_bank_name
            continue
        # 题目边界：### {question_code}
        m = _QUESTION_HEADER_RE.match(line)
        if m and current_bank_code is not None:
            _flush_question()
            current_question_code = m.group("code")
            question_block = []
            continue
        # 题目块内的行
        if current_question_code is not None:
            question_block.append(line)

    # 收尾：最后一个题目
    _flush_question()

    logger.info(f"题目块切分完成：题库 {len(bank_name_map)} 个，题目块 {len(blocks)} 个")
    return blocks, bank_name_map


@step_log("step_3_parse_question_records")
def step_3_parse_question_records(blocks: list, source_file: str, source_path: str):
    """
    逐题目块解析字段并做答案归一化，组装题目记录
    :return: (question_records, failed)
    """
    question_records = []
    failed = []

    for block in blocks:
        question_code = block["question_code"]
        bank_code = block["bank_code"]
        bank_name = block["bank_name"]

        record, failure = _parse_question_block(
            block["lines"], bank_code, bank_name, source_file, source_path
        )
        if record is None:
            failed.append({
                "entity_type": "question",
                "bank_code": bank_code,
                "question_code": question_code,
                **(failure or {}),
            })
            logger.warning(f"题目[{question_code}]解析失败：{failure}")
            continue

        record["question_code"] = question_code
        question_records.append(record)
        # 源数据异常（已宽容处理）单独记入异常清单，不混入失败清单
        if record.get("data_anomaly"):
            failed.append({
                "entity_type": "question_anomaly",
                "bank_code": bank_code,
                "question_code": question_code,
                "reason": record["data_anomaly"],
            })

    return question_records, failed


if __name__ == "__main__":
    # 独立验证：直接解析实测数据
    from collections import Counter
    data_path = Path("/Users/zdy/Downloads/教育/数据/题目资料.md")
    md_text = data_path.read_text(encoding="utf-8")
    result = node_question_parser({"md_text": md_text, "source_file": data_path.name, "source_path": str(data_path)})
    questions = result["question_records"]
    banks = result["bank_name_map"]
    failed = result["parse_failed_entities"]
    print(f"题库数: {len(banks)}  (期望 73)")
    print(f"题目数: {len(questions)}  (期望 1752)")
    hard_failures = [f_ for f_ in failed if f_.get("entity_type") == "question"]
    anomalies = [f_ for f_ in failed if f_.get("entity_type") == "question_anomaly"]
    print(f"硬失败数（未入库）: {len(hard_failures)}")
    print(f"数据异常数（已宽容入库）: {len(anomalies)}")
    for a in anomalies:
        print("  异常样例:", a)

    # 题型分布验证（对照 PRD 4.2 配置表实测数量）
    type_dist = Counter(q["question_type"] for q in questions)
    print("\n题型分布:")
    for t, c in sorted(type_dist.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")

    # 判断题答案归一化验证（期望 true 145 / false 90）
    tf_dist = Counter(q["answer"] for q in questions if q["question_type"] == "true_false")
    print(f"\n判断题答案分布（期望 true=145 / false=90）: {dict(tf_dist)}")

    # 多选题答案分隔符验证（应全部为半角逗号 + 字母序）
    multi_answers = [q["answer"] for q in questions if q["question_type"] == "multiple_choice"]
    bad_multi = [a for a in multi_answers if ("、" in a or "，" in a or a != ",".join(sorted(a.split(","))))]
    print(f"多选题答案归一化异常数（期望 0）: {len(bad_multi)}")

    # 抽样验证
    print("\n首条题目:")
    first = questions[0]
    print({k: v for k, v in first.items() if k not in ("content", "analysis")})
    print("content 预览:", first["content"][:120].replace("\n", "\\n"))
