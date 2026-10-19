"""
课程目录 Parser 节点
对应 PRD V1.3 3.2「课程介绍解析规则」：
- 系列边界：## {系列名称}
- 系列字段锚点：- **系列编码**: / - **描述**: / - **课程分类**: / - **适合人群**: / - **学习目标**: / - **适合年级**:
- 课程模块：系列下「### 课程」小节内的 - **{课程名}**，子字段：编码/课时/学时/描述
产出：course_series_records（PRD 4.3 Collection 1）+ course_module_records（Collection 2）
"""
import re
from pathlib import Path

from common.logging.logger import logger, node_log, step_log


# 系列标题行：## 通用编程入门班
_SERIES_HEADER_RE = re.compile(r"^##\s+(?P<name>.+?)\s*$")
# 系列字段锚点：- **系列编码**: xxx（原生 MD）/ - 系列编码: xxx（docx 归一化产物，无 ** 加粗）
_SERIES_FIELD_RES = {
    "series_code": re.compile(r"^-\s*(?:\*\*)?系列编码(?:\*\*)?\s*[:：]\s*(?P<value>.+?)\s*$"),
    "description": re.compile(r"^-\s*(?:\*\*)?描述(?:\*\*)?\s*[:：]\s*(?P<value>.+?)\s*$"),
    "course_category": re.compile(r"^-\s*(?:\*\*)?课程分类(?:\*\*)?\s*[:：]\s*(?P<value>.+?)\s*$"),
    "target_audience": re.compile(r"^-\s*(?:\*\*)?适合人群(?:\*\*)?\s*[:：]\s*(?P<value>.+?)\s*$"),
    "learning_goal": re.compile(r"^-\s*(?:\*\*)?学习目标(?:\*\*)?\s*[:：]\s*(?P<value>.+?)\s*$"),
    "suitable_grade": re.compile(r"^-\s*(?:\*\*)?适合年级(?:\*\*)?\s*[:：]\s*(?P<value>.+?)\s*$"),
}
# 「### 课程」小节标题（模块列表开始）
_COURSE_SECTION_RE = re.compile(r"^###\s*课程\s*$")
# 课程模块条目：- **语法基础与开发环境**（原生 MD）/ - 语法基础与开发环境（docx 归一化产物）
_MODULE_ITEM_RE = re.compile(r"^-\s*(?:\*\*)?(?P<name>.+?)(?:\*\*)?\s*$")
# 模块子字段（缩进行）：编码: xxx, 课时: 8, 学时: 16.00, 描述: xxx
_MODULE_CODE_RE = re.compile(r"编码\s*[:：]\s*(?P<value>.+?)\s*[,，]")
_MODULE_DURATION_RE = re.compile(r"课时\s*[:：]\s*(?P<value>\d+)")
_MODULE_CREDIT_RE = re.compile(r"学时\s*[:：]\s*(?P<value>[\d.]+)")
_MODULE_DESC_RE = re.compile(r"描述\s*[:：]\s*(?P<value>.+?)\s*$")


@node_log("node_course_parser")
def node_course_parser(state: dict) -> dict:
    """
    课程目录解析节点
    读取归一化后的 Markdown 全文，解析出系列与课程模块两类记录
    单实体解析失败（如缺少系列编码）仅跳过该实体并记录失败清单
    :param state: 导入图状态
    :return: 状态更新（course_series_records / course_module_records / parse_failed_entities）
    """
    # 1. 校验并获取归一化 MD 与溯源信息
    data = step_1_validate_and_get_data(state)
    if data is None:
        return {"parse_failed_entities": [{"reason": "归一化后 Markdown 为空"}]}
    md_text, source_file, source_path = data

    # 2. 按「## 系列标题」切分系列块
    blocks = step_2_split_series_blocks(md_text)

    # 3. 逐块解析系列字段（业务主键缺失则记失败清单）
    series_pairs, failed = step_3_parse_series_records(blocks, source_file, source_path)

    # 4. 逐系列解析「### 课程」小节内的课程模块
    module_records, module_failed = step_4_parse_module_records(series_pairs)
    failed.extend(module_failed)

    series_records = [record for record, _ in series_pairs]
    logger.info(
        f"课程目录解析完成：系列 {len(series_records)} 条，课程模块 {len(module_records)} 条，失败 {len(failed)} 条"
    )
    return {
        "course_series_records": series_records,
        "course_module_records": module_records,
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


@step_log("step_2_split_series_blocks")
def step_2_split_series_blocks(md_text: str):
    """
    将 Markdown 全文按「## 系列标题」切分为系列块
    :param md_text: 归一化后的 Markdown 全文
    :return: 系列块列表 [(series_name, block_lines), ...]
    """
    lines = md_text.splitlines()
    blocks = []
    current_name = None
    current_lines = []
    for line in lines:
        m = _SERIES_HEADER_RE.match(line)
        if m:
            # 保存上一个系列块
            if current_name is not None:
                blocks.append((current_name, current_lines))
            current_name = m.group("name").strip()
            current_lines = []
        elif current_name is not None:
            current_lines.append(line)
    # 收尾：最后一个系列块
    if current_name is not None:
        blocks.append((current_name, current_lines))
    logger.info(f"系列块切分完成：{len(blocks)} 个系列")
    return blocks


@step_log("step_3_parse_series_records")
def step_3_parse_series_records(blocks: list, source_file: str, source_path: str):
    """
    逐系列块解析系列字段并组装系列记录
    :return: ([(series_record, block_lines), ...], failed)
    """
    series_pairs = []
    failed = []

    for series_name, block_lines in blocks:
        # ---- 第一遍：提取系列字段 ----
        series_fields = {}
        in_course_section = False
        for line in block_lines:
            # 命中「### 课程」小节，后续行属于模块列表
            if _COURSE_SECTION_RE.match(line):
                in_course_section = True
                continue
            # 模块列表内的行不再匹配系列字段
            if in_course_section:
                continue
            for field_name, field_re in _SERIES_FIELD_RES.items():
                m = field_re.match(line)
                if m:
                    series_fields[field_name] = m.group("value").strip()

        # 系列编码为业务主键，缺失则该实体失败（不影响其他系列）
        series_code = series_fields.get("series_code")
        if not series_code:
            failed.append({
                "entity_type": "course_series",
                "series_name": series_name,
                "reason": "缺少系列编码（业务主键）",
            })
            logger.warning(f"系列[{series_name}]缺少系列编码，跳过该实体")
            continue

        series_record = {
            "series_code": series_code,
            "series_name": series_name,
            "description": series_fields.get("description", ""),
            "course_category": series_fields.get("course_category", ""),
            "target_audience": series_fields.get("target_audience", ""),
            "learning_goal": series_fields.get("learning_goal", ""),
            "suitable_grade": series_fields.get("suitable_grade", ""),
            "source_file": source_file,
            "source_path": source_path,
        }
        series_record["content"] = _build_series_content(series_record)
        series_pairs.append((series_record, block_lines))

    logger.info(f"系列解析完成：成功 {len(series_pairs)} 条，失败 {len(failed)} 条")
    return series_pairs, failed


@step_log("step_4_parse_module_records")
def step_4_parse_module_records(series_pairs: list):
    """
    逐系列解析「### 课程」小节内的课程模块列表
    :param series_pairs: [(series_record, block_lines), ...]
    :return: (module_records, failed)
    """
    module_records = []
    failed = []

    for series_record, block_lines in series_pairs:
        in_course_section = False
        current_module = None
        for line in block_lines:
            if _COURSE_SECTION_RE.match(line):
                in_course_section = True
                continue
            if not in_course_section:
                continue
            # 模块条目行：- 课程名（排除含冒号的字段行，如 - 编码: xxx）
            m = _MODULE_ITEM_RE.match(line)
            if m and ":" not in line and "：" not in line:
                # 保存上一个模块
                if current_module is not None:
                    _finish_module(current_module, series_record, module_records, failed)
                current_module = {"course_name": m.group("name").strip()}
                continue
            # 模块子字段行（缩进行或列表项）：编码: xxx, 课时: 8, 学时: 16.00 / 描述: xxx
            # 注意：docx 归一化产物中子字段可能是同级列表项（- 编码: xxx），strip 后统一匹配
            if current_module is not None:
                stripped = line.strip()
                if not stripped:
                    continue
                stripped = stripped.lstrip("-•· ").strip()  # 兼容 docx 列表项前缀
                m = _MODULE_CODE_RE.search(stripped)
                if m and "course_code" not in current_module:
                    current_module["course_code"] = m.group("value").strip()
                m = _MODULE_DURATION_RE.search(stripped)
                if m and "duration" not in current_module:
                    current_module["duration"] = int(m.group("value"))
                m = _MODULE_CREDIT_RE.search(stripped)
                if m and "credit_hours" not in current_module:
                    current_module["credit_hours"] = float(m.group("value"))
                m = _MODULE_DESC_RE.search(stripped)
                if m and "description" not in current_module:
                    current_module["description"] = m.group("value").strip()
        # 收尾：最后一个模块
        if current_module is not None:
            _finish_module(current_module, series_record, module_records, failed)

    logger.info(f"课程模块解析完成：成功 {len(module_records)} 条，失败 {len(failed)} 条")
    return module_records, failed


def _build_series_content(record: dict) -> str:
    """
    拼接系列全文（用于语义召回与展示，PRD 4.3 Collection 1 content 字段）
    """
    parts = [
        f"课程系列：{record['series_name']}",
        f"系列编码：{record['series_code']}",
    ]
    if record.get("description"):
        parts.append(f"描述：{record['description']}")
    if record.get("course_category"):
        parts.append(f"课程分类：{record['course_category']}")
    if record.get("target_audience"):
        parts.append(f"适合人群：{record['target_audience']}")
    if record.get("learning_goal"):
        parts.append(f"学习目标：{record['learning_goal']}")
    if record.get("suitable_grade"):
        parts.append(f"适合年级：{record['suitable_grade']}")
    return "\n".join(parts)


def _build_module_content(record: dict) -> str:
    """
    拼接课程模块全文（PRD 4.3 Collection 2 content 字段）
    """
    parts = [
        f"课程模块：{record['course_name']}",
        f"编码：{record['course_code']}",
    ]
    if record.get("series_name"):
        parts.append(f"所属系列：{record['series_name']}")
    if record.get("duration") is not None:
        parts.append(f"课时：{record['duration']}")
    if record.get("credit_hours") is not None:
        parts.append(f"学时：{record['credit_hours']}")
    if record.get("description"):
        parts.append(f"描述：{record['description']}")
    return "\n".join(parts)


def _finish_module(module: dict, series_record: dict, module_records: list, failed: list):
    """
    结束一个课程模块的解析，校验业务主键后组装完整记录
    :param module: 待完成的模块字段字典
    :param series_record: 所属系列记录（用于回填 series_code/series_name）
    :param module_records: 模块记录输出列表
    :param failed: 失败实体清单
    """
    course_code = module.get("course_code")
    if not course_code:
        failed.append({
            "entity_type": "course_module",
            "series_code": series_record["series_code"],
            "course_name": module.get("course_name", ""),
            "reason": "缺少课程编码（业务主键）",
        })
        logger.warning(f"模块[{module.get('course_name')}]缺少编码，跳过该实体")
        return

    record = {
        "course_code": course_code,
        "course_name": module.get("course_name", ""),
        "series_code": series_record["series_code"],
        "series_name": series_record["series_name"],
        "duration": module.get("duration"),
        "credit_hours": module.get("credit_hours"),
        "description": module.get("description", ""),
        "source_file": series_record["source_file"],
        "source_path": series_record["source_path"],
    }
    record["content"] = _build_module_content(record)
    module_records.append(record)


if __name__ == "__main__":
    # 独立验证：直接解析实测数据
    data_path = Path("/Users/zdy/Downloads/教育/数据/课程介绍.md")
    md_text = data_path.read_text(encoding="utf-8")
    result = node_course_parser({"md_text": md_text, "source_file": data_path.name, "source_path": str(data_path)})
    series = result["course_series_records"]
    modules = result["course_module_records"]
    print(f"系列数: {len(series)}  (期望 219)")
    print(f"模块数: {len(modules)}  (期望 657)")
    print(f"失败数: {len(result['parse_failed_entities'])}")
    # 抽样验证
    print("\n首条系列:", {k: v for k, v in series[0].items() if k != "content"})
    print("\n首条模块:", {k: v for k, v in modules[0].items() if k != "content"})
