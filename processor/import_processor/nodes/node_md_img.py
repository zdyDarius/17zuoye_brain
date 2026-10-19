"""
Markdown 图片处理节点
适配教育知识库导入图（参考 shopkeeper-brain node_md_img），支持 zip 资料包内的多文档 + 本地图片：
1. 扫描 MD 引用的本地图片：Markdown `![](...)` 与 HTML `<img src="...">` 双语法，
   相对路径按「MD 所在目录 → 资料包根目录 → 同名文件兜底」定位
2. VL 多模态模型生成图片摘要（图片语义 → 文本，提升检索能力）
3. 图片上传 MinIO（公开读桶，前端可访问 URL）
4. MD 中的图片引用替换为：![VL摘要](MinIO URL) —— 摘要参与向量化，URL 供前端展示
5. 新 MD 备份落盘（xxx_new.md）
"""
import base64
from mimetypes import guess_type
from pathlib import Path
from urllib.parse import quote

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from minio.deleteobjects import DeleteObject

from common.config.lm_config import lm_config
from common.config.minio_config import minio_config
from common.logging.logger import logger, node_log, step_log
from utils.clients.minio_utils import get_minio_client
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt
from utils.md_image_utils import (
    IMAGE_EXTENSIONS, ResolvedImage, build_image_key,
    extract_image_refs, is_remote_src, render_image_markdown, resolve_image_path,
)
from utils.rate_limit_utils import apply_api_rate_limit

# 图片上下文字符数（上文/下文各取 100 字符）
_CONTEXT_WINDOW = 100


def is_supported_image(filename: str) -> bool:
    """判断文件是否为支持的图片格式（后缀不区分大小写）"""
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS


@node_log("node_md_img")
def node_md_img(state: dict) -> dict:
    """
    图片处理节点：扫描 → VL 识别 → MinIO 上传 → MD 替换 → 备份
    MD 中没有可定位的本地图片时直接跳过，不阻塞主链路
    :param state: 导入图状态（normalized_md_path / md_text / local_dir）
    :return: 状态更新（md_text 为替换后的新 MD）
    """
    # 1. 校验并获取 MD 内容与定位根目录（非 MD 链路直接跳过）
    data = step_1_validate_and_get_data(state)
    if data is None:
        return {}
    md_path_obj, md_content, search_root = data

    # 2. 扫描图片引用并定位本地文件
    image_list = step_2_scan_images(md_content, md_path_obj, search_root)
    if not image_list:
        logger.info(f"{md_path_obj.name} 无可定位的本地图片，跳过图片处理")
        return {}

    # 3. VL 模型生成图片摘要 {图片键: 摘要}
    summary_image_dict = step_3_image_summary(image_list, md_path_obj.stem)

    # 4. 图片上传 MinIO {图片键: URL}
    image_url_dict = step_4_upload_images_get_url(image_list, md_path_obj.stem)

    # 5. MD 图片引用替换（![摘要](MinIO URL)）
    md_content_new = step_5_md_content_image_replace(
        md_content, image_list, summary_image_dict, image_url_dict,
    )

    # 6. 新 MD 备份落盘
    step_6_backup_new_md(md_content_new, md_path_obj)

    return {"md_text": md_content_new}


@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state: dict):
    """
    校验并获取 MD 路径/内容 + 图片定位根目录
    :return: (md_path_obj, md_content, search_root) 或 None（无需处理图片）
    """
    md_path = state.get("normalized_md_path")
    md_content = state.get("md_text", "")
    # 资料包根目录：图片集中放在包根 images/、而 MD 在子目录时靠它兜底定位
    search_root = state.get("local_dir") or ""

    # 无 MD 路径（docx 直转内存 MD 无图片）或无内容：跳过
    if not md_path or not md_content:
        logger.info("无 MD 文件路径（非 MD 链路），跳过图片处理")
        return None

    md_path_obj = Path(md_path)
    if not md_path_obj.is_file():
        logger.warning(f"MD 文件不存在，跳过图片处理：{md_path}")
        return None

    return md_path_obj, md_content, search_root


@step_log("step_2_scan_images")
def step_2_scan_images(md_content: str, md_path_obj: Path,
                       search_root: str | Path = "") -> list[ResolvedImage]:
    """
    扫描 MD 中的图片引用并解析为本地文件（同一图片被多次引用时合并，保留每处引用位置）
    :param md_content: MD 原文
    :param md_path_obj: MD 文件路径
    :param search_root: 资料包根目录（可为空）
    :return: ResolvedImage 列表
    """
    md_dir = md_path_obj.parent
    refs = extract_image_refs(md_content)
    if not refs:
        logger.info(f"{md_path_obj.name} 未发现图片引用")
        return []

    resolved: dict[str, ResolvedImage] = {}
    missing: list[str] = []
    remote_count = 0

    for ref in refs:
        if is_remote_src(ref.src):
            remote_count += 1
            continue
        image_path = resolve_image_path(ref.src, md_dir, search_root or None)
        if image_path is None:
            missing.append(ref.src)
            continue

        key = build_image_key(image_path, md_dir, search_root or None)
        item = resolved.get(key)
        if item is None:
            item = ResolvedImage(key=key, path=image_path)
            resolved[key] = item
        item.refs.append(ref)
        if item.context is None:
            item.context = (
                md_content[max(ref.start - _CONTEXT_WINDOW, 0):ref.start],
                md_content[ref.end:min(ref.end + _CONTEXT_WINDOW, len(md_content))],
            )

    if missing:
        logger.warning(f"{md_path_obj.name} 有 {len(missing)} 处图片引用未找到本地文件，"
                       f"示例：{missing[:5]}")
    logger.info(f"{md_path_obj.name} 图片识别完成：引用 {len(refs)} 处 → 本地图片 {len(resolved)} 张"
                f"（远程图片跳过 {remote_count} 处）")
    return list(resolved.values())


@step_log("step_3_image_summary")
def step_3_image_summary(image_list: list[ResolvedImage], root_folder: str) -> dict:
    """
    VL 多模态模型生成图片摘要（base64 内联图片 + 上下文提示）
    :return: {图片键: 摘要文本}
    """
    summary_image_dict = {}
    vl_model = get_llm_client(lm_config.vl_model)

    for item in image_list:
        prompt_text = load_prompt(
            name="image_summary",
            root_folder=root_folder,
            image_content=item.context or ("", ""),
        )
        mime_type = guess_type(item.path.name)[0] or "image/jpeg"
        image_data = base64.b64encode(item.path.read_bytes()).decode("utf-8")
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                },
            ]
        )
        chains = vl_model | StrOutputParser()
        apply_api_rate_limit()
        image_summary = chains.invoke([message])
        summary_image_dict[item.key] = image_summary.strip()
        logger.info(f"视觉识别完成：{item.key} → {image_summary[:50]}")
    return summary_image_dict


@step_log("step_4_upload_images_get_url")
def step_4_upload_images_get_url(image_list: list[ResolvedImage], stem: str) -> dict:
    """
    图片上传 MinIO（先清理同前缀旧对象实现幂等重导，再 fput 上传）
    :return: {图片键: 公开访问 URL}
    """
    minio_client = get_minio_client()

    # 幂等：清理同前缀旧图片（重导入场景避免 MinIO 垃圾累积）
    old_objects = minio_client.list_objects(
        bucket_name=minio_config.bucket_name,
        prefix=f"{minio_config.minio_img_dir.lstrip('/')}/{stem}",
        recursive=True,
    )
    errors = minio_client.remove_objects(
        bucket_name=minio_config.bucket_name,
        delete_object_list=[DeleteObject(obj.object_name) for obj in old_objects],
    )
    for error in errors:
        logger.warning(f"删除旧图片出现问题：{error}")

    image_url_dict = {}
    for item in image_list:
        # 子目录层级压平为 __，避免不同目录同名图片互相覆盖
        object_name = f"{minio_config.minio_img_dir}/{stem}/{item.key.replace('/', '__')}"
        minio_client.fput_object(
            bucket_name=minio_config.bucket_name,
            object_name=object_name,
            file_path=str(item.path),
            content_type=guess_type(item.path.name)[0] or "image/jpeg",
        )
        url = (f"http://{minio_config.endpoint}/{minio_config.bucket_name}"
               f"/{quote(object_name)}")
        image_url_dict[item.key] = url
        logger.info(f"{item.key} 已上传 MinIO：{url}")
    return image_url_dict


@step_log("step_5_md_content_image_replace")
def step_5_md_content_image_replace(md_content: str, image_list: list[ResolvedImage],
                                    summary_image_dict: dict, image_url_dict: dict) -> str:
    """
    MD 图片引用替换：![原图引用] / <img src=...> → ![VL 摘要](MinIO URL)
    按原文偏移倒序替换，保证未处理段落的位置不被扰动
    """
    replacements = []
    for item in image_list:
        image_url = image_url_dict.get(item.key)
        image_summary = summary_image_dict.get(item.key)
        if not image_url or not image_summary:
            continue
        new_text = render_image_markdown(image_summary, image_url)
        for ref in item.refs:
            replacements.append((ref.start, ref.end, new_text))

    for start, end, new_text in sorted(replacements, reverse=True):
        md_content = md_content[:start] + new_text + md_content[end:]
    logger.info(f"MD 图片引用替换完成：{len(replacements)} 处")
    return md_content


@step_log("step_6_backup_new_md")
def step_6_backup_new_md(md_content_new: str, md_path_obj: Path):
    """新 MD 备份落盘（原文件名_new.md，便于审计对照；zip 导入时不会被二次扫描）"""
    md_path_obj_new = md_path_obj.with_name(f"{md_path_obj.stem}_new.md")
    md_path_obj_new.write_text(md_content_new, encoding="utf-8")
    logger.info(f"新 MD 已备份：{md_path_obj_new}")
