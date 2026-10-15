"""
Markdown 图片处理节点
参考 shopkeeper-brain 参考项目 node_md_img 实现，适配教育知识库导入图：
1. 扫描归一化 MD 引用的本地图片（MinerU 结果包 images/ 目录）
2. VL 多模态模型生成图片摘要（图片语义 → 文本，提升检索能力）
3. 图片上传 MinIO（公开读桶，前端可访问 URL）
4. MD 中的图片引用替换为：![VL摘要](MinIO URL) —— 摘要参与向量化，URL 供前端展示
5. 新 MD 备份落盘（xxx_new.md）
"""
import base64
import re
from mimetypes import guess_type
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from minio.deleteobjects import DeleteObject

from common.config.lm_config import lm_config
from common.config.minio_config import minio_config
from common.logging.logger import logger, node_log, step_log
from utils.clients.minio_utils import get_minio_client
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt
from utils.rate_limit_utils import apply_api_rate_limit

# MinIO 支持的图片格式（与参考项目一致）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def is_supported_image(filename: str) -> bool:
    """判断文件是否为支持的图片格式（后缀不区分大小写）"""
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS


@node_log("node_md_img")
def node_md_img(state: dict) -> dict:
    """
    图片处理节点：扫描 → VL 识别 → MinIO 上传 → MD 替换 → 备份
    无图片（images 目录不存在或为空）时直接跳过，不阻塞主链路
    :param state: 导入图状态（需含 normalized_md_path 或 local_dir/images）
    :return: 状态更新（md_text 为替换后的新 MD）
    """
    md_path = state.get("normalized_md_path")
    md_content = state.get("md_text", "")
    local_dir = state.get("local_dir", "")

    # 无 MD 路径（docx 直转内存 MD 无图片）或无内容：跳过
    if not md_path or not md_content:
        logger.info("无 MD 文件路径（非 PDF 链路），跳过图片处理")
        return {}

    md_path_obj = Path(md_path)
    if not md_path_obj.is_file():
        logger.warning(f"MD 文件不存在，跳过图片处理：{md_path}")
        return {}

    # 图片目录：MD 同级 images/
    md_images_dir_obj = md_path_obj.parent / "images"
    if (not md_images_dir_obj.exists()) or not any(md_images_dir_obj.iterdir()):
        logger.info(f"{md_path_obj.name} 无图片内容，跳过图片处理")
        return {}

    # 1. 扫描图片及上下文 [(name, path, (上文, 下文))]
    image_info_list = step_1_scan_images(md_content, md_images_dir_obj)
    if not image_info_list:
        logger.info("MD 中无被引用的本地图片，跳过图片处理")
        return {}

    # 2. VL 模型生成图片摘要 {图片名: 摘要}
    summary_image_dict = step_2_image_summary(image_info_list, md_path_obj.stem)

    # 3. 图片上传 MinIO {图片名: URL}
    image_url_dict = step_3_upload_images_get_url(image_info_list, md_path_obj.stem)

    # 4. MD 图片引用替换（![摘要](MinIO URL)）
    md_content_new = step_4_md_content_image_replace(md_content, summary_image_dict, image_url_dict)

    # 5. 新 MD 备份落盘
    step_5_backup_new_md(md_content_new, md_path_obj)

    return {"md_text": md_content_new}


@step_log("step_1_scan_images")
def step_1_scan_images(md_content: str, md_images_dir_obj: Path) -> list:
    """
    扫描本地图片目录，匹配 MD 中的引用并提取上下文（与参考项目同构）
    :return: [(图片名, 图片Path, (上文, 下文))]
    """
    image_info_list = []
    for md_file_obj in md_images_dir_obj.iterdir():
        image_name = md_file_obj.name
        if not is_supported_image(image_name):
            logger.warning(f"跳过非图片文件：{image_name}")
            continue
        rep = re.compile(r"!\[.*?]\(.*?" + re.escape(image_name) + r".*?\)")
        search_match = rep.search(md_content)
        if not search_match:
            logger.warning(f"{image_name} 未被 MD 引用，跳过")
            continue
        start, end = search_match.start(), search_match.end()
        pre_content = md_content[max(start - 100, 0):start]
        post_content = md_content[end: min(end + 100, len(md_content))]
        image_info_list.append((image_name, md_file_obj, (pre_content, post_content)))

    logger.info(f"图片上下文识别完成，数量：{len(image_info_list)}")
    return image_info_list


@step_log("step_2_image_summary")
def step_2_image_summary(image_info_list: list, root_folder: str) -> dict:
    """
    VL 多模态模型生成图片摘要（base64 内联图片 + 上下文提示）
    :return: {图片名: 摘要文本}
    """
    summary_image_dict = {}
    vl_model = get_llm_client(lm_config.vl_model)

    for image_name, image_path_obj, image_content in image_info_list:
        prompt_text = load_prompt(
            name="image_summary",
            root_folder=root_folder,
            image_content=image_content,
        )
        image_data = base64.b64encode(image_path_obj.read_bytes()).decode("utf-8")
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{guess_type(image_name)[0]};base64,{image_data}"},
                },
            ]
        )
        chains = vl_model | StrOutputParser()
        apply_api_rate_limit()
        image_summary = chains.invoke([message])
        summary_image_dict[image_name] = image_summary.strip()
        logger.info(f"视觉识别完成：{image_name} → {image_summary[:50]}")
    return summary_image_dict


@step_log("step_3_upload_images_get_url")
def step_3_upload_images_get_url(image_info_list: list, stem: str) -> dict:
    """
    图片上传 MinIO（先清理同前缀旧对象实现幂等重导，再 fput 上传）
    :return: {图片名: 公开访问 URL}
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
    for image_name, image_path_obj, _ in image_info_list:
        minio_client.fput_object(
            bucket_name=minio_config.bucket_name,
            object_name=f"{minio_config.minio_img_dir}/{stem}/{image_name}",
            file_path=str(image_path_obj),
            content_type=guess_type(image_name)[0],
        )
        url = (f"http://{minio_config.endpoint}/{minio_config.bucket_name}"
               f"{minio_config.minio_img_dir}/{stem}/{image_name}")
        image_url_dict[image_name] = url
        logger.info(f"{image_name} 已上传 MinIO：{url}")
    return image_url_dict


@step_log("step_4_md_content_image_replace")
def step_4_md_content_image_replace(md_content: str, summary_image_dict: dict, image_url_dict: dict) -> str:
    """
    MD 图片引用替换：![原图引用] → ![VL 摘要](MinIO URL)
    摘要作为图片语义文本参与后续向量化（检索价值），URL 供前端展示
    """
    for image_name, image_summary in summary_image_dict.items():
        image_url = image_url_dict.get(image_name)
        if not image_url:
            continue
        rep = re.compile(r"!\[.*?]\(.*?" + re.escape(image_name) + r".*?\)")
        md_content = rep.sub(lambda _: f"![{image_summary}]({image_url})", md_content)
    return md_content


@step_log("step_5_backup_new_md")
def step_5_backup_new_md(md_content_new: str, md_path_obj: Path):
    """新 MD 备份落盘（原文件名_new.md，便于审计对照）"""
    md_path_obj_new = md_path_obj.with_name(f"{md_path_obj.stem}_new.md")
    md_path_obj_new.write_text(md_content_new, encoding="utf-8")
    logger.info(f"新 MD 已备份：{md_path_obj_new}")
