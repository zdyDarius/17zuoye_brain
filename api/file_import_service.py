"""
导入服务（FastAPI :8000）
对应 PRD V1.3 3.2 / 5.4：
- GET /import.html：导入操作页面
- POST /upload：多文件上传 + content_type 元数据，后台执行导入图
  · 支持 .md/.pdf/.docx 单文件，以及 .zip 资料包（包内多个 md + images/ 图片，
    解压后每个 md 独立生成一个导入任务）
- GET /status/{task_id}：任务进度查询（全局状态 + 已完成/进行中节点 + 失败实体数）
"""
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse

from common.logging.logger import logger
from processor.import_processor.main_graph import kb_import_app
from processor.import_processor.nodes.node_entry import VALID_EXTENSIONS, MAX_FILE_SIZE
from processor.import_processor.state import create_import_default_state
from utils.path_util import PROJECT_ROOT
from utils.task_utils import (
    get_running_task_list, get_done_task_list, get_task_status,
    add_running_task, add_done_task, update_task_status, set_task_result,
    clear_task,
    TASK_STATUS_PROCESSING, TASK_STATUS_FAILED, TASK_STATUS_COMPLETED,
)
from utils.zip_utils import (
    ZipExtractError, MAX_MD_PER_ZIP, discover_md_files, extract_zip,
)

app = FastAPI(title="education-brain import service")

# 上传白名单 = 导入图支持的单文件格式 + zip 资料包
UPLOAD_EXTENSIONS = VALID_EXTENSIONS | {".zip"}
# zip 资料包体积上限（内含图片，单独放宽）
MAX_ZIP_SIZE = 500 * 1024 * 1024  # 500MB

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/import.html")
def import_page():
    """导入操作页面"""
    page_path = PROJECT_ROOT / "page" / "import.html"
    if not page_path.exists():
        raise HTTPException(status_code=404, detail=f"页面不存在：{page_path}")
    return FileResponse(page_path, media_type="text/html")


# ---- 导入图后台执行（stream 模式，节点级进度）----
def _run_graph_with_progress(task_id: str, local_dir: str, local_file_path: str,
                             content_type: str, project_name: str = "", source_file: str = ""):
    """
    后台任务：执行导入图全流程并更新任务状态
    :param local_dir: 任务工作目录（zip 导入时为解压根目录，供图片兜底定位）
    :param source_file: 来源文件名（zip 导入时为包内相对路径）
    """
    try:
        clear_task(task_id)
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        logger.info(f"[{task_id}] 开始导入流程：{source_file or local_file_path}（{content_type}）")

        init_state = create_import_default_state(
            task_id=task_id, local_dir=local_dir,
            local_file_path=local_file_path, content_type=content_type,
            project_name=project_name, source_file=source_file,
        )

        final_state = init_state
        for event in kb_import_app.stream(init_state, stream_mode="updates"):
            for node_name, node_result in event.items():
                add_done_task(task_id, node_name)
                logger.info(f"[{task_id}] 节点完成：{node_name}")
                if isinstance(node_result, dict):
                    final_state.update(node_result)

        error = (final_state.get("format_convert_error") or final_state.get("embedding_error")
                 or final_state.get("import_error"))
        failed_entities = final_state.get("parse_failed_entities") or []
        import_result = final_state.get("import_result") or {}

        if error:
            update_task_status(task_id, TASK_STATUS_FAILED)
            set_task_result(task_id, "error", str(error))
        else:
            update_task_status(task_id, TASK_STATUS_COMPLETED)
        set_task_result(task_id, "import_result", str(import_result))
        set_task_result(task_id, "failed_entities_count", str(len(failed_entities)))
        set_task_result(task_id, "failed_entities", str(failed_entities[:20]))
        logger.info(f"[{task_id}] 导入流程结束：{import_result}，失败实体 {len(failed_entities)} 个")

    except Exception as e:
        update_task_status(task_id, TASK_STATUS_FAILED)
        set_task_result(task_id, "error", str(e))
        logger.exception(f"[{task_id}] 导入图执行失败")


async def _save_upload(file: UploadFile, dest: Path, max_size: int) -> int:
    """
    流式落盘（zip 可达数百 MB，不整块读入内存）；超限即删除半成品并报错
    :return: 已写入字节数
    """
    size = 0
    with open(dest, "wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > max_size:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=400,
                    detail=f"文件 {file.filename} 超过 {max_size // 1024 // 1024}MB 上限",
                )
            out.write(chunk)
    return size


def _expand_zip(zip_path: Path, task_local_dir: Path, upload_name: str,
                content_type: str, project_name: str,
                background_tasks: BackgroundTasks) -> list[dict]:
    """
    zip 资料包 → 解压 → 每个 md 生成一个独立导入任务
    包内通常为「多个 md + images/」，图片由 node_md_img 按 md 内相对引用自动定位
    :return: 任务清单 [{task_id, file_name, source}]
    """
    extract_dir = task_local_dir / "extracted"
    try:
        extract_zip(zip_path, extract_dir)
    except ZipExtractError as e:
        raise HTTPException(status_code=400, detail=f"资料包 {upload_name} 解析失败：{e}")

    md_files = discover_md_files(extract_dir)
    if not md_files:
        raise HTTPException(status_code=400,
                            detail=f"资料包 {upload_name} 中未找到 .md/.markdown 文档")
    if len(md_files) > MAX_MD_PER_ZIP:
        raise HTTPException(status_code=400,
                            detail=f"资料包 {upload_name} 含 {len(md_files)} 个文档，"
                                   f"超过单次 {MAX_MD_PER_ZIP} 个上限")

    # course_doc 项目文档标记：显式传入优先，缺省用压缩包名（如「掌柜智库」）
    zip_project_name = project_name or zip_path.stem

    tasks = []
    for md_path in md_files:
        task_id = str(uuid.uuid4())
        rel_name = md_path.relative_to(extract_dir).as_posix()
        add_done_task(task_id, "upload_file")
        background_tasks.add_task(
            _run_graph_with_progress, task_id, str(extract_dir), str(md_path),
            content_type, zip_project_name, rel_name,
        )
        logger.info(f"[{task_id}] 已提交后台导入（资料包 {upload_name}）：{rel_name}")
        tasks.append({"task_id": task_id, "file_name": rel_name, "source": upload_name})
    return tasks


@app.post("/upload")
async def upload_file(
        files: list[UploadFile] = File(...),
        content_type: str = Form(...),
        project_name: str = Form(""),
        background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    文件上传 + 后台导入
    - 白名单：.md/.pdf/.docx（单文件 ≤50MB）+ .zip 资料包（≤500MB，包内每个 md 一个任务）
    - 单次 ≤20 个文件；zip 内 ≤50 个 md
    - content_type：course_catalog / question_bank / course_doc（枚举校验）
    - project_name：可选，course_doc 用于标记项目文档（zip 导入缺省取包名）
    """
    valid_types = {"course_catalog", "question_bank", "course_doc"}
    if content_type not in valid_types:
        raise HTTPException(status_code=400,
                            detail=f"非法 content_type：{content_type}（合法值：{sorted(valid_types)}）")
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="单次批量上限 20 个文件")

    today_str = datetime.now().strftime("%Y%m%d")
    tasks: list[dict] = []

    for file in files:
        ext = Path(file.filename or "").suffix.lower()
        if ext not in UPLOAD_EXTENSIONS:
            raise HTTPException(status_code=400,
                                detail=f"文件 {file.filename} 类型 {ext} 不在白名单"
                                       f"（.md/.pdf/.docx/.zip）")

        task_id = str(uuid.uuid4())
        task_local_dir: Path = PROJECT_ROOT / "output" / today_str / task_id
        task_local_dir.mkdir(parents=True, exist_ok=True)
        local_file_path = task_local_dir / (file.filename or f"upload{ext}")

        if ext == ".zip":
            await _save_upload(file, local_file_path, MAX_ZIP_SIZE)
            tasks.extend(_expand_zip(
                local_file_path, task_local_dir, file.filename or local_file_path.name,
                content_type, project_name, background_tasks,
            ))
            continue

        await _save_upload(file, local_file_path, MAX_FILE_SIZE)
        add_done_task(task_id, "upload_file")
        background_tasks.add_task(
            _run_graph_with_progress, task_id, str(task_local_dir), str(local_file_path),
            content_type, project_name, "",
        )
        logger.info(f"[{task_id}] 已提交后台导入：{file.filename}（{content_type}）")
        tasks.append({"task_id": task_id, "file_name": file.filename, "source": file.filename})

    return {
        "code": 200,
        "message": f"已接收 {len(files)} 个文件，共 {len(tasks)} 个导入任务",
        "task_ids": [t["task_id"] for t in tasks],
        "tasks": tasks,
    }


class TaskStatusResponse(BaseModel):
    code: int = 200
    task_id: str
    status: str
    done_list: list
    running_list: list
    error: str = ""
    import_result: str = ""
    failed_entities_count: str = ""


@app.get("/status/{task_id}")
def get_task_progress(task_id: str) -> TaskStatusResponse:
    """导入任务进度查询"""
    from utils.task_utils import get_task_result
    return TaskStatusResponse(
        task_id=task_id,
        status=get_task_status(task_id) or "unknown",
        done_list=get_done_task_list(task_id),
        running_list=get_running_task_list(task_id),
        error=get_task_result(task_id, "error"),
        import_result=get_task_result(task_id, "import_result"),
        failed_entities_count=get_task_result(task_id, "failed_entities_count"),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
