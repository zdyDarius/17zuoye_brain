"""
导入服务（FastAPI :8000）
对应 PRD V1.3 3.2 / 5.4：
- GET /import.html：导入操作页面
- POST /upload：多文件上传 + content_type 元数据，后台执行导入图
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
from processor.import_processor.state import create_import_default_state
from utils.path_util import PROJECT_ROOT
from utils.task_utils import (
    get_running_task_list, get_done_task_list, get_task_status,
    add_running_task, add_done_task, update_task_status, set_task_result,
    clear_task,
    TASK_STATUS_PROCESSING, TASK_STATUS_FAILED, TASK_STATUS_COMPLETED,
)

app = FastAPI(title="education-brain import service")

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


# ---- 导入图后台执行 ----
def run_graph_task(task_id: str, local_dir: str, local_file_path: str, content_type: str):
    """后台任务：执行导入图全流程并更新任务状态"""
    try:
        clear_task(task_id)
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        logger.info(f"[{task_id}] 开始导入流程：{local_file_path}（{content_type}）")

        init_state = create_import_default_state(
            task_id=task_id, local_dir=local_dir,
            local_file_path=local_file_path, content_type=content_type,
        )

        # 流式更新节点进度（task_utils 中文节点名）
        final_state = kb_import_app.invoke(
            init_state,
            config={"callbacks": [_TaskProgressCallback(task_id)]},
        )

        # 结果统计与状态
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


from langchain_core.callbacks import BaseCallbackHandler


class _TaskProgressCallback(BaseCallbackHandler):
    """LangChain 回调：节点开始/结束时更新任务进度（供 /status 查询）"""

    def __init__(self, task_id: str):
        self.task_id = task_id

    def on_chain_start(self, serialized, inputs, **kwargs):
        node = (serialized or {}).get("name") or (inputs or {}).get("node") or ""
        if node and not node.startswith("__"):
            add_running_task(self.task_id, node)

    def on_chain_end(self, outputs, **kwargs):
        pass


# 简化版进度跟踪：LangGraph stream 更直观——改用直接包装（上面回调对 LangGraph 节点粒度支持有限）
def _run_graph_with_progress(task_id: str, local_dir: str, local_file_path: str, content_type: str):
    """后台任务（带节点级进度）：stream 模式执行导入图"""
    try:
        clear_task(task_id)
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        logger.info(f"[{task_id}] 开始导入流程：{local_file_path}（{content_type}）")

        init_state = create_import_default_state(
            task_id=task_id, local_dir=local_dir,
            local_file_path=local_file_path, content_type=content_type,
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


@app.post("/upload")
async def upload_file(
        files: list[UploadFile] = File(...),
        content_type: str = Form(...),
        background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    文件上传 + 后台导入
    - 白名单：.md/.pdf/.docx（PRD 3.2 校验规则）；单文件 ≤50MB；单次 ≤20 个
    - content_type：course_catalog / question_bank / course_doc（枚举校验）
    """
    valid_types = {"course_catalog", "question_bank", "course_doc"}
    if content_type not in valid_types:
        raise HTTPException(status_code=400,
                            detail=f"非法 content_type：{content_type}（合法值：{sorted(valid_types)}）")
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="单次批量上限 20 个文件")

    from processor.import_processor.nodes.node_entry import VALID_EXTENSIONS, MAX_FILE_SIZE

    today_str = datetime.now().strftime("%Y%m%d")
    task_ids = []

    for file in files:
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)

        ext = Path(file.filename).suffix.lower()
        if ext not in VALID_EXTENSIONS:
            raise HTTPException(status_code=400,
                                detail=f"文件 {file.filename} 类型 {ext} 不在白名单（.md/.pdf/.docx）")

        # 保存到任务目录
        task_local_dir: Path = PROJECT_ROOT / "output" / today_str / task_id
        task_local_dir.mkdir(parents=True, exist_ok=True)
        local_file_path = task_local_dir / file.filename

        data = await file.read()
        if len(data) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400,
                                detail=f"文件 {file.filename} 超过 50MB 上限")
        local_file_path.write_bytes(data)

        add_done_task(task_id, "upload_file")
        background_tasks.add_task(
            _run_graph_with_progress, task_id, str(task_local_dir), str(local_file_path), content_type,
        )
        logger.info(f"[{task_id}] 已提交后台导入：{file.filename}（{content_type}）")

    return {"code": 200, "message": f"已接收 {len(files)} 个文件", "task_ids": task_ids}


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
