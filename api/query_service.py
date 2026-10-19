"""
问答服务（FastAPI :8001）
对应 PRD V1.3 3.7 / 5.4：
- POST /query：问答入口（非流式同步返回；流式后台执行 + SSE 推送）
  返回 answer / references / image_urls / degraded
- GET /stream/{session_id}：SSE 流式通道（事件自增 id + Last-Event-ID 重连，复审 P2-2）
- GET /history/{session_id} / DELETE /history/{session_id}：历史管理（含参考图）
- 多轮对话：查询前拉取最近 6 轮（12 条）历史供改写
"""
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from common.logging.logger import logger
from processor.query_processor.main_graph import query_app
from processor.query_processor.state import create_query_default_state
from utils.clients.mongo_history_utils import get_recent_messages, save_chat_message, clear_history
from utils.sse_utils import (
    create_sse_queue, push_to_session, sse_generator, remove_sse_queue,
    schedule_session_cleanup, SSEEvent,
)
from utils.task_utils import update_task_status, TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED, TASK_STATUS_FAILED
from utils.path_util import PROJECT_ROOT

app = FastAPI(title="education-brain query service")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# 健康检查
@app.get("/health")
def health():
    return {"ok": True}


# 问答页面
@app.get("/chat.html")
def chat_page():
    page_path = PROJECT_ROOT / "page" / "chat.html"
    if not page_path.exists():
        raise HTTPException(status_code=404, detail=f"页面不存在：{page_path}")
    return FileResponse(page_path, media_type="text/html")


# 请求结构
class QueryRequest(BaseModel):
    query: str = Field(..., description="用户问题")
    session_id: str = Field(None, description="会话ID（多轮对话；空则新建")
    is_stream: bool = Field(False, description="是否流式返回")


# ---- 查询图后台执行（流式模式）----
def run_query_graph(session_id: str, user_query: str, history: list):
    """后台任务：执行查询图，节点级 step 事件 + final 事件 SSE 推送"""
    try:
        create_sse_queue(session_id)
        update_task_status(session_id, TASK_STATUS_PROCESSING)

        state = create_query_default_state(
            session_id=session_id, original_query=user_query, is_stream=True, history=history,
        )
        push_to_session(session_id, "step", {"node": "start", "message": "开始处理您的问题"})

        # 流式执行：每个节点完成推送 step 事件
        final_state = dict(state)
        for event in query_app.stream(state, stream_mode="updates"):
            for node_name, node_result in event.items():
                push_to_session(session_id, "step", {"node": node_name})
                if isinstance(node_result, dict):
                    final_state.update(node_result)

        answer = final_state.get("answer", "")
        references = final_state.get("references", [])
        image_urls = final_state.get("image_urls", [])
        degraded = bool(final_state.get("degraded"))

        # 历史写入（user + assistant 两条，助手消息带引用与配图）
        try:
            save_chat_message(session_id=session_id, role="user", text=user_query)
            save_chat_message(session_id=session_id, role="assistant", text=answer,
                              references=references, image_urls=image_urls)
        except Exception as e:
            logger.warning(f"[{session_id}] 历史写入失败：{e}")

        update_task_status(session_id, TASK_STATUS_COMPLETED)
        push_to_session(session_id, SSEEvent.FINAL, {
            "answer": answer, "references": references, "image_urls": image_urls,
            "degraded": degraded, "status": "completed",
        })
        # 移除队列引用（重放缓冲保留，供断线重连补发 final）
        remove_sse_queue(session_id)
        # 缓冲到期回收（默认 5 分钟），避免长期运行内存随会话数增长
        schedule_session_cleanup(session_id)

    except Exception as e:
        update_task_status(session_id, TASK_STATUS_FAILED)
        logger.exception(f"[{session_id}] 流式查询执行失败")
        push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})
        remove_sse_queue(session_id)
        schedule_session_cleanup(session_id)


@app.post("/query")
def query(query_request: QueryRequest, background_tasks: BackgroundTasks):
    """
    问答入口：
    - 非流式：同步执行返回完整结果
    - 流式：后台执行，前端通过 GET /stream/{session_id} 接收 step/final/error 事件
    """
    user_query = query_request.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="query 不能为空")

    session_id = query_request.session_id or str(uuid.uuid4())
    logger.info(f"[{session_id}] 用户提问（stream={query_request.is_stream}）：{user_query}")

    # ---- 拉取历史（多轮指代消解输入）----
    # get_recent_messages 按 ts 倒序返回（新→旧），这里统一转为时间正序（旧→新），
    # 下游节点取 history[-N:] 才能拿到"最近"的对话。
    history = []
    try:
        history = list(reversed(get_recent_messages(session_id, limit=12)))
    except Exception as e:
        logger.warning(f"[{session_id}] 历史拉取失败（降级单轮模式）：{e}")

    # ---- 流式模式：后台执行 + SSE ----
    if query_request.is_stream:
        create_sse_queue(session_id)
        background_tasks.add_task(run_query_graph, session_id, user_query, history)
        return {"session_id": session_id, "message": "结果正在处理中，请通过 /stream/{session_id} 订阅事件"}

    # ---- 非流式模式：同步执行 ----
    try:
        state = create_query_default_state(
            session_id=session_id, original_query=user_query, is_stream=False, history=history,
        )
        final_state = query_app.invoke(state)
    except Exception as e:
        logger.exception(f"[{session_id}] 查询图执行失败")
        raise HTTPException(status_code=500, detail=f"query error: {e}")

    answer = final_state.get("answer", "")
    references = final_state.get("references", [])
    image_urls = final_state.get("image_urls", [])
    degraded = bool(final_state.get("degraded"))

    # ---- 写入历史 ----
    try:
        save_chat_message(session_id=session_id, role="user", text=user_query)
        save_chat_message(session_id=session_id, role="assistant", text=answer,
                          references=references, image_urls=image_urls)
    except Exception as e:
        logger.warning(f"[{session_id}] 历史写入失败：{e}")

    return {
        "session_id": session_id,
        "answer": answer,
        "references": references,
        "image_urls": image_urls,
        "degraded": degraded,
    }


# ---- SSE 流式通道 ----
@app.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):
    """SSE 事件流：step（节点进度）/ final（完整答案）/ error（异常）；支持 Last-Event-ID 重连"""
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/history/{session_id}")
def get_history(session_id: str):
    """查询会话历史记录"""
    try:
        records = get_recent_messages(session_id, limit=50)
        items = [
            {
                "_id": str(r.get("_id")) if r.get("_id") is not None else "",
                "session_id": session_id,
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "references": r.get("references", []),
                "image_urls": r.get("image_urls", []),
                "ts": r.get("ts"),
            }
            for r in records
        ]
        return {"session_id": session_id, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {e}")


@app.delete("/history/{session_id}")
def clear_chat_history(session_id: str):
    """清空会话历史"""
    deleted_count = clear_history(session_id)
    logger.info(f"清空 {session_id} 历史记录，删除 {deleted_count} 条")
    return {"message": "History cleared", "deleted_count": deleted_count}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
