"""
问答服务（FastAPI :8001）
对应 PRD V1.3 3.7 / 5.4：
- POST /query：问答入口（M2 实现非流式；SSE 流式为 M3 范围）
- GET /history/{session_id} / DELETE /history/{session_id}：历史管理
- 多轮对话：查询前拉取最近 6 轮（12 条）历史供改写
"""
import uuid
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from common.logging.logger import logger
from processor.query_processor.main_graph import query_app
from processor.query_processor.state import create_query_default_state
from utils.clients.mongo_history_utils import get_recent_messages, save_chat_message, clear_history

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


# 请求结构
class QueryRequest(BaseModel):
    query: str = Field(..., description="用户问题")
    session_id: str = Field(None, description="会话ID（多轮对话；空则新建")
    is_stream: bool = Field(False, description="是否流式返回（M3 实现，当前忽略并以非流式返回）")


@app.post("/query")
def query(query_request: QueryRequest):
    """
    问答入口：
    1. session_id 为空则新建
    2. 拉取最近 12 条历史（6 轮）供意图路由与改写
    3. 执行查询图
    4. 写入 MongoDB 历史（user + assistant 两条）
    """
    user_query = query_request.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="query 不能为空")

    session_id = query_request.session_id or str(uuid.uuid4())
    logger.info(f"[{session_id}] 用户提问：{user_query}")

    # ---- 拉取历史（多轮指代消解输入）----
    history = []
    try:
        history = get_recent_messages(session_id, limit=12)
    except Exception as e:
        logger.warning(f"[{session_id}] 历史拉取失败（降级单轮模式）：{e}")

    # ---- 执行查询图 ----
    try:
        state = create_query_default_state(
            session_id=session_id,
            original_query=user_query,
            is_stream=query_request.is_stream,
            history=history,
        )
        final_state = query_app.invoke(state)
    except Exception as e:
        logger.exception(f"[{session_id}] 查询图执行失败")
        raise HTTPException(status_code=500, detail=f"query error: {e}")

    answer = final_state.get("answer", "")
    references = final_state.get("references", [])
    degraded = bool(final_state.get("degraded"))

    # ---- 写入历史（user + assistant 两条，PRD 3.7 一轮 = 2 条）----
    try:
        save_chat_message(session_id=session_id, role="user", text=user_query)
        save_chat_message(
            session_id=session_id,
            role="assistant",
            text=answer,
            references=references,
        )
    except Exception as e:
        logger.warning(f"[{session_id}] 历史写入失败：{e}")

    return {
        "session_id": session_id,
        "answer": answer,
        "references": references,
        "degraded": degraded,
    }


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
