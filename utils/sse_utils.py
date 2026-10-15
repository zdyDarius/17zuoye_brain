import json
import queue
import asyncio
from typing import Dict, Any, Optional, AsyncGenerator, List
from fastapi import Request


class SSEEvent:
    READY = "ready"         # 连接建立
    PROGRESS = "progress"   # 任务节点进度
    DELTA = "delta"         # LLM 流式输出增量
    FINAL = "final"         # 最终完整答案
    ERROR = "error"         # 错误信息
    CLOSE = "__close__"     # 关闭连接信号（内部）


# 全局 SSE 会话队列存储
# Key: session_id, Value: queue.Queue
_session_stream: Dict[str, queue.Queue] = {}

# 会话事件重放缓冲（PRD 3.7 / 复审 P2-2：事件带自增 id，Last-Event-ID 重连补发）
# Key: session_id, Value: List[{"id": int, "event": str, "data": dict}]
_session_replay: Dict[str, List[Dict[str, Any]]] = {}

# 各会话的下一个事件 id（自增）
_session_next_id: Dict[str, int] = {}


def get_sse_queue(session_id: str) -> Optional["queue.Queue"]:
    """获取指定 session 的队列"""
    return _session_stream.get(session_id)


def create_sse_queue(session_id: str) -> "queue.Queue":
    """创建并注册一个新的 SSE 队列（同时初始化重放缓冲与事件 id 计数）"""
    if session_id not in _session_replay:
        _session_replay[session_id] = []
    if session_id not in _session_next_id:
        _session_next_id[session_id] = 1
    q = _session_stream.get(session_id)
    if q is None:
        q = queue.Queue()
        _session_stream[session_id] = q
    return q


def remove_sse_queue(session_id: str):
    """移除指定 session 的队列（重放缓冲保留一段时间供重连，由 clear_session 清理）"""
    _session_stream.pop(session_id, None)


def clear_session(session_id: str):
    """彻底清理会话（任务终态后调用）：队列 + 重放缓冲 + id 计数"""
    _session_stream.pop(session_id, None)
    _session_replay.pop(session_id, None)
    _session_next_id.pop(session_id, None)


def _sse_pack(event: str, data: Dict[str, Any], event_id: Optional[int] = None) -> str:
    """打包 SSE 消息格式（含自增 id，支持 Last-Event-ID 断线重连）"""
    payload = json.dumps(data, ensure_ascii=False)
    if event_id is not None:
        return f"id: {event_id}\nevent: {event}\ndata: {payload}\n\n"
    return f"event: {event}\ndata: {payload}\n\n"


def push_to_session(session_id: str, event: str, data: Dict[str, Any]):
    """
    通过 session_id 推送事件：
    1. 分配自增事件 id
    2. 写入重放缓冲（供重连补发）
    3. 投递到活跃队列（若有监听者）
    """
    # 分配事件 id 并写入缓冲
    event_id = _session_next_id.get(session_id, 1)
    _session_next_id[session_id] = event_id + 1
    replay = _session_replay.setdefault(session_id, [])
    replay.append({"id": event_id, "event": event, "data": data})
    # 缓冲上限保护（仅保留最近 200 条事件）
    if len(replay) > 200:
        del replay[: len(replay) - 200]

    # 投递到活跃队列
    stream_queue = get_sse_queue(session_id)
    if stream_queue:
        stream_queue.put({"event": event, "data": data, "id": event_id})


def _parse_last_event_id(request: Request) -> Optional[int]:
    """解析请求头中的 Last-Event-ID（断线重连时客户端上报的最后收到的事件 id）"""
    header = request.headers.get("last-event-id") if request else None
    if header:
        try:
            return int(header)
        except ValueError:
            pass
    return None


async def sse_generator(session_id: str, request: Request):
    """
    SSE 生成器，用于 FastAPI 的 StreamingResponse
    支持 Last-Event-ID 断线重连：重连时从该 id 之后补发缓冲事件（复审 P2-2）
    """
    stream_queue = get_sse_queue(session_id)
    if stream_queue is None:
        # 无对应队列：任务可能已结束，输出重放缓冲后结束（按 Last-Event-ID 过滤补发）
        replay = _session_replay.get(session_id) or []
        last_id = _parse_last_event_id(request)
        missed = [item for item in replay if (last_id is None or item["id"] > last_id)]
        if missed:
            for item in missed:
                yield _sse_pack(item["event"], item["data"], item["id"])
            yield _sse_pack("close", {"message": "replay done"})
        else:
            yield _sse_pack("error", {"error": f"session {session_id} 无待补发事件"})
        return

    # 断线重连：解析 Last-Event-ID 头，补发该 id 之后的事件
    last_event_id = _parse_last_event_id(request)
    if last_event_id is not None:
        replay = _session_replay.get(session_id) or []
        missed = [item for item in replay if item["id"] > last_event_id]
        for item in missed:
            yield _sse_pack(item["event"], item["data"], item["id"])

    loop = asyncio.get_running_loop()
    try:
        # 发送连接建立信号
        yield _sse_pack("ready", {"replayed": last_event_id is not None})

        while True:
            # 若客户端断开，尽快退出
            if await request.is_disconnected():
                break

            try:
                # 使用 run_in_executor 避免阻塞 async 事件循环
                msg = await loop.run_in_executor(None, stream_queue.get, True, 1.0)
            except queue.Empty:
                continue

            event = msg.get("event")
            data = msg.get("data")
            event_id = msg.get("id")

            # 特殊关闭事件
            if event == "__close__":
                break

            yield _sse_pack(event, data, event_id)
    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
        # 生成器被取消/对端断开：静默退出（重放缓冲保留供重连）
        return
    except Exception:
        return
    finally:
        # 仅移除队列引用，保留重放缓冲（断线可重连；终态由 clear_session 清理）
        _session_stream.pop(session_id, None)
