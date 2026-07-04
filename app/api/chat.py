"""
聊天 API 模块。

提供 POST /api/chat/stream 接口，接收用户消息并以 SSE（Server-Sent Events）
流式返回 Agent 的回复与工具调用事件。
"""
from fastapi import APIRouter
from starlette.responses import StreamingResponse

from dietitian import getTravel  # Agent 流式生成器
from schemas.chat import ChatRequest  # 请求体：含 message 字段

from app.emailAgent.emailAgent import email_agent
from fastapi.responses import JSONResponse
router = APIRouter()


@router.post("/chat/stream")
async def chat_route(request: ChatRequest):
    """
    流式对话接口。

    将 request.message 交给婚礼协调 Agent，响应体为 text/event-stream，
    客户端可逐条解析 data: {...} 事件（token / tool_start / tool_end / done / error）。

    Args:
        request: 用户输入，例如婚礼地点、人数、风格等。

    Returns:
        StreamingResponse: SSE 流式 HTTP 响应。
    """
    return StreamingResponse(
        getTravel(request.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )




@router.post("/chat/send", tags=["聊天"])
async def send_chat(request: ChatRequest):
    """发送聊天消息，使用 SSE 流式返回 Agent 响应。同时处理 HITL 中断决策。"""
    if not request.thread_id:
        return JSONResponse(status_code=400, content={"error": "thread_id is required"})



    return StreamingResponse(
        email_agent.generate_sse(
            request.thread_id,
            request.message or "",
            request.interrupt_decision,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )







