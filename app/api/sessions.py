from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from app.models.session import (
    SessionCreate,
    search_param,
    SessionResponse,
    create_session,
    delete_session,
    upSessionTitle
)
from app.emailAgent.emailAgent import email_agent
from app.models.session import UpdateTitleRequest


router = APIRouter()

#创建新会话列表
@router.post("/setSessions", response_model=SessionResponse, tags=["会话"])
def create_new_session():
    """创建新会话"""
    return create_session()

#查询会话列表(现在没加权限管理，没分配角色，默认测试只是标题查询)
@router.get("/getSessions", response_model=List[SessionResponse], tags=["会话"])
def search_param_session(title:Optional[str] = Query(None, description="用户ID")):
    return search_param(title)

# @router.get("/sessions", response_model=List[SessionResponse], tags=["会话"])
# def list_sessions(
#     user_id: Optional[str] = Query(None, description="用户ID"),
#     biz_type: Optional[str] = Query(None, description="业务类型")
# ):
#     """查询会话列表，支持按 user_id 和 biz_type 筛选"""
#     return get_sessions(user_id=user_id, biz_type=biz_type)


@router.delete("/sessions/{thread_id}", tags=["会话"])
async def remove_session(thread_id: str):
    """删除会话列表记录及 LangGraph checkpoint 聊天历史"""
    deleted_db = delete_session(thread_id)
    deleted_checkpoint = await email_agent.clear_messages(thread_id)

    if not deleted_db and not deleted_checkpoint:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "message": "Session deleted successfully",
        "thread_id": thread_id,
    }


@router.patch("/sessions/{thread_id}/messages", tags=["会话"])
async def get_session_messages(thread_id: str):
    """获取会话的历史消息"""
    try:
        result = await email_agent.get_messages(thread_id)
        return result
    except Exception as e:
        return {"messages": [], "error": str(e)}

@router.post("/sessions/uptitle", tags=["会话"])
async def up_session_title(request_body: UpdateTitleRequest):
    """当聊天时，user输入第一句话动态修改name标题"""
    try:
        upSessionTitle(request_body.thread_id,request_body.name)
        return {"message":"更新标题成功"}
    except Exception as e:
        return {"messages": [], "error": str(e)}