from typing import Optional, Dict, Any,List

from pydantic import BaseModel

# ---  数据模型 ---
class ChatRequest(BaseModel):
    # 消息内容
    message: str =None
    thread_id: str | None = None
    interrupt_decision: Optional[Dict[str, Any]] = None