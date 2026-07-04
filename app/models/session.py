import sqlite3
import uuid
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from pydantic import BaseModel


# Pydantic 模型
class SessionCreate(BaseModel):
    """创建会话的请求模型"""
    user_id: str | None = None
    biz_type: str | None = None
    name: str


class UpdateTitleRequest(BaseModel):
    thread_id: str
    name: str

class SessionResponse(BaseModel):
    """会话响应模型"""
    thread_id: str
    user_id: str
    biz_type: str
    name: str
    created_at: str
    updated_at: str

DB_PATH = Path(__file__).parent.parent / "db/sessions.db"



def get_db_connection():
    """获取数据库连接"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化数据库，创建 sessions 表"""

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            thread_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            biz_type TEXT NOT NULL,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def create_session() -> SessionResponse:
    """创建新会话"""
    conn = get_db_connection()
    cursor = conn.cursor()

    thread_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    user_id="0111"
    biz_type="emailAgent"
    name = "新会话"

    cursor.execute(
        """
        INSERT INTO sessions (thread_id, user_id, biz_type, name, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (thread_id, user_id, biz_type, name, now, now)
    )
    conn.commit()
    conn.close()

    return SessionResponse(
        thread_id=thread_id,
        user_id=user_id,
        biz_type=biz_type,
        name=name,
        created_at=now,
        updated_at=now
    )

#默认只是标题查询，现在没加权限管理，没分配角色
def search_param(title=None)-> List[SessionResponse]:
    #默认只是查询标题
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "SELECT thread_id, user_id, biz_type, name, created_at, updated_at FROM sessions"
    if title:
        query += " WHERE name LIKE ?"
        cursor.execute(query, (f"%{title}%",))
    else:
        cursor.execute(query, ())
    rows = cursor.fetchall()
    conn.close()

    return [
        SessionResponse(
            thread_id=row["thread_id"],
            user_id=row["user_id"],
            biz_type=row["biz_type"],
            name=row["name"],
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        )
        for row in rows
    ]

# def get_sessions(user_id: Optional[str] = None, biz_type: Optional[str] = None) -> List[SessionResponse]:
#     """查询会话列表，支持按 user_id 和 biz_type 筛选"""
#     conn = get_db_connection()
#     cursor = conn.cursor()
#
#     query = "SELECT thread_id, user_id, biz_type, name, created_at, updated_at FROM sessions"
#     params = []
#
#     if user_id or biz_type:
#         conditions = []
#         if user_id:
#             conditions.append("user_id = ?")
#             params.append(user_id)
#         if biz_type:
#             conditions.append("biz_type = ?")
#             params.append(biz_type)
#         query += " WHERE " + " AND ".join(conditions)
#
#     query += " ORDER BY updated_at DESC"
#
#     cursor.execute(query, params)
#     rows = cursor.fetchall()
#     conn.close()
#
#     return [
#         SessionResponse(
#             thread_id=row["thread_id"],
#             user_id=row["user_id"],
#             biz_type=row["biz_type"],
#             name=row["name"],
#             created_at=row["created_at"],
#             updated_at=row["updated_at"]
#         )
#         for row in rows
#     ]


def delete_session(thread_id: str) -> bool:
    """删除会话,删除的列表记录"""

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM sessions WHERE thread_id = ?", (thread_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()

    return deleted

def upSessionTitle(thread_id,name):

    conn = get_db_connection()
    cursor = conn.cursor()

    update_query = "UPDATE sessions SET name = ? WHERE thread_id = ?"

    # 执行更新（注意参数顺序要和 ? 一一对应）
    cursor.execute(update_query, (name,thread_id))
    conn.commit()  # 别忘了提交事务！
    conn.close()

if __name__ == "__main__":
    # session_data = SessionCreate(
    #     user_id="user_123",  # 替换为实际的用户ID
    #     biz_type="email",  # 替换为实际的业务类型
    #     name="新会话"  # 会话名称
    # )
    #thread_id='emailAgent_01'
    print(search_param())
    #print(upSessionTitle("emailAgent_01","查询邮箱"))
# 初始化数据库
init_db()