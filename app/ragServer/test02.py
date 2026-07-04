import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

def get_chat_list(db_path: str, limit: int = 20):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. 获取最近的 thread_id 列表 (利用 checkpoint_id 排序)
    query = """
        SELECT thread_id
        FROM checkpoints
        GROUP BY thread_id
        ORDER BY MAX(checkpoint_id) DESC
        LIMIT ?
    """
    cursor.execute(query, (limit,))
    threads = [row[0] for row in cursor.fetchall()]

    chat_list = []

    # 2. 遍历 thread_id，获取具体详情
    with SqliteSaver.from_conn_string(db_path) as saver:
        for tid in threads:
            config = {"configurable": {"thread_id": tid}}

            # 获取该线程的最新状态
            state_snapshot = saver.get_tuple(config)

            if state_snapshot:
                metadata = state_snapshot.metadata
                state_values = state_snapshot.checkpoint['channel_values']

                # --- 提取标题逻辑 ---
                # 假设你的 State 中 messages 是第一个字段
                messages = state_values.get('messages', [])
                title = "新对话"
                if messages:
                    # 取第一条人类消息的内容作为标题，截取前15个字
                    first_msg = messages[0]
                    content = first_msg.content if hasattr(first_msg, 'content') else str(first_msg)
                    title = content[:15] + "..." if len(content) > 15 else content

                # --- 提取时间逻辑 (可选) ---
                # 如果 metadata 里没有存时间，这里只能显示 "刚刚" 或忽略
                updated_at = metadata.get('created_at', '刚刚')

                chat_list.append({
                    "thread_id": tid,
                    "title": title,
                    "updated_at": updated_at
                })

    conn.close()
    return chat_list

if __name__=="__main__":
    db_path = "D:/newProject/langchain_v3/chatdb/my_app.db"
    param = get_chat_list(db_path)
    print(param)






