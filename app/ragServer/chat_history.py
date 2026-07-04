# -*- coding: utf-8 -*-
"""从 chatdb/my_app.db 读取 LangGraph checkpoint 中的会话记录。"""
from __future__ import annotations

import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

CHAT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "chatdb/my_app.db"


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content) if content else ""


def _normalize_user_content(content: str) -> str:
    marker = "用户问题："
    if marker in content:
        return content.split(marker, 1)[1].strip()
    return content


def serialize_message(msg: Any, include_tools: bool = False) -> dict | None:
    if isinstance(msg, dict):
        role = msg.get("type") or msg.get("role") or "unknown"
        content = _extract_text(msg.get("content", ""))
        name = msg.get("name")
    else:
        role = getattr(msg, "type", None) or msg.__class__.__name__.replace("Message", "").lower()
        content = _extract_text(getattr(msg, "content", ""))
        name = getattr(msg, "name", None)

    if role == "human":
        content = _normalize_user_content(content)

    if role == "tool" and not include_tools:
        return None

    if not content.strip():
        return None

    item = {"role": role, "content": content}
    if role == "tool" and name:
        item["name"] = name
    return item


def parse_checkpoint_messages(checkpoint: dict | None, include_tools: bool = False) -> list[dict]:
    channel_values = (checkpoint or {}).get("channel_values") or {}
    raw_messages = channel_values.get("messages") or []

    messages: list[dict] = []
    for msg in raw_messages:
        item = serialize_message(msg, include_tools=include_tools)
        if item:
            messages.append(item)
    return messages


class ChatHistoryReader:
    def __init__(
        self,
        db_path: Path = CHAT_DB_PATH,
        executor: ThreadPoolExecutor | None = None,
    ):
        self.db_path = db_path
        self.executor = executor or ThreadPoolExecutor(max_workers=2)
        self._checkpointer: AsyncSqliteSaver | None = None
        self._checkpointer_ctx = None

    async def _ensure_checkpointer(self) -> AsyncSqliteSaver | None:
        if self._checkpointer is not None:
            return self._checkpointer
        if not self.db_path.exists():
            return None

        self._checkpointer_ctx = AsyncSqliteSaver.from_conn_string(str(self.db_path))
        self._checkpointer = await self._checkpointer_ctx.__aenter__()
        return self._checkpointer

    def _list_thread_ids_sync(self, thread_id: str | None) -> list[str]:
        if not self.db_path.exists():
            return []

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            if thread_id:
                cursor.execute(
                    """
                    SELECT DISTINCT thread_id
                    FROM checkpoints
                    WHERE thread_id = ?
                    ORDER BY thread_id
                    """,
                    (thread_id,),
                )
            else:
                cursor.execute(
                    """
                    SELECT DISTINCT thread_id
                    FROM checkpoints
                    ORDER BY thread_id
                    """
                )
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

    async def list_thread_ids(self, thread_id: str | None = None) -> list[str]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._list_thread_ids_sync, thread_id)

    async def load_thread(self, thread_id: str, include_tools: bool = False) -> dict:
        checkpointer = await self._ensure_checkpointer()
        if checkpointer is None:
            return {
                "thread_id": thread_id,
                "updated_at": None,
                "message_count": 0,
                "messages": [],
            }

        config = {"configurable": {"thread_id": thread_id}}
        checkpoint_tuple = await checkpointer.aget_tuple(config)
        if checkpoint_tuple is None:
            return {
                "thread_id": thread_id,
                "updated_at": None,
                "message_count": 0,
                "messages": [],
            }

        checkpoint = checkpoint_tuple.checkpoint or {}
        messages = parse_checkpoint_messages(checkpoint, include_tools=include_tools)
        return {
            "thread_id": thread_id,
            "updated_at": checkpoint.get("ts"),
            "message_count": len(messages),
            "messages": messages,
        }

    async def get_history(self, thread_id: str | None = None, include_tools: bool = False) -> dict:
        base = {"db_path": str(self.db_path)}

        if not self.db_path.exists():
            if thread_id:
                return {**base, "thread_id": thread_id, "message_count": 0, "messages": []}
            return {**base, "total": 0, "threads": []}

        thread_ids = await self.list_thread_ids(thread_id)
        if not thread_ids:
            if thread_id:
                return {"thread_id": thread_id, "message_count": 0, "messages": []}
            return {"total": 0, "threads": []}

        if thread_id:
            return await self.load_thread(thread_id, include_tools=include_tools)

        threads = await asyncio.gather(
            *[self.load_thread(tid, include_tools=include_tools) for tid in thread_ids]
        )
        return {"total": len(threads), "threads": list(threads)}
