"""Diagnostic script for RAG agent tool calling."""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain.messages import AIMessage, ToolMessage
from schemas.chat import ChatRequest
from app.ragServer.ragServerTool import RagServerTool


QUERY = (
    "学python,掌握langchain的相关问题，市面上最主流的AI模型是哪种，"
    "今天学习，明天会过时吗？"
)


async def test_agent_events():
    server = RagServerTool()
    await server._ensure_agent()

    thread_id = "diag_tool_call_01"
    config = {"configurable": {"thread_id": thread_id}}

    print("=== Agent astream_events ===")
    tool_calls = []
    tool_results = []
    ai_tokens = []

    async for event in server.agent.astream_events(
        {"messages": [{"role": "user", "content": QUERY}]},
        config=config,
        version="v2",
    ):
        kind = event.get("event")
        name = event.get("name", "")

        if kind == "on_chat_model_stream":
            chunk = event["data"].get("chunk")
            if chunk and getattr(chunk, "content", None):
                ai_tokens.append(chunk.content)

        if kind == "on_tool_start" and "search_knowledge_base" in name:
            tool_calls.append(event.get("data", {}).get("input"))
            print(f"[tool_start] input={event.get('data', {}).get('input')}")

        if kind == "on_tool_end" and "search_knowledge_base" in name:
            output = event.get("data", {}).get("output")
            tool_results.append(output)
            preview = str(output)[:300] if output else output
            print(f"[tool_end] output_preview={preview!r}")

        if kind == "on_chain_error":
            print(f"[chain_error] {event.get('data')}")

    print(f"\nTool calls count: {len(tool_calls)}")
    print(f"Tool results count: {len(tool_results)}")
    print(f"Streamed AI token chunks: {len(ai_tokens)}")
    if ai_tokens:
        print("AI response preview:", "".join(ai_tokens)[:500])


async def test_get_knowledge_sse():
    server = RagServerTool()
    request = ChatRequest(message=QUERY, thread_id="diag_sse_01")

    print("\n=== getKnowledge SSE (current impl) ===")
    events = []
    async for line in server.getKnowledge(request):
        events.append(line)
        print(line.strip())
        if len(events) >= 20:
            break

    print(f"SSE events received: {len(events)}")


def test_search_direct():
    print("\n=== search_knowledge_base direct ===")
    server = RagServerTool()
    try:
        result = server.search_knowledge_base("langchain agent 是什么")
        print(f"result_len={len(result)}")
        print(f"preview={result[:300]!r}")
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")


async def main():
    test_search_direct()
    await test_agent_events()
    await test_get_knowledge_sse()


if __name__ == "__main__":
    asyncio.run(main())
