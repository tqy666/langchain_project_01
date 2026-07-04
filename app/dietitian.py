import asyncio
import json

from langchain.agents import AgentState, create_agent
from langchain.messages import HumanMessage

from dietitian_detail import search_flights, search_venues, update_state
from config.Settings import Settings



SYSTEM_PROMPT = """
You are a wedding coordinator. Delegate tasks to your specialists for flights, venues and playlists.
First find all the information you need to update the state. Once that is done you can delegate the tasks.
Once you have received their answers, coordinate the perfect wedding for me.
"""


class WeddingState(AgentState):
    origin: str
    destination: str
    guest_count: str
    genre: str


coordinator = create_agent(
    model=Settings.Llm_qwen,
    tools=[search_flights, search_venues, update_state],
    state_schema=WeddingState,
    system_prompt=SYSTEM_PROMPT,
)


def _extract_text(content) -> str:
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


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def getTravel(text: str = "我来自伦敦，我想在巴黎举办一场100人的婚礼，爵士风格的"):
    """
    流式询问旅行计划
    :param text: 用户说的话
    :yield: SSE 格式字符串
    """
    try:
        async for event in coordinator.astream_events(
            {"messages": [HumanMessage(content=text)]},
            version="v2",
        ):
            #print(f"当前event的数据：{event}")
            kind = event["event"]
            #print(f"每个事件的event数据：{kind}")
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                content = _extract_text(getattr(chunk, "content", ""))
                if content:
                    result = _sse({"type": "token", "content": content})
                    #print(f"打印获取到的on_chat_model_stream数据：{result}")
                    yield result
            # elif kind == "on_tool_start":
            #     tool_start = _sse({"type": "tool_start", "name": event.get("name", "")})
            #     print(f"打印获取到的on_tool_start数据：{tool_start}")
            #     yield tool_start
            # elif kind == "on_tool_end":
            #     tool_end = _sse({"type": "tool_end", "name": event.get("name", "")})
            #     print(f"打印获取到的on_tool_end数据：{tool_end}")
            #     yield tool_end

        #yield _sse({"type": "done"})
    except Exception as exc:
        yield _sse({"type": "error", "content": str(exc)})


async def _run_demo():
    async for chunk in getTravel():
        print(chunk, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(_run_demo())
