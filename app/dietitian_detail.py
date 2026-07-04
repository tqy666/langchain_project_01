from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent  # 如果确实存在，保留；否则用 create_react_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from langgraph.types import Command
from typing import Dict, Any
import datetime

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
import asyncio
from config.Settings import Settings
from common.logger import logger

load_dotenv()

# 初始化模型（deepseek 兼容 OpenAI API）



################################################1.获取航班的相关信息####################################################
# --- 新增：预加载 MCP 工具 ---

_travel_agent = None
_travel_agent_lock = asyncio.Lock()


async def _initialize_mcp_tools():
    # MCP客户端，包含Time、Kiwi两个MCP
    client = MultiServerMCPClient(
        {
            "travel_server": {
                "transport": "http",
                "url": "https://mcp.kiwi.com"
            },
            "time": {
                "transport": "stdio",
                "command": "uvx",
                "args": [
                    "mcp-server-time",
                    "--local-timezone=Asia/Shanghai"
                ]
            }
        }
    )

    return await client.get_tools()


def _build_travel_system_prompt() -> str:
    tz = datetime.timezone(datetime.timedelta(hours=8))
    future_time = datetime.datetime.now(tz) + datetime.timedelta(days=3)
    now_str = future_time.isoformat()
    return f"""
       You are a playlist specialist. TIME: {now_str}. Query the sql database and curate the perfect playlist for a wedding given a genre.
       Once you have your playlist, calculate the total duration and cost of the playlist, each song has an associated price.
       If you run into errors when querying the database, try to fix them by making changes to the query.
       Do not come back empty handed, keep trying to query the db until you find a list of songs.
       You may need to make multiple queries to iteratively find the best options.

       """


async def _ensure_travel_agent():
    """懒加载 travel agent，避免启动时连接 MCP 失败导致整个服务无法启动。"""
    global _travel_agent
    if _travel_agent is not None:
        return _travel_agent

    async with _travel_agent_lock:
        if _travel_agent is not None:
            return _travel_agent

        tools = []
        try:
            tools = await _initialize_mcp_tools()
        except Exception as exc:
            logger.warning("MCP 工具加载失败，航班查询将不可用: %s", exc)

        _travel_agent = create_agent(
            model=Settings.Llm_qwen,
            tools=tools,
            system_prompt=_build_travel_system_prompt(),
        )
        return _travel_agent

################################################2.获取场地的相关信息####################################################
# 定义Tavily web_search工具
tavily_client = TavilySearch(max_results=3, topic="general")


@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information"""

    return tavily_client.invoke({"query": query})


# 创建 Venue agent
venue_agent = create_agent(
    model=Settings.Llm_qwen,
    tools=[web_search],
    system_prompt="""
    You are a venue specialist. Search for venues in the desired location, and with the desired capacity.
    You are not allowed to ask any more follow up questions, you must find the best venue options based on the following criteria:
    - Price (lowest)
    - Capacity (exact match)
    - Reviews (highest)
    You may need to make multiple searches to iteratively find the best options.
    """
)


# 定义一个普通工具函数，供协调器调用（注意：不使用 runtime）
##########1.定义航班查询函数############
@tool
async def search_flights(runtime: ToolRuntime) -> str:
    """Travel agent searches for flights to the desired destination wedding location."""

    origin = runtime.state["origin"]
    destination = runtime.state["destination"]
    #departure_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")  # 默认一周后
    travel_agent = await _ensure_travel_agent()
    response = await travel_agent.ainvoke(
        {"messages": [HumanMessage(content=f"Find flights from {origin} to {destination} ")]})
    return response['messages'][-1].content


########2.定义场地寻找函数###############
@tool
def search_venues(runtime: ToolRuntime) -> str:
    """Venue agent chooses the best venue for the given location and capacity."""


    destination = runtime.state["destination"]

    capacity = runtime.state["guest_count"]
    query = f"Find wedding venues in {destination} for {capacity} guests"
    response = venue_agent.invoke({"messages": [HumanMessage(content=query)]})
    return response['messages'][-1].content


#########4.更新state数据

@tool
def update_state(origin: str, destination: str, guest_count: str, genre: str, runtime: ToolRuntime) -> str:
    """Update the state when you know all of the values: origin, destination, guest_count, genre"""

    return Command(update={
        "origin": origin,
        "destination": destination,
        "guest_count": guest_count,
        "messages": [ToolMessage("Successfully updated state", tool_call_id=runtime.tool_call_id)]}
    )




