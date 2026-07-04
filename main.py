
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat,sessions
from common.logger import setup_logging
from app.emailAgent.emailAgent import email_agent
from contextlib import asynccontextmanager
from common.logger import logger
# 初始化日志配置
setup_logging()


# 使用 lifespan 管理 checkpointer 生命周期
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化 checkpointer 和 agent
    try:
        logger.info("开始初始化 Email Agent...")
        await email_agent.init()
        logger.info("Email Agent 初始化成功！")
    except Exception as e:
        logger.error(f"Email Agent 初始化失败: {e}", exc_info=True)
        raise e  # 抛出异常让 FastAPI 知道启动失败了!
    yield

    # --- 关闭阶段 ---
    try:
        await email_agent.close()
        logger.info("Email Agent 连接已关闭")
    except Exception as e:
        logger.error(f"关闭 Email Agent 失败: {e}")
app = FastAPI(
    title="Travel API",
    description="私人旅行聊天室&&RAG检索&&emailAgent",
    version="0.1.0",
    lifespan=lifespan
)

# 1. 配置跨域资源共享 (CORS)
# 插件开发中，由于请求来自浏览器扩展环境，必须正确配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议指定插件的 ID 或具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2.挂载路由
app.include_router(chat.router, prefix="/api", tags=["会话"])
app.include_router(rag.router, prefix="/api", tags=["RAG检索"])
app.include_router(sessions.router, prefix="/api/v1", tags=["emailAgent会话管理"])

if __name__ == "__main__":
    import uvicorn
    # 启动命令：python -m app.main
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)