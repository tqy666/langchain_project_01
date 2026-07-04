import socket
from turtledemo import __main__
import asyncio
from langchain_core.tools import tool

async def main():



    """自动获取当前运行环境的局域网IP地址。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
    except Exception:
        local_ip = '127.0.0.1'
    finally:
        s.close()
    print(f"打印：{local_ip}")
    return local_ip

# 运行
if __name__ == "__main__":
    asyncio.run(main())