import json
import os
from pathlib import Path
from typing import Dict, List, Tuple,Optional

import bm25s
import jieba

from dotenv import load_dotenv
from langchain.agents import create_agent

from langchain.tools import tool

from langchain_ollama import OllamaEmbeddings


from config.Settings import Settings
from schemas.chat import ChatRequest
from app.ragServer.chat_history import CHAT_DB_PATH
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain.messages import AIMessage

import time

import sqlite3
import json
import msgpack
import uuid

load_dotenv()


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

SYSTEM_PROMPT = (
    "你是一个知识库检索 LangChain 专家，精通 Python。"
    "用户消息中会附带「参考资料」，请优先基于资料回答。"
    "只有在资料明显不足时，才调用 search_knowledge_base 工具补充检索。"
    "回答应清晰、简洁、准确。"
)


top_k = 3
k1=1.5
b=0.75

bm25_dir = Path(__file__).resolve().parent.parent.parent / "db/my_index.bm25"

SYSTEM_PROMPT = (
    "你是一个知识库检索 LangChain 专家，精通 Python。"
    "用户消息中会附带「参考资料」，请优先基于资料回答。"
    "如果用户问了一个你不确定的问题，或者涉及langchian和Python的问题，你必须使用 search_knowledge_base 工具补充检索。"
    "在引用文档时，要清楚地总结包括内容中的相关上下文"
    "回答应清晰、简洁、准确。"
    "如果获取文档失败，请告诉用户，并以您最好的专家理解继续进行"
)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


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

class RagServerTool:
    def __init__(self):
        self.collection_name = "nomic_embeddings_v3"
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vector_db =Settings.Chroma
        self.bm25_dir = Path(__file__).resolve().parent.parent.parent / "db/my_index.bm25"
        self.llm = Settings.Llm_qwen
        self.Reranker_model =Settings.Reranker_model
        self.chat_db_path = CHAT_DB_PATH
        self.chat_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._checkpointer = None
        self._checkpointer_ctx = None
        self.agent = None
        self.search_tool = self._create_search_tool()

    async def _ensure_checkpointer(self):
        if self._checkpointer is not None:
            return self._checkpointer

        self._checkpointer_ctx = AsyncSqliteSaver.from_conn_string(str(self.chat_db_path))
        self._checkpointer = await self._checkpointer_ctx.__aenter__()
        return self._checkpointer

    async def _ensure_agent(self):
        """懒加载 Agent，复用已打开的 SQLite checkpointer。"""
        if self.agent is not None:
            return

        await self._ensure_checkpointer()
        self.agent = create_agent(
            model=self.llm,
            tools=[self.search_tool],
            system_prompt=SYSTEM_PROMPT,
            checkpointer=self._checkpointer,
        )

    def _create_search_tool(self):
        server = self

        @tool
        def search_knowledge_base(query: str) -> str:
            """搜索知识库，获取 LangChain、Python 相关的技术文档与框架说明。"""
            return server.search_knowledge_base(query)

        return search_knowledge_base





    def create_bm25_index(self,metadata_corpus,index_path=bm25_dir,k1=k1,b=b):
        if os.path.exists(index_path):
            print(f"索引文件已存在，正在加载: {index_path}")

        else:
            print(f"未找到索引文件，正在创建新索引...")
            # 对语料进行分词
            corpus_tokens = [jieba.cut(doc["content"]) for doc in metadata_corpus]
            # 基于原始文档创建索引库，将来检索出来的也是原始文档
            retriever = bm25s.BM25(k1=k1, b=b, corpus=metadata_corpus)
            # 创建索引
            retriever.index(corpus_tokens)
            # 保存到本地
            retriever.save(index_path)
            print(f"索引已保存至: {index_path}")

    def bm25_search(self, query: str, k: int = 3) -> List[Tuple[Dict, float]]:
        # 查询分词
        query_tokens = [jieba.lcut(query)]
        print(f"提示词切分出来{query_tokens}")
        # exit()
        # 检索,返回top-k结果，形式为(docs, scores). docs和scores都是二维数组 shape (n_queries, k).
        bm25_retriever=bm25s.BM25.load(bm25_dir, load_corpus=True)
        results, scores = bm25_retriever.retrieve(query_tokens, k=k)
        print(f"查找到的数据：{results}，查找的分数：{scores}")
        # 封装结果
        return [(results[0, i], scores[0, i]) for i in range(results.shape[1])]


    def reciprocal_rank_fusion(self,ranked_lists: List[List[Dict]], k=60):
        """
            ranked_lists: List[List[Dict]]
                          每个检索器返回的文档dict列表，包含id和content，按排名从高到低。
                          注意：这里不需要score，在 RRF 中只使用排名位置。
            """
        rrf_scores = {}  # 字典，key=doc_id, value=累加的 RRF 贡献
        results = {}  # 字典，key=doc_id, value=doc
        for rank_list in ranked_lists:
            for rank, doc in enumerate(rank_list, start=1):
                # 每个文档在每个列表中独立贡献
                doc_id = doc["id"]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank)
                results[doc_id] = doc
        # 按 RRF 总分降序排序
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        return [results[doc_id] for doc_id, score in sorted_docs]

    def rerank_with_qwen(self,query: str, documents: list[str], top_n: int = 5) -> list[dict]:
        """
        使用千问重排序 API 替换 CrossEncoder
        """
        import dashscope

        # 1. 初始化客户端（建议在环境变量中配置 DASHSCOPE_API_KEY）
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")


        try:
            # 2. 调用云端重排序接口
            response = dashscope.TextReRank.call(
                model="qwen3-rerank",  # 指定千问重排序模型
                query=query,
                documents=documents,
                top_n=top_n  # 让云端直接返回排名前 N 的结果，减少网络传输
            )

            # 3. 处理返回结果
            if response.status_code == 200:
                reranked_results = []
                for result in response.results:
                    reranked_results.append({
                        "document": result.document,
                        "relevance_score": result.relevance_score
                    })
                return reranked_results
            else:
                print(f"重排序 API 调用失败: {response.code} - {response.message}")
                return []

        except Exception as e:
            print(f"重排序请求异常: {e}")
            return []

    def cross_encoder_rerank(self,query: str, docs: List[Dict], top_k: int = 3):
        # 打分(用的本地模型)
        scores = self.Reranker_model.predict([(query, doc["content"]) for doc in docs])
        print(f"重排序打印query：{query}")
        print(f"重排序打印docs：{docs}")
        # #用替换为千问 API 接口
        # top_docs = self.rerank_with_qwen(
        #     query=query,
        #     documents=docs,
        #     top_n=3
        # )
        #
        # # 打印重排序后的结果
        # for i,doc in top_docs:
        #     print(f"分数: {doc['relevance_score']:.4f} | 内容: {doc['document']}")
        #     docs[i]["score"] = doc['relevance_score']


        # # 分数写入文档dict
        for i, s in enumerate(scores):
            docs[i]["score"] = s

        # 排序
        sorted_docs = sorted(docs, key=lambda x: x["score"], reverse=True)

        # 去掉0分以下的
        positive_docs = [doc for doc in sorted_docs if doc["score"] > 0]

        return positive_docs[0: min(top_k, len(positive_docs))]

    def search_knowledge_base(self, query: str) -> str:
        """query: 用户传入的提示词"""
        start = time.perf_counter()
        # 1.稠密检索（向量）
        vector_results = self.vector_db.similarity_search(query, k=top_k)
        print(f"稠密搜索出来的消息{vector_results}")
        end = time.perf_counter()
        print(f"稠密检索完成,耗时:{(end - start) * 1000}ms~")
        start = end
        metadata_corpus = [
            {"id": doc.id, "content": doc.page_content} for doc in vector_results
        ]
        #print(metadata_corpus)

        #判断有没有建BM25索引库，有就不用建，没有就建设一个索引库
        self.create_bm25_index(metadata_corpus)

        #BM25索引库查询
        bm25_results = self.bm25_search(query, k=top_k)
        end = time.perf_counter()
        print(f"稀疏检索完成,耗时:{(end - start) * 1000}ms~")
        print(bm25_results)
        start = end

        if not vector_results or not bm25_results:
            return "未找到相关文档"

        # 3.RRF
        # 3.1.处理成List[dict], dict包含id和content
        bm25_rs = [doc for doc, _ in bm25_results]
        end = time.perf_counter()
        print(f"rrf前置文档处理完成,耗时:{(end - start) * 1000}ms~")
        start = end

        # 3.2.rrf
        rrf_results = self.reciprocal_rank_fusion([metadata_corpus, bm25_rs])
        end = time.perf_counter()
        print(f"rrf完成,耗时:{(end - start) * 1000}ms~")
        start = end

        # 4.cross-encoder精排
        final_docs = self.cross_encoder_rerank(query, rrf_results, top_k)
        end = time.perf_counter()
        print(f"cross-encoder完成,耗时:{(end - start) * 1000}ms~")

        # 5.拼接文档
        docs_content = "\n\n".join(doc["content"] for doc in final_docs)

        return docs_content




    async def getKnowledge(self,request: ChatRequest):
        await self._ensure_agent()

        query = request.message
        thread_id = request.thread_id or str(uuid.uuid4())
        print(f"新生成的thread_id：{thread_id}")
        config = {"configurable": {"thread_id": thread_id}}

        try:
            async for chunk, metadata in self.agent.astream(
                    {"messages": [{"role": "user", "content": query}]},
                    stream_mode="messages",
                    config=config,
            ):
                if isinstance(chunk, AIMessage) and chunk.content:
                    yield _sse({"type": "token", "content": chunk.content})
        except Exception as exc:
            yield _sse({"type": "error", "content": str(exc)})


    def comm_sql_data(self, thread_id:str):

        conn = sqlite3.connect(self.chat_db_path)
        cursor = conn.cursor()

        cursor.execute(
                "SELECT  checkpoint FROM checkpoints WHERE  thread_id = ?   ",
                (thread_id,)
            )

        rows = cursor.fetchall()
        return rows

    # 聊天记录列表
    def read_chat_history_fixed(self):
        limit: int = 20
        from langgraph.checkpoint.sqlite import SqliteSaver
        conn = sqlite3.connect(self.chat_db_path)
        cursor = conn.cursor()

        # 1. 获取最近的 thread_id 列表 (利用 checkpoint_id 排序)
        query = """
                SELECT thread_id
                FROM checkpoints
                GROUP BY thread_id
                ORDER BY MAX(checkpoint_id) DESC LIMIT ? \
                """
        cursor.execute(query, (limit,))
        threads = [row[0] for row in cursor.fetchall()]

        chat_list = []

        # 2. 遍历 thread_id，获取具体详情
        with SqliteSaver.from_conn_string(self.chat_db_path) as saver:
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

    # 聊天详情
    def get_chat_info(self,thread_id):

        rows = self.comm_sql_data(thread_id)
        #print(f"sql获取到的数据：{rows}")
        print(f"sql获取到的数据总共有：{len(rows)}条")
        chat_info = []
        for row in rows:
            binary_data = row[0]
            if not binary_data:
                continue
        try:
            # 1. 使用 msgpack 解码
            data = msgpack.unpackb(binary_data, raw=False)
            print(f"原始数据：{data}")
            # 2. 获取 messages 字段
            channel_values = data.get("channel_values", {})
            print(f"打印channel_values:{channel_values}")
            messages = channel_values.get("messages", [])
            print(f"获取messages：{messages}")
            print(isinstance(messages, list))
            if isinstance(messages, list):

                # print(f"解析{msgpack.unpackb(messages[0][1], raw=False)}")

                for msg in messages:
                    content_raw = msg[1]
                    if isinstance(content_raw, bytes):
                        content = msgpack.unpackb(content_raw, raw=False)
                        print(content)
                        if content[2]['content']:
                            chat_info.append({"role": content[1], "content": content[2]['content']})

                #print(chat_info)
                print(f"消息详情有多少条数据{len(chat_info)}")

        except Exception as e:
            print(f"解析某条记录出错: {e}")
        return chat_info


if __name__ == "__main__":
    import asyncio

    db_path = "D:/newProject/langchain_v3/chatdb/my_app.db"
    thread_id = "project_01"
    server = RagServerTool()
    param = server.get_chat_info('12fabc8b-45e9-45dd-a582-0b5a2be35fa2')
    print(param)
    #asyncio.run(server.getKnowledge("学习langchain"))












