import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple,Optional

import bm25s
import jieba
import torch
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from sentence_transformers import CrossEncoder
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from sqlalchemy.engine import default

from config.Settings import Settings
from schemas.chat import ChatRequest
from app.ragServer.chat_history import ChatHistoryReader, CHAT_DB_PATH
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain.messages import AIMessage

import time


load_dotenv()


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

SYSTEM_PROMPT = (
    "你是一个知识库检索 LangChain 专家，精通 Python。"
    "用户消息中会附带「参考资料」，请优先基于资料回答。"
    "只有在资料明显不足时，才调用 search_knowledge_base 工具补充检索。"
    "回答应清晰、简洁、准确。"
)


# 1. 连接数据库（如果文件不存在会自动创建）
#连接sqlite
checkpointer_dir = Path(__file__).resolve().parent.parent.parent / "chatdb/my_app.db"
conn = sqlite3.connect(checkpointer_dir,check_same_thread=False)
#初始化checkpinter
checkpointer = SqliteSaver(conn)
#自动创建
checkpointer.setup()


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
        self.checkpointer_dir = Path(__file__).resolve().parent.parent.parent / "chatdb/my_app.db"





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

    def bm25_search(sef,query: str, k: int = 3) -> List[Tuple[Dict, float]]:
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

    def cross_encoder_rerank(self,query: str, docs: List[Dict], top_k: int = 3):
        # 打分
        scores = self.Reranker_model.predict([(query, doc["content"]) for doc in docs])

        # 分数写入文档dict
        for i, s in enumerate(scores):
            docs[i]["score"] = s

        # 排序
        sorted_docs = sorted(docs, key=lambda x: x["score"], reverse=True)

        # 去掉0分以下的
        positive_docs = [doc for doc in sorted_docs if doc["score"] > 0]

        return positive_docs[0: min(top_k, len(positive_docs))]

    @tool
    def search_knowledge_base(self,query):
        """
        query:用户传入的提示词
        """
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
    #async def getKnowledge(self):

        #query = "你好langchain"
        #search_data = self.search_knowledge_base(query)
        query = request.message
        thread_id = request.thread_id or "project_01"



        rag_agent = create_agent(
            model=self.llm,
            tools=[self.search_knowledge_base],
            system_prompt=SYSTEM_PROMPT,
            checkpointer=checkpointer
        )

        # query = "学习Python"
        # thread_id = "project_01"
        config = {"configurable": {"thread_id": thread_id}}

        try:
            # 注意：这里使用 astream 而不是 stream，以匹配异步环境
            async for chunk, metadata in rag_agent.astream(
                    {"messages": [{"role": "user", "content": query}]},
                    stream_mode="messages",
                    config=config,
            ):
                if isinstance(chunk, AIMessage) and chunk.content:
                    yield _sse({"type": "token", "content": chunk.content})
        except Exception as exc:
            yield _sse({"type": "error", "content": str(exc)})

        # response = rag_agent.stream(
        #     {"messages": [{"role": "user", "content": query}]},
        #     stream_mode="messages",
        #     config=config,
        # )
        #
        # try:
        #     for chunk, metadata in response:
        #         if isinstance(chunk, AIMessage) and chunk.content:
        #             print(chunk.content, end="")
        #             if chunk.content:
        #                 yield _sse({"type": "token", "content": chunk.content})
        #
        # except Exception as exc:
        #     yield _sse({"type": "error", "content": str(exc)})

    #     yield _sse({"type": "error", "content": str(exc)})
        # try:
        #     async for event in rag_agent.astream_events(
        #             {"messages": [HumanMessage(content=query)]},
        #             config=config,
        #             version="v2",
        #     ):
        #         if event["event"] != "on_chat_model_stream":
        #             continue
        #
        #         chunk = event["data"]["chunk"]
        #         content = _extract_text(getattr(chunk, "content", ""))
        #         if content:
        #             yield _sse({"type": "token", "content": content})
        #
        #     yield _sse({"type": "done"})
        # except Exception as exc:
        #     yield _sse({"type": "error", "content": str(exc)})



    def test(self):
        import openvino as ov

        core = ov.Core()
        # 打印所有可用的推理设备
        print("打印")
        print("可用设备:", core.available_devices)

if __name__ == "__main__":
    import asyncio

    server = RagServerTool()
    asyncio.run(server.getKnowledge())
    #asyncio.run(server.test())











