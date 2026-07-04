import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

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

from config.Settings import Settings
from schemas.chat import ChatRequest
from app.ragServer.chat_history import ChatHistoryReader, CHAT_DB_PATH
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

load_dotenv()
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

SYSTEM_PROMPT = (
    "你是一个知识库检索 LangChain 专家，精通 Python。"
    "用户消息中会附带「参考资料」，请优先基于资料回答。"
    "只有在资料明显不足时，才调用 search_knowledge_base 工具补充检索。"
    "回答应清晰、简洁、准确。"
)

TOP_K = 3
RRF_K = 60
MAX_DOC_CHARS = 800
USE_RERANKER = os.getenv("RAG_USE_RERANKER", "0") == "1"


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



class RagServer:
    def __init__(self):
        self.collection_name = "nomic_embeddings_v3"
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vector_db = Chroma(
            persist_directory=str(Settings.Path),
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
        )
        self.bm25_dir = Path(__file__).resolve().parent.parent.parent / "db/my_index.bm25"
        self.bm25_dir.parent.mkdir(parents=True, exist_ok=True)

        self.chat_db_path = CHAT_DB_PATH
        self.chat_db_path.parent.mkdir(parents=True, exist_ok=True)

        self._checkpointer = None
        self._checkpointer_ctx = None
        self.agent = None
        self._bm25_retriever = None
        self._reranker = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.history_reader = ChatHistoryReader(self.chat_db_path, self.executor)
        self.search_tool = self._create_search_tool()

        if self.bm25_dir.exists():
            self._get_bm25_retriever()

    async def _ensure_checkpointer(self):
        if self._checkpointer is not None:
            return self._checkpointer

        self._checkpointer_ctx = AsyncSqliteSaver.from_conn_string(str(self.chat_db_path))
        self._checkpointer = await self._checkpointer_ctx.__aenter__()
        self.history_reader._checkpointer = self._checkpointer
        self.history_reader._checkpointer_ctx = self._checkpointer_ctx
        return self._checkpointer

    async def _ensure_agent(self):
        """懒加载 Agent，复用已打开的 SQLite checkpointer。"""
        if self.agent is not None:
            return

        await self._ensure_checkpointer()

        self.agent = create_agent(
            model=Settings.Llm_qwen,
            tools=[self.search_tool],
            system_prompt=SYSTEM_PROMPT,
            checkpointer=self._checkpointer,
        )



    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder(
                "Qwen/Qwen3-Reranker-0.6B",
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        return self._reranker

    def _create_search_tool(self):
        server = self

        @tool
        def search_knowledge_base(query: str) -> str:
            """搜索知识库，获取 LangChain、Python 相关的技术文档与框架说明。"""
            return server.search(query)

        return search_knowledge_base

    def _doc_to_dict(self, doc_id: str, content: str) -> Dict[str, str]:
        text = content[:MAX_DOC_CHARS] if len(content) > MAX_DOC_CHARS else content
        return {"id": doc_id, "content": text}

    def _ensure_bm25_index(self) -> bool:
        """从 Chroma 全量语料构建 BM25 索引（仅首次）。"""
        if self.bm25_dir.exists():
            return True

        data = self.vector_db.get()
        documents = data.get("documents") or []
        ids = data.get("ids") or []
        if not documents:
            return False

        metadata_corpus = [
            self._doc_to_dict(doc_id, content)
            for doc_id, content in zip(ids, documents)
        ]
        self.create_bm25_index(metadata_corpus)
        self._bm25_retriever = None
        return self.bm25_dir.exists()

    def create_bm25_index(self, metadata_corpus: List[Dict], k1: float = 1.5, b: float = 0.75):
        if self.bm25_dir.exists():
            return

        corpus_tokens = [jieba.lcut(doc["content"]) for doc in metadata_corpus]
        retriever = bm25s.BM25(k1=k1, b=b, corpus=metadata_corpus)
        retriever.index(corpus_tokens)
        retriever.save(self.bm25_dir)

    def _get_bm25_retriever(self):
        if self._bm25_retriever is None:
            self._bm25_retriever = bm25s.BM25.load(self.bm25_dir, load_corpus=True)
        return self._bm25_retriever

    def bm25_search(self, query: str, k: int = TOP_K) -> List[Tuple[Dict, float]]:
        query_tokens = [jieba.lcut(query)]
        retriever = self._get_bm25_retriever()
        results, scores = retriever.retrieve(query_tokens, k=k)
        return [(results[0, i], scores[0, i]) for i in range(results.shape[1])]

    @staticmethod
    def reciprocal_rank_fusion(ranked_lists: List[List[Dict]], k: int = RRF_K) -> List[Dict]:
        rrf_scores: Dict[str, float] = {}
        results: Dict[str, Dict] = {}

        for rank_list in ranked_lists:
            for rank, doc in enumerate(rank_list, start=1):
                doc_id = doc["id"]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank)
                results[doc_id] = doc

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [results[doc_id] for doc_id, _ in sorted_docs]

    def cross_encoder_rerank(self, query: str, docs: List[Dict], top_k: int = TOP_K) -> List[Dict]:
        if not docs:
            return []

        scores = self.reranker.predict([(query, doc["content"]) for doc in docs])
        for doc, score in zip(docs, scores):
            doc["score"] = float(score)

        sorted_docs = sorted(docs, key=lambda x: x["score"], reverse=True)
        positive_docs = [doc for doc in sorted_docs if doc["score"] > 0]
        return positive_docs[: min(top_k, len(positive_docs))]

    def _finalize_docs(self, query: str, docs: List[Dict], top_k: int = TOP_K) -> List[Dict]:
        if not docs:
            return []
        if USE_RERANKER:
            return self.cross_encoder_rerank(query, docs[: top_k * 2], top_k)
        return docs[:top_k]

    def search_fast(self, query: str) -> str:
        """轻量检索：向量 + BM25 + RRF，默认跳过重排序模型。"""
        top_k = TOP_K

        vector_results = self.vector_db.similarity_search(query, k=top_k)
        dense_docs = [
            self._doc_to_dict(getattr(doc, "id", str(i)), doc.page_content)
            for i, doc in enumerate(vector_results)
        ]

        if not self._ensure_bm25_index():
            final_docs = self._finalize_docs(query, dense_docs, top_k)
            if not final_docs:
                return "未找到相关文档"
            return "\n\n".join(doc["content"] for doc in final_docs)

        bm25_results = self.bm25_search(query, k=top_k)
        if not dense_docs and not bm25_results:
            return "未找到相关文档"

        bm25_docs = [doc for doc, _ in bm25_results]
        rrf_results = self.reciprocal_rank_fusion([dense_docs, bm25_docs])
        final_docs = self._finalize_docs(query, rrf_results, top_k)
        if not final_docs:
            return "未找到相关文档"

        return "\n\n".join(doc["content"] for doc in final_docs)

    def search(self, query: str) -> str:
        return self.search_fast(query)

    async def _retrieve_context(self, query: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self.search_fast, query)

    async def getKnowledge(self, request: ChatRequest):
        await self._ensure_agent()

        query = request.message
        thread_id = request.thread_id or "default"
        config = {"configurable": {"thread_id": thread_id}}

        context = await self._retrieve_context(query)
        user_message = (
            f"参考资料：\n{context}\n\n"
            f"用户问题：{query}"
        )

        try:
            async for event in self.agent.astream_events(
                {"messages": [HumanMessage(content=user_message)]},
                config=config,
                version="v2",
            ):
                if event["event"] != "on_chat_model_stream":
                    continue

                chunk = event["data"]["chunk"]
                content = _extract_text(getattr(chunk, "content", ""))
                if content:
                    yield _sse({"type": "token", "content": content})

            yield _sse({"type": "done"})
        except Exception as exc:
            yield _sse({"type": "error", "content": str(exc)})

    async def get_chat_history(self,thread_id: str | None = None):
        """从 chatdb/my_app.db 查询聊天记录"""
        import msgpack
        import sqlite3
        path = Path(__file__).resolve().parent.parent.parent / "chatdb/my_app.db"

        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if  thread_id:
            cursor.execute(f"SELECT * FROM checkpoints where thread_id = {thread_id} limit 1")
        else:
            cursor.execute("SELECT * FROM checkpoints")
        rows = cursor.fetchall()  # 获取全部数据
        return

        cursor.close()
        conn.close()


if __name__ == "__main__":
    import asyncio

    # async def _demo():
    #     server = RagServer()
    #     request = ChatRequest(message="python 和 langchain 是什么关系？",thread_id="1")
    #     async for chunk in server.getKnowledge(request):
    #         print(chunk, end="", flush=True)
    #
    # asyncio.run(_demo())

    server = RagServer()
    print(asyncio.run(server.get_chat_history("1")))