from langchain.chat_models import init_chat_model
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv
import os
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import torch
from sentence_transformers import CrossEncoder
load_dotenv()



@dataclass
class Settings:
    Path= Path(__file__).resolve().parent.parent / "chroma_dir"
    Llm_qwen =init_chat_model(
    model="qwen3.6-flash",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)
    Llm_deepseek = init_chat_model( model="deepseek-chat")
    #MongoDB=MongoDBSaver(MongoClient("mongodb://localhost:27017"))
    collection_name = "nomic_embeddings_v3"
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    Chroma=Chroma(
            persist_directory=str(Path),
            embedding_function=embeddings,
            collection_name=collection_name,
        )

    Reranker_model = CrossEncoder(
        "Qwen/Qwen3-Reranker-0.6B",
        device="cuda" if torch.cuda.is_available() else "cpu",

    )


Settings = Settings()






